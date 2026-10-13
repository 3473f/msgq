#include "msgq/visionipc/visionbuf.h"

#include <cassert>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <cuda.h>
#include <fcntl.h>
#include <linux/dma-buf.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <unistd.h>

// CUDA VMM backend. Each buffer is a cuMemCreate allocation exported as a POSIX
// fd (a dma-buf on Jetson). The fd travels over SCM_RIGHTS like every other
// backend, the receiver imports it into CUDA, and both sides mmap the same fd
// to get the CPU address in `addr`.

static void cu_check(CUresult r, const char *what) {
  if (r == CUDA_SUCCESS) return;
  const char *msg = "unknown";
  cuGetErrorString(r, &msg);
  fprintf(stderr, "visionbuf_cuda: %s failed: %s (%d)\n", what, msg, (int)r);
  abort();
}

// Make sure the calling thread has a CUDA context, reusing the primary context
// so it is shared with any CUDA/TensorRT code in the same process.
static CUdevice ensure_context() {
  cu_check(cuInit(0), "cuInit");

  CUcontext ctx = nullptr;
  cu_check(cuCtxGetCurrent(&ctx), "cuCtxGetCurrent");
  CUdevice dev;
  if (ctx != nullptr) {
    cu_check(cuCtxGetDevice(&dev), "cuCtxGetDevice");
    return dev;
  }

  const char *env = getenv("VISIONIPC_CUDA_DEVICE");
  cu_check(cuDeviceGet(&dev, env ? atoi(env) : 0), "cuDeviceGet");
  cu_check(cuDevicePrimaryCtxRetain(&ctx, dev), "cuDevicePrimaryCtxRetain");
  cu_check(cuCtxSetCurrent(ctx), "cuCtxSetCurrent");
  return dev;
}

static CUmemAllocationProp alloc_prop(CUdevice dev) {
  CUmemAllocationProp prop = {};
  prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
  prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  prop.location.id = (int)dev;
  prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
  return prop;
}

static void map_for_device(CUdevice dev, size_t size, CUmemGenericAllocationHandle handle, CUdeviceptr *ptr) {
  cu_check(cuMemAddressReserve(ptr, size, 0, 0, 0), "cuMemAddressReserve");
  cu_check(cuMemMap(*ptr, size, 0, handle, 0), "cuMemMap");

  CUmemAccessDesc access = {};
  access.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  access.location.id = (int)dev;
  access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
  cu_check(cuMemSetAccess(*ptr, size, &access, 1), "cuMemSetAccess");
}

static void *map_for_cpu(int fd, size_t size) {
  void *addr = mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
  if (addr == MAP_FAILED) {
    fprintf(stderr, "visionbuf_cuda: mmap of exported fd failed: %s\n", strerror(errno));
    abort();
  }
  return addr;
}

void VisionBuf::allocate(size_t length) {
  CUdevice dev = ensure_context();

  CUmemAllocationProp prop = alloc_prop(dev);
  size_t granularity = 0;
  cu_check(cuMemGetAllocationGranularity(&granularity, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM),
           "cuMemGetAllocationGranularity");

  size_t size = length + sizeof(uint64_t);
  size = ((size + granularity - 1) / granularity) * granularity;

  CUmemGenericAllocationHandle handle;
  cu_check(cuMemCreate(&handle, size, &prop, 0), "cuMemCreate");

  int export_fd = -1;
  cu_check(cuMemExportToShareableHandle(&export_fd, handle, CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR, 0),
           "cuMemExportToShareableHandle");

  CUdeviceptr dptr = 0;
  map_for_device(dev, size, handle, &dptr);

  this->addr = map_for_cpu(export_fd, size);
  memset(this->addr, 0, size);

  this->len = length;
  this->mmap_len = size;
  this->fd = export_fd;
  this->cuda_ptr = (uint64_t)dptr;
  this->cuda_handle = (uint64_t)handle;
  this->frame_id = (uint64_t*)((uint8_t*)this->addr + this->len);
}

void VisionBuf::import() {
  assert(this->fd >= 0);
  CUdevice dev = ensure_context();

  // Import from a dup so the original fd stays valid for the CPU mmap.
  int import_fd = dup(this->fd);
  assert(import_fd >= 0);

  CUmemGenericAllocationHandle handle;
  cu_check(cuMemImportFromShareableHandle(&handle, (void *)(intptr_t)import_fd,
                                          CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR),
           "cuMemImportFromShareableHandle");
  close(import_fd);

  CUdeviceptr dptr = 0;
  map_for_device(dev, this->mmap_len, handle, &dptr);

  this->addr = map_for_cpu(this->fd, this->mmap_len);
  this->cuda_ptr = (uint64_t)dptr;
  this->cuda_handle = (uint64_t)handle;
  this->frame_id = (uint64_t*)((uint8_t*)this->addr + this->len);
}

void VisionBuf::init_yuv(size_t init_width, size_t init_height, size_t init_stride, size_t init_uv_offset) {
  this->width = init_width;
  this->height = init_height;
  this->stride = init_stride;
  this->uv_offset = init_uv_offset;

  this->y = (uint8_t *)this->addr;
  this->uv = this->y + this->uv_offset;
}

// One-shot cache maintenance for the CPU mapping. Not every dma-buf exporter
// implements DMA_BUF_IOCTL_SYNC; if this one doesn't, the mapping is already
// coherent in practice (verified on Orin) and we skip it from then on.
static int dmabuf_sync(int fd, uint64_t rw) {
  static bool supported = true;
  if (!supported) return 0;

  struct dma_buf_sync sync = {};
  sync.flags = DMA_BUF_SYNC_START | rw;
  if (ioctl(fd, DMA_BUF_IOCTL_SYNC, &sync) != 0) {
    if (errno == ENOTTY || errno == ENOSYS || errno == EINVAL || errno == EOPNOTSUPP) {
      supported = false;
      return 0;
    }
    return -1;
  }
  sync.flags = DMA_BUF_SYNC_END | rw;
  return ioctl(fd, DMA_BUF_IOCTL_SYNC, &sync);
}

int VisionBuf::sync(int dir) {
  assert(dir == VISIONBUF_SYNC_FROM_DEVICE || dir == VISIONBUF_SYNC_TO_DEVICE);

  if (dir == VISIONBUF_SYNC_FROM_DEVICE) {
    // device wrote, CPU is about to read: wait for GPU work, then invalidate
    ensure_context();
    if (cuCtxSynchronize() != CUDA_SUCCESS) return -1;
    return dmabuf_sync(this->fd, DMA_BUF_SYNC_READ);
  }
  // CPU wrote, device is about to read: clean CPU caches
  return dmabuf_sync(this->fd, DMA_BUF_SYNC_WRITE);
}

int VisionBuf::free() {
  int ret = 0;
  if (munmap(this->addr, this->mmap_len) != 0) ret = -1;

  if (this->cuda_ptr != 0) {
    ensure_context();
    if (cuMemUnmap((CUdeviceptr)this->cuda_ptr, this->mmap_len) != CUDA_SUCCESS) ret = -1;
    if (cuMemAddressFree((CUdeviceptr)this->cuda_ptr, this->mmap_len) != CUDA_SUCCESS) ret = -1;
    if (cuMemRelease((CUmemGenericAllocationHandle)this->cuda_handle) != CUDA_SUCCESS) ret = -1;
    this->cuda_ptr = 0;
    this->cuda_handle = 0;
  }

  if (close(this->fd) != 0) ret = -1;
  return ret;
}

uint64_t VisionBuf::get_frame_id() {
  return *frame_id;
}

void VisionBuf::set_frame_id(uint64_t id) {
  *frame_id = id;
}
