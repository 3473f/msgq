import ctypes
import multiprocessing as mp
import unittest

from msgq.visionipc import VisionIpcClient, VisionIpcServer, visionbuf_backend_name

WIDTH, HEIGHT = 128, 64
NAME = "cuda_test"


def _libcuda():
  try:
    return ctypes.CDLL("libcuda.so.1")
  except OSError:
    return None


def _cuda_backend_active():
  return visionbuf_backend_name() == "cuda" and _libcuda() is not None


def _device_read(cuda, ptr, n):
  host = (ctypes.c_ubyte * n)()
  assert cuda.cuMemcpyDtoH_v2(ctypes.byref(host), ctypes.c_uint64(ptr), ctypes.c_size_t(n)) == 0
  return bytes(host)


def _device_write(cuda, ptr, data):
  buf = (ctypes.c_ubyte * len(data)).from_buffer_copy(data)
  assert cuda.cuMemcpyHtoD_v2(ctypes.c_uint64(ptr), ctypes.byref(buf), ctypes.c_size_t(len(data))) == 0
  assert cuda.cuCtxSynchronize() == 0


def _client_proc(conn):
  """Receives frames, checks the CPU view, then writes through the device pointer."""
  cuda = _libcuda()
  cuda.cuInit(0)
  client = VisionIpcClient(NAME, 0, False)
  assert client.connect(True)  # import() retains the primary context on this thread
  conn.send("ready")

  buf = client.recv(5000)
  assert buf is not None and buf.cuda_ptr is not None
  n = client.buffer_len
  expected = bytes((i * 7 + 3) & 0xFF for i in range(n))

  # CPU mapping sees what the server's CPU write put there
  assert bytes(buf.data) == expected, "CPU view mismatch"
  # device pointer sees the same bytes
  assert _device_read(cuda, buf.cuda_ptr, n) == expected, "device view mismatch"

  # a device-side write is visible through the CPU mapping
  pattern = bytes((i * 11 + 5) & 0xFF for i in range(n))
  _device_write(cuda, buf.cuda_ptr, pattern)
  assert bytes(buf.data) == pattern, "device write not visible on CPU"
  conn.send("done")


@unittest.skipUnless(_cuda_backend_active(), "visionipc CUDA backend not active")
class TestVisionIpcCuda(unittest.TestCase):
  def test_cross_process_cpu_and_device_views(self):
    server = VisionIpcServer(NAME)
    server.create_buffers(0, 2, WIDTH, HEIGHT)
    server.start_listener()

    parent, child = mp.get_context("spawn").Pipe()
    proc = mp.get_context("spawn").Process(target=_client_proc, args=(child,))
    proc.start()
    try:
      assert parent.poll(30) and parent.recv() == "ready"
      n = WIDTH * HEIGHT * 3 // 2
      server.send(0, bytes((i * 7 + 3) & 0xFF for i in range(n)), frame_id=7)
      assert parent.poll(30) and parent.recv() == "done"
    finally:
      proc.join(30)
      if proc.is_alive():
        proc.kill()
    self.assertEqual(proc.exitcode, 0)


if __name__ == "__main__":
  unittest.main()
