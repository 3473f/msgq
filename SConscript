import os

Import('env', 'envCython', 'common', 'arch')


visionipc_dir = Dir('msgq/visionipc')

# Build msgq
msgq_objects = env.SharedObject([
  'msgq/ipc.cc',
  'msgq/event.cc',
  'msgq/impl_msgq.cc',
  'msgq/impl_fake.cc',
  'msgq/msgq.cc',
])
msgq = env.Library('msgq', msgq_objects)
msgq_python = envCython.Program('msgq/ipc_pyx.so', 'msgq/ipc_pyx.pyx', LIBS=envCython["LIBS"]+[msgq]+common)

# Build Vision IPC
vipc_files = ['visionipc.cc', 'visionipc_server.cc', 'visionipc_client.cc']

# Backend: VISIONBUF_BACKEND=shm|ion|cuda overrides auto-detection.
# Auto: ion if /dev/ion exists, cuda on Jetson (Tegra) with a CUDA toolkit, else shm.
cuda_home = next((p for p in [os.environ.get('CUDA_HOME'), '/usr/local/cuda'] if p and os.path.isdir(f'{p}/include')), None)
backend = os.environ.get('VISIONBUF_BACKEND')
if backend is None:
  if File('/dev/ion').exists():
    backend = 'ion'
  elif os.path.exists('/etc/nv_tegra_release') and cuda_home:
    backend = 'cuda'
  else:
    backend = 'shm'
assert backend in ('shm', 'ion', 'cuda'), f"unknown VISIONBUF_BACKEND: {backend}"
print(f"visionipc backend: {backend}")

vipc_extra_libs = []
vipc_env = env
if backend == 'cuda':
  assert cuda_home, "VISIONBUF_BACKEND=cuda needs a CUDA toolkit (set CUDA_HOME)"
  cuda_libdirs = [f'{cuda_home}/lib64', f'{cuda_home}/targets/{arch}-linux/lib', f'{cuda_home}/lib64/stubs']
  vipc_env = env.Clone()
  vipc_env.Append(CPPPATH=[f'{cuda_home}/include'], LIBPATH=cuda_libdirs, RPATH=[f'{cuda_home}/lib64'])
  vipc_extra_libs = ['cuda']
  envCython.Append(LIBPATH=cuda_libdirs, RPATH=[f'{cuda_home}/lib64'])

vipc_files += {'ion': ['visionbuf_ion.cc'], 'cuda': ['visionbuf_cuda.cc'], 'shm': ['visionbuf.cc']}[backend]
vipc_sources = [f'{visionipc_dir.abspath}/{f}' for f in vipc_files]

vipc_env = vipc_env.Clone()
vipc_env.Append(CPPDEFINES=[('VISIONBUF_BACKEND_NAME', f'\\"{backend}\\"')])
vipc_objects = vipc_env.SharedObject(vipc_sources)
visionipc = env.Library('visionipc', vipc_objects)


vipc_libs = envCython["LIBS"] + [visionipc, msgq] + vipc_extra_libs + common
envCython.Program(f'{visionipc_dir.abspath}/visionipc_pyx.so', f'{visionipc_dir.abspath}/visionipc_pyx.pyx',
                  LIBS=vipc_libs)

if GetOption('extras'):
  env.Program('msgq/test_runner', ['msgq/msgq_tests.cc'], LIBS=[msgq]+common)

Export('visionipc', 'msgq', 'msgq_python')
