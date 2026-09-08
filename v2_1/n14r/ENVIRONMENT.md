# N14R execution environment

This environment was recorded before any N14R fit.  The exact versions of the
58 locked Python distributions are in `requirements-lock.txt`; unrelated extra
distributions in the environment are neither required nor rejected.

The only authorized execution interpreter on the registered host is
`E:\科研\SER\ser_gpu\Scripts\python.exe`. The runner compares every locked
distribution with `requirements-lock.txt` and separately enforces the core
Python/NumPy/SciPy/scikit-learn/PyTorch/CUDA/cuDNN/librosa/GPU values in
`spec.py`; any mismatch stops before a fit. The run-level
`execution_environment.json` records the resolved executable, platform string,
locked package versions, and enforced core values and is the final machine
receipt. The OS build, nominal VRAM, and driver below are descriptive host
details; they are not separately enforced or recorded as dedicated fields.

- Python: 3.13.9, Anaconda build, MSC v.1929 64 bit
- OS: Windows 11, build 26200, 64 bit
- GPU: NVIDIA GeForce RTX 5070, 12,227 MiB
- NVIDIA driver: 610.62
- PyTorch: 2.11.0+cu128
- CUDA runtime reported by PyTorch: 12.8
- cuDNN reported by PyTorch: 9.19.0
- NumPy: 2.4.6
- SciPy: 1.18.0
- scikit-learn: 1.9.0
- librosa: 0.11.0

The CUDA wheels in this environment are the `cu128` builds; recreating it with
pip requires the PyTorch CUDA 12.8 wheel index rather than the default CPU
index. The frozen run reuses the already verified environment instead of
reinstalling packages after registration.

The inherited frozen training engine calls `set_seed` before construction and
training, enables `torch.backends.cudnn.deterministic`, and disables cuDNN
benchmarking.  It does not call `torch.use_deterministic_algorithms`; therefore
the study claims seeded repeatability, not guaranteed bitwise identity on a
different GPU/driver stack.
