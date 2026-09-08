# N14R2 portable execution environment

This is a fresh experiment, not a continuation of the Windows N14R run.
The scientific engine and feature bytes remain unchanged. The prospective
execution environment is Linux x86_64 / Python 3.12.3 / PyTorch 2.11.0+cu128 /
CUDA 12.8 / cuDNN 91900, with four distinct RTX 4090 GPUs.

`requirements-lock.txt` is compiled for Linux / Python 3.12 from
`requirements.in`, with every resolved distribution version and hash pinned.
Installation requires hash verification in an isolated virtual environment.
The base image is `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404`;
its system Torch 2.8 is not the experiment interpreter.

The disposable interpreter lives at `/root/n14r2-env` on container-local disk.
Code, input feature cache, immutable attempts, logs and receipts live on the
persistent `/workspace` mount. Rebuilding an interpreter is allowed only from
the same lock and followed by the frozen runtime/health checks. No dependency
upgrades are permitted after preregistration.

Before formal fitting, four outcome-free environment receipts are collected
under `environment/`, then their exact hashes, GPU UUIDs and driver versions
are bound to node0 through node3 in `ENVIRONMENT_BINDINGS.json`. These files
are included in PINS and the public preregistration commit. Whole draws remain
on one bound node; node number equals draw ID modulo four.

One Torch intra-op and inter-op CPU thread per worker, eight independent
processes per GPU. Each fit reseeds the unchanged engine. No thread-shared RNG,
DDP, gradient accumulation, mixed precision, altered model, batch size or
early-stopping rule is introduced. GPU failures pause dispatch and require a
healthy fresh process before any eligible retry.

The inputs are the existing frozen CREMA-D log-mel NPZ and metadata, not fitted
results. Reusing these deterministic input features is not reusing N14R draws,
checkpoints, hyperparameter selections or outcomes. No old fitted output is
uploaded into this run's input or output directories.

## Operational cost control

The expected approximately USD 16 is an estimate, not a quote or hard cap.
The agent applies an estimated USD 25 pause-and-stop guard for this run,
including setup and a storage allowance. This is not an outcome-dependent
scientific stopping rule; incompleteness cannot produce a scientific verdict.
The Runpod control API used here has no verified native hard TTL. Stopping GPU
billing depends on the external monitor being online and the control plane
responding. Stopped persistent disks still incur storage costs. Pods are
deleted only after their new-run artifacts are copied out and hash-verified.
