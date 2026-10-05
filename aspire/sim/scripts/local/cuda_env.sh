#!/usr/bin/env bash
# Source (don't execute) to build CUDA extensions for a Blackwell GPU (RTX 50xx, sm_120)
# without touching the system CUDA: nvcc 12.8 + a pinned host gcc live in an isolated conda env.
#
# One-time toolkit env:
#   conda create -y -p "$ASPIRE_CUDA_ENV" -c nvidia/label/cuda-12.8.1 -c conda-forge \
#     cuda-nvcc=12.8 cuda-cudart-dev cuda-libraries-dev cuda-nvtx cuda-profiler-api \
#     gcc_linux-64=13 gxx_linux-64=13
export ASPIRE_CUDA_ENV="${ASPIRE_CUDA_ENV:-$HOME/miniconda3/envs/aspire-cuda128}"
export CUDA_HOME="$ASPIRE_CUDA_ENV"
export PATH="$CUDA_HOME/bin:$PATH"
export CC="$CUDA_HOME/bin/x86_64-conda-linux-gnu-gcc" CXX="$CUDA_HOME/bin/x86_64-conda-linux-gnu-g++"
export CUDAHOSTCXX="$CXX"
# conda puts CUDA headers/libs under targets/, which torch.utils.cpp_extension does not search
export CPATH="$CUDA_HOME/targets/x86_64-linux/include${CPATH:+:$CPATH}"
export LIBRARY_PATH="$CUDA_HOME/lib:$CUDA_HOME/targets/x86_64-linux/lib${LIBRARY_PATH:+:$LIBRARY_PATH}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-12.0}"   # RTX 5090 = sm_120
export MAX_JOBS="${MAX_JOBS:-8}"
