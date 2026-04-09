#!/bin/bash
# One-time setup: creates the beta_splatting conda env on the cluster.
# Run once from the login node: bash scripts/setup_env.sh

set -e

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_NAME="beta_splatting"

echo "=== Setting up $ENV_NAME conda environment ==="
echo "    Repo: $REPO_DIR"
echo "    Note: nvcc must be in PATH (run 'nvcc --version' to verify)"

# Target A100 (sm_80). Set this so CUDA extensions compile correctly even
# when no GPU is present on the login node — nvcc compiles for the specified
# arch without needing a physical device.
export TORCH_CUDA_ARCH_LIST="8.0"
export MAX_JOBS=8

# Create env
conda create -y -n "$ENV_NAME" python=3.10

# Install PyTorch — cu118 is forward-compatible with CUDA 12.0 toolkit
conda run -n "$ENV_NAME" pip install torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu118

# Install CUDA extensions that need torch visible at build time (no isolation)
TORCH_CUDA_ARCH_LIST="8.0" conda run -n "$ENV_NAME" \
    pip install --no-build-isolation \
    "fused-ssim @ git+https://github.com/rahul-goel/fused-ssim@1272e21a282342e89537159e4bad508b19b34157"

conda run -n "$ENV_NAME" pip install --no-build-isolation \
    "plas @ git+https://github.com/fraunhoferhhi/PLAS.git"

# Build custom gsplat submodule (CUDA extension — compiles without a GPU)
TORCH_CUDA_ARCH_LIST="8.0" conda run -n "$ENV_NAME" \
    pip install --no-build-isolation "$REPO_DIR/submodules"

# Install main package (submodule already built above)
conda run -n "$ENV_NAME" pip install --no-build-isolation "$REPO_DIR"

echo "=== Done! Activate with: conda activate $ENV_NAME ==="
