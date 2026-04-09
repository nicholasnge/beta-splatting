#!/bin/bash
# One-time setup: creates the beta_splatting conda env on the cluster.
# Run once from the login node: bash scripts/setup_env.sh

set -e

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_NAME="beta_splatting"

echo "=== Setting up $ENV_NAME conda environment ==="

# Create env
conda create -y -n "$ENV_NAME" python=3.10

# Install PyTorch with CUDA 12.0-compatible build (cu118 is forward-compatible)
conda run -n "$ENV_NAME" pip install torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu118

# Install CUDA extensions that need torch visible at build time
conda run -n "$ENV_NAME" pip install --no-build-isolation \
    "fused-ssim @ git+https://github.com/rahul-goel/fused-ssim@1272e21a282342e89537159e4bad508b19b34157"

conda run -n "$ENV_NAME" pip install --no-build-isolation \
    "plas @ git+https://github.com/fraunhoferhhi/PLAS.git"

# Install custom gsplat submodule (CUDA extension)
conda run -n "$ENV_NAME" pip install --no-build-isolation "$REPO_DIR/submodules"

# Install main package (submodule build already done above)
conda run -n "$ENV_NAME" pip install --no-build-isolation "$REPO_DIR"

echo "=== Done! Activate with: conda activate $ENV_NAME ==="
