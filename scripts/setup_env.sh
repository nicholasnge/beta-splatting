#!/bin/bash
# One-time setup: creates the beta_splatting conda env on the cluster.
# Best run as a SLURM job: sbatch scripts/setup_slurm.sh

set -e

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# When called from setup_slurm.sh, SLURM_SUBMIT_DIR is more reliable
if [[ -n "$SLURM_SUBMIT_DIR" ]]; then
    REPO_DIR="$SLURM_SUBMIT_DIR"
fi

ENV_NAME="beta_splatting"

echo "=== Setting up $ENV_NAME conda environment ==="
echo "    Repo: $REPO_DIR"

# A100 = sm_80. Explicit arch means nvcc compiles without querying a GPU.
export TORCH_CUDA_ARCH_LIST="8.0"
export MAX_JOBS=8

# ── Source conda ──────────────────────────────────────────────────────────────
CONDA_BASE="$(conda info --base 2>/dev/null || echo "$HOME/miniconda3")"
source "$CONDA_BASE/etc/profile.d/conda.sh"

# ── Create env ────────────────────────────────────────────────────────────────
if conda env list | grep -q "^${ENV_NAME} "; then
    echo "    Env '$ENV_NAME' already exists — skipping create"
else
    conda create -y -n "$ENV_NAME" python=3.10 -c conda-forge
fi

conda activate "$ENV_NAME"
echo "    Python: $(which python)"

# ── PyTorch cu124 ─────────────────────────────────────────────────────────────
# PYTHONNOUSERSITE=1 prevents pip from seeing ~/.local packages as "already
# satisfied", ensuring all nvidia-* deps are installed in the conda env
# (not left in ~/.local where their .so files won't be on LD_LIBRARY_PATH).
PYTHONNOUSERSITE=1 pip install torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu124

# ── Fix LD_LIBRARY_PATH for pip-wheel nvidia libs ─────────────────────────────
# PyTorch pip wheels install .so files under site-packages/nvidia/*/lib/.
SITE_PKG=$(python -c "import site; print(site.getsitepackages()[0])")
NVIDIA_LIBS=$(find "$SITE_PKG/nvidia" -name "lib" -type d 2>/dev/null | tr '\n' ':')
export LD_LIBRARY_PATH="${NVIDIA_LIBS}${LD_LIBRARY_PATH:-}"
echo "    LD_LIBRARY_PATH set: $NVIDIA_LIBS"

# ── CUDA extensions (need torch visible — no build isolation) ─────────────────
pip install --no-build-isolation \
    "fused-ssim @ git+https://github.com/rahul-goel/fused-ssim@1272e21a282342e89537159e4bad508b19b34157"

pip install --no-build-isolation \
    "plas @ git+https://github.com/fraunhoferhhi/PLAS.git"

# ── Custom gsplat submodule ───────────────────────────────────────────────────
pip install --no-build-isolation "$REPO_DIR/submodules"

# ── Main package ─────────────────────────────────────────────────────────────
pip install --no-build-isolation "$REPO_DIR"

echo "=== Done! Activate with: conda activate $ENV_NAME ==="
