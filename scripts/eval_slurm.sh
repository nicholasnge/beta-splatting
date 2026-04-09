#!/bin/bash
#SBATCH --job-name=beta-eval
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err
#SBATCH --gres=gpu:a100-40
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00

# Usage:
#   sbatch scripts/eval_slurm.sh --scene room --data_root ~/3DGSDATASETS

set -e

SCENE=""
DATA_ROOT="$HOME/3DGSDATASETS"
OUT_ROOT="$SLURM_SUBMIT_DIR/output"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --scene)     SCENE="$2";     shift 2 ;;
        --data_root) DATA_ROOT="$2"; shift 2 ;;
        --out_root)  OUT_ROOT="$2";  shift 2 ;;
        *) shift ;;
    esac
done

if [[ -z "$SCENE" ]]; then
    echo "ERROR: --scene <name> is required"
    exit 1
fi

REPO_DIR="$SLURM_SUBMIT_DIR"
CONDA_BASE="$(conda info --base 2>/dev/null || echo "$HOME/miniconda3")"
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate beta_splatting

cd "$REPO_DIR"
python eval.py \
    -s "$DATA_ROOT/$SCENE" \
    -m "$OUT_ROOT/$SCENE"
