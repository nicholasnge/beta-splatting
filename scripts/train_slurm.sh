#!/bin/bash
#SBATCH --job-name=beta-splat
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err
#SBATCH --gres=gpu:a100-40
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00

# Usage:
#   sbatch --job-name=room scripts/train_slurm.sh \
#       --scene room \
#       --data_root ~/3DGSDATASETS \
#       --out_root ~/beta-splatting/output
#
# Or with extra train.py args:
#   sbatch scripts/train_slurm.sh --scene bicycle --resolution 2 --cap_max 500000

set -e

# ── Parse args ────────────────────────────────────────────────────────────────
SCENE=""
DATA_ROOT="$HOME/3DGSDATASETS"
OUT_ROOT="$SLURM_SUBMIT_DIR/output"
RESOLUTION=4
CAP_MAX=300000
EXTRA_ARGS=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --scene)       SCENE="$2";       shift 2 ;;
        --data_root)   DATA_ROOT="$2";   shift 2 ;;
        --out_root)    OUT_ROOT="$2";    shift 2 ;;
        --resolution)  RESOLUTION="$2";  shift 2 ;;
        --cap_max)     CAP_MAX="$2";     shift 2 ;;
        *)             EXTRA_ARGS="$EXTRA_ARGS $1"; shift ;;
    esac
done

if [[ -z "$SCENE" ]]; then
    echo "ERROR: --scene <name> is required"
    exit 1
fi

# ── Environment ───────────────────────────────────────────────────────────────
REPO_DIR="$SLURM_SUBMIT_DIR"
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate beta_splatting

mkdir -p "$OUT_ROOT" slurm_logs

# ── Validate scene structure ──────────────────────────────────────────────────
SCENE_PATH="$DATA_ROOT/$SCENE"
OUT_PATH="$OUT_ROOT/$SCENE"

if [[ ! -d "$SCENE_PATH/sparse/0" ]]; then
    echo "ERROR: Expected $SCENE_PATH/sparse/0/ — not found."
    echo "       Contents of $SCENE_PATH: $(ls "$SCENE_PATH" 2>/dev/null)"
    exit 1
fi
if [[ ! -f "$SCENE_PATH/sparse/0/cameras.bin" && ! -f "$SCENE_PATH/sparse/0/cameras.txt" ]]; then
    echo "ERROR: No cameras.bin or cameras.txt in $SCENE_PATH/sparse/0/"
    exit 1
fi

# ── Run ───────────────────────────────────────────────────────────────────────
echo "Scene:      $SCENE"
echo "Data:       $SCENE_PATH"
echo "Output:     $OUT_PATH"
echo "Resolution: $RESOLUTION"
echo "Cap:        $CAP_MAX"
echo "Node:       $(hostname)"
echo "GPU:        $(nvidia-smi --query-gpu=name --format=csv,noheader)"

cd "$REPO_DIR"
python train.py \
    -s "$SCENE_PATH" \
    -m "$OUT_PATH" \
    -r "$RESOLUTION" \
    --cap_max "$CAP_MAX" \
    --eval \
    --disable_viewer \
    $EXTRA_ARGS

echo "=== Done: $SCENE ==="
