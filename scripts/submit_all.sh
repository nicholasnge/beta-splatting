#!/bin/bash
# Submit training jobs for all scenes in a dataset root.
# Usage: bash scripts/submit_all.sh [--data_root ~/3DGSDATASETS] [--resolution 4] [--cap_max 300000]
#
# Scenes available on xlogin1:
#   360:  bicycle bonsai garden kitchen
#   T&T:  Caterpillar Family Francis Horse Ignatius Train Truck

set -e

DATA_ROOT="$HOME/3DGSDATASETS"
OUT_ROOT="$HOME/beta-splatting/output"
RESOLUTION=1
CAP_MAX=300000
TIME="02:00:00"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --data_root)  DATA_ROOT="$2";  shift 2 ;;
        --out_root)   OUT_ROOT="$2";   shift 2 ;;
        --resolution) RESOLUTION="$2"; shift 2 ;;
        --cap_max)    CAP_MAX="$2";    shift 2 ;;
        --time)       TIME="$2";       shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

SCRIPT="$(dirname "$0")/train_slurm.sh"
mkdir -p slurm_logs

# Detect available scenes
SCENES=()
for d in "$DATA_ROOT"/*/; do
    scene=$(basename "$d")
    SCENES+=("$scene")
done

echo "Submitting ${#SCENES[@]} jobs from $DATA_ROOT"
echo "Output root: $OUT_ROOT"
echo ""

for SCENE in "${SCENES[@]}"; do
    JOB_ID=$(sbatch \
        --job-name="bs_$SCENE" \
        --time="$TIME" \
        "$SCRIPT" \
            --scene "$SCENE" \
            --data_root "$DATA_ROOT" \
            --out_root "$OUT_ROOT" \
            --resolution "$RESOLUTION" \
            --cap_max "$CAP_MAX" \
        | awk '{print $NF}')
    echo "  Submitted $SCENE -> job $JOB_ID"
done

echo ""
echo "Monitor with: squeue -u \$USER"
