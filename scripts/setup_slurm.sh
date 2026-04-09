#!/bin/bash
#SBATCH --job-name=beta-setup
#SBATCH --output=slurm_logs/setup_%j.out
#SBATCH --error=slurm_logs/setup_%j.err
#SBATCH --gres=gpu:a100-40
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00

# Runs setup_env.sh on a compute node where RAM and GPU are available.
# Usage (from repo root on login node):
#   mkdir -p slurm_logs
#   sbatch scripts/setup_slurm.sh

set -e
# SLURM_SUBMIT_DIR = directory where sbatch was called (the repo root)
REPO_DIR="$SLURM_SUBMIT_DIR"
echo "Repo dir: $REPO_DIR"
bash "$REPO_DIR/scripts/setup_env.sh"
