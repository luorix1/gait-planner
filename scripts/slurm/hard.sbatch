#!/usr/bin/env bash
# One single-clip FlashSAC check on one GPU: scripts/hard/run.sh in the foreground (train, then evaluate).
# Submitted by scripts/slurm/submit_hard.sh, which adds the account, partition, time, job name and log path.
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
echo "job $SLURM_JOB_ID on $(hostname), $(date -Is): scripts/hard/run.sh $*"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true
exec scripts/hard/run.sh "$@"
