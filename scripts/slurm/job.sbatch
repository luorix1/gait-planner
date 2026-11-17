#!/usr/bin/env bash
# One Watch Your Step run on one GPU: scripts/run.sh in the foreground (train, SPRINT Studio export).
# Submitted by scripts/slurm/submit.sh, which adds the account, partition, time, job name and log path; the
# arguments are scripts/run.sh's. No conda or module here: run.sh uses .venv/bin/python (scripts/slurm/setup.sh).
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"        # submit.sh submits from the repository root
echo "job $SLURM_JOB_ID on $(hostname), $(date -Is): scripts/run.sh $*"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true
exec scripts/run.sh "$@"
