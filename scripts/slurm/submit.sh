#!/usr/bin/env bash
# Submit one run (scripts/run.sh: train, SPRINT Studio export) as a one-GPU SLURM job; NCSA Delta by default.
#
#   scripts/slurm/submit.sh --name NAME [--hours 12] [--worlds 8192] [--seed 0] [--smoke]
#                           [--time HH:MM:SS] [--partition P] [--after JOBID] [--dry-run]
#
#   --hours      training hours for this job (default 12); --time must leave room for start-up and the export
#   --time       wall-clock limit (default: --hours + 1 h). A run the limit cuts off continues when the same command is
#                submitted again: train_wys.py resumes from runs/NAME/checkpoint.pt (saved every 10 minutes).
#   --partition  default $GP_PARTITION, else gpuA100x4 (Delta; gpuA40x4 is the other common one, `sinfo -s` lists all)
#   --after      start only once job JOBID has ended (afterany): a continuation queued behind its first part
#   --dry-run    print the sbatch command instead of submitting
#
# Environment: GP_ACCOUNT, the allocation (e.g. bxxx-delta-gpu; scripts/slurm/setup.sh writes it to ~/.bashrc).
# SLURM's log: runs/slurm/<job id>_NAME.out. The run's own: runs/NAME/stdout.log and LAUNCH.txt.
set -euo pipefail
cd "$(dirname "$0")/../.."

usage() { sed -n '2,16p' "$0" | sed 's/^# \{0,1\}//'; exit "${1:-1}"; }
[[ $# -ge 1 && $1 != -h && $1 != --help ]] || usage 0

args=() name="" hours=12 time="" partition=${GP_PARTITION:-gpuA100x4} after="" dry=0
while [[ $# -gt 0 ]]; do
  case $1 in
    --name) name=$2; args+=(--name "$2"); shift 2 ;;
    --hours) hours=$2; args+=(--hours "$2"); shift 2 ;;
    --time) time=$2; shift 2 ;;
    --partition) partition=$2; shift 2 ;;
    --after) after=$2; shift 2 ;;
    --dry-run) dry=1; shift ;;
    *) args+=("$1"); shift ;;
  esac
done
[[ -n $name ]] || { echo "--name is required" >&2; exit 1; }
[[ -n $time ]] || time=$(python3 -c "import math; m = math.ceil(($hours + 1) * 60); print(f'{m // 60:02d}:{m % 60:02d}:00')")
GP_ACCOUNT=${GP_ACCOUNT:-${NT_ACCOUNT:-}}                    # neural_test's allocation serves too
[[ -n $GP_ACCOUNT || $dry == 1 ]] || { echo "no SLURM account: run scripts/slurm/setup.sh once (it sets GP_ACCOUNT)" >&2; exit 1; }
[[ -x .venv/bin/python || $dry == 1 ]] || { echo "no .venv: run scripts/slurm/setup.sh on the login node first" >&2; exit 1; }
[[ ! -e runs/$name ]] || echo "runs/$name exists: the job continues it"

mkdir -p runs/slurm
cmd=(sbatch --account "${GP_ACCOUNT:-<account>}" --partition "$partition" --time "$time" --job-name "gp_$name"
     --output "runs/slurm/%j_$name.out" ${after:+--dependency "afterany:$after"} scripts/slurm/job.sbatch "${args[@]}")
if [[ $dry == 1 ]]; then
  echo "${cmd[*]}"
else
  "${cmd[@]}"
fi
