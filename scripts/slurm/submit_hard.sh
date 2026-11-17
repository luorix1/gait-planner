#!/usr/bin/env bash
# Submit one single-clip FlashSAC job per clip (scripts/hard/run.sh), skipping clips already done (runs/hard/CLIP/eval.json).
#
#   scripts/slurm/submit_hard.sh [--clips FILE] [--frames 150e6] [--minutes 1200] [--time HH:MM:SS] [--partition P]
#                                [--max N] [--dry-run]
#
#   --clips      one clip per line (default data/hard_ncsa/CLIPS.txt: every clip scripts/hard/build.sh built)
#   --minutes    training cap per clip (default 1200 = 20 h); --frames the budget (default 150e6), whichever ends first
#   --time       wall-clock limit per job (default --minutes + 40 min: start-up, kernel compile, evaluation)
#   --max        submit at most N jobs this time (the rest on a later call; done clips are skipped)
#   --partition  default $GP_PARTITION, else gpuA100x4
#
# Then: squeue -u $USER; .venv/bin/python hard_report.py (runs/hard/summary.md); on your machine sync.sh pull-hard.
set -euo pipefail
cd "$(dirname "$0")/../.."
usage() { sed -n '2,14p' "$0" | sed 's/^# \{0,1\}//'; exit "${1:-1}"; }
list=data/hard_ncsa/CLIPS.txt frames=150e6 minutes=1200 time="" partition=${GP_PARTITION:-gpuA100x4} max=1000 dry=0
while [[ $# -gt 0 ]]; do
  case $1 in
    --clips) list=$2; shift 2 ;;
    --frames) frames=$2; shift 2 ;;
    --minutes) minutes=$2; shift 2 ;;
    --time) time=$2; shift 2 ;;
    --partition) partition=$2; shift 2 ;;
    --max) max=$2; shift 2 ;;
    --dry-run) dry=1; shift ;;
    -h|--help) usage 0 ;;
    *) echo "unknown option $1 (--help)" >&2; exit 1 ;;
  esac
done
[[ -f $list ]] || { echo "no $list: run scripts/hard/build.sh on your machine, then scripts/ncsa/sync.sh push" >&2; exit 1; }
[[ -n $time ]] || time=$(python3 -c "m = int($minutes) + 40; print(f'{m // 60:02d}:{m % 60:02d}:00')")
GP_ACCOUNT=${GP_ACCOUNT:-${NT_ACCOUNT:-}}                    # neural_test's allocation serves too
[[ -n $GP_ACCOUNT || $dry == 1 ]] || { echo "no SLURM account: run scripts/slurm/setup.sh once (it sets GP_ACCOUNT)" >&2; exit 1; }
[[ -x .venv/bin/python || $dry == 1 ]] || { echo "no .venv: run scripts/slurm/setup.sh on the login node first" >&2; exit 1; }
mkdir -p runs/slurm
n=0
while read -r clip; do
  [[ -z $clip || $clip == \#* ]] && continue
  [[ -f runs/hard/$clip/eval.json ]] && { echo "done: $clip"; continue; }
  (( n < max )) || break
  cmd=(sbatch --account "${GP_ACCOUNT:-<account>}" --partition "$partition" --time "$time" --job-name "gp_hard_$clip"
       --output "runs/slurm/%j_hard_$clip.out" scripts/slurm/hard.sbatch --clip "$clip" --frames "$frames" --minutes "$minutes")
  if [[ $dry == 1 ]]; then echo "${cmd[*]}"; else "${cmd[@]}"; fi
  n=$((n + 1))
done < "$list"
echo "$n job(s) $([[ $dry == 1 ]] && echo 'would be ' )submitted"
