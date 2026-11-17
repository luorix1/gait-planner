#!/usr/bin/env bash
# One single-clip FlashSAC check (imitate_clip.py): can our body reproduce this clip? Trains, then evaluates the best
# policy from the clip's first frame (32 episodes) into runs/hard/CLIP/eval.json.
#
#   scripts/hard/run.sh --clip NAME [--frames 150e6] [--minutes 1200] [--worlds 2048] [--data data/hard_ncsa] [--smoke]
#
#   --frames/--minutes  training budget, whichever ends first (default 150M frames, STC's running test; 20 h cap)
#                       learner and effort follow STC's running test too (imitate_clip.py's defaults)
#   --data     the dataset holding the clip (default data/hard_ncsa, built by scripts/hard/build.sh)
#   --smoke    a pipeline check in minutes (small learner, 256 worlds, 3 minutes, 4 evaluation episodes)
#
# A finished clip (eval.json present) is skipped; an interrupted one is moved aside to CLIP.interrupted_<date> and
# started again (FlashSAC's replay buffer is not resumed). LAUNCH.txt and stdout.log are in runs/hard/CLIP/.
set -euo pipefail
cd "$(dirname "$0")/../.."
clip="" frames=150e6 minutes=1200 worlds=2048 dat=data/hard_ncsa smoke=0
while [[ $# -gt 0 ]]; do
  case $1 in
    --clip) clip=$2; shift 2 ;;
    --frames) frames=$2; shift 2 ;;
    --minutes) minutes=$2; shift 2 ;;
    --worlds) worlds=$2; shift 2 ;;
    --data) dat=$2; shift 2 ;;
    --smoke) smoke=1; shift ;;
    -h|--help) sed -n '2,14p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown option $1 (--help)" >&2; exit 1 ;;
  esac
done
[[ -n $clip ]] || { echo "--clip is required (--help)" >&2; exit 1; }
py=${GP_PYTHON:-.venv/bin/python}
[[ -x $py ]] || { echo "no $py: build it (./setup_env.sh here, scripts/slurm/setup.sh on a cluster)" >&2; exit 1; }
unset PYTHONPATH
[[ -z ${GP_MODEL:-} && -f data/models/Rajagopal_lower_collision_v3.osim ]] && export GP_MODEL=$PWD/data/models/Rajagopal_lower_collision_v3.osim
out=runs/hard/$clip
if [[ -f $out/eval.json ]]; then echo "$clip: done ($out/eval.json); skipped"; exit 0; fi
[[ -d $out ]] && mv "$out" "$out.interrupted_$(date +%Y%m%d-%H%M%S)"
extra=(); [[ $smoke == 1 ]] && { extra=(--smoke --episodes 4 --capacity 100000 --learning-starts 2048); worlds=256; minutes=3; }
cmd=("$py" imitate_clip.py "$clip" --data "$dat" --out "$out" --worlds "$worlds" --minutes "$minutes" --frames "$frames" "${extra[@]}")
mkdir -p runs/hard
log=$(mktemp)
{
  echo "date:    $(date -Is)"
  echo "host:    $(hostname)${SLURM_JOB_ID:+ (SLURM job $SLURM_JOB_ID)}"
  echo "gpu:     $(nvidia-smi --query-gpu=name --format=csv,noheader 2> /dev/null | head -1)"
  echo "commit:  $(git rev-parse --short HEAD 2> /dev/null)$(git diff --quiet HEAD -- gp imitate_clip.py 2> /dev/null || echo ' (code dirty)')"
  echo "command: ${cmd[*]}"
  echo
} > "$log"
"${cmd[@]}" 2>&1 | tee "$out.stdout.log"
mv "$log" "$out/LAUNCH.txt" && mv "$out.stdout.log" "$out/stdout.log"
