#!/usr/bin/env bash
# One Watch Your Step run end to end, in the foreground: train, then export it for SPRINT Studio.
#
#   scripts/run.sh --name NAME [--hours 12] [--worlds 8192] [--seed 0] [--steps N] [--ladder L0|L1] [--gaitamp] [--smoke]
#
#   --name     the run directory runs/NAME/; running the same name again continues it (train_wys.py resumes)
#   --hours    training wall-clock for this start (default 12); a resubmitted job adds another --hours
#   --steps    also stop at this many environment steps in total (e.g. 5e8)
#   --ladder   a rung of the footstep planner ladder (gp.ladder, build B1) instead: L0 whole map, L1 patches,
#              L0.5 patches without the command in the policy's input
#   --pretrain stage 0 of B1: L1's policy learns to walk by imitating AB01's level clips (gp.pretrain)
#   --init-from RUN  with --ladder: start from that run's policy, critic and guesser (e.g. a --pretrain run)
#   --gaitamp  with --ladder: GAIT_amp's training conditions (flat, no delays/noise/pushes, 60 deg falls, walking starts)
#   --smoke    a pipeline check in minutes: 256 worlds, 3 minutes of training, then the export
#
# After training: export_wys.py runs/NAME (runs/NAME/studio_<iteration>/: one 10 s episode per terrain type at
# level 5, a folder SPRINT Studio opens; also copied into Studio's results folder when this machine has Studio).
# Locally, detach it:  setsid nohup scripts/run.sh ... > /dev/null 2>&1 < /dev/null &
# Each start appends to runs/NAME/LAUNCH.txt (date, host, GPU, commit, command); output goes to runs/NAME/stdout.log.
set -euo pipefail
cd "$(dirname "$0")/.."

name="" hours=12 worlds=8192 seed=0 smoke=0 ladder=() steps=() extra=()
while [[ $# -gt 0 ]]; do
  case $1 in
    --name) name=$2; shift 2 ;;
    --hours) hours=$2; shift 2 ;;
    --steps) steps=(--steps "$2"); shift 2 ;;
    --worlds) worlds=$2; shift 2 ;;
    --seed) seed=$2; shift 2 ;;
    --ladder) case $2 in L0) ladder=(--task ladder --perception map) ;; L1) ladder=(--task ladder --perception patches) ;;
              L0.5) ladder=(--task ladder --perception patches --no-policy-command) ;;
              *) echo "--ladder takes L0, L0.5 or L1" >&2; exit 1 ;; esac; shift 2 ;;
    --gaitamp) extra+=(--gaitamp-conditions); shift ;;
    --pretrain) ladder=(--task pretrain --perception patches); extra+=(--drive-floor 0); shift ;;
    --init-from) extra+=(--init-from "$2"); shift 2 ;;
    --smoke) smoke=1; shift ;;
    -h|--help) sed -n '2,20p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown option $1 (--help)" >&2; exit 1 ;;
  esac
done
[[ -n $name ]] || { echo "--name is required (--help)" >&2; exit 1; }
[[ $smoke == 1 ]] && { worlds=256; hours=0.05; }

py=${GP_PYTHON:-.venv/bin/python}
[[ -x $py ]] || { echo "no $py: build it (./setup_env.sh here, scripts/slurm/setup.sh on a cluster)" >&2; exit 1; }
unset PYTHONPATH
# the model: shipped beside the checkout by scripts/ncsa/sync.sh on a cluster, else gp.data's default
[[ -z ${GP_MODEL:-} && -f data/models/Rajagopal_lower_collision_v3.osim ]] && export GP_MODEL=$PWD/data/models/Rajagopal_lower_collision_v3.osim
out=runs/$name
mkdir -p "$out"
cmd=("$py" train_wys.py --out "$out" --hours "$hours" --worlds "$worlds" --seed "$seed" "${ladder[@]}" "${steps[@]}" "${extra[@]}")
{
  echo "date:    $(date -Is)"
  echo "host:    $(hostname)${SLURM_JOB_ID:+ (SLURM job $SLURM_JOB_ID)}"
  echo "gpu:     $(nvidia-smi --query-gpu=name --format=csv,noheader 2> /dev/null | head -1)"
  echo "commit:  $(git rev-parse --short HEAD 2> /dev/null)$(git diff --quiet HEAD -- gp train_wys.py 2> /dev/null || echo ' (code dirty)')"
  echo "command: ${cmd[*]}"
  echo
} >> "$out/LAUNCH.txt"

"${cmd[@]}" 2>&1 | tee -a "$out/stdout.log"
studio=(); [[ -d ${XDG_CONFIG_HOME:-$HOME/.config}/gait-studio ]] && studio=(--to-studio)   # SPRINT Studio installed here
"$py" export_wys.py "$out" "${studio[@]}" 2>&1 | tee -a "$out/stdout.log"
