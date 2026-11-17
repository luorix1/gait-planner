#!/usr/bin/env bash
# One-time setup on the cluster's LOGIN node (downloads and installs only; nothing runs on a GPU here).
#
#   scripts/ncsa/sync.sh push                     # on your machine: this checkout, SPRINT and the model -> the cluster
#   scripts/slurm/setup.sh [--account NAME]       # then on the login node, in that checkout
#
# 1. SLURM account: --account, else $GP_ACCOUNT, else neural_test's $NT_ACCOUNT, else your only *-gpu account
#    (sacctmgr; on Delta bxxx-delta-gpu).
#    Written to ~/.bashrc as `export GP_ACCOUNT=...`, so later logins have it.
# 2. SPRINT: the exact engine every gait_planner run uses (branch gait-planner-engine: v1.0.0 a44107e plus the
#    specific-tension commit 42c0606, which exists only on the lab machine), pushed by sync.sh to ../sprint-gait-planner.
# 3. .venv: Python 3.11 with requirements.lock, the exact local set (torch 2.15 + CUDA 13.2, warp 1.19), sprint-sim
#    from ../sprint-gait-planner. With uv if it is on PATH, else a conda env at .venv (conda from $GP_CONDA_MODULE,
#    default miniforge3-python, Delta's). Several GB of wheels: set PIP_CACHE_DIR / UV_CACHE_DIR first if $HOME is tight.
set -euo pipefail
cd "$(dirname "$0")/../.."

account=${GP_ACCOUNT:-${NT_ACCOUNT:-}}
while [[ $# -gt 0 ]]; do
  case $1 in
    --account) account=$2; shift 2 ;;
    -h|--help) sed -n '2,15p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown option $1 (--help)" >&2; exit 1 ;;
  esac
done
[[ -f data/models/Rajagopal_lower_collision_v3.osim ]] || { echo "no data/models: run scripts/ncsa/sync.sh push on your machine" >&2; exit 1; }

# -- 1. SLURM account -> ~/.bashrc ---------------------------------------------------------------------------------
if [[ -z $account ]]; then
  mapfile -t gpu < <(sacctmgr -n -P show assoc user="$USER" format=account 2> /dev/null | grep -- '-gpu$' | sort -u)
  case ${#gpu[@]} in
    1) account=${gpu[0]} ;;
    0) echo "no *-gpu SLURM account found for $USER (run \`accounts\`); pass --account NAME" >&2; exit 1 ;;
    *) echo "several GPU accounts: ${gpu[*]}; pass --account NAME" >&2; exit 1 ;;
  esac
fi
line="export GP_ACCOUNT=$account"
if grep -q '^export GP_ACCOUNT=' ~/.bashrc 2> /dev/null; then
  sed -i "s|^export GP_ACCOUNT=.*|$line|" ~/.bashrc
else
  printf '\n# gait_planner: SLURM allocation for scripts/slurm (setup.sh)\n%s\n' "$line" >> ~/.bashrc
fi
echo "GP_ACCOUNT=$account (in ~/.bashrc)"

# -- 2. SPRINT ---------------------------------------------------------------------------------------------------------
wt=$(realpath -m ../sprint-gait-planner)
[[ -f $wt/pyproject.toml && -f $wt/SPRINT_COMMIT ]] || { echo "no SPRINT at $wt: run scripts/ncsa/sync.sh push" >&2; exit 1; }
echo "SPRINT $(cat "$wt/SPRINT_COMMIT") at $wt"

# -- 3. .venv --------------------------------------------------------------------------------------------------------
lock=$(mktemp --suffix=.txt)
trap 'rm -f "$lock"' EXIT
sed "s#^sprint-sim @ file://.*#sprint-sim @ file://$wt#" requirements.lock > "$lock"
if [[ -x .venv/bin/python ]]; then
  echo ".venv exists (delete it to rebuild)"
elif command -v uv > /dev/null; then
  uv venv .venv --python 3.11
  uv pip install --python .venv/bin/python -r "$lock"
else
  command -v conda > /dev/null || module load "${GP_CONDA_MODULE:-miniforge3-python}"
  conda create -y -p "$PWD/.venv" python=3.11 pip
  .venv/bin/python -m pip install -r "$lock"
fi
# Import check only: login nodes have no GPU (the first job compiles and caches the kernels, ~/.cache/warp).
.venv/bin/python -c "import sys, torch, warp, sprint; print(sys.executable, '| torch', torch.__version__, '| warp', warp.__version__, '| sprint', sprint.__version__)"
echo "done. Open a new shell (or: source ~/.bashrc), then: scripts/slurm/smoke.sh"
