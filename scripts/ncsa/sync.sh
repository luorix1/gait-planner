#!/usr/bin/env bash
# Move this checkout to the cluster, and runs back.
#
#   scripts/ncsa/sync.sh push                  # tracked files and .git -> $GP_REMOTE; the SPRINT engine -> beside it
#                                              # (../sprint-gait-planner); the model and its meshes -> data/models/
#                                              # and data/hard_ncsa if built (scripts/hard/build.sh)
#   scripts/ncsa/sync.sh pull NAME... [--all]  # runs/NAME <- $GP_REMOTE: logs, record, progress, predictor models,
#                                              # policy and the Studio folders (--all: checkpoint.pt too)
#   scripts/ncsa/sync.sh pull-hard             # runs/hard <- $GP_REMOTE (no checkpoints), then hard_report.py
#
# GP_REMOTE is the checkout on the cluster as rsync writes it, e.g.
#   export GP_REMOTE=<user>@login.delta.ncsa.illinois.edu:/u/<user>/gait_planner
# Each command is one rsync login (Delta asks for your password and Duo; an ssh ControlMaster saves the repeats).
# push carries .git, so runs on the cluster record their commit; it never touches the cluster's runs/ or .venv.
# After a pull, each run's Studio folders are copied into SPRINT Studio's results folder.
set -euo pipefail
cd "$(dirname "$0")/../.."
usage() { sed -n '2,15p' "$0" | sed 's/^# \{0,1\}//'; exit "${1:-1}"; }
[[ $# -ge 1 && $1 != -h && $1 != --help ]] || usage 0
[[ -n ${GP_REMOTE:-} ]] || { echo "set GP_REMOTE, e.g. <user>@login.delta.ncsa.illinois.edu:/u/<user>/gait_planner" >&2; exit 1; }

sprint=${GP_SPRINT_SRC:-$HOME/Jinwoo/worktrees/sprint-gait-planner}
model=${GP_MODEL:-$HOME/Jinwoo/neural_test/models/Rajagopal_lower_collision_v3.osim}
case $1 in
  push)
    # Everything goes in ONE rsync (one login, one Duo prompt): staged locally first in ~/.cache/gait_planner_sync,
    # laid out as the cluster's parent folder: <checkout>/ (tracked files, .git, data/models, data/hard_ncsa) and
    # sprint-gait-planner/ (the engine's tracked files, SPRINT_COMMIT).
    git diff --quiet HEAD || echo "note: uncommitted changes go too; runs will record the commit as dirty"
    git -C "$sprint" diff --quiet HEAD || echo "note: $sprint has uncommitted changes; they go too"
    stage=${GP_SYNC_STAGE:-$HOME/.cache/gait_planner_sync}
    repo=$stage/${GP_REMOTE##*/}
    mkdir -p "$repo/data/models" "$stage/sprint-gait-planner"
    { git ls-files; echo .git; } | rsync -a -r --delete --files-from=- ./ "$repo/"
    git -C "$sprint" ls-files | rsync -a --files-from=- "$sprint/" "$stage/sprint-gait-planner/"
    git -C "$sprint" rev-parse HEAD > "$stage/sprint-gait-planner/SPRINT_COMMIT"
    cp "$model" "$repo/data/models/"
    rsync -a --copy-links "$(dirname "$model")/Geometry/" "$repo/data/models/Geometry/"   # its meshes are symlinks here
    if [[ -d data/hard_ncsa ]]; then
      [[ -f data/hard_ncsa/CLIPS.txt ]] || { echo "data/hard_ncsa has no CLIPS.txt: rebuild it with scripts/hard/build.sh" >&2; exit 1; }
      rsync -a --delete data/hard_ncsa "$repo/data/"
    fi
    rsync -az "$stage/" "${GP_REMOTE%/*}/"
    echo "pushed $(git rev-parse --short HEAD) ($(git branch --show-current)), SPRINT $(cut -c1-7 "$stage/sprint-gait-planner/SPRINT_COMMIT"), $(basename "$model")$([[ -d data/hard_ncsa ]] && echo " and data/hard_ncsa ($(wc -l < data/hard_ncsa/CLIPS.txt) clips)") to $GP_REMOTE"
    ;;
  pull)
    shift
    all=0 names=()
    for a in "$@"; do [[ $a == --all ]] && all=1 || names+=("$a"); done
    [[ ${#names[@]} -gt 0 ]] || usage
    skip=(); [[ $all == 1 ]] || skip=(--exclude 'checkpoint.pt')
    src=(); for n in "${names[@]}"; do src+=("$GP_REMOTE/runs/$n"); done
    rsync -az "${skip[@]}" "${src[@]}" runs/                     # one login for every run
    dest=$(env -u PYTHONPATH .venv/bin/python -c "from sprint.export import format as f; print(f.results_dir())" 2> /dev/null || true)
    for n in "${names[@]}"; do
      it=$(python3 -c "import json; r = json.load(open('runs/$n/record.json')); print(r.get('iterations') or (r.get('resumes') or [{}])[-1].get('iterations', '?'))" 2> /dev/null || echo "?")
      echo "runs/$n: iterations $it, predictor refits $(ls runs/"$n" | grep -c '^foot_model_')"
      if [[ -n $dest ]]; then
        for d in runs/"$n"/studio_*; do [[ -d $d ]] && rm -rf "$dest/$n.$(basename "$d")" && cp -r "$d" "$dest/$n.$(basename "$d")" && echo "  -> $dest/$n.$(basename "$d")"; done
      fi
    done
    ;;
  pull-hard)
    rsync -az --exclude 'checkpoint.pt' "$GP_REMOTE/runs/hard" runs/
    env -u PYTHONPATH .venv/bin/python hard_report.py runs/hard | head -5
    echo "table: runs/hard/summary.md"
    ;;
  *) usage ;;
esac
