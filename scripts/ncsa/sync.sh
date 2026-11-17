#!/usr/bin/env bash
# Move this checkout to the cluster, and runs back.
#
#   scripts/ncsa/sync.sh push                  # tracked files and .git -> $GP_REMOTE; the SPRINT engine -> beside it
#                                              # (../sprint-gait-planner); the model and its meshes -> data/models/
#   scripts/ncsa/sync.sh pull NAME... [--all]  # runs/NAME <- $GP_REMOTE: logs, record, progress, predictor models,
#                                              # policy and the Studio folders (--all: checkpoint.pt too)
#
# GP_REMOTE is the checkout on the cluster as rsync writes it, e.g.
#   export GP_REMOTE=<user>@login.delta.ncsa.illinois.edu:/u/<user>/gait_planner
# Each rsync is one login (Delta asks for your password and Duo; an ssh ControlMaster saves the repeats).
# push carries .git, so runs on the cluster record their commit; it never touches the cluster's runs/ or .venv.
# After a pull, each run's Studio folders are copied into SPRINT Studio's results folder.
set -euo pipefail
cd "$(dirname "$0")/../.."
usage() { sed -n '2,13p' "$0" | sed 's/^# \{0,1\}//'; exit "${1:-1}"; }
[[ $# -ge 1 && $1 != -h && $1 != --help ]] || usage 0
[[ -n ${GP_REMOTE:-} ]] || { echo "set GP_REMOTE, e.g. <user>@login.delta.ncsa.illinois.edu:/u/<user>/gait_planner" >&2; exit 1; }

sprint=${GP_SPRINT_SRC:-$HOME/Jinwoo/worktrees/sprint-gait-planner}
model=${GP_MODEL:-$HOME/Jinwoo/neural_test/models/Rajagopal_lower_collision_v3.osim}
case $1 in
  push)
    git diff --quiet HEAD || echo "note: uncommitted changes go too; runs will record the commit as dirty"
    { git ls-files; echo .git; } | rsync -az -r --files-from=- ./ "$GP_REMOTE/"
    stage=$(mktemp -d); trap 'rm -rf "$stage"' EXIT
    git -C "$sprint" rev-parse HEAD > "$stage/SPRINT_COMMIT"
    git -C "$sprint" diff --quiet HEAD || echo "note: $sprint has uncommitted changes; they go too"
    { git -C "$sprint" ls-files; } | rsync -az -r --files-from=- "$sprint/" "${GP_REMOTE%/*}/sprint-gait-planner/"
    rsync -az "$stage/SPRINT_COMMIT" "${GP_REMOTE%/*}/sprint-gait-planner/"
    mkdir -p "$stage/models" && cp "$model" "$stage/models/" && cp -r "$(dirname "$model")/Geometry" "$stage/models/"
    rsync -az "$stage/models" "$GP_REMOTE/data/"
    echo "pushed $(git rev-parse --short HEAD) ($(git branch --show-current)), SPRINT $(cut -c1-7 "$stage/SPRINT_COMMIT") and $(basename "$model") to $GP_REMOTE"
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
  *) usage ;;
esac
