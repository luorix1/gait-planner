#!/usr/bin/env bash
# Rebuild .venv: SPRINT from the pinned worktree of branch gait-planner-engine (v1.0.0 = a44107e plus the
# specific-tension commit, 42c0606), with the recipes and dev extras. requirements.lock is the resolved set
# (neural_test's, with this engine).
set -euo pipefail
cd "$(dirname "$0")"
W="$HOME/Jinwoo/worktrees/sprint-gait-planner"
[ -d "$W" ] || git -C "$HOME/Jinwoo/GAIT" worktree add "$W" gait-planner-engine
[ "$(git -C "$W" rev-parse HEAD)" = "$(git -C "$HOME/Jinwoo/GAIT" rev-parse gait-planner-engine)" ] || { echo "worktree not at gait-planner-engine"; exit 1; }
uv venv .venv --python 3.11
VIRTUAL_ENV= uv pip install --python .venv/bin/python -r requirements.lock
