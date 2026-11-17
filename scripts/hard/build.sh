#!/usr/bin/env bash
# Build the cluster's single-clip dataset, data/hard_ncsa, from a clip list (default scripts/hard/clips.txt), on THIS
# machine (it reads the seed library).
#
#   scripts/hard/build.sh [scripts/hard/clips.txt]
#
# Then scripts/ncsa/sync.sh push carries data/hard_ncsa to the cluster.
set -euo pipefail
cd "$(dirname "$0")/../.."
list=${1:-scripts/hard/clips.txt}
mapfile -t clips < <(grep -v '^\s*#' "$list" | sed 's/\s.*//' | grep -v '^$')
py=${GP_PYTHON:-.venv/bin/python}
rm -rf data/hard_ncsa
"$py" - "${clips[@]}" <<'PY'
import sys
from pathlib import Path
from gp import data
clips = sys.argv[1:]
out = Path("data/hard_ncsa")
data.build("hard", out=out, clips=clips)
D = data.load("train", src=out)
names = [str(n) for n in D["name"]]
missing = sorted(set(clips) - set(names))
(out / "CLIPS.txt").write_text("\n".join(names) + "\n")
print(f"data/hard_ncsa: {len(names)} clips" + (f"; not built (dropped or unknown): {missing}" if missing else ""))
PY
