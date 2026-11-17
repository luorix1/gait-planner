#!/usr/bin/env bash
# A first check on the cluster, 30-minute limit: a --smoke run (256 worlds, 3 minutes of training), then the SPRINT
# Studio export, as at the end of a full run. Its first job also compiles and caches the GPU kernels (~/.cache/warp);
# if it hits the limit doing so, submit it again.
#
#   scripts/slurm/smoke.sh [--partition P] [--dry-run]
#
# Afterwards:  tail runs/slurm/<job id>_smoke_*.out     # node, GPU, then the run's own output
#              ls runs/smoke_*/studio_*                 # the Studio folder (an untrained walker falls at once)
set -euo pipefail
cd "$(dirname "$0")/../.."
[[ ${1:-} != -h && ${1:-} != --help ]] || { sed -n '2,10p' "$0" | sed 's/^# \{0,1\}//'; exit 0; }
scripts/slurm/submit.sh --name "smoke_$(date +%Y%m%d-%H%M%S)" --smoke --hours 0.05 --time 00:30:00 "$@"
