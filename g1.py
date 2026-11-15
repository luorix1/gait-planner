"""Gate G1 (plan doc): the planner against the full-pose baseline at matched compute, from both runs' eval.json.

1. **Acceptance**: at least 50 % of the planner's level and ramp episodes on held-out subjects pass the acceptance
   filter (incomplete while a check has no threshold, e.g. style before accept.json is calibrated with the prior).
2. **Kinetics**: the planner's right-stance hip, knee and ankle moments and vertical force lie inside the real band
   at least as often as the baseline's (mean share in band over episodes, held-out subjects and terrain).
3. **Throughput**: the planner trained at 10k steps/s or more (median over the second half of its progress.csv).

Run: .venv/bin/python g1.py runs/planner_v1 runs/baseline_v1
Out: <planner run>/g1.json
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

from gp import data

LEVEL_RAMP = ("level", "ramp_up", "ramp_down")
HELD = data.HELD_OUT                    # held-out motion: Camargo subjects (v1), AB01 trials (ab01)
MIN_ACCEPT, MIN_SPS = 0.5, 10_000


def main() -> None:
    planner, baseline = Path(sys.argv[1]), Path(sys.argv[2])
    ep, eb = (json.loads((r / "eval.json").read_text()) for r in (planner, baseline))
    rows = [r for r in ep[HELD] if r["terrain_class"] in LEVEL_RAMP]
    acc = float(np.mean([r["accepted"] for r in rows])) if rows else float("nan")
    unchecked = sorted({c for r in rows for c in r.get("unchecked", [])})
    fails: dict[str, int] = {}
    for r in rows:
        for c in r.get("failed", []):
            fails[c] = fails.get(c, 0) + 1
    band = {}
    for name, e in (("planner", ep), ("baseline", eb)):
        band[name] = {sp: float(np.nanmean([r.get("in_band_mean", np.nan) for r in e[sp]]))
                      for sp in (HELD, "test_terrain") if sp in e}
    both = [sp for sp in band["planner"] if sp in band["baseline"]]
    kin_pass = all(band["planner"][sp] >= band["baseline"][sp] for sp in both) if both else None
    prog = list(csv.DictReader((planner / "progress.csv").open()))
    sps = [float(r["steps_per_s"]) for r in prog[len(prog) // 2:]]
    med = float(np.median(sps)) if sps else float("nan")
    rep = dict(
        acceptance=dict(rate=acc, episodes=len(rows), failed_by_check=fails, unchecked=unchecked,
                        passed=None if unchecked else bool(acc >= MIN_ACCEPT)),
        kinetics=dict(in_band=band, passed=kin_pass),
        throughput=dict(median_steps_per_s=med, passed=bool(med >= MIN_SPS)),
        prior=ep.get("prior"), accept_json=ep.get("accept_json"))
    rep["g1"] = None if None in (rep["acceptance"]["passed"], kin_pass) else bool(
        rep["acceptance"]["passed"] and kin_pass and rep["throughput"]["passed"])
    (planner / "g1.json").write_text(json.dumps(rep, indent=1))
    say = {True: "PASS", False: "FAIL", None: "INCOMPLETE"}
    print(f"1. acceptance {acc:.0%} of {len(rows)} level/ramp episodes (need {MIN_ACCEPT:.0%}): "
          f"{say[rep['acceptance']['passed']]}" + (f"  (no threshold yet: {', '.join(unchecked)})" if unchecked else "")
          + (f"  failures: {fails}" if fails else ""))
    print("2. curves in the real band, planner vs baseline: " + ", ".join(
        f"{sp} {band['planner'][sp]:.0%} vs {band['baseline'][sp]:.0%}" for sp in both) + f": {say[kin_pass]}")
    print(f"3. throughput {med:,.0f} steps/s (need {MIN_SPS:,}): {say[rep['throughput']['passed']]}")
    print(f"G1 {say[rep['g1']]} -> {planner / 'g1.json'}")


if __name__ == "__main__":
    main()
