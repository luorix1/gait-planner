"""Summarise the single-clip FlashSAC checks (runs/hard/*/eval.json, imitate_clip.py): can our body do each clip?

Verdict per clip, from its 32 evaluation episodes started at the clip's first frame:
- **pass**: no episode fell and they cover at least 95 % of the playable clip;
- **partial**: they cover at least half of it on average;
- **fail**: less.
Flight (both feet under 5 % of body weight) is compared with the clip's over the same frames: a run or jump the policy
completes with much less flight than the clip ("flight < 70 %") is reproduced only in a slower, grounded form.

Run: .venv/bin/python hard_report.py [runs/hard] [more run folders ...]
Out: <first folder>/summary.md, and the table printed
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def verdict(e: dict) -> str:
    if e["fell_frac"] == 0 and e["episode_frac"] >= 0.95:
        return "pass"
    return "partial" if e["episode_frac"] >= 0.5 else "fail"


def main() -> None:
    roots = [Path(a) for a in sys.argv[1:]] or [Path("runs/hard")]
    rows, pending = [], []
    for root in roots:
        for d in sorted(p for p in root.iterdir() if p.is_dir() and not p.name.startswith(("slurm", ".")) and ".interrupted_" not in p.name):
            ev, rec = d / "eval.json", d / "record.json"
            if not ev.exists():
                pending.append(d.name)
                continue
            r = json.loads(rec.read_text()) if rec.exists() else {}
            for clip, e in json.loads(ev.read_text())["clips"].items():
                ratio = e["flight_total_s"] / e["ref_flight_total_s"] if e["ref_flight_total_s"] > 0 else float("nan")
                rows.append(dict(clip=clip, verdict=verdict(e), frames_m=round(r.get("env_steps", 0) / 1e6, 1),
                                 hours=round(r.get("train_seconds", 0) / 3600, 2), playable_s=e["playable_s"],
                                 episode_s=e["episode_s"], fell=e["fell_frac"], joint_deg=e["joint_error_deg"],
                                 pelvis_m=e["pelvis_error_m"], flight=ratio,
                                 rise=f'{e["pelvis_rise_m"]:.2f}/{e["ref_pelvis_rise_m"]:.2f}'))
    head = "| clip | verdict | frames (M) | hours | playable (s) | survived (s) | fell | joint err (deg) | pelvis err (m) | flight vs clip | pelvis rise (m, sim/clip) |"
    lines = [head, "|" + "---|" * 11]
    for x in rows:
        fl = "–" if x["flight"] != x["flight"] else f'{100 * x["flight"]:.0f} %' + (" (< 70 %)" if x["flight"] < 0.7 else "")
        lines.append(f'| {x["clip"]} | {x["verdict"]} | {x["frames_m"]} | {x["hours"]} | {x["playable_s"]} | {x["episode_s"]} | '
                     f'{100 * x["fell"]:.0f} % | {x["joint_deg"]} | {x["pelvis_m"]} | {fl} | {x["rise"]} |')
    count = {v: sum(x["verdict"] == v for x in rows) for v in ("pass", "partial", "fail")}
    text = (f"# Single-clip checks\n\n{len(rows)} clips: {count['pass']} pass, {count['partial']} partial, "
            f"{count['fail']} fail" + (f"; {len(pending)} not finished ({', '.join(pending)})" if pending else "")
            + "\n\n" + "\n".join(lines) + "\n\n" + __doc__.split("Run:")[0].split("\n\n", 1)[1])
    (roots[0] / "summary.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
