"""EMG timing of trained policies against Camargo's EMG on held-out subjects (gp/emg.py), with the real-data ceiling.

For each terrain class: per muscle, the Pearson r of the policy's mean stride profile against the real one, and on/off
agreement; their means over the scored muscles (rectus femoris is shown, not scored). The ceiling is the same score for the
training subjects' real EMG against the held-out subjects': what subject-to-subject variation alone leaves.

Run: .venv/bin/python emg_report.py runs/planner_v1 [runs/baseline_v1] [--split test_subject]
     (after evaluate.py, which writes <run>/emg_strides.npz)
Out: <first run>/emg_timing.json and <first run>/emg_<class>.png per class
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from gp import data, emg


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--split", default=data.HELD_OUT)
    a = ap.parse_args()
    runs = [Path(r) for r in a.runs]
    _, channels, _, unreliable = emg.load_emg()
    ref, ceiling_ref = emg.reference(a.split), emg.reference("train")
    sims = {}
    for r in runs:
        z = np.load(r / "emg_strides.npz")
        keep = z["split"] == a.split
        sims[r.name] = {c: emg.summarise(z["profiles"][keep & (z["terrain_class"] == c)])
                        for c in sorted(set(z["terrain_class"][keep])) if ((z["terrain_class"] == c) & keep).sum() >= 3}
    report: dict = dict(split=a.split, unreliable=list(unreliable), classes={})
    print(f"{'class':12s} {'real strides':>12s} {'ceiling r / agree':>18s} " + " ".join(f"{n[:22]:>24s}" for n in sims))
    for cls, rf in sorted(ref.items()):
        ceil = emg.compare(rf, dict(mean=ceiling_ref[cls]["mean"], on_prob=ceiling_ref[cls]["on_prob"]), unreliable) \
            if cls in ceiling_ref else None
        row = dict(real_strides=rf["strides"], ceiling=ceil)
        cells = []
        for n, sm in sims.items():
            if cls in sm:
                row[n] = dict(emg.compare(rf, sm[cls], unreliable), strides=sm[cls]["strides"])
                cells.append(f"{row[n]['mean_r']:.2f} / {row[n]['mean_on_off_agreement']:.2f} ({sm[cls]['strides']})")
            else:
                cells.append("no strides")
        report["classes"][cls] = row
        c = f"{ceil['mean_r']:.2f} / {ceil['mean_on_off_agreement']:.2f}" if ceil else "-"
        print(f"{cls:12s} {rf['strides']:12d} {c:>18s} " + " ".join(f"{x:>24s}" for x in cells))
        plot(runs[0] / f"emg_{cls}.png", cls, channels, unreliable, rf, {n: sm.get(cls) for n, sm in sims.items()})
    (runs[0] / "emg_timing.json").write_text(json.dumps(report, indent=1, default=float))
    print("->", runs[0] / "emg_timing.json")


def plot(path: Path, cls: str, channels: list[str], unreliable: tuple[str, ...], ref: dict, sims: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    x = np.linspace(0, 100, 101)
    fig, ax = plt.subplots(3, 4, figsize=(13, 8), sharex=True)
    colours = ["tab:blue", "0.45", "tab:orange"]
    for k, ch in enumerate(channels):
        a = ax.flat[k]
        on = ref["on_prob"][k] >= 0.5
        for lo, hi in emg.accept.runs(on):
            a.axvspan(x[lo], x[hi], color="0.9", lw=0)
        a.plot(x, ref["mean"][k], color="k", lw=1.6, label="real (held-out subjects)")
        for (n, sm), col in zip(sims.items(), colours, strict=False):
            if sm is not None:
                a.plot(x, sm["mean"][k], color=col, lw=1.3, label=n)
        a.set_title(ch + ("  (not scored)" if ch in unreliable else ""), fontsize=9)
        a.set_yticks([])
    ax.flat[-1].axis("off")
    ax.flat[0].legend(fontsize=7, frameon=False, loc="upper right")
    for a in ax[-1]:
        a.set_xlabel("% right stride (touchdown to touchdown)", fontsize=8)
    fig.suptitle(f"{cls}: mean EMG / activation over the right stride, each normalised to its 95th percentile; "
                 "grey = real muscle on in most strides", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


if __name__ == "__main__":
    main()
