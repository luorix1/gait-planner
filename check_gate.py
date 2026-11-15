"""The style gate's test (G0 v2): per kind of motion, does a gate built on the style prior keep real motion and reject
bad copies of it? The loop keeps episodes by this gate, so it must work per kind of motion, not on average.

Windows: held-out real windows (test_trial + test_terrain) and five bad copies of each (check_prior.degraded):
time warp (moving windows only: a still window played fast is the same window), foot skate, crouch, stiff knee,
jitter. Two gates, both rejecting a window that scores below the 5th percentile of real training windows:
- ``class``: one limit per terrain class (what the filter uses today, on single windows);
- ``context``: the limit of the window's ``K`` nearest real training windows by situation: horizontal speed, turning
  rate and vertical speed of the root over the window. A bad copy takes its real window's situation, as an episode
  takes its plan's, so running faster or slower than asked earns nothing.
Per group (terrain class / mode): the share of real windows rejected (false rejects) and of each kind of bad copy
accepted (false accepts). Target (``TARGET``): at most 10 % each, in every group.

Run: .venv/bin/python check_gate.py runs/prior_v1/prior.pt   (CPU is fine: CUDA_VISIBLE_DEVICES= ...)
Out: <prior folder>/gate.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

from check_prior import WARP, degraded, score
from gp import data
from gp.smp import Prior, W

N_REF = 20_000             # real training windows scored as the reference
N_TEST = 3000              # held-out windows (each with its bad copies)
K = 100                    # neighbours for the context gate
PCT = 5                    # reject below this percentile of the reference scores
TARGET = 0.10
KINDS = ("time_warp", "foot_skate", "crouch", "stiff_knee", "jitter")
SCALE = np.array([0.5, 0.5, 0.2])     # context units: m/s, rad/s, m/s


def context(rec: np.ndarray) -> np.ndarray:
    """Per window ``[n, W + 1, D]`` of style records: root horizontal speed, turning rate, vertical speed."""
    lay = data.record_layout(rec.shape[-1] - 7 - 3 * len(data.KEY_BODIES) - 1)
    p, q = rec[..., lay["root_pos"]], rec[..., lay["root_quat"]]
    dt = W / data.HZ
    x, y, z, w = (q[..., i] for i in range(4))
    psi = np.arctan2(-2 * (x * z - w * y), 1 - 2 * (y * y + z * z))
    turn = np.angle(np.exp(1j * (psi[:, -1] - psi[:, 0]))) / dt
    return np.stack([np.linalg.norm((p[:, -1] - p[:, 0])[:, [0, 2]], axis=1) / dt, turn,
                     (p[:, -1, 1] - p[:, 0, 1]) / dt], 1)


def main() -> None:
    from sprint.recipes.motion import model_layout
    path = Path(sys.argv[1])
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    prior = Prior.load(path, dev)
    model = data.load_model()
    L = model_layout(model)
    rng = np.random.default_rng(0)
    # reference: real training windows, their scores, situations and classes
    Dr = data.load("train")
    ends = np.concatenate([np.arange(s + W, s + n) for s, n in zip(Dr["start"], Dr["frames"], strict=True)])
    ends = rng.choice(ends, min(N_REF, len(ends)), replace=False)
    rec = Dr["record"][ends[:, None] + np.arange(-W, 1)]
    ref_s, ref_c = score(prior, rec), context(rec)
    ref_cls = np.asarray(Dr["terrain_class"])[np.searchsorted(Dr["start"], ends, side="right") - 1]
    lim_cls = {c: float(np.percentile(ref_s[ref_cls == c], PCT)) for c in sorted(set(ref_cls))}
    # held-out windows and their bad copies
    D = data.load((data.HELD_OUT, "test_terrain"))
    margin = int(np.ceil(W * WARP)) + 1
    valid = np.concatenate([np.arange(s + margin, s + n) for s, n in zip(D["start"], D["frames"], strict=True)])
    ends = rng.choice(valid, min(N_TEST, len(valid)), replace=False)
    ci = np.searchsorted(D["start"], ends, side="right") - 1
    cls = np.asarray(D["terrain_class"])[ci]
    grp = np.array([f"{D['terrain_class'][c]}/{D['mode'][c]}" for c in ci])
    real_rec = degraded(D, model, L, ends, "real")
    ctx = context(real_rec)
    d2 = (((ctx[:, None] - ref_c[None]) / SCALE) ** 2).sum(-1)
    nn = np.argpartition(d2, K, axis=1)[:, :K]
    lim_ctx = np.percentile(ref_s[nn], PCT, axis=1)
    lim_c = np.array([lim_cls.get(c, np.nan) for c in cls])
    moving = ctx[:, 0] > 0.3
    scores = {"real": score(prior, real_rec)}
    for k in KINDS:
        scores[k] = score(prior, degraded(D, model, L, ends, k))
    out: dict = {"prior": str(path), "n_ref": len(ref_s), "k": K, "pct": PCT, "target": TARGET, "groups": {}}
    for g in sorted(set(grp)):
        m = grp == g
        row = {"windows": int(m.sum())}
        for gate, lim in (("class", lim_c), ("context", lim_ctx)):
            r = {"false_reject": float((scores["real"][m] < lim[m]).mean())}
            for k in KINDS:
                mk = m & moving if k == "time_warp" else m
                r[f"accept_{k}"] = float((scores[k][mk] >= lim[mk]).mean()) if mk.sum() >= 10 else None
            r["pass"] = bool(r["false_reject"] <= TARGET and all(v is None or v <= TARGET for kk, v in r.items()
                                                                  if kk.startswith("accept_")))
            row[gate] = r
        out["groups"][g] = row
        print(f"{g:22s} n {m.sum():4d}  " + "  |  ".join(
            f"{gate}: rej {row[gate]['false_reject']:.2f} acc " + " ".join(
                f"{k.split('_')[0][:5]} {row[gate][f'accept_{k}']:.2f}" if row[gate][f"accept_{k}"] is not None
                else f"{k.split('_')[0][:5]}  -  " for k in KINDS) for gate in ("class", "context")), flush=True)
    for gate in ("class", "context"):
        out[f"{gate}_pass"] = all(r[gate]["pass"] for r in out["groups"].values())
    (path.parent / "gate.json").write_text(json.dumps(out, indent=1))
    print("class gate", "PASS" if out["class_pass"] else "FAIL", "| context gate", "PASS" if out["context_pass"] else "FAIL",
          "->", path.parent / "gate.json")


if __name__ == "__main__":
    main()
