"""The gatekeeper's exam (G0 v2): per kind of motion, does the gate keep real motion and reject bad copies of it? The
loop keeps episodes by this gate, so it must work per kind of motion, not on average.

Windows: held-out real windows (test_trial + test_terrain) and five bad copies of each (check_prior.degraded):
time warp and stiff knee (moving windows only: a still window played fast, or with its knees capped, is the same
window), foot skate, crouch, jitter. Every window and its copies take the real window's situation (root horizontal
speed, turning rate, vertical speed, change of speed), as an episode takes its plan's, so moving faster or slower than
asked earns nothing.

Every limit comes from one recipe, the same for each measure (no per-measure rules): a straight-line trend of the
measure against the situation over all real training windows (so limits extend to situations never seen, such as
steeper ground), plus a percentile of the residuals of the window's ``K`` nearest real windows (situations scaled by
their own spread in the data). Measures and percentiles:
- style score: reject below the 5th (``PCT``);
- over 2 s (``kin``): the knees' peak flexion below the 2nd (a stiff knee), their least flexion above the 98th (a
  crouch), the planted feet's sliding speed above the 98th (skating; the simulation's contact check measures this too).
Gates: ``class`` (style score above its terrain class's 5th percentile: what the filter uses today, for reference),
``style`` (the recipe), ``kin`` (the recipe, three checks), ``combined`` (``style`` and ``kin``).
Per group (split, terrain class / mode): the share of real windows rejected (false rejects) and of each kind of bad copy
accepted (false accepts). Target (``TARGET``): at most 10 % each, in every group. test_trial holds familiar situations;
test_terrain the +-15 deg ramps, steeper than any training ground.

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

N_REF = 20_000             # real training windows as the reference
N_TEST = 3000              # held-out windows (each with its bad copies)
K = 100                    # neighbours for the context limits
PCT = 5                    # style: reject below this percentile of the neighbours' scores
KIN_PCT = 2                # kinematics: reject beyond this percentile (each tail used)
KIN = int(2 * data.HZ)     # kinematic window: 2 s
PLANT_M = 0.02             # a foot point within this of its lowest in the window counts as planted
TARGET = 0.10
KINDS = ("time_warp", "foot_skate", "crouch", "stiff_knee", "jitter")
MOVING_ONLY = ("time_warp", "stiff_knee")
GATES = ("class", "style", "kin", "combined")


def _layout(rec: np.ndarray) -> dict:
    return data.record_layout(rec.shape[-1] - 7 - 3 * len(data.KEY_BODIES) - 1)


def context(rec: np.ndarray) -> np.ndarray:
    """Per window ``[n, F, D]`` of style records, the situation: root horizontal speed, turning rate (either way),
    vertical speed, and change of horizontal speed (second half against first)."""
    lay = _layout(rec)
    p, q = rec[..., lay["root_pos"]], rec[..., lay["root_quat"]]
    F = rec.shape[1]
    dt, m = (F - 1) / data.HZ, F // 2
    x, y, z, w = (q[..., i] for i in range(4))
    psi = np.arctan2(-2 * (x * z - w * y), 1 - 2 * (y * y + z * z))
    turn = np.abs(np.angle(np.exp(1j * (psi[:, -1] - psi[:, 0])))) / dt
    v = lambda a, b: np.linalg.norm((p[:, b] - p[:, a])[:, [0, 2]], axis=1) / ((b - a) / data.HZ)  # noqa: E731
    return np.stack([v(0, F - 1), turn, (p[:, -1, 1] - p[:, 0, 1]) / dt, (v(m, F - 1) - v(0, m)) / (dt / 2)], 1)


def kinematics(rec: np.ndarray, clips: np.ndarray, D: dict) -> np.ndarray:
    """Per window ``[n, F, D]`` (of clips ``clips`` of ``D``): the knees' peak flexion and least flexion (deg, over both
    knees), and the planted feet's sliding speed (m/s: per heel and toe origin, the median horizontal speed over the
    frames it is within ``PLANT_M`` of its lowest above the ground; the fastest of the four)."""
    lay = _layout(rec)
    J = rec[..., lay["joints"]]
    tr = list(D["tracked"])
    knees = np.degrees(J[..., [tr.index("knee_angle_l"), tr.index("knee_angle_r")]])
    feet = rec[..., lay["key_pos"]].reshape(*rec.shape[:2], -1, 3)[:, :, :4]           # calcn l, r, toes l, r
    slip = np.zeros(len(rec))
    for i, c in enumerate(clips):
        f = feet[i]
        h = f[..., 1] - data.ground_under(D, int(c), f[..., 0].astype(np.float64), f[..., 2].astype(np.float64))
        v = np.r_[np.zeros((1, 4)), np.linalg.norm(np.diff(f[..., [0, 2]], axis=0), axis=-1) * data.HZ]
        planted = h <= h.min(0) + PLANT_M
        slip[i] = max(float(np.median(v[planted[:, k], k])) for k in range(4))
    return np.stack([knees.max((1, 2)), knees.min((1, 2)), slip], 1)


class Limits:
    """The recipe (module docstring) for one set of reference windows: situations ``ctx`` ``[n, c]``, measures ``y``
    ``[n, m]``. ``limit(x, pct)``: per window at situations ``x``, per measure, trend plus the ``pct`` percentile of the
    ``K`` nearest reference windows' residuals."""

    def __init__(self, ctx: np.ndarray, y: np.ndarray) -> None:
        self.mu, self.sd = ctx.mean(0), ctx.std(0) + 1e-9
        self.z = (ctx - self.mu) / self.sd
        X = np.c_[np.ones(len(ctx)), self.z]
        self.beta = np.linalg.lstsq(X, y, rcond=None)[0]
        self.res = y - X @ self.beta

    def limit(self, x: np.ndarray, pct: list[float]) -> np.ndarray:
        z = (x - self.mu) / self.sd
        trend = np.c_[np.ones(len(z)), z] @ self.beta
        nn = np.concatenate([np.argpartition(((z[a:a + 500, None] - self.z[None]) ** 2).sum(-1), K, axis=1)[:, :K]
                             for a in range(0, len(z), 500)])            # chunks keep the distance matrix small
        return trend + np.stack([np.percentile(self.res[nn, j], p, axis=1) for j, p in enumerate(pct)], 1)


def main() -> None:
    from sprint.recipes.motion import model_layout
    path = Path(sys.argv[1])
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    prior = Prior.load(path, dev)
    model = data.load_model()
    L = model_layout(model)
    rng = np.random.default_rng(0)
    # reference: real training windows (style: the last W + 1 frames; kinematics: the last KIN frames)
    Dr = data.load("train")
    ends = np.concatenate([np.arange(s + KIN - 1, s + n) for s, n in zip(Dr["start"], Dr["frames"], strict=True)])
    ends = rng.choice(ends, min(N_REF, len(ends)), replace=False)
    rclip = np.searchsorted(Dr["start"], ends, side="right") - 1
    rec_s, rec_k = Dr["record"][ends[:, None] + np.arange(-W, 1)], Dr["record"][ends[:, None] + np.arange(1 - KIN, 1)]
    ref_s = score(prior, rec_s)
    style_lim = Limits(context(rec_s), ref_s[:, None])
    kin_lim = Limits(context(rec_k), kinematics(rec_k, rclip, Dr))
    ref_cls = np.asarray(Dr["terrain_class"])[rclip]
    lim_cls = {c: float(np.percentile(ref_s[ref_cls == c], PCT)) for c in sorted(set(ref_cls))}
    # held-out windows and their bad copies
    D = data.load((data.HELD_OUT, "test_terrain"))
    margin = int(np.ceil(KIN * WARP)) + 1
    valid = np.concatenate([np.arange(s + margin, s + n) for s, n in zip(D["start"], D["frames"], strict=True)])
    ends = rng.choice(valid, min(N_TEST, len(valid)), replace=False)
    ci = np.searchsorted(D["start"], ends, side="right") - 1
    cls = np.asarray(D["terrain_class"])[ci]
    grp = np.array([f"{D['split'][c]}: {D['terrain_class'][c]}/{D['mode'][c]}" for c in ci])
    real_s, real_k = degraded(D, model, L, ends, "real"), degraded(D, model, L, ends, "real", KIN)
    lim_s = style_lim.limit(context(real_s), [PCT])[:, 0]
    kin_lo, kin_hi1, kin_hi2 = kin_lim.limit(context(real_k), [KIN_PCT, 100 - KIN_PCT, 100 - KIN_PCT]).T
    lim_c = np.array([lim_cls.get(c, np.nan) for c in cls])
    moving = context(real_k)[:, 0] > 0.3

    def verdicts(rs: np.ndarray, rk: np.ndarray) -> dict:
        s, k = score(prior, rs), kinematics(rk, ci, D)
        kin_ok = (k[:, 0] >= kin_lo) & (k[:, 1] <= kin_hi1) & (k[:, 2] <= kin_hi2)
        return dict(**{"class": s >= lim_c, "style": s >= lim_s}, kin=kin_ok, combined=(s >= lim_s) & kin_ok)

    acc = {"real": verdicts(real_s, real_k)}
    for kind in KINDS:
        acc[kind] = verdicts(degraded(D, model, L, ends, kind), degraded(D, model, L, ends, kind, KIN))
        print(f"scored {kind}", flush=True)
    out: dict = {"prior": str(path), "n_ref": len(ref_s), "k": K, "pct": PCT, "kin_pct": KIN_PCT, "target": TARGET,
                 "groups": {}}
    for g in sorted(set(grp)):
        m = grp == g
        row: dict = {"windows": int(m.sum())}
        for gate in GATES:
            r = {"false_reject": float((~acc["real"][gate][m]).mean())}
            for kind in KINDS:
                mk = m & moving if kind in MOVING_ONLY else m
                r[f"accept_{kind}"] = float(acc[kind][gate][mk].mean()) if mk.sum() >= 10 else None
            r["pass"] = bool(r["false_reject"] <= TARGET and all(v is None or v <= TARGET for kk, v in r.items()
                                                                  if kk.startswith("accept_")))
            row[gate] = r
        out["groups"][g] = row
    for gate in GATES:
        out[f"{gate}_pass"] = all(r[gate]["pass"] for r in out["groups"].values())
        print(f"\n{gate} gate {'PASS' if out[f'{gate}_pass'] else 'FAIL'}: false reject | false accepts "
              + " ".join(k.split("_")[0][:6] for k in KINDS))
        for g, r in out["groups"].items():
            x = r[gate]
            print(f"  {g:34s} n {r['windows']:4d}  {x['false_reject']:.2f} | " + " ".join(
                f"{x[f'accept_{k}']:6.2f}" if x[f"accept_{k}"] is not None else "     -" for k in KINDS))
    (path.parent / "gate.json").write_text(json.dumps(out, indent=1))
    print("->", path.parent / "gate.json")


if __name__ == "__main__":
    main()
