"""The gatekeeper's exam (G0 v2): per kind of motion, does the gate keep real motion and reject bad copies of it? The
loop keeps episodes by this gate, so it must work per kind of motion, not on average.

Windows: held-out real windows (test_trial + test_terrain) and five bad copies of each (check_prior.degraded):
time warp and stiff knee (moving windows only: a still window played fast, or with its knees capped, is the same
window), foot skate, crouch, jitter. Every window and its copies take the real window's plan as their situation
(:func:`plan_features`: the planned centre-of-mass path and height, turning, contact pattern and pelvis tilt), as an
episode takes its own plan's, so moving faster, lower or differently than asked earns nothing, and motions that are
not locomotion (squats, lunges) would be told apart by what their plans ask for.

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

Style limits are in-sample unless ``--oos`` gives two fold priors (train_prior.py --fold): the prior scores motion it
was trained on higher than new real motion, so limits from its own training windows reject new real motion.
``--exclude-steep G`` is the one-step-beyond test: a prior trained without ground steeper than G (train_prior.py
--max-grade), the reference without it, and those steeper real clips as a test group.

Run: .venv/bin/python check_gate.py runs/prior_v1/prior.pt [--oos runs/prior_fold0/prior.pt,runs/prior_fold1/prior.pt]
     .venv/bin/python check_gate.py runs/prior_le5/prior.pt --exclude-steep 7.5   (CPU is fine: CUDA_VISIBLE_DEVICES=)
Out: <prior folder>/gate.json (or --out)
"""

from __future__ import annotations

import json
import zlib
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


def plan_features(D: dict, ends: np.ndarray, F: int) -> np.ndarray:
    """Per window of ``F`` frames ending at ``ends``, what its plan asks for (gp.data plans; the pelvis orientation is
    the clip's, as PlanTracking's plan): the centre of mass's horizontal speed, vertical speed and change of speed; the
    planned pelvis's turning rate (either way); the centre of mass's mean height above the ground under it, and its
    range; the shares of frames with both feet down and with neither; the planned pelvis's mean sideways and forward
    tilt (its orientation less its heading)."""
    from scipy.spatial.transform import Rotation
    idx = ends[:, None] + np.arange(1 - F, 1)
    com = np.asarray(D["com"])[idx].astype(np.float64)
    pc = np.asarray(D["plan_contact"])[idx]
    q = np.asarray(D["record"])[idx][..., data.record_layout(len(D["tracked"]))["root_quat"]].astype(np.float64)
    dt, m = (F - 1) / data.HZ, F // 2
    v = lambda a, b: np.linalg.norm((com[:, b] - com[:, a])[:, [0, 2]], axis=1) / ((b - a) / data.HZ)  # noqa: E731
    x, y, z, w = (q[..., i] for i in range(4))
    psi = np.arctan2(-2 * (x * z - w * y), 1 - 2 * (y * y + z * z))
    turn = np.abs(np.angle(np.exp(1j * (psi[:, -1] - psi[:, 0])))) / dt
    yaw = Rotation.from_rotvec(np.c_[np.zeros(psi.size), psi.ravel(), np.zeros(psi.size)])
    tilt = (yaw.inv() * Rotation.from_quat(q.reshape(-1, 4))).as_rotvec().reshape(*psi.shape, 3)
    clips = np.searchsorted(D["start"], ends, side="right") - 1
    h = np.stack([com[i, :, 1] - data.ground_under(D, int(c), com[i, :, 0], com[i, :, 2]) for i, c in enumerate(clips)])
    return np.stack([v(0, F - 1), (com[:, -1, 1] - com[:, 0, 1]) / dt, (v(m, F - 1) - v(0, m)) / (dt / 2), turn,
                     h.mean(1), h.max(1) - h.min(1), pc.all(-1).mean(1), (~pc.any(-1)).mean(1),
                     tilt[..., 0].mean(1), tilt[..., 2].mean(1)], 1)


def sample_ends(D: dict, n: int, before: int, rng: np.random.Generator) -> np.ndarray:
    """``n`` window ends with at least ``before`` frames of the same clip before them."""
    valid = np.concatenate([np.arange(s + before, s + k) for s, k in zip(D["start"], D["frames"], strict=True) if k > before])
    return rng.choice(valid, min(n, len(valid)), replace=False)


def main() -> None:
    import argparse

    from sprint.recipes.motion import model_layout
    ap = argparse.ArgumentParser()
    ap.add_argument("prior", help="the gate's prior")
    ap.add_argument("--oos", default="", help="F0,F1: priors trained on the two motion-group folds (train_prior.py "
                                              "--fold); the reference windows are scored by the one that never saw them")
    ap.add_argument("--exclude-steep", type=float, default=None, help="G: reference without ground steeper than G deg, "
                                                                       "and those steeper training clips as a test group")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    path = Path(a.prior)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    prior = Prior.load(path, dev)
    model = data.load_model()
    L = model_layout(model)
    rng = np.random.default_rng(0)
    # reference: real training windows (style: the last W + 1 frames; kinematics: the last KIN frames)
    Dr = data.load("train")
    steep = np.abs(np.asarray(Dr["grade_deg"])) > (a.exclude_steep if a.exclude_steep is not None else np.inf)
    Ds, Dr = data.take(Dr, np.flatnonzero(steep)) if steep.any() else None, data.take(Dr, np.flatnonzero(~steep))
    ends = sample_ends(Dr, N_REF, KIN - 1, rng)
    rclip = np.searchsorted(Dr["start"], ends, side="right") - 1
    rec_s, rec_k = Dr["record"][ends[:, None] + np.arange(-W, 1)], Dr["record"][ends[:, None] + np.arange(1 - KIN, 1)]
    if a.oos:
        folds = [Prior.load(f, dev) for f in a.oos.split(",")]
        assert len(folds) == 2, "--oos takes the two fold priors"
        fold = np.array([zlib.crc32(data.motion_group(str(Dr["name"][c])).encode()) % 2 for c in rclip])
        ref_s = np.empty(len(ends))
        for j in (0, 1):                                   # fold j's windows, scored by the prior trained on the other
            ref_s[fold == j] = score(folds[1 - j], rec_s[fold == j])
    else:
        ref_s = score(prior, rec_s)
    style_lim = Limits(plan_features(Dr, ends, W + 1), ref_s[:, None])
    kin_lim = Limits(plan_features(Dr, ends, KIN), kinematics(rec_k, rclip, Dr))
    ref_cls = np.asarray(Dr["terrain_class"])[rclip]
    lim_cls = {c: float(np.percentile(ref_s[ref_cls == c], PCT)) for c in sorted(set(ref_cls))}
    # test windows and their bad copies: held-out splits, and the steep training clips when excluded from the reference
    sets = [(None, data.load((data.HELD_OUT, "test_terrain")))]
    if Ds is not None:
        sets.append((f"steeper than {a.exclude_steep:g} deg (unseen)", Ds))
    margin = int(np.ceil(KIN * WARP)) + 1
    grp, moving = [], []
    acc: dict = {k: {g: [] for g in GATES} for k in ("real", *KINDS)}
    for label, D in sets:
        te = sample_ends(D, N_TEST, margin, rng)
        ci = np.searchsorted(D["start"], te, side="right") - 1
        grp += [f"{label or D['split'][c]}: {D['terrain_class'][c]}/{D['mode'][c]}" for c in ci]
        lim_s = style_lim.limit(plan_features(D, te, W + 1), [PCT])[:, 0]
        lo, hi1, hi2 = kin_lim.limit(plan_features(D, te, KIN), [KIN_PCT, 100 - KIN_PCT, 100 - KIN_PCT]).T
        lim_c = np.array([lim_cls.get(str(D["terrain_class"][c]), np.nan) for c in ci])
        real_k = degraded(D, model, L, te, "real", KIN)
        moving.append(context(real_k)[:, 0] > 0.3)
        for kind in ("real", *KINDS):
            rs, rk = degraded(D, model, L, te, kind), (real_k if kind == "real" else degraded(D, model, L, te, kind, KIN))
            sc, k = score(prior, rs), kinematics(rk, ci, D)
            kin_ok = (k[:, 0] >= lo) & (k[:, 1] <= hi1) & (k[:, 2] <= hi2)
            for g, ok in (("class", sc >= lim_c), ("style", sc >= lim_s), ("kin", kin_ok), ("combined", (sc >= lim_s) & kin_ok)):
                acc[kind][g].append(ok)
        print(f"scored {label or 'held-out'}", flush=True)
    grp, moving = np.array(grp), np.concatenate(moving)
    acc = {k: {g: np.concatenate(v) for g, v in d.items()} for k, d in acc.items()}
    out: dict = {"prior": str(path), "oos": a.oos, "exclude_steep": a.exclude_steep, "n_ref": len(ref_s), "k": K,
                 "pct": PCT, "kin_pct": KIN_PCT, "target": TARGET, "situation": "plan features", "groups": {}}
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
            print(f"  {g:48s} n {r['windows']:4d}  {x['false_reject']:.2f} | " + " ".join(
                f"{x[f'accept_{k}']:6.2f}" if x[f"accept_{k}"] is not None else "     -" for k in KINDS))
    dest = Path(a.out) if a.out else path.parent / "gate.json"
    dest.write_text(json.dumps(out, indent=1))
    print("->", dest)


if __name__ == "__main__":
    main()
