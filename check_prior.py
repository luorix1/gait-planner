"""Gate G0 for the style prior: does it score held-out subjects' real motion above degraded copies of the same
windows, and does it do so for every subject?

Degradations, each built from the real window's coordinates and re-run through the same forward kinematics:
- ``time_warp``: the same motion played 1.5x faster;
- ``foot_skate``: the whole body slid forward at an extra 0.5 m/s, joints unchanged (planted feet skate);
- ``crouch``: hips +15 deg, knees +25 deg, ankles +10 deg dorsiflexion, the root lowered to keep the feet down;
- ``stiff_knee`` (check_gate.py): knee flexion capped at 15 deg;
- ``jitter`` (check_gate.py): 3 deg of white noise on every rotational coordinate, frame by frame.

Score = minus the prior's mean SDS loss (4 noise draws, fixed seed). Pass (the plan's G0): real beats its degraded
copy in >= 95 % of pairs, and with the threshold at the degraded copies' 95th-percentile score, no held-out
subject has more than 10 % of its real windows below it. Held-out terrain clips are reported beside it.

Unseen skills (AB01 prior only): Camargo's stair clips are real motion the prior never saw. They are scored beside
Camargo level walking (same subjects, same welded trunk), so the gap between the two is the new skill's cost alone.
Stairs below the threshold about as often as Camargo level = a frozen walking prior can admit stairs in the loop.

Run: .venv/bin/python check_prior.py runs/prior_v1/prior.pt
Out: <prior folder>/g0.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

from gp import data
from gp.smp import Prior, W

N_WINDOWS = 1500
WARP = 1.5
SKATE_MPS = 0.5
CROUCH_DEG = {"hip_flexion": 15.0, "knee_angle": 25.0, "ankle_angle": 10.0}
STIFF_KNEE_DEG = 15.0
JITTER_DEG = 3.0


def degraded(D: dict, model: object, L: object, ends: np.ndarray, kind: str, frames: int = W + 1) -> np.ndarray:
    """Style records ``[n, frames, D]`` of the windows ending at frames ``ends`` (``kind`` = real or a degradation)."""
    from sprint.recipes.motion import forward_kinematics_np
    names = D["coord_names"]
    key = [list(model.body_names).index(b) for b in data.KEY_BODIES]
    clip = np.searchsorted(D["start"], ends, side="right") - 1
    out = []
    for e, c in zip(ends, clip, strict=True):
        s0 = D["start"][c]
        k = np.arange(1 - frames, 1, dtype=float)
        idx = e + (k * WARP if kind == "time_warp" else k)
        idx = np.clip(idx, s0, None)
        lo = np.floor(idx).astype(int)
        hi = np.minimum(lo + 1, s0 + D["frames"][c] - 1)
        q = D["qpos"][lo] + (idx - lo)[:, None] * (D["qpos"][hi] - D["qpos"][lo])
        if kind == "foot_skate":
            rot = q[:, names.index("pelvis_rotation")]
            t = (k - k[0]) / data.HZ
            q[:, names.index("pelvis_tx")] += SKATE_MPS * t * np.cos(rot)
            q[:, names.index("pelvis_tz")] -= SKATE_MPS * t * np.sin(rot)
        if kind == "stiff_knee":
            for side in ("_l", "_r"):
                j = names.index("knee_angle" + side)
                q[:, j] = np.minimum(q[:, j], np.radians(STIFF_KNEE_DEG))
        if kind == "jitter":
            rot = [j for j, n in enumerate(names) if not n.startswith("pelvis_t")]
            q[:, rot] += np.random.default_rng(int(e)).normal(0.0, np.radians(JITTER_DEG), (len(q), len(rot)))
        xp, xq = forward_kinematics_np(model, q.astype(np.float32))
        if kind == "crouch":
            from sprint.recipes.motion import sphere_bottoms
            low0 = sphere_bottoms(model, q.astype(np.float32), L)[..., 1].min(1)
            for n, deg in CROUCH_DEG.items():
                for s in ("_l", "_r"):
                    q[:, names.index(n + s)] += np.radians(deg)
            q[:, names.index("pelvis_ty")] -= (sphere_bottoms(model, q.astype(np.float32), L)[..., 1].min(1) - low0).mean()
            xp, xq = forward_kinematics_np(model, q.astype(np.float32))
        ground = data.ground_under(D, c, xp[:, 0, 0].astype(np.float64), xp[:, 0, 2].astype(np.float64))
        out.append(data.style_records(xp.astype(np.float64), xq.astype(np.float64), q, L.tracked, key, ground))
    return np.stack(out).astype(np.float32)


@torch.no_grad()
def score(prior: Prior, rec: np.ndarray, draws: int = 4, c: torch.Tensor | None = None) -> np.ndarray:
    """Minus the mean SDS loss; ``c``: a conditional prior's labels (Prior.labels; None = its null label)."""
    torch.manual_seed(0)
    x = prior.features(torch.as_tensor(rec, device=prior.ab.device))
    return -np.mean([prior.sds(x, c=c).mean(1).cpu().numpy() for _ in range(draws)], 0)


def unseen_skills(prior: Prior, model: object, L: object, thr: float, rng: np.random.Generator) -> dict:
    """Share of real Camargo windows below ``thr`` and median score, per class (empty for the v1 prior: it saw them)."""
    src = data.REF
    if data.DATASET == "v1" or not (src / "clips.npz").exists():
        return {}
    D = data.load(("train", "test_subject", "test_terrain"), src)
    out = {}
    for cls in ("level", "stairs_up", "stairs_down"):
        keep = np.flatnonzero((D["terrain_class"] == cls) & (D["source"] == "camargo2021"))
        valid = np.concatenate([np.arange(D["start"][c] + W + 1, D["start"][c] + D["frames"][c]) for c in keep])
        e = rng.choice(valid, min(N_WINDOWS // 3, len(valid)), replace=False)
        s = score(prior, degraded(D, model, L, e, "real"), c=prior.labels(D, np.searchsorted(D["start"], e, side="right") - 1))
        out[f"camargo_{cls}"] = {"below_threshold": float((s < thr).mean()), "score_median": float(np.median(s))}
    return out


def main() -> None:
    from sprint.recipes.motion import model_layout
    path = Path(sys.argv[1])
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    prior = Prior.load(path, dev)
    model = data.load_model()
    L = model_layout(model)
    rng = np.random.default_rng(0)
    report: dict = {}
    for split in (data.HELD_OUT, "test_terrain"):
        D = data.load(split)
        margin = int(np.ceil(W * WARP)) + 1
        valid = np.concatenate([np.arange(s + margin, s + n) for s, n in zip(D["start"], D["frames"], strict=True)])
        ends = rng.choice(valid, min(N_WINDOWS, len(valid)), replace=False)
        clip = np.searchsorted(D["start"], ends, side="right") - 1
        subj, lab = D["subject"][clip], prior.labels(D, clip)
        real = score(prior, degraded(D, model, L, ends, "real"), c=lab)
        r: dict = {"windows": len(ends), "real_score_median": float(np.median(real))}
        deg_all = []
        for kind in ("time_warp", "foot_skate", "crouch"):
            d = score(prior, degraded(D, model, L, ends, kind), c=lab)
            deg_all.append(d)
            r[f"win_vs_{kind}"] = float((real > d).mean())
        deg = np.concatenate(deg_all)
        thr = float(np.percentile(deg, 95))
        r["win_overall"] = float(np.mean([(real > d).mean() for d in deg_all]))
        r["threshold"] = thr
        r["below_threshold_by_subject"] = {str(s): float((real[subj == s] < thr).mean()) for s in sorted(set(subj))}
        r["pass"] = bool(r["win_overall"] >= 0.95 and max(r["below_threshold_by_subject"].values()) <= 0.10)
        report[split] = r
        print(split, json.dumps(r, indent=1), flush=True)
    report["unseen_skills"] = unseen_skills(prior, model, L, report[data.HELD_OUT]["threshold"], rng)
    if report["unseen_skills"]:
        print("unseen skills", json.dumps(report["unseen_skills"], indent=1), flush=True)
    report["g0_pass"] = report[data.HELD_OUT]["pass"]
    (path.parent / "g0.json").write_text(json.dumps(report, indent=1))
    print("G0", "PASS" if report["g0_pass"] else "FAIL", "->", path.parent / "g0.json")


if __name__ == "__main__":
    main()
