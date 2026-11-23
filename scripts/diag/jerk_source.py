"""Where does planner jerk come from? (2026-11-23, before v12)

Splits the scoreboard's jerk into: inside one plan vs at replan seams, joint vs pelvis trigger, and the foot-hold
post-process vs none. Open loop = one 24-frame plan from real history (no stitching); closed loop = the scoreboard's
replay + course rollouts at keep 4 and keep 8 (keep 8 leaves 3rd-difference windows that touch no seam).

Run (bolt env, from gait_planner/): GP_DATASET=v5 python scripts/diag/jerk_source.py RUN CKPT [--steps N] [--out FILE]
(the 11-23 results in runs/diag_jerk ran without GP_DATASET=v5: held-out set minus the SEED stair replays)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from export_pose2 import NS, circmean, decode, frame_features, hold_feet, rollout, start  # noqa: E402
from gp import data, footfk  # noqa: E402
from gp.planner import FUT, HIST, HZ, STRIDE, to_frame  # noqa: E402
from gp.pose_planner import PosePlanner, patch  # noqa: E402
from scoreboard import HOLD, Scoreboard  # noqa: E402


def d3(x: np.ndarray) -> np.ndarray:
    """|3rd difference| * HZ^3 along axis 1: joints -> max over joints (rad/s^3), pelvis -> norm (m/s^3)."""
    return np.diff(x, 3, axis=1) * HZ ** 3


def acc(x: np.ndarray) -> np.ndarray:
    return (x[:, 2:] - 2 * x[:, 1:-1] + x[:, :-2]) * HZ ** 2


def flags(j: np.ndarray, p: np.ndarray, thr: tuple[float, float]) -> tuple[np.ndarray, np.ndarray]:
    return np.abs(d3(j)).max(-1) > thr[0], np.linalg.norm(d3(p), axis=-1) > thr[1]


def closed(B: Scoreboard, P: PosePlanner, keep: int, hold: int) -> dict:
    W, dev = B.W, B.dev
    a_rep, last = B.a_rep, B.clip_last
    n = int(round(8.0 * HZ / keep))
    gen = torch.Generator(device=dev).manual_seed(0)

    def replay_goal(i, o, psi):
        tgt = torch.minimum(a_rep + i * keep * STRIDE + STRIDE * FUT, last[W.fr.clip[a_rep]])
        return to_frame(W.fr.pcom[tgt][:, None, [0, 2]], o, psi)[:, 0]

    def course_goal(i, o, psi):
        t = (i * keep + FUT) / HZ
        ahead = torch.as_tensor(np.stack([B.track_at(c, np.array([t]))[0] for c in range(len(B.courses))]),
                                dtype=torch.float32, device=dev)
        return to_frame(ahead[:, None], o, psi)[:, 0]
    with torch.enable_grad():
        r1 = rollout(P, W.fr.G, W.fr.clip[a_rep], *start(W, a_rep), replay_goal, n, keep, 1.0, gen, hold)
        r2 = rollout(P, B.G, B.gid, *B.c_start, course_goal, n, keep, 1.0, gen, hold)
    j = np.concatenate([r1["joints"], r2["joints"]])
    p = np.concatenate([r1["pelvis"], r2["pelvis"]])
    cls = [f"replay {B.D['terrain_class'][k]}" for k in B.rep] + [f"course {c}" for _, c, _, _ in B.courses]
    fj, fp = flags(j, p, B.thr)
    i = np.arange(fj.shape[1])                       # window i covers frames i..i+3; frame 0 = last given frame
    touch = np.array([any(i0 <= b < i0 + 3 for b in range(0, fj.shape[1] + 3, keep)) for i0 in i])
    a_j = np.linalg.norm(np.degrees(acc(j)), axis=-1)             # deg/s^2 at frames 1..n-2
    a_p = np.linalg.norm(acc(p), axis=-1)                          # m/s^2
    f = np.arange(1, j.shape[1] - 1)
    seam = (f % keep == 0) | (f % keep == 1)
    box = np.array(["box" in c for c in cls])
    return dict(keep=keep, hold=hold,
                jerk=float((fj | fp).mean()), jerk_joint=float(fj.mean()), jerk_pelvis=float(fp.mean()),
                jerk_box=float((fj | fp)[box].mean()),
                jerk_no_seam=float((fj | fp)[:, ~touch].mean()) if (~touch).any() else None,
                jerk_seam=float((fj | fp)[:, touch].mean()),
                acc_joint_seam=float(a_j[:, seam].mean()), acc_joint_mid=float(a_j[:, ~seam].mean()),
                acc_pelvis_seam=float(a_p[:, seam].mean()), acc_pelvis_mid=float(a_p[:, ~seam].mean()))


def open_loop(B: Scoreboard, P: PosePlanner, hold: int) -> dict:
    """One plan per replay start; frames = the 8 given + 24 generated. Real = the clip's own 32 frames."""
    W, dev = B.W, B.dev
    a = B.a_rep
    s, ph = start(W, a)
    o, a_psi = s["pcom"][:, -1, [0, 2]], circmean(ph[:, -NS:])
    g0 = W.fr.G(W.fr.clip[a], s["pcom"][:, -1:, 0], s["pcom"][:, -1:, 2])[:, 0]
    hist = P.with_segments(frame_features(s, o, a_psi, g0))
    tgt = torch.minimum(a + STRIDE * FUT, B.clip_last[W.fr.clip[a]])
    goal = to_frame(W.fr.pcom[tgt][:, None, [0, 2]], o, a_psi)[:, 0]
    gen = torch.Generator(device=dev).manual_seed(0)
    fut = P.generate(hist, patch(W.fr.G, W.fr.clip[a], o, a_psi, g0, P.grid), goal, gen)
    if hold:
        with torch.enable_grad():
            fut = hold_feet(P, hist, fut, hold)
    new = decode(fut, o, a_psi, g0)
    j = torch.cat([s["joints"], new["joints"]], 1).cpu().numpy()
    k = (a[:, None] + STRIDE * torch.arange(-(HIST - 1), FUT + 1, device=dev))
    k = torch.minimum(k, B.clip_last[W.fr.clip[a]][:, None])
    jr = W.fr.joints[k].cpu().numpy()
    # pelvis: the com channel is what the planner outputs; compare com to com
    pc = torch.cat([s["pcom"], new["pcom"]], 1).cpu().numpy()
    pr = W.fr.pcom[k].cpu().numpy()
    out = {}
    for name, jj, pp in (("gen", j, pc), ("real", jr, pr)):
        fj, fp = flags(jj, pp, B.thr)
        a_j = np.linalg.norm(np.degrees(acc(jj)), axis=-1)
        a_p = np.linalg.norm(acc(pp), axis=-1)
        # windows: 0..4 inside the given frames, 5..7 straddle the junction, 8.. inside the plan
        out[name] = dict(jerk_inside=float((fj | fp)[:, HIST:].mean()), jerk_junction=float((fj | fp)[:, HIST - 3:HIST].mean()),
                         jerk_joint_inside=float(fj[:, HIST:].mean()), jerk_pelvis_inside=float(fp[:, HIST:].mean()),
                         jerk_joint_junction=float(fj[:, HIST - 3:HIST].mean()),
                         jerk_pelvis_junction=float(fp[:, HIST - 3:HIST].mean()),
                         acc_joint_junction=float(a_j[:, HIST - 2:HIST].mean()), acc_joint_inside=float(a_j[:, HIST:].mean()),
                         acc_pelvis_junction=float(a_p[:, HIST - 2:HIST].mean()), acc_pelvis_inside=float(a_p[:, HIST:].mean()),
                         jitter_joint=float(np.degrees(np.abs(np.diff(jj[:, HIST:], 3, axis=1))).mean()),
                         jitter_pelvis_mm=float(1000 * np.linalg.norm(np.diff(pp[:, HIST:], 3, axis=1), axis=-1).mean()))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("ckpt")
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    dev = "cuda"
    D = data.load(("test_trial", "test_terrain"))
    fk = footfk.fit(D, dev, data.ROOT / "data" / "foot_fk.pt")
    B = Scoreboard(D, fk, dev)
    P = PosePlanner.load(Path(a.run) / a.ckpt, dev).attach(fk)
    if a.steps:
        P.cfg["sample_steps"] = a.steps
    res = dict(run=a.run, ckpt=a.ckpt, steps=P.cfg.get("sample_steps", P.cfg["T"]), thr=B.thr,
               open={f"hold{h}": open_loop(B, P, h) for h in (0, HOLD)},
               closed=[closed(B, P, k, h) for k, h in ((4, HOLD), (4, 0), (8, HOLD), (8, 0))])
    s = json.dumps(res, indent=1)
    print(s)
    if a.out:
        Path(a.out).write_text(s)


if __name__ == "__main__":
    main()
