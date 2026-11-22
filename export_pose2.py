"""A SPRINT Studio folder from a version-2 full-pose planner (train_planner.py --pose2), rolled out on its own with no
physics (kinematic playback), plus a score of how physical the motion is.

As export_pose_planner.py: open loop, each plan's first ``--keep`` frames kept (0.2 s), the next plan from the
planner's own last 0.4 s, re-anchored at its last pelvis centre of mass and smoothed heading. Episodes:
- **replay**: from held-out windows (spread over modes and terrain); goal = where the clip's pelvis is 1.2 s ahead,
  seen from the generated anchor, so a drift off the clip's path is steered back and the goal shrinks to 0 where the
  clip stops (its last frame is held); the clip is the ghost. (Until 11-22 the clip's own 1.2 s travel in its own
  heading frame: dead reckoning, every heading error became sideways drift, and the last goal was held after the
  clip ended, so a stopping clip kept walking.)
- **command**: from a held-out level walk; goal = speed x 1.2 s along a world-fixed direction (the start's smoothed
  heading + the angle): heading drift is turned back, and an angle is one turn, then straight.

Score (summary.json, also printed), from SPRINT's exact forward kinematics of the decoded poses against each clip's
terrain: the lowest foot sphere's height (5/50/95th percentile), the share of frames with both feet more than 3 cm up
("floating") or a sphere 2 cm under ("sinking"), the median horizontal speed of spheres within 2 cm of the ground
("stance slip"), and the path speed of the pelvis. The same numbers for the real clip, where there is one.

Run: GP_DATASET=v4 .venv/bin/python export_pose2.py runs/pose2_planner_v4 [--guidance 2] [--to-studio]
Out: <run>/studio_pose2/ (+ summary.json)
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path

import numpy as np
import torch
from sprint.export import FrameSource
from sprint.export import format as fmt
from sprint.export.meshes import bake
from sprint.recipes.imitation import _qmul
from sprint.recipes.motion import _rotate, forward_kinematics_np, model_layout

from export_pose_planner import to_qpos
from gp import data, footfk
from gp.planner import FUT, HIST, HZ, STRIDE, from_frame, qexp, tilt, to_frame, yaw_quat
from gp.pose_planner import GEO_SCALE, J0, P0, SMOOTH_S, PosePlanner, Windows, _rot, geometry, patch
from train import build

NS = int(round(SMOOTH_S * HZ))              # plan frames in the heading average


def circmean(psi: torch.Tensor) -> torch.Tensor:
    return torch.atan2(psi.sin().sum(-1), psi.cos().sum(-1))


def frame_features(s: dict, o: torch.Tensor, a_psi: torch.Tensor, g0: torch.Tensor) -> torch.Tensor:
    """World frames ``s`` (each ``[B, K, ...]``) -> ``[B, K, C]`` in the anchor frame, heights over ``g0``."""
    pts = s["points"]
    pt = torch.cat([to_frame(pts[..., [0, 2]], o, a_psi), (pts[..., 1] - g0[:, None, None])[..., None]], -1).flatten(-2)
    dpsi = s["psi"] - a_psi[:, None]
    return torch.cat([s["contact"].float() * 2 - 1, to_frame(s["hold"], o, a_psi).flatten(-2),
                      to_frame(s["pcom"][..., [0, 2]], o, a_psi), (s["pcom"][..., 1] - g0[:, None])[..., None],
                      dpsi.cos()[..., None], dpsi.sin()[..., None], tilt(s["quat"], s["psi"]), s["joints"], pt], -1)


def decode(x: torch.Tensor, o: torch.Tensor, a_psi: torch.Tensor, g0: torch.Tensor) -> dict:
    """Generated frames ``[B, K, C]`` -> world frames (the inverse of :func:`frame_features`)."""
    B, K = x.shape[:2]
    pxz = from_frame(x[..., 8:10], o, a_psi)
    pcom = torch.stack([pxz[..., 0], x[..., 10] + g0[:, None], pxz[..., 1]], -1)
    psi = a_psi[:, None] + torch.atan2(x[..., 12], x[..., 11])
    quat = _qmul(yaw_quat(psi), qexp(x[..., 13:16]))
    pf = x[..., P0:].view(B, K, 4, 3)
    ptxz = from_frame(pf[..., :2], o, a_psi)
    pty = pf[..., 2] + g0[:, None, None]
    return dict(pcom=pcom, quat=quat, psi=psi, contact=x[..., :4] > 0,
                hold=from_frame(x[..., 4:8].view(B, K, 2, 2), o, a_psi), joints=x[..., J0:J0 + 17],
                points=torch.stack([ptxz[..., 0], pty, ptxz[..., 1]], -1))


def foot_hold(P: PosePlanner, hist: torch.Tensor, iters: int, from_t: int, lr: float = 0.02):
    """A ``guide`` for PosePlanner.generate: every foot point the history ends on the ground stays where it was (as
    the joints place it, which playback shows, and in the point channels) until the plan's own flag lifts it; ``iters``
    Adam steps on each clean estimate at noise steps <= ``from_t``. Replans slid the stance foot at every seam (v6 stair
    descent: 17 cm/s at seams, 2-3 within a plan, 2026-11-22)."""
    tgt = geometry(P.fk, hist[:, -1:])[1][:, 0]                      # [B, 4, 3] the last given frame's foot points
    down = hist[:, -1, :4] > 0

    def guide(x0: torch.Tensor, t: int) -> torch.Tensor:
        if t > from_t:
            return x0
        x = x0.detach().clone().requires_grad_(True)
        opt = torch.optim.Adam([x], lr=lr)
        with torch.enable_grad():
            for _ in range(iters):
                xu = x * P.x_std + P.x_mean
                ch, kin, _ = geometry(P.fk, xu)
                held = ((xu[..., :4] > 0).int().cumprod(1).bool() & down[:, None]).detach()   # until the flag lifts
                err = ((kin - tgt[:, None]).square().sum(-1) + (ch - tgt[:, None]).square().sum(-1)) / GEO_SCALE ** 2
                opt.zero_grad()
                (err * held).sum().backward()
                opt.step()
        return x.detach()
    return guide


def rollout(P: PosePlanner, W: Windows, a0: torch.Tensor, goal_of, n_plans: int, keep: int, w: float,
            gen: torch.Generator, hold: int = 0, hold_from: int = 49) -> dict[str, np.ndarray]:
    fr = W.fr
    clip = fr.clip[a0]
    k = a0[:, None] + W.k[:HIST]
    s = dict(pcom=fr.pcom[k], quat=fr.quat[k], psi=fr.psi[k], contact=fr.contact[k], hold=fr.hold[k],
             joints=fr.joints[k], points=fr.points[k])
    starts = torch.as_tensor(np.repeat(np.asarray(W.starts), np.asarray(W.lengths)), device=a0.device)[a0]
    past = torch.maximum(a0[:, None] - STRIDE * torch.arange(NS - 1, -1, -1, device=a0.device), starts[:, None])
    psi_hist = fr.psi[past]                                          # the last SMOOTH_S of headings (20 Hz)
    seq = {key: [v] for key, v in s.items()}
    for i in range(n_plans):
        last = {key: torch.cat(v, 1)[:, -HIST:] for key, v in seq.items()}
        o, a_psi = last["pcom"][:, -1, [0, 2]], circmean(psi_hist[:, -NS:])
        g0 = fr.G(clip, last["pcom"][:, -1:, 0], last["pcom"][:, -1:, 2])[:, 0]
        hist = frame_features(last, o, a_psi, g0)
        guide = foot_hold(P, hist, hold, hold_from) if hold else None
        fut = P.generate(hist, patch(fr.G, clip, o, a_psi, g0), goal_of(i, o, a_psi), gen, w=w, guide=guide)[:, :keep]
        new = decode(fut, o, a_psi, g0)
        for key in seq:
            seq[key].append(new[key])
        psi_hist = torch.cat([psi_hist, new["psi"]], 1)
    full = {key: torch.cat(v, 1)[:, HIST - 1:] for key, v in seq.items()}
    origin = full["pcom"] - _rot(full["quat"], P.fk.bcom.expand_as(full["pcom"]))
    return dict(com=full["pcom"].cpu().numpy(), quat=full["quat"].cpu().numpy(), pelvis=origin.cpu().numpy(),
                joints=full["joints"].cpu().numpy())


class Scorer:
    """Foot contact numbers from exact forward kinematics of poses on a clip's terrain."""

    def __init__(self, W: Windows) -> None:
        self.model = data.load_model()
        L, h = model_layout(self.model), self.model.host
        sph = L.contact_spheres
        self.body, self.pos, self.rad = h["geom_body"][sph].astype(int), h["geom_pos"][sph], np.asarray(h["geom_size"][sph][:, 0])
        self.side = np.array([self.model.body_names[b][-1] for b in self.body])
        self.G = W.fr.G

    def __call__(self, q: np.ndarray, clip: int) -> dict:
        xp, xq = forward_kinematics_np(self.model, q.astype(np.float32))
        c = xp[:, self.body] + _rotate(xq[:, self.body], np.broadcast_to(self.pos, xp[:, self.body].shape))
        b = c - np.stack([0 * self.rad, self.rad, 0 * self.rad], -1)
        dev = self.G.X.device
        g = self.G(torch.full((1,), clip, device=dev), torch.as_tensor(b[None, ..., 0].reshape(1, -1), device=dev),
                   torch.as_tensor(b[None, ..., 2].reshape(1, -1), device=dev)).cpu().numpy().reshape(b.shape[:2])
        gap = b[..., 1] - g
        low = np.stack([gap[:, self.side == s].min(1) for s in "lr"], 1)
        v = np.linalg.norm(np.diff(b[..., [0, 2]], axis=0), axis=-1) * data.HZ
        vf = np.stack([v[:, self.side == s].mean(1) for s in "lr"], 1)
        stance = low[1:] < 0.02
        either = low.min(1)
        p = xp[::20, 0][:, [0, 2]]
        vz = np.abs(np.diff(xp[:, 0, 1])) * data.HZ             # pelvis vertical speed: a 55 cm teleport reads 22 m/s
        return dict(pelvis_vz_max_mps=round(float(vz.max()), 2), deep_frac=round(float((gap.min(1) < -0.05).mean()), 3),min_foot_gap_cm=np.round(100 * np.percentile(either, [5, 50, 95]), 1).tolist(),
                    floating=round(float((either > 0.03).mean()), 3), sinking=round(float((either < -0.02).mean()), 3),
                    stance_slip_cms=round(float(100 * np.median(vf[stance])), 1) if stance.any() else None,
                    path_speed_mps=round(float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum() / max(len(p) - 1, 1) / 0.5), 2))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--checkpoint", default="planner_best.pt")
    ap.add_argument("--seconds", type=float, default=8.0)
    ap.add_argument("--keep", type=int, default=4)
    ap.add_argument("--guidance", type=float, default=1.0, help="1 = no guidance (11-21: w=2 worsened feet, speed already held)")
    ap.add_argument("--replays", type=int, default=8)
    ap.add_argument("--speeds", default="0.8,1.2,1.6,2.0")
    ap.add_argument("--turn-deg", type=float, default=30.0)
    ap.add_argument("--to-studio", action="store_true")
    a = ap.parse_args()
    run, dev = Path(a.run), "cuda" if torch.cuda.is_available() else "cpu"
    D = data.load(("test_trial", "test_terrain"))
    fk = footfk.fit(D, dev, data.ROOT / "data" / "foot_fk.pt")
    P = PosePlanner.load(run / a.checkpoint, dev).attach(fk)
    W = Windows(D, dev, fk)
    W.starts, W.lengths = D["start"], D["frames"]
    names = list(D["coord_names"])
    gen = torch.Generator(device=dev).manual_seed(0)
    n_plans = int(round(a.seconds * HZ / a.keep))
    room = np.asarray(D["frames"]) - STRIDE * (HIST + FUT)

    # replays: the longest clip of each (source, mode, terrain) group, then the next longest, up to --replays
    groups: dict[tuple, list[int]] = {}
    for k in np.argsort(-room):
        if room[k] > 0:
            groups.setdefault((str(D["source"][k]), str(D["mode"][k]), str(D["terrain_class"][k])), []).append(int(k))
    picked: list[int] = []
    while len(picked) < a.replays and any(groups.values()):
        for g in list(groups):
            if groups[g] and len(picked) < a.replays:
                picked.append(groups[g].pop(0))
    a_rep = torch.as_tensor([int(D["start"][k]) + STRIDE * (HIST - 1) for k in picked], device=dev)

    clip_last = torch.as_tensor(D["start"] + D["frames"] - 1, device=dev)

    def replay_goal(i: int, o: torch.Tensor, psi: torch.Tensor) -> torch.Tensor:
        tgt = torch.minimum(a_rep + i * a.keep * STRIDE + STRIDE * FUT, clip_last[W.fr.clip[a_rep]])
        return to_frame(W.fr.pcom[tgt][:, None, [0, 2]], o, psi)[:, 0]

    walk = [k for k in range(len(D["name"])) if str(D["mode"][k]) == "walk" and str(D["terrain_class"][k]) == "level"
            and str(D["source"][k]) == "ab01" and room[k] > 0]
    k_cmd = max(walk, key=lambda k: float(D["frames"][k]))
    cmds = [(float(v), 0.0) for v in a.speeds.split(",")] + ([(1.2, a.turn_deg)] if a.turn_deg else [])
    a_cmd = torch.full((len(cmds),), int(D["start"][k_cmd]) + STRIDE * (HIST - 1), device=dev)
    th = W.fr.psi_s[a_cmd] + torch.as_tensor([np.radians(d) for _, d in cmds], dtype=torch.float32, device=dev)
    reach = torch.as_tensor([v * FUT / HZ for v, _ in cmds], dtype=torch.float32, device=dev)

    def command_goal(i: int, o: torch.Tensor, psi: torch.Tensor) -> torch.Tensor:
        ahead = o + reach[:, None] * torch.stack([th.cos(), -th.sin()], -1)       # to_frame's forward convention
        return to_frame(ahead[:, None], o, psi)[:, 0]
    r_rep = rollout(P, W, a_rep, replay_goal, n_plans, a.keep, a.guidance, gen)
    r_cmd = rollout(P, W, a_cmd, command_goal, n_plans, a.keep, a.guidance, gen)
    episodes = [("replay", k, to_qpos(r_rep, b, names)) for b, k in enumerate(picked)]
    episodes += [(f"command {v:.1f} m/s" + (f" at {d:+.0f} deg" if d else ""), k_cmd, to_qpos(r_cmd, b, names))
                 for b, (v, d) in enumerate(cmds)]

    score = Scorer(W)
    env, task = build("baseline", D, 1, 0, "cpu", starts=[(k, STRIDE * (HIST - 1)) for _, k, _ in episodes])
    task.export_frame = None
    task.reference_pose = None
    src = FrameSource(env, meshes=bake(data.MODEL))
    from sprint.dynamics.forward import forward_kinematics
    from sprint.pipeline import forward
    out = run / "studio_pose2"
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    shutil.copy2(data.MODEL, out / "model.osim")
    fmt.write_model(out, src.model())
    m, d = env.model, env.data
    clips, summary = [], []
    for ep, (kind, k, q) in enumerate(episodes):
        env.reset()
        s0 = int(D["start"][k]) + STRIDE * (HIST - 1)
        real = D["qpos"][s0:min(s0 + len(q), int(D["start"][k] + D["frames"][k]))]
        src.reference = real.astype(np.float32) if kind == "replay" else None
        qd = np.gradient(q, 1.0 / data.HZ, axis=0)
        frames = []
        for t in range(len(q)):
            d.set("qpos", q[t:t + 1].astype(np.float32))
            d.set("qvel", qd[t:t + 1].astype(np.float32))
            forward_kinematics(m, d)
            forward(m, d, env.opt)
            d.body_fext_avg.zero_()
            d.con_force_avg.zero_()
            frames.append(src.frame(t / data.HZ))
        clips.append(f"{kind} from {D['name'][k]} ({D['mode'][k]}, {D['terrain_class'][k]})")
        slug = re.sub(r"[^A-Za-z0-9.+-]", "", kind.replace("command", "cmd"))
        fmt.write_episode(out, fmt.episode_name(ep, slug, 0), src.meta(len(frames), frames), frames)
        row = dict(episode=ep, kind=kind, clip=str(D["name"][k]), mode=str(D["mode"][k]),
                   terrain=str(D["terrain_class"][k]), generated=score(q, k))
        if kind == "replay":
            row["real"] = score(real, k)
        summary.append(row)
        print(row, flush=True)
    fmt.write_config(out, {"producer": "sprint.export.studio", "model_file": fmt.MODEL_NAME, "dt_ctrl": 1.0 / data.HZ,
                           "run": str(run), "clips": clips,
                           "policy": f"full-pose planner v2 {a.checkpoint}, guidance {a.guidance}, kinematic open loop"})
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(f"-> {out}  ({len(episodes)} episodes)")
    if a.to_studio:
        dest = fmt.results_dir() / f"{run.name}.{out.name}"
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(out, dest)
        print(f"-> {dest}")


if __name__ == "__main__":
    main()
