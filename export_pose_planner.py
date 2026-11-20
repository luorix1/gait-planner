"""A SPRINT Studio folder from a full-pose planner (train_planner.py --pose): the planner rolled out on its own, with no
physics (kinematic playback), so its motion can be watched before any muscle policy follows it.

Open loop, like the deployment loop minus the body: each plan's first ``--keep`` frames (default 4 = 0.2 s, the
replanning interval) are kept, and the next plan starts from the planner's own last 0.4 s. Two kinds of episode,
both from held-out clips:
- **replay**: from a held-out window's real first 0.4 s; the goal at each replan is where that clip's centre of mass
  really went over the next 1.2 s, in the clip's heading frame (its intent, as gp.planner.Replanner uses). The
  clip itself is the ghost.
- **command**: from a held-out level walk's first 0.4 s; the goal is speed x 1.2 s along a fixed direction relative
  to the current facing (0 deg = straight on; a nonzero angle turns continuously). No ghost.

Run: GP_DATASET=v3 .venv/bin/python export_pose_planner.py runs/pose_planner_v3 [--to-studio]
Out: <run>/studio_pose/ (episodes named by kind), plus <run>/studio_pose/summary.json (achieved speeds)
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation
from sprint.export import FrameSource
from sprint.export import format as fmt
from sprint.export.meshes import bake

from gp import data
from gp.planner import FUT, HIST, HZ, POSE_COORDS, STRIDE, C, Planner, Windows, features, from_frame, patch, world
from train import build

PELVIS_EULER = "ZXY"          # pelvis_tilt, pelvis_list, pelvis_rotation, intrinsic (checked against FK to 1e-8)


def rollout(P: Planner, W: Windows, a0: torch.Tensor, goal_of, n_plans: int, keep: int, gen: torch.Generator
            ) -> dict[str, np.ndarray]:
    """Plan frames (20 Hz, world) from anchors ``a0``: the real anchor frame, then ``n_plans * keep`` generated ones."""
    k = a0[:, None] + W.k[:HIST]
    seq = [[W.down[k], W.hold[k], W.com[k], W.psi[k], W.quat[k], W.pelvis[k], W.joints[k]]]
    a_com, a_psi = W.com[a0], W.psi[a0]
    hist = features(*seq[0][:5], a_com, a_psi, *seq[0][5:])
    for i in range(n_plans):
        fut = P.generate(hist, patch(W.G, W.clip[a0], a_com, a_psi), goal_of(i, a_com, a_psi), gen)[:, :keep]
        down, hold, com, psi, quat = world(fut[..., :C], a_com, a_psi)
        pxz = from_frame(fut[..., [C, C + 2]], a_com[:, [0, 2]], a_psi)     # channels: forward, up, lateral
        pel = torch.stack([pxz[..., 0], fut[..., C + 1] + a_com[:, 1:2], pxz[..., 1]], -1)
        seq.append([down, hold, com, psi, quat, pel, fut[..., C + 3:]])
        a_com, a_psi = com[:, -1], psi[:, -1]
        recent = [torch.cat([s[j] for s in seq], 1)[:, -HIST:] for j in range(7)]
        hist = features(*recent[:5], a_com, a_psi, *recent[5:])
    full = [torch.cat([s[j] for s in seq], 1)[:, HIST - 1:].cpu().numpy() for j in range(7)]
    return dict(zip(("down", "hold", "com", "psi", "quat", "pelvis", "joints"), full, strict=True))


def to_qpos(r: dict[str, np.ndarray], b: int, names: list[str]) -> np.ndarray:
    """Episode ``b``'s 20 Hz plan frames -> qpos ``[F, nv]`` on the 40 Hz control grid."""
    n = len(r["com"][b])
    eul = Rotation.from_quat(r["quat"][b]).as_euler(PELVIS_EULER)
    eul = np.unwrap(eul, axis=0)
    q20 = np.zeros((n, len(names)))
    for j, c in enumerate(("pelvis_tilt", "pelvis_list", "pelvis_rotation")):
        q20[:, names.index(c)] = eul[:, j]
    for j, c in enumerate(("pelvis_tx", "pelvis_ty", "pelvis_tz")):
        q20[:, names.index(c)] = r["pelvis"][b][:, j]
    for j, c in enumerate(POSE_COORDS):
        q20[:, names.index(c)] = r["joints"][b][:, j]
    t20 = np.arange(n) / HZ
    t40 = np.arange(int((n - 1) * STRIDE) + 1) / data.HZ
    return np.stack([np.interp(t40, t20, q20[:, j]) for j in range(len(names))], 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--checkpoint", default="planner_best.pt")
    ap.add_argument("--seconds", type=float, default=8.0)
    ap.add_argument("--keep", type=int, default=4, help="plan frames (20 Hz) kept per plan; 4 = 0.2 s")
    ap.add_argument("--replays", type=int, default=6)
    ap.add_argument("--speeds", default="0.8,1.2,1.6,2.0")
    ap.add_argument("--turn-deg", type=float, default=30.0, help="also a 1.2 m/s command this far off the facing")
    ap.add_argument("--to-studio", action="store_true")
    a = ap.parse_args()
    run, dev = Path(a.run), "cuda" if torch.cuda.is_available() else "cpu"
    P = Planner.load(run / a.checkpoint, dev)
    D = data.load(("test_trial", "test_terrain"))
    W = Windows(D, dev, pose=True)
    names = list(D["coord_names"])
    gen = torch.Generator(device=dev).manual_seed(0)
    n_plans = int(round(a.seconds * HZ / a.keep))
    last_anchor = torch.as_tensor(D["start"] + D["frames"] - STRIDE * FUT - 1, device=dev)  # per clip

    # replay starts: clips with the most room, spread over sources and modes
    room = np.asarray(D["frames"]) - STRIDE * (HIST + FUT)
    order = [k for k in np.argsort(-room) if room[k] > 0]
    picked, seen = [], set()
    for k in order:
        g = (str(D["source"][k]), str(D["mode"][k]), str(D["terrain_class"][k]))
        if g not in seen or len(seen) >= a.replays:
            picked.append(int(k))
            seen.add(g)
        if len(picked) == a.replays:
            break
    a_rep = torch.as_tensor([int(D["start"][k]) + STRIDE * (HIST - 1) for k in picked], device=dev)

    def replay_goal(i: int, a_com: torch.Tensor, a_psi: torch.Tensor) -> torch.Tensor:
        f = torch.minimum(a_rep + i * a.keep * STRIDE, last_anchor[W.clip[a_rep]])
        return W(f)[2]

    # command starts: one held-out level walk, every command from its first window
    walk = [k for k in range(len(D["name"])) if str(D["mode"][k]) == "walk" and str(D["terrain_class"][k]) == "level"
            and room[k] > 0]
    k_cmd = max(walk, key=lambda k: float(D["frames"][k])) if walk else picked[0]
    cmds = [(float(v), 0.0) for v in a.speeds.split(",")] + ([(1.2, a.turn_deg)] if a.turn_deg else [])
    a_cmd = torch.full((len(cmds),), int(D["start"][k_cmd]) + STRIDE * (HIST - 1), device=dev)
    g_cmd = torch.as_tensor([[v * FUT / HZ * np.cos(np.radians(d)), v * FUT / HZ * np.sin(np.radians(d))]
                             for v, d in cmds], dtype=torch.float32, device=dev)

    r_rep = rollout(P, W, a_rep, replay_goal, n_plans, a.keep, gen)
    r_cmd = rollout(P, W, a_cmd, lambda i, c, p: g_cmd, n_plans, a.keep, gen)

    episodes = [("replay", k, to_qpos(r_rep, b, names), r_rep["com"][b]) for b, k in enumerate(picked)]
    episodes += [(f"command {v:.1f} m/s" + (f" at {d:+.0f} deg" if d else ""), k_cmd, to_qpos(r_cmd, b, names),
                  r_cmd["com"][b]) for b, (v, d) in enumerate(cmds)]

    # a 1-world env only to draw frames (meshes, terrain); posed, never stepped
    env, task = build("baseline", D, 1, 0, "cpu", starts=[(k, STRIDE * (HIST - 1)) for _, k, _, _ in episodes])
    task.export_frame = None                     # no tracking channels: nothing is tracked
    task.reference_pose = None                   # the ghost comes from FrameSource.reference
    src = FrameSource(env, meshes=bake(data.MODEL))
    from sprint.dynamics.forward import forward_kinematics
    from sprint.pipeline import forward

    out = run / "studio_pose"
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    shutil.copy2(data.MODEL, out / "model.osim")
    fmt.write_model(out, src.model())
    clips = []
    summary = []
    m, d = env.model, env.data
    for ep, (kind, k, q, com) in enumerate(episodes):
        env.reset()
        s0 = int(D["start"][k]) + STRIDE * (HIST - 1)
        src.reference = (D["qpos"][s0:int(D["start"][k] + D["frames"][k])].astype(np.float32)
                         if kind == "replay" else None)
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
        label = f"{kind} from {D['name'][k]}"
        clips.append(label)
        slug = re.sub(r"[^A-Za-z0-9.+-]", "", kind.replace("command", "cmd"))      # "m/s" must not reach a filename
        fmt.write_episode(out, fmt.episode_name(ep, slug, 0), src.meta(len(frames), frames), frames)
        dist = np.linalg.norm(com[-1, [0, 2]] - com[0, [0, 2]])
        summary.append(dict(episode=ep, kind=kind, clip=str(D["name"][k]), seconds=round(len(q) / data.HZ, 2),
                            net_speed_mps=round(float(dist / ((len(com) - 1) / HZ)), 3),
                            clip_speed_mps=round(float(D["speed_mps"][k]), 3)))
        print(summary[-1], flush=True)
    fmt.write_config(out, {"producer": "sprint.export.studio", "model_file": fmt.MODEL_NAME,
                           "dt_ctrl": 1.0 / data.HZ, "run": str(run), "clips": clips,
                           "policy": f"full-pose planner {a.checkpoint}, kinematic open-loop rollout (no physics)"})
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
