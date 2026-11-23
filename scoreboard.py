"""The P0 scoreboard (roadmap): a full-pose planner (train_planner.py --pose2) judged by its rollouts, not by its
denoising loss. The held-out loss was one random batch, raw weights, ~80 % level walking; v9's moved 0.066-0.086 at
random while its stairs were memorised.

Episodes, fixed (the same every call; open loop, plans kept 0.2 s, foot hold on, as export_pose2):
- **replays**: up to 2 held-out clips per (source, mode, terrain class), 4 for terrain other than level, 8 s from the
  clip's own first 0.4 s; goal = the clip's pelvis 1.2 s ahead (export_pose2's world-anchored replay goal);
- **courses**: 23 fixed gp.terrain_gen courses (stairs 12/16/20 cm up and down and boxes 20/40/60 cm at 0.8 m/s, ramps
  5/10/15 deg up and down and 90 deg turns at 1.2 m/s, level at 0.8-2.0 m/s), from one held-out AB01 level walk set
  down at the start; goal = the centreline point 1.2 s ahead of a walker moving at the course speed.

Metrics per episode, all lower = better (SPRINT's exact forward kinematics on the episode's terrain):
  goal_m    mean horizontal distance of the pelvis from its track (the clip's pelvis; the course point at speed x t)
  slip_cms  median horizontal speed of foot spheres within 2 cm of the ground (export_pose2.Scorer)
  sink      share of frames with a sphere more than 2 cm under the terrain; deep: more than 5 cm
  float     share of frames with both feet more than 3 cm up
  seam      joint acceleration at the frames a replan seam touches minus elsewhere (deg/s^2; 0 = seamless; a ratio
            blew up on standing clips, where any seam wobble over ~0 motion read 4.7)
  jerk      share of 20 Hz frames whose joint or pelvis jerk passes the 99.9th percentile of real held-out frames
  style     mean SDS loss of the frozen style prior (runs/prior_v4) over 13-frame windows
  gait      replays only: mean |log(generated / real)| of stride time, step length and duty factor
Per class (replay terrain class, course kind) the episodes' mean; ``balanced`` = the mean over classes, so a staircase
counts as much as level walking. ``select`` ranks checkpoints by their mean rank over the balanced metrics.

Run: GP_DATASET=v5 .venv/bin/python scoreboard.py runs/pose2_planner_v9 [--checkpoint planner_best.pt | --all]
Out: <run>/scoreboard.json (one row per checkpoint); with --all also <run>/planner_score_best.pt
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sprint.recipes.imitation import _qmul
from sprint.recipes.motion import _rotate, forward_kinematics_np, model_layout

from export_pose2 import NS, Scorer, circmean, frame_features, rollout, start
from export_pose_planner import to_qpos
from gp import data
from gp.planner import FUT, HIST, HZ, POSE_COORDS, STRIDE, from_frame, to_frame, yaw_quat
from gp.pose_planner import PosePlanner, Windows
from gp.smp import Prior
from gp.smp import W as STYLE_W
from gp.terrain import Ground, atlas, pad
from gp.terrain_gen import CourseSpec, _turn_length, course_tile

SECONDS, KEEP, HOLD = 8.0, 4, 50
STYLE_PRIOR = data.ROOT / "runs" / "prior_v4" / "prior.pt"
METRICS = ("goal_m", "slip_cms", "sink", "deep", "float", "seam", "jerk", "style", "gait")
START_X = 1.0                              # m along the course's first level stretch


def courses() -> list[tuple[str, str, float, list[dict[str, Any]]]]:
    """(name, class, speed m/s, gp.terrain_gen segments) of the fixed courses."""
    def lv(m: float) -> dict[str, Any]:
        return dict(kind="level", length=m, turn=0.0)
    out = []
    for r in (0.12, 0.16, 0.20):
        for d, cls in ((1, "stairs_up"), (-1, "stairs_down")):
            out.append((f"{cls} {100 * r:.0f} cm", cls, 0.8,
                        [lv(3.0), dict(kind="stairs", steps=6, rise=r, tread=0.30, length=1.8, turn=0.0, dir=d), lv(4.0)]))
    for h in (0.2, 0.4, 0.6):
        out.append((f"box {100 * h:.0f} cm", "box", 0.8, [lv(3.0), dict(kind="platform", height=h, length=2.0, turn=0.0), lv(4.0)]))
    for g in (5.0, 10.0, 15.0):
        for d, cls in ((1, "ramp_up"), (-1, "ramp_down")):
            out.append((f"{cls} {g:.0f} deg", cls, 1.2,
                        [lv(3.0), dict(kind="ramp", length=3.0, grade_deg=g, turn=0.0, dir=d), lv(5.0)]))
    for side, nm in ((1.0, "left"), (-1.0, "right")):
        out.append((f"turn {nm}", "turn", 1.2, [lv(3.0), dict(kind="turn", length=_turn_length(CourseSpec()), turn=side * math.pi / 2), lv(7.0)]))
    for v in (0.8, 1.2, 1.6, 2.0):
        out.append((f"level {v:.1f} m/s", "level_cmd", v, [lv(20.0)]))
    for *_, segs in out:
        h = 0.0
        for s in segs:
            s["h0"] = h
            if s["kind"] == "stairs":
                h += s["dir"] * s["steps"] * s["rise"]
            elif s["kind"] == "ramp":
                h += s["dir"] * s["length"] * math.tan(math.radians(s["grade_deg"]))
    return out


def place(s: dict, psi_hist: torch.Tensor, o: torch.Tensor, a_psi: torch.Tensor, g0: torch.Tensor,
          xz: torch.Tensor, psi: torch.Tensor, g: torch.Tensor) -> tuple[dict, torch.Tensor]:
    """Start frames (export_pose2.start) moved rigidly so their anchor (pelvis com ``o``, smoothed heading ``a_psi``,
    ground ``g0``) lands at ``xz``, heading ``psi``, ground ``g`` (all ``[B, ...]``); anchor-frame features unchanged."""
    d = psi - a_psi

    def mv(p: torch.Tensor) -> torch.Tensor:
        return from_frame(to_frame(p, o, a_psi), xz, psi)

    def mv3(p: torch.Tensor) -> torch.Tensor:
        q = mv(p[..., [0, 2]])
        dy = (g - g0).view(-1, *[1] * (p.dim() - 2))
        return torch.stack([q[..., 0], p[..., 1] + dy, q[..., 1]], -1)
    yq = yaw_quat(d)[:, None].expand_as(s["quat"])
    out = dict(s, pcom=mv3(s["pcom"]), quat=_qmul(yq, s["quat"]), psi=s["psi"] + d[:, None], hold=mv(s["hold"]),
               points=mv3(s["points"]))
    return out, psi_hist + d[:, None]


def jerk_flags(x20: np.ndarray, p20: np.ndarray, thr: tuple[float, float]) -> np.ndarray:
    """Per 20 Hz frame (from the 4th): does any joint's (rad) or the pelvis's (m) jerk pass ``thr``?"""
    j = np.abs(np.diff(x20, 3, axis=0)).max(1) * HZ ** 3
    p = np.linalg.norm(np.diff(p20, 3, axis=0), axis=1) * HZ ** 3
    return (j > thr[0]) | (p > thr[1])


def seam_excess(x20: np.ndarray, keep: int) -> float:
    """Mean joint acceleration (deg/s^2) at the frames a replan seam touches minus the mean elsewhere (20 Hz frames,
    frame 0 = the last given one)."""
    acc = np.linalg.norm(np.degrees(x20[2:] - 2 * x20[1:-1] + x20[:-2]), axis=1) * HZ ** 2     # at frames 1 .. n-2
    j = np.arange(1, len(x20) - 1)
    seam = (j % keep == 0) | (j % keep == 1)
    return float(acc[seam].mean() - acc[~seam].mean())


class Scoreboard:
    """The fixed benchmark: build once (held-out data ``D``, the planner's foot stand-in ``fk``), call on a planner."""

    def __init__(self, D: dict[str, Any], fk: Any, device: Any, seconds: float = SECONDS) -> None:
        self.dev, self.fk = device, fk
        self.n_plans = int(round(seconds * HZ / KEEP))
        self.W = W = Windows(D, device, fk)
        W.starts, W.lengths = D["start"], D["frames"]
        self.D, self.names = D, list(D["coord_names"])
        self.ji = [self.names.index(n) for n in POSE_COORDS]
        self.pi = [self.names.index(n) for n in ("pelvis_tx", "pelvis_ty", "pelvis_tz")]
        self.model = data.load_model()
        L = model_layout(self.model)
        self.tracked, self.key = L.tracked, [list(self.model.body_names).index(b) for b in data.KEY_BODIES]
        self.prior = Prior.load(STYLE_PRIOR, device)

        # replays
        room = np.asarray(D["frames"]) - STRIDE * (HIST + FUT)
        groups: dict[tuple, list[int]] = {}
        for k in np.argsort(-room, kind="stable"):
            if room[k] > 0:
                groups.setdefault((str(D["source"][k]), str(D["mode"][k]), str(D["terrain_class"][k])), []).append(int(k))
        self.rep = [k for g, ks in sorted(groups.items()) for k in ks[:2 if g[2] == "level" else 4]]
        self.a_rep = torch.as_tensor([int(D["start"][k]) + STRIDE * (HIST - 1) for k in self.rep], device=device)
        self.clip_last = torch.as_tensor(np.asarray(D["start"]) + np.asarray(D["frames"]) - 1, device=device)

        # courses on one heightfield atlas, from the longest held-out AB01 level walk
        self.courses = courses()
        tiles = [course_tile(segs) for *_, segs in self.courses]
        hf = atlas(tiles, tiles[0]["cell"])
        n = len(tiles)
        X, Y = pad([np.array([[-1e3, 0.0], [1e3, 0.0]])] * n)
        self.G = Ground(X, Y, np.ones(n, bool), hf, hf["offsets"], device)
        self.gid = torch.arange(n, device=device)
        self.paths = [t["path"] for t in tiles]
        self.speed = np.array([v for _, _, v, _ in self.courses])
        walk = [k for k in range(len(D["name"])) if str(D["mode"][k]) == "walk" and str(D["terrain_class"][k]) == "level"
                and str(D["source"][k]) == "ab01" and room[k] > 0]
        self.k_walk = max(walk, key=lambda k: float(D["frames"][k]))
        a = torch.full((n,), int(D["start"][self.k_walk]) + STRIDE * (HIST - 1), device=device)
        s, ph = start(W, a)
        o, a_psi = s["pcom"][:, -1, [0, 2]], circmean(ph[:, -NS:])
        g0 = W.fr.G(W.fr.clip[a], s["pcom"][:, -1:, 0], s["pcom"][:, -1:, 2])[:, 0]
        xz = torch.tensor([[START_X, 0.0]], device=device).expand(n, 2)
        g = self.G(self.gid, xz[:, :1], xz[:, 1:])[:, 0]
        self.c_start = place(s, ph, o, a_psi, g0, xz, torch.zeros(n, device=device), g)
        moved = frame_features(self.c_start[0], xz, torch.zeros(n, device=device), g)
        assert torch.allclose(moved, frame_features(s, o, a_psi, g0), atol=1e-4), "course start not rigidly placed"

        # the jerk threshold and the real clips' own scores
        x20 = [np.asarray(D["qpos"][st:st + f:STRIDE]) for st, f in zip(D["start"], D["frames"], strict=True)]
        jj = np.concatenate([np.abs(np.diff(x[:, self.ji], 3, axis=0)).max(1) for x in x20 if len(x) > 3]) * HZ ** 3
        pp = np.concatenate([np.linalg.norm(np.diff(x[:, self.pi], 3, axis=0), axis=1) for x in x20 if len(x) > 3]) * HZ ** 3
        self.thr = (float(np.percentile(jj, 99.9)), float(np.percentile(pp, 99.9)))
        self.scorer = Scorer(W.fr.G)
        self.c_scorer = Scorer(self.G)
        self.real = []
        for b in range(len(self.rep)):
            q = self.real_q(b)
            self.real.append(self.metrics(q, self.scorer, int(W.fr.clip[self.a_rep[b]]), q[::STRIDE][:, self.ji],
                                          q[::STRIDE][:, self.pi], None, None))

    def real_q(self, b: int) -> np.ndarray:
        s0, k = int(self.a_rep[b]), self.rep[b]
        n = int((self.n_plans * KEEP) * STRIDE) + 1
        return np.asarray(self.D["qpos"][s0:min(s0 + n, int(self.D["start"][k] + self.D["frames"][k]))])

    def track_at(self, c: int, t: np.ndarray) -> np.ndarray:
        """Course ``c``'s centreline point a walker at the course speed reaches at times ``t`` (s)."""
        p = self.paths[c]
        s = np.minimum(START_X + self.speed[c] * t, p["s"][-1])
        return np.c_[np.interp(s, p["s"], p["x"]), np.interp(s, p["s"], p["z"])]

    def gait(self, xp: np.ndarray, xq: np.ndarray, sc: Scorer, gid: int) -> dict[str, float] | None:
        """Stride time (s), step length (m) and duty factor from the lowest sphere of each foot (down = within 2 cm)."""
        c = xp[:, sc.body] + _rotate(xq[:, sc.body], np.broadcast_to(sc.pos, xp[:, sc.body].shape))
        b = c - np.stack([0 * sc.rad, sc.rad, 0 * sc.rad], -1)
        dev = sc.G.X.device
        g = sc.G(torch.full((1,), gid, device=dev), torch.as_tensor(b[None, ..., 0].reshape(1, -1), device=dev),
                 torch.as_tensor(b[None, ..., 2].reshape(1, -1), device=dev)).cpu().numpy().reshape(b.shape[:2])
        gap = b[..., 1] - g
        strides, tds, duty = [], [], []
        for side in "lr":
            m = sc.side == side
            on = gap[:, m].min(1) < 0.02
            td = np.flatnonzero(on[1:] & ~on[:-1]) + 1
            duty.append(on.mean())
            strides += list(np.diff(td) / data.HZ)
            tds += [(t, b[t, m][:, [0, 2]].mean(0)) for t in td]
        if len(strides) < 2:
            return None
        tds.sort(key=lambda e: e[0])
        steps = [float(np.linalg.norm(p1 - p0)) for (_, p0), (_, p1) in zip(tds[:-1], tds[1:], strict=True)]
        return dict(stride_s=float(np.median(strides)), step_m=float(np.median(steps)), duty=float(np.mean(duty)))

    def metrics(self, q: np.ndarray, sc: Scorer, gid: int, x20: np.ndarray, p20: np.ndarray,
                com20: np.ndarray | None, track: np.ndarray | None) -> dict[str, Any]:
        xp, xq = forward_kinematics_np(self.model, q.astype(np.float32))
        f = sc(q, gid, (xp, xq))
        ground = sc.G(torch.full((1,), gid, device=self.dev), torch.as_tensor(xp[None, :, 0, 0], device=self.dev),
                      torch.as_tensor(xp[None, :, 0, 2], device=self.dev)).cpu().numpy()[0]
        rec = torch.as_tensor(data.style_records(xp, xq, q, self.tracked, self.key, ground), dtype=torch.float32,
                              device=self.dev)
        ends = torch.arange(STYLE_W, len(rec), 4, device=self.dev)
        torch.manual_seed(0)
        style = float(self.prior.sds(self.prior.features(rec[ends[:, None] + torch.arange(-STYLE_W, 1, device=self.dev)])).mean())
        out = dict(slip_cms=f["stance_slip_cms"], sink=f["sinking"], deep=f["deep_frac"], float=f["floating"],
                   seam=seam_excess(x20, KEEP), jerk=float(jerk_flags(x20, p20, self.thr).mean()), style=style,
                   pelvis_vz_max_mps=f["pelvis_vz_max_mps"], path_speed_mps=f["path_speed_mps"],
                   gait_raw=self.gait(xp, xq, sc, gid))
        if com20 is not None:
            n = min(len(com20), len(track))
            out["goal_m"] = float(np.linalg.norm(com20[:n] - track[:n], axis=1).mean())
        return out

    @torch.no_grad()
    def __call__(self, P: PosePlanner, seed: int = 0, w: float = 1.0, blend: float | None = None, xfade: int = 0,
                 hold_smooth: float = 0.0) -> dict[str, Any]:
        was = P.training
        P.eval()
        W, D, dev = self.W, self.D, self.dev
        gen = torch.Generator(device=dev).manual_seed(seed)
        t0 = time.time()
        a_rep, last = self.a_rep, self.clip_last

        def replay_goal(i: int, o: torch.Tensor, psi: torch.Tensor) -> torch.Tensor:
            tgt = torch.minimum(a_rep + i * KEEP * STRIDE + STRIDE * FUT, last[W.fr.clip[a_rep]])
            return to_frame(W.fr.pcom[tgt][:, None, [0, 2]], o, psi)[:, 0]

        def course_goal(i: int, o: torch.Tensor, psi: torch.Tensor) -> torch.Tensor:
            t = (i * KEEP + FUT) / HZ
            ahead = torch.as_tensor(np.stack([self.track_at(c, np.array([t]))[0] for c in range(len(self.courses))]),
                                    dtype=torch.float32, device=dev)
            return to_frame(ahead[:, None], o, psi)[:, 0]
        with torch.enable_grad():                                        # hold_feet's Adam steps need it
            r_rep = rollout(P, W.fr.G, W.fr.clip[a_rep], *start(W, a_rep), replay_goal, self.n_plans, KEEP, w, gen, HOLD,
                            blend, xfade, hold_smooth)
            r_cou = rollout(P, self.G, self.gid, *self.c_start, course_goal, self.n_plans, KEEP, w, gen, HOLD, blend,
                            xfade, hold_smooth)
        t_roll = time.time() - t0

        eps = []
        t20 = np.arange(self.n_plans * KEEP + 1) / HZ
        for b, k in enumerate(self.rep):
            q = to_qpos(r_rep, b, self.names)
            tgt = np.minimum(int(a_rep[b]) + STRIDE * np.arange(len(t20)), int(last[W.fr.clip[a_rep[b]]]))
            m = self.metrics(q, self.scorer, int(W.fr.clip[a_rep[b]]), r_rep["joints"][b], r_rep["pelvis"][b],
                             r_rep["com"][b][:, [0, 2]], W.fr.pcom[torch.as_tensor(tgt, device=dev)][:, [0, 2]].cpu().numpy())
            g, gr = m.pop("gait_raw"), self.real[b]["gait_raw"]
            m["gait"] = (float(np.mean([abs(math.log(max(g[x], 1e-3) / max(gr[x], 1e-3))) for x in g]))
                         if g and gr else None)
            eps.append(dict(kind="replay", cls=f"replay {D['terrain_class'][k]}", name=str(D["name"][k]),
                            mode=str(D["mode"][k]), **m))
        for c, (name, cls, v, _) in enumerate(self.courses):
            q = to_qpos(r_cou, c, self.names)
            m = self.metrics(q, self.c_scorer, c, r_cou["joints"][c], r_cou["pelvis"][c], r_cou["com"][c][:, [0, 2]],
                             self.track_at(c, t20))
            m.pop("gait_raw")
            eps.append(dict(kind="course", cls=f"course {cls}", name=name, speed=v, gait=None, **m))

        hist = torch.zeros(512, HIST, P.c, device=dev) + P.norm_at(0, HIST)[0]
        patch0 = torch.zeros(512, len(P.grid[0]), len(P.grid[1]), device=dev)
        torch.cuda.synchronize() if str(dev).startswith("cuda") else None
        t1 = time.time()
        P.generate(hist, patch0, torch.zeros(512, 2, device=dev), gen, w=w)
        torch.cuda.synchronize() if str(dev).startswith("cuda") else None
        P.train(was)
        return dict(summarize(eps), episodes=eps, rollout_s=round(t_roll, 1), gen_s_512=round(time.time() - t1, 2),
                    jerk_thr=self.thr, real=summarize([dict(e, cls=f"replay {D['terrain_class'][k]}")
                                                       for e, k in zip(self.real, self.rep, strict=True)])["classes"])


def summarize(eps: list[dict[str, Any]]) -> dict[str, Any]:
    """Per class the mean of each metric over its episodes (Nones skipped); ``balanced`` = the mean over classes."""
    classes: dict[str, dict[str, float]] = {}
    for cls in sorted({e["cls"] for e in eps}):
        rows = [e for e in eps if e["cls"] == cls]
        classes[cls] = {m: round(float(np.mean(v)), 4) for m in METRICS
                        if (v := [e[m] for e in rows if e.get(m) is not None])}
        classes[cls]["n"] = len(rows)
    balanced = {m: round(float(np.mean(v)), 4) for m in METRICS
                if (v := [c[m] for c in classes.values() if m in c])}
    return dict(balanced=balanced, classes=classes)


def select(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """The row with the lowest mean rank over the balanced metrics (ties: earlier row)."""
    ms = [m for m in METRICS if all(m in r["balanced"] for r in rows)]
    ranks = np.mean([np.argsort(np.argsort([r["balanced"][m] for r in rows], kind="stable"), kind="stable") for m in ms], 0)
    for r, k in zip(rows, ranks, strict=True):
        r["mean_rank"] = round(float(k), 2)
    return rows[int(np.argmin(ranks))]


def table(rows: list[dict[str, Any]]) -> str:
    head = f"{'checkpoint':<24}" + "".join(f"{m:>10}" for m in METRICS)
    lines = [head] + [f"{r['checkpoint']:<24}" + "".join(f"{r['balanced'].get(m, float('nan')):>10.4f}" for m in METRICS)
                      for r in rows]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--checkpoint", default="planner_best.pt")
    ap.add_argument("--blend", type=float, nargs="*", default=None, help="history blend s values to score each checkpoint at (models trained with --hist-drop; default the model's)")
    ap.add_argument("--steps", type=int, default=None, help="DDIM sampling steps (default the model's)")
    ap.add_argument("--all", action="store_true", help="every ckpt_*.pt in the run (train_planner.py --score-every)")
    ap.add_argument("--xfade", type=int, nargs="*", default=[0], help="frames each new plan fades in from the previous "
                    "plan (v12 seam test; one score per value)")
    ap.add_argument("--hold-smooth", type=float, nargs="*", default=[0.0], help="foot-hold smoothness weight (v12; one "
                    "score per value)")
    ap.add_argument("--out", default=None, help="result file name in the run folder (default by the options)")
    a = ap.parse_args()
    from gp import footfk
    run, dev = Path(a.run), "cuda" if torch.cuda.is_available() else "cpu"
    D = data.load(("test_trial", "test_terrain"))
    fk = footfk.fit(D, dev, data.ROOT / "data" / "foot_fk.pt")
    board = Scoreboard(D, fk, dev)
    paths = sorted(run.glob("ckpt_*.pt")) if a.all else [run / a.checkpoint]
    rows = []
    for p in paths:
        P = PosePlanner.load(p, dev).attach(fk)
        if a.steps:
            P.cfg["sample_steps"] = a.steps
        for blend, xf, hs in itertools.product(a.blend or [None], a.xfade, a.hold_smooth):
            r = board(P, blend=blend, xfade=xf, hold_smooth=hs)
            r["file"] = p.name
            r["checkpoint"] = (p.name + ("" if blend is None else f" s={blend}") + (f" {a.steps} steps" if a.steps else "")
                               + (f" xf{xf}" if xf else "") + (f" hs{hs:g}" if hs else ""))
            r["iters"] = torch.load(p, map_location="cpu", weights_only=False).get("iters")
            rows.append(r)
            print(f"{r['checkpoint']}: {r['balanced']}  ({r['rollout_s']} s rollouts, {r['gen_s_512']} s per 512 plans)",
                  flush=True)
    best = select(rows) if len(rows) > 1 else rows[0]
    out = a.out or ("scoreboard_blend.json" if a.blend or a.steps else "scoreboard.json")
    (run / out).write_text(json.dumps(rows, indent=1))
    print(table(rows))
    print("real held-out clips per class:", json.dumps(rows[0]["real"]))
    if a.all:
        shutil.copy2(run / best["file"], run / "planner_score_best.pt")
        print(f"best by mean rank: {best['checkpoint']} -> planner_score_best.pt")


if __name__ == "__main__":
    main()
