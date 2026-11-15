"""Score a trained planner (train_planner.py) before any physics.

- **Held-out windows** (test_trial + test_terrain, and train for comparison): from each window's real history, ground
  and goal, generate the future and compare with the real plan: per-frame contact agreement; each foot's next
  touchdown (whether both have one, and if so its timing and foothold error); the centre of mass 0.6 s on.
- **Generated courses** (gp.terrain_gen, difficulties ``--difficulty``): roll the planner out from a standing start
  (gp.planner.plan_course) and check every foothold: on the course (the ground there is the course's height at the
  nearest centreline point, and within the causeway), not in a gap, not across an edge (the ground within 8 cm varies
  by more than 2 cm), and whether the plan reaches the course end. The stand-in plan (gp.free.stand_in_plan) is scored
  the same way as the reference. One top-down picture per course: ground, footholds (left blue, right red), centre of
  mass.

Run: .venv/bin/python check_planner.py runs/planner_net_v1/planner.pt [--courses 6]
Out: <planner dir>/check/check.json and course_*.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from gp import data
from gp.free import REFINE, stand_in_plan
from gp.planner import HIST, HZ, Planner, Windows, plan_course
from gp.terrain import hf_height
from gp.terrain_gen import CourseSpec, course_tile, sample_course2d

EDGE_R, EDGE_DH, ON_DH = 0.08, 0.02, 0.02


def first_touchdown(down: torch.Tensor, prev: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per window and foot, the first frame that is down after a frame that is not ``[B, 2]``, and whether there is one."""
    td = down & ~torch.cat([prev[:, None], down[:, :-1]], 1)
    has = td.any(1)
    return td.float().argmax(1), has


def window_metrics(planner: Planner, W: Windows, n: int, seed: int = 0) -> dict[str, float]:
    gen = torch.Generator(device=W.anchors.device).manual_seed(seed)
    a = W.anchors[torch.randperm(len(W.anchors), generator=gen, device=W.anchors.device)[:n]]
    x, p, g = W(a)
    fut, true = planner.generate(x[:, :HIST], p, g, gen), x[:, HIST:]
    dp, dt, prev = fut[..., :2] > 0, true[..., :2] > 0, x[:, HIST - 1, :2] > 0
    kp, hp = first_touchdown(dp, prev)
    kt, ht = first_touchdown(dt, prev)
    both = hp & ht
    b = torch.arange(len(x), device=x.device)[:, None].expand(-1, 2)
    foot = torch.arange(2, device=x.device).expand(len(x), -1)
    hold = lambda y, k: y[..., 2:6].unflatten(-1, (2, 2))[b, k, foot]  # noqa: E731
    err = (hold(fut, kp) - hold(true, kt)).norm(dim=-1)[both]
    dt_s = (kp - kt).abs()[both].float() / HZ
    k6 = int(0.6 * HZ) - 1
    q = lambda v, f: float(torch.quantile(v, f)) if v.numel() else float("nan")  # noqa: E731
    return dict(windows=n, contact_agreement=float((dp == dt).float().mean()), touchdown_exists_agree=float((hp == ht).float().mean()),
                touchdown_time_med_s=q(dt_s, 0.5), touchdown_time_p90_s=q(dt_s, 0.9),
                foothold_err_med_m=q(err, 0.5), foothold_err_p90_m=q(err, 0.9),
                com_err_0p6s_med_m=q((fut[:, k6, [6, 8]] - true[:, k6, [6, 8]]).norm(dim=1), 0.5))


def course_metrics(P: dict, T: dict, spec: CourseSpec) -> dict[str, float]:
    p = T["path"]
    ground = lambda x, z: hf_height(T["heights"], T["block"], T["cell"], T["origin"], x, z)  # noqa: E731
    holds = np.array([P["foothold"][a, k] for k in (0, 1) for a, _ in data._intervals(P["plan_contact"][:, k])])
    d2 = (p["x"][None] - holds[:, :1]) ** 2 + (p["z"][None] - holds[:, 2:3]) ** 2
    j = d2.argmin(1)
    g = ground(holds[:, 0], holds[:, 2])
    kinds = np.array([s["kind"] for s in T["segments"]])[p["seg"][j]]
    gap = kinds == "gap"
    on = (np.abs(g - p["h"][j]) < ON_DH) & (np.sqrt(d2.min(1)) < spec.causeway_m / 2 - 0.05) & ~gap
    ang = np.linspace(0, 2 * np.pi, 8, endpoint=False)
    ring = ground(holds[:, :1] + EDGE_R * np.cos(ang), holds[:, 2:3] + EDGE_R * np.sin(ang))
    edge = np.abs(ring - g[:, None]).max(1) > EDGE_DH
    tds = np.sort(np.concatenate([np.flatnonzero(np.diff(P["plan_contact"][:, k].astype(int)) == 1) for k in (0, 1)]))
    end = np.array([p["x"][-1], p["z"][-1]])
    s_end = float(p["s"][-1]) - 1.0
    k_end = np.searchsorted(p["s"], s_end)
    return dict(footholds=len(holds), on_course=float(on.mean()), in_gap=float(gap.mean()), on_edge=float(edge.mean()),
                reached_end=bool(np.linalg.norm(P["com"][-1, [0, 2]] - [p["x"][k_end], p["z"][k_end]]) < 0.5),
                end_dist_m=float(np.linalg.norm(P["com"][-1, [0, 2]] - end)),
                flight_frac=float((~P["plan_contact"].any(1)).mean()),
                step_time_med_s=float(np.median(np.diff(tds)) / data.HZ) if len(tds) > 1 else float("nan"))


def picture(P: dict, T: dict, path: Path, title: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    H, (x0, z0), c = T["heights"], T["origin"], T["cell"]
    fig, ax = plt.subplots(figsize=(10, 10 * H.shape[0] / H.shape[1] + 0.6))
    ax.imshow(H, origin="lower", extent=(x0, x0 + c * (H.shape[1] - 1), z0, z0 + c * (H.shape[0] - 1)), cmap="Greys_r")
    ax.plot(P["com"][:, 0], P["com"][:, 2], color="#e0a020", lw=1)
    for k, col in ((0, "#2060d0"), (1, "#d03030")):
        h = np.array([P["foothold"][a, k] for a, _ in data._intervals(P["plan_contact"][:, k])])
        if len(h):
            ax.scatter(h[:, 0], h[:, 2], s=14, color=col, edgecolors="white", linewidths=0.4, zorder=3)
    ax.set_title(title, fontsize=9)
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("planner")
    ap.add_argument("--courses", type=int, default=6)
    ap.add_argument("--difficulty", type=float, nargs="+", default=[0.3, 0.6, 0.9])
    ap.add_argument("--speed", type=float, default=1.2)
    ap.add_argument("--windows", type=int, default=1024, help="0: courses only")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    out = Path(a.planner).parent / "check"
    out.mkdir(exist_ok=True)
    planner = Planner.load(a.planner, a.device)
    res: dict = {"planner": a.planner, "speed": a.speed, "windows": {}}
    for split in (("train", ("test_trial", "test_terrain")) if a.windows else ()):
        name = split if isinstance(split, str) else "held_out"
        res["windows"][name] = window_metrics(planner, Windows(data.load(split), a.device), a.windows)
        print(name, json.dumps(res["windows"][name]), flush=True)
    D = data.load("train")
    stand = next(i for i in range(len(D["name"])) if D["mode"][i] == "stand" and D["terrain_class"][i] == "level")
    cell, spec = D["hf"]["cell"] / REFINE, CourseSpec()
    rows = []
    for d in a.difficulty:
        tiles = [course_tile(sample_course2d(np.random.default_rng(1000 + k), d), cell=cell) for k in range(a.courses)]
        plans = plan_course(planner, tiles, D, stand, speed=a.speed, device=a.device)
        for k, (T, P) in enumerate(zip(tiles, plans, strict=True)):
            kinds = "+".join(s["kind"] for s in T["segments"] if s["kind"] != "level")
            ref = course_metrics(stand_in_plan(T, 0.98), T, spec)
            m = course_metrics(P, T, spec)
            rows.append(dict(difficulty=d, course=k, kinds=kinds, planner=m, stand_in=ref))
            print(f"d {d:.1f} #{k} {kinds[:60]:60s} on {m['on_course']:.2f} gap {m['in_gap']:.2f} edge {m['on_edge']:.2f} "
                  f"end {m['reached_end']} flight {m['flight_frac']:.2f} step {m['step_time_med_s']:.2f} s | "
                  f"stand-in on {ref['on_course']:.2f} edge {ref['on_edge']:.2f}", flush=True)
            picture(P, T, out / f"course_d{d:.1f}_{k}.png", f"d {d:.1f} #{k}: {kinds}")
    res["courses"] = rows
    for who in ("planner", "stand_in"):
        res[f"courses_{who}"] = {k: float(np.mean([r[who][k] for r in rows]))
                                 for k in ("on_course", "in_gap", "on_edge", "reached_end", "flight_frac")}
    print(json.dumps({k: res[k] for k in ("courses_planner", "courses_stand_in")}, indent=1))
    (out / "check.json").write_text(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    main()
