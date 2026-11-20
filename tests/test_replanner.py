"""gp.planner.Replanner in the recipe teacher's environment (CPU, mc021, an untrained planner): the student's plan
inputs have the clip plan's layout and stay finite, timings stay in [0, cap], a new episode's history is its clip's
past plus the measured first frame (which matches the clip at a recipe start), and generations happen at the start,
on the timer and at touchdowns."""

import numpy as np
import torch

from distill import plan_inputs
from gp import data
from gp.planner import HIST, STRIDE, Planner, Replanner, Windows
from train import build


def test_replanner_on_recipe_env() -> None:
    D = data.load(("train", "test_trial", "test_terrain"), src=data.ROOT / "data" / "ab01")
    D = data.take(D, [k for k, n in enumerate(D["name"]) if str(n).startswith("ab01_mc021_")])
    env, task = build("baseline", D, 4, 0, "cpu")
    P = Planner(layers=1, heads=2, head_dim=16)
    P.set_normaliser(*Windows(D).batch(512))
    r = Replanner(P.eval(), task, D, every=8, stride=25)
    clip = plan_inputs(task, D)
    env.reset()
    s = env.state
    x = r(s)
    assert x.shape == clip(s).shape == (4, 15 * len(task.taps)) and torch.isfinite(x).all()
    # the measured frame at a recipe start is the clip's (centre of mass within 2 cm, heading within 0.05 rad)
    f = task.frame
    assert (r.h_com[:, -1] - r.W.com[f]).norm(dim=1).max() < 0.02
    assert (torch.remainder(r.h_psi[:, -1] - r.W.psi[f] + np.pi, 2 * np.pi) - np.pi).abs().max() < 0.05
    assert torch.equal(r.h_com[:, -2], r.W.com[torch.maximum(f - 1, task.clip_start[task.clip])])
    assert r.replans == 4
    for _ in range(20):
        env.step(torch.full((4, task.nmus), 0.05))
        x = r(env.state)
        assert torch.isfinite(x).all()
        assert (r.td >= 0).all() and (r.td <= data.PLAN_CAP_S).all() and (r.lo >= 0).all()
    assert r.replans >= 4 + 4 * (20 // 8)
    assert r.h_down.shape[1] == STRIDE * (HIST - 1) + 1
