"""The ladder task (gp.ladder) on a few clips, on the CPU: the style judge's input is the same function of a pose on
the simulation's side as on the clips', its mirror map is an involution, and a step gives finite rewards."""

import numpy as np
import torch
from sprint.dynamics import forward_kinematics

from gp import data
from gp import wys_terrain as WT
from gp.ladder import Ladder, LadderConfig


def test_ladder_pairs_and_step() -> None:
    import sprint

    D = data.load("train", src=data.ROOT / "data" / "ab01")
    pick = [k for k, (m, t) in enumerate(zip(D["mode"], D["terrain_class"], strict=True))
            if m in ("walk", "stand") and t == "level"][:4]
    D = data.take(D, pick)
    task = Ladder(WT.build(), D, LadderConfig(), seed=0)
    env = sprint.VecEnv(data.load_model("cpu"), 4, task=task, seed=0, device="cpu",
                        step_options=sprint.StepOptions(dt_ctrl=0.025, n_sub=75, implicit_friction=True,
                                                        implicit_normal=True, strict=False, average_fext=True),
                        perturb_slots=2)
    s = env.state
    for j in (0, len(task.amp.pairs) // 2, len(task.amp.pairs) - 1):
        f, poses = int(task.ex_frame[j]), []
        for ff in (f, f + 1):
            s.qpos[0] = task.ex_q[ff]
            forward_kinematics(env.model, env.data)
            poses.append(task.pose(s.qpos[:1], s.body_pos[:1], s.body_quat[:1], task.ex_gy[ff:ff + 1]))
        assert torch.allclose(task.pair(*poses)[0], task.amp.pairs[j], atol=1e-4), j
    P, S = task.pair_mirror()
    assert (P[P] == np.arange(len(P))).all() and np.allclose(S * S[P], 1.0)
    env.reset()
    _, r, *_ = env.step(torch.full((4, task.nmus), 0.05))
    assert torch.isfinite(torch.as_tensor(r)).all()
    assert env.obs.shape[-1] == sum(b[1] for b in task.blocks)
    m = task.metrics                    # the effort takes effort_share of the task + style reward
    paid = (m["effort_scale"] * m["effort"]).mean()
    earned = (task.c.w_task * m["r_task"] + task.c.w_style * m["r_style"]).mean()
    assert torch.isclose(paid / earned, torch.tensor(task.c.effort_share), rtol=1e-4)
