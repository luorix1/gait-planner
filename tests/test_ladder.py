"""The ladder task (gp.ladder) on a few clips, on the CPU: the style judge's input is the same function of a pose on
the simulation's side as on the clips', its mirror map is an involution, and a step gives finite rewards."""

from dataclasses import replace

import numpy as np
import torch
from sprint.dynamics import forward_kinematics
from sprint.recipes.study import default_ppo

from gp import data
from gp import wys_terrain as WT
from gp.ladder import Ladder, LadderConfig
from train_wys import FloorPPO


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
    # sensor noise is in physical units, before the scales: at most 0.2 x n_ang and 0.05 x n_qd on the scaled readings
    task.c = replace(task.c, spike_p=0.0)
    noisy, clean = task.body_obs(task.body_state(env.state))
    assert (noisy[1] - clean[1]).abs().max() <= 0.2 * task.c.n_ang + 1e-6
    assert (noisy[5] - clean[5]).abs().max() <= 0.05 * task.c.n_qd + 1e-6
    task.c = replace(task.c, spike_p=1e-4)
    # L0.5: the actor's command columns read zeros, the critic's keep the command
    o = [b[0] for b in task.blocks]
    at = lambda name: sum(b[1] for b in task.blocks[:o.index(name)])  # noqa: E731
    task.cmd[:] = torch.tensor([1.0, 0.0, 0.3])
    task.c = replace(task.c, policy_command=False)
    obs = task.observation(env.state)
    assert (obs[:, at("command"):at("command") + 3] == 0).all() and (obs[:, at("critic_command")] == 1.0).all()
    task.c = replace(task.c, policy_command=True)
    # the drive floor: silent while every muscle's mean is above it (a fresh policy sits at 0.05), active below
    agent = FloorPPO(env, replace(default_ppo(), rollout_steps=24, compile_actor=False), device="cpu")
    obs = env.torch(env.obs)
    base = agent._symmetry_loss(obs)
    agent.floor = 0.04
    assert torch.allclose(agent._symmetry_loss(obs), base)
    agent.floor = 0.2
    assert agent._symmetry_loss(obs) > base + 1e-3
