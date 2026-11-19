"""Stage 0 (gp.pretrain) on a few clips, on the CPU: a body put exactly on its reference frame scores ~1 on every
tracking term (frames, heading frames, COMs and heights line up), and a step gives finite rewards."""

import numpy as np
import torch
from sprint.dynamics import forward_kinematics

from gp import data
from gp import wys_terrain as WT
from gp.pretrain import Pretrain, PretrainConfig


def test_reference_scores_one() -> None:
    import sprint

    D = data.load("train", src=data.ROOT / "data" / "ab01")
    pick = [k for k, (m, t) in enumerate(zip(D["mode"], D["terrain_class"], strict=True))
            if m == "walk" and t == "level"][:4]
    D = data.take(D, pick)
    T = WT.build()
    T.heights[:] = 0.0
    T.block[:] = np.nan
    task = Pretrain(T, D, PretrainConfig(), seed=0)
    env = sprint.VecEnv(data.load_model("cpu"), 4, task=task, seed=0, device="cpu",
                        step_options=sprint.StepOptions(dt_ctrl=0.025, n_sub=75, implicit_friction=True,
                                                        implicit_normal=True, strict=False, average_fext=True),
                        perturb_slots=2)
    env.reset()
    s, f = env.state, task.ref_f.clone()
    q = task.ex_q[f].clone()
    q[:, task.pel["ty"]] -= task.ex_gy[f]                 # the clip's height over its ground, on flat ground at 0
    s.qpos[:], s.qvel[:] = q, task.ex_qd[f]
    forward_kinematics(env.model, env.data)
    terms, eh, angle = task.track(s, f)
    for k, v in terms.items():
        assert (v > 0.99).all(), (k, v)
    assert eh.abs().max() < 1e-3 and angle.max() < 1e-3
    assert (task.ex_speed_f[f] > 0.3).all()               # walking clips: the command follows their speed
    env.reset()
    _, r, *_ = env.step(torch.full((4, task.nmus), 0.05))
    assert torch.isfinite(torch.as_tensor(r)).all()


def test_ramp_starts_line_up() -> None:
    """Ramp clips start on the strip of their grade: right after a reset every world sits on its reference frame
    (all tracking terms ~1, pelvis height within its reset lift), and the ground under a ramp world rises at the
    strip's grade along +x."""
    from train_wys import build

    trials = ("ab01_mc019", "ab01_mc026", "ab01_mc031", "ab01_mc010")      # level, up 5, up 10, down 10
    env, task = build(8, 0, "cpu", None, PretrainConfig(ramp_share=0.5), kind="pretrain", trials=trials)
    env.reset()
    s = env.state
    terms, eh, angle = task.track(s, task.ref_f)
    for k, v in terms.items():
        assert (v > 0.95).all(), (k, v)
    assert eh.abs().max() < 0.05 and angle.max() < 1e-3
    assert task.wgrade.tolist() == [0, 0, 0, 0, 1, 1, 2, 2]
    x, z = s.body_pos[:, task.root, 0:1], s.body_pos[:, task.root, 2:3]
    slope = (task.ground(x + 0.5, z) - task.ground(x - 0.5, z))[:, 0]
    want = torch.tan(torch.deg2rad(torch.tensor([0.0, 5.0, 10.0])))[task.wgrade]
    assert torch.allclose(slope, want, atol=0.01), (slope, want)
