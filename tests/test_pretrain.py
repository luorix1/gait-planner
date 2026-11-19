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


def test_clip_estimator_inputs_match_the_simulation() -> None:
    """The estimator's input computed from a clip equals the simulation's ``system_state`` with the body on that
    clip frame (velocities from finite differences: within a few cm/s), and clip labels are finite."""
    from dataclasses import replace

    from train_wys import build

    env, task = build(1, 0, "cpu", None, replace(PretrainConfig(), nominal_jitter=0.0), kind="pretrain",
                      trials=("ab01_mc021",))
    env.reset()
    s = env.state
    a = int(torch.unique(task.ex_first_f)[0])
    f = a + 150
    q = task.ex_q[f].clone()
    q[task.pel["ty"]] -= task.ex_gy[f]
    s.qpos[0], s.qvel[0] = q, task.ex_qd[f]
    forward_kinematics(env.model, env.data)
    task.cmd[:] = torch.tensor([float(task.ex_speed_f[f]), 0.0, 0.0])
    sim = task.system_state(task.body_state(s))[0]
    clip = task.clip_inputs(a, int(task.ex_last_f[a]) + 1)[150]
    assert torch.allclose(sim[6:], clip[6:], atol=1e-4), (sim[6:] - clip[6:]).abs().max()
    assert torch.allclose(sim[:3], clip[:3], atol=0.05) and torch.allclose(sim[3:6], clip[3:6], atol=0.15), (sim[:6], clip[:6])
    x, y = task.clip_estimator_data()
    assert len(x) == len(y) > 0 and torch.isfinite(x).all() and torch.isfinite(y).all()


def test_teacher_sees_the_clip() -> None:
    """The teacher's extra inputs: right after a reset (body on the clip) the joint-angle error at tap t is the clip's
    own change over t frames, the layout matches the observation, and the mirror map is an involution."""
    from dataclasses import replace

    from train_wys import build

    env, task = build(2, 0, "cpu", None, replace(PretrainConfig(), teacher=True, warp=0), kind="pretrain",
                      trials=("ab01_mc021",))
    env.reset()
    obs = env.torch(env.obs)
    assert obs.shape[1] == sum(b[1] for b in task.blocks) and torch.isfinite(obs).all()
    names = [b[0] for b in task.blocks]
    at = sum(b[1] for b in task.blocks[:names.index("ref_joint[5]")])
    f = task.ref_f
    want = task.ex_q[torch.minimum(f + 5, task.ex_last_f[f])][:, task.jidx] - task.ex_q[f][:, task.jidx]
    assert torch.allclose(obs[:, at:at + task.nj], want, atol=1e-4)
    P, S, _ = task.mirror_maps()
    assert (P[P] == np.arange(len(P))).all() and np.allclose(S * S[P], 1.0)
    assert task.n_actor > 687                              # L1's inputs plus the clip's
