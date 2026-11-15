"""CPU smoke checks: terrain lookups match SPRINT, both tasks step with consistent observations and finite rewards,
the prior trains and scores, and PPO runs an iteration. Run: .venv/bin/python -m pytest tests -q
(needs data/v1: python -m gp.data)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from gp import data


@pytest.fixture(scope="module")
def D() -> dict:
    full = data.load("train")
    cls = np.array(full["terrain_class"])
    pick = [int(np.flatnonzero(cls == c)[0]) for c in ("level", "ramp_up", "ramp_down", "stairs_up", "stairs_down")]
    return data.take(full, pick)


def test_terrain_matches_sprint(D: dict) -> None:
    from sprint.terrain import height as sprint_height

    from gp.terrain import height
    rng = np.random.default_rng(0)
    for i in range(len(D["name"])):
        P = np.c_[D["profile_x"][i][: D["profile_n"][i]], D["profile_y"][i][: D["profile_n"][i]]]
        x = rng.uniform(P[0, 0] - 5, P[-1, 0] + 5, 200)
        x = x[np.min(np.abs(x[:, None] - P[None, :, 0]), 1) > 1e-4]          # not exactly on a riser
        ref = np.array([sprint_height(P, v) for v in x])
        ours = height(D["profile_x"], D["profile_y"], np.full(1, i), x[None])[0]
        tours = height(torch.as_tensor(D["profile_x"]), torch.as_tensor(D["profile_y"]), torch.full((1,), i),
                       torch.as_tensor(x[None]))[0].numpy()
        assert np.abs(ours - ref).max() < 1e-6 and np.abs(tours - ref).max() < 1e-6, D["name"][i]


def _prior(D: dict, tmp_path) -> str:
    from gp.smp import Prior, window_features
    from train_prior import batch, windows
    rec, ends = windows(D, "cpu")
    p = Prior(nt=len(D["tracked"]), dt=1.0 / D["hz"])
    p.set_normaliser(window_features(batch(rec, ends, 512), p.cfg["nt"], p.cfg["dt"]))
    opt = torch.optim.AdamW(p.net.parameters(), lr=2e-4)
    first = None
    for _ in range(30):
        loss = p.loss(p.features(batch(rec, ends, 64)))
        opt.zero_grad()
        loss.backward()
        opt.step()
        first = first or float(loss.detach())
    assert float(loss.detach()) < first
    p.ema.module.load_state_dict(p.net.state_dict())
    path = str(tmp_path / "prior.pt")
    p.save(path)
    return path


def _env(task, worlds: int = 6):
    import sprint

    from gp.tasks import recipe_config
    model = data.load_model()
    c = task.config
    opts = sprint.StepOptions(dt_ctrl=1.0 / 40, n_sub=c.physics_steps, implicit_friction=True,
                              implicit_normal=c.implicit_normal, strict=False)
    assert recipe_config().physics_steps == 75
    return sprint.VecEnv(model, worlds, task=task, seed=0, step_options=opts, device="cpu", delays=c.delays)


def _roll(env, steps: int = 12) -> list:
    obs = env.reset()
    rewards = []
    for _ in range(steps):
        a = torch.full((env.num_worlds, env.action_dim), 0.05)
        obs, r, *_ = env.step(a)
        rewards.append(env.torch(r) if hasattr(env, "torch") else torch.as_tensor(r))
    o = env.torch(obs) if hasattr(env, "torch") else torch.as_tensor(obs)
    assert o.shape[1] == env.task.obs_dim and torch.isfinite(o).all()
    assert all(torch.isfinite(x).all() for x in rewards)
    return rewards


def test_terrain_imitation_steps(D: dict) -> None:
    from gp.tasks import TerrainImitation
    env = _env(TerrainImitation(D))
    _roll(env)
    perm, sign, act = env.task.mirror_maps()
    assert len(perm) == env.task.obs_dim and sorted(act.tolist()) == list(range(env.action_dim))


def test_plan_tracking_steps(D: dict, tmp_path) -> None:
    from gp.tasks import PlanConfig, PlanTracking
    task = PlanTracking(D, plan=PlanConfig(prior=_prior(D, tmp_path)))
    env = _env(task)
    _roll(env)
    cols = task.actor_columns()
    offsets = np.cumsum([0] + [b[1] for b in task.blocks])[:-1]
    actor_blocks = {b[0].split("[")[0] for b, o in zip(task.blocks, offsets, strict=True) if o in set(cols.tolist())}
    assert "joint_angle_error" not in actor_blocks and "plan_feet" in actor_blocks and "height_scan" in actor_blocks
    for k in ("r_contact", "r_foothold", "r_com", "r_style"):
        v = task.metrics[k]
        assert torch.isfinite(v).all() and (v >= 0).all() and (v <= 1).all(), k


def test_ppo_iteration(D: dict, tmp_path) -> None:
    from dataclasses import replace

    from sprint.recipes.ppo import PPO
    from sprint.recipes.study import default_ppo

    from gp.tasks import PlanConfig, PlanTracking
    env = _env(PlanTracking(D, plan=PlanConfig(prior=_prior(D, tmp_path))), worlds=8)
    agent = PPO(env, replace(default_ppo(), hidden=(64, 64), minibatches=2, rollout_steps=4), device="cpu")
    agent.train(1)
    assert agent.env_steps > 0


def test_sim_record_matches_data(D: dict) -> None:
    """The simulation's style record at a reset equals the data's, but for the reset lift on the root height."""
    from gp.data import record_layout
    from gp.tasks import TerrainImitation
    starts = [(i, 10) for i in range(len(D["name"]))]
    env = _env(TerrainImitation(D, starts=starts), worlds=len(starts))
    env.reset()
    task = env.task
    sim = task.record(env.state).cpu().numpy()
    f = (D["start"] + 10).astype(int)
    ref = D["record"][f].copy()
    ref[:, 1] += D["lift"][f]
    lay = record_layout(len(D["tracked"]))
    q_s, q_r = sim[:, lay["root_quat"]], ref[:, lay["root_quat"]]
    assert np.abs(np.abs((q_s * q_r).sum(1)) - 1).max() < 1e-5
    for k in ("root_pos", "joints", "ground"):
        assert np.abs(sim[:, lay[k]] - ref[:, lay[k]]).max() < 1e-4, k
    kp = np.abs(sim[:, lay["key_pos"]] - ref[:, lay["key_pos"]]).reshape(len(f), -1, 3)
    assert kp[..., [0, 2]].max() < 1e-4 and np.abs(kp[..., 1] - D["lift"][f][:, None]).max() < 1e-4


def test_specific_tension(D: dict) -> None:
    """Rajagopal's muscle masses at 0.6 MPa: from the same state, the metabolic cost drops (heat is charged per kg of
    muscle, work is not) and no force changes."""
    from gp.tasks import TerrainImitation
    starts = [(i, 20) for i in range(len(D["name"]))]
    out = {}
    for pa in (0.25e6, data.SPECIFIC_TENSION):
        env = _env(TerrainImitation(D, starts=starts), worlds=len(starts))
        assert np.allclose(env.model.host["mus_tension"], data.SPECIFIC_TENSION)
        env.model.set_specific_tension(pa)
        env.reset()
        env.step(torch.full((env.num_worlds, env.action_dim), 0.2))
        s = env.state
        out[pa] = (s.metabolic_power.sum(1).cpu().numpy(), s.muscle_force.cpu().numpy())
    (m25, f25), (m60, f60) = out[0.25e6], out[data.SPECIFIC_TENSION]
    assert np.array_equal(f25, f60)
    assert ((m60 < 0.9 * m25) & (m60 > m25 / 2.4)).all(), (m25, m60)
