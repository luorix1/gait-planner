"""distill.py's StudentView on the recipe teacher's environment (CPU, mc021): the student's clip frame is the
recipe's reference frame, its observation has L1's layout (687 actor inputs) and stays finite over steps and resets,
and the command follows the clip."""

from dataclasses import replace

import torch

from distill import StudentView, student_loc, student_net
from gp import data
from gp.pretrain import PretrainConfig
from train import build


def test_student_view_on_recipe_env() -> None:
    from sprint.recipes.study import default_ppo

    D = data.load(("train", "test_trial", "test_terrain"), src=data.ROOT / "data" / "ab01")
    D = data.take(D, [k for k, n in enumerate(D["name"]) if str(n).startswith("ab01_mc021_")])
    env, task = build("baseline", D, 4, 0, "cpu")
    view = StudentView(None, D, replace(PretrainConfig(), teacher=False, warp=0, bootstrap_iters=0, refit=False),
                       seed=0, foot_model="models/fpm_ab01.pt")
    view.bind(env.model)
    view.attach(env, task)
    assert view.n_actor == 687
    obs_dim = sum(b[1] for b in view.blocks)
    env.reset()
    for _ in range(30):
        f = view.ref_frame()
        ref = task.ref["qpos"][torch.minimum(task.frame, task.clip_last[task.clip])]
        assert torch.allclose(view.ex_q[f][:, view.jidx], ref[:, view.jidx], atol=1e-5)
        so = view.observe(env.state)
        assert so.shape == (4, obs_dim) and torch.isfinite(so).all()
        assert torch.allclose(view.cmd[:, 0], view.ex_speed_f[f]) and (view.cmd[:, 0] > 0.8).all()   # 1.2 m/s trial
        env.step(torch.full((4, task.nmus), 0.05))
        view.prev_action = env.state.excitation.clone()
    net = student_net(view, env, replace(default_ppo(), rollout_steps=view.c.steps_per_iter), obs_dim)
    assert student_loc(net, so[:, :687]).shape == (4, task.nmus)
