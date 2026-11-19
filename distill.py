"""Distil an imitation teacher (train.py --task baseline: SPRINT's recipe) into L1's student, the muscle policy that
reads L1's inputs (body state, command, muscle states, the estimator's two foot patches), by DAgger.

One simulation, the teacher's (the recipe's environment, reward and episodes, untouched). At every state both read
their own inputs: the teacher the recipe's observation, the student L1's, computed from the same body by
:class:`StudentView` (gp.pretrain's own observation code, with the ground, the clip frame and the episode ends taken
from the recipe). Each episode is driven by the teacher with probability beta, which falls linearly from 1 to 0 over
``--beta-iters`` iterations, else by the student; every visited state is labelled with the teacher's action, and the
student is fitted to all labels so far (a ring buffer), on the teacher's pre-squash mean (MSE).

The student is SPRINT's ActorCritic exactly as stage 1 builds it (train_wys.py's PPO on the ladder task), saved as
``checkpoint.pt`` + ``task_state.pt`` (the estimator), so ``train_wys.py --init-from <out>`` starts from it. Its
critic is untrained: stage 1 trains a fresh one.

Run (GPU, detached; a fresh --out per run):
  setsid nohup .venv/bin/python distill.py --teacher runs/recipe_teacher_mc021 --trials ab01_mc021 \\
      --out runs/distill_mc021 --minutes 30 > runs/distill_mc021.log 2>&1 < /dev/null &
Out: record.json, progress.csv, checkpoint.pt, task_state.pt, teacher.pt (the snapshot distilled), eval.json (the
student and the teacher alone, deterministic, from the recipe's starts).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import subprocess
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import numpy as np
import sprint
import torch
from sprint.recipes.ppo import PPO, ActorCritic, RunningNorm, env_action_bounds
from sprint.recipes.study import default_ppo

from gp import fpm as FPM
from gp.pretrain import Pretrain, PretrainConfig
from gp.wys import _heading, _wrap
from train import build as build_recipe
from train import load_clips


class StudentView(Pretrain):
    """L1's observation of a body that another task simulates (the recipe teacher's ``Imitation``). Only the parts of
    WatchYourStep / Ladder / Pretrain setup that the observation reads are set; the environment is never written."""

    def attach(self, env: Any, teacher: Any) -> None:
        c, model = self.c, env.model
        self.tt, self.n = teacher, env.num_worlds
        dev = self.dev = teacher.dev
        self.gen = torch.Generator(device=dev).manual_seed(self.seed)
        self.q0 = torch.as_tensor(np.asarray(model.host["coord_default"][: len(self.coords)], np.float32), device=dev)
        self.jidx = torch.as_tensor(self.joints, device=dev)
        self.e_y = torch.tensor([0.0, 1.0, 0.0], device=dev)
        n = self.n
        z = lambda *k: torch.zeros(n, *k, device=dev)   # noqa: E731
        self.cmd, self.heading_target, self.prev_action = z(3), z(), z(self.nmus)
        self.xy_drift, self.z_drift, self.nominal = z(2), z(1), self.q0[self.jidx].expand(n, -1).clone()
        g = torch.arange(-(self.nx // 2), self.nx // 2 + 1, device=dev) * c.scan_res
        lat = torch.arange(-(self.ny // 2), self.ny // 2 + 1, device=dev) * c.scan_res
        self.grid_f, self.grid_l = g.repeat_interleave(self.ny), lat.repeat(self.nx)
        k = c.patch // 2
        off = torch.arange(-k, k + 1, device=dev)
        self.off_x, self.off_y = off.repeat_interleave(c.patch), off.repeat(c.patch)
        self.fpm = FPM.NormalizedFootPredictor(FPM.FootPredictor(self.state_dim, 4), torch.zeros(self.state_dim),
                                               torch.ones(self.state_dim)).to(dev)
        self.fpm.load_state_dict(torch.load(self.foot_model_path, map_location=dev))
        self.fpm_ready, self.steps = True, 1
        self._expert(model)
        self._speed_table()
        # each recipe clip's first frame in the clip tables (_expert: the clips of style_modes, in D's order)
        first = np.full(len(self.D["name"]), -1)
        first[self.ex_clips] = np.r_[0, np.cumsum(self.D["frames"][self.ex_clips])[:-1]]
        self.clip_first = torch.as_tensor(first, device=dev)
        self.clip_frames = torch.as_tensor(np.asarray(self.D["frames"]), device=dev)
        self.new_episode(torch.ones(n, dtype=torch.bool, device=dev))

    def ground(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        return self.tt.ground(x, z)

    def ref_frame(self) -> torch.Tensor:
        """Each world's current reference frame in the clip tables (the recipe's frame, held at its clip's end)."""
        tt = self.tt
        j = (tt.frame - tt.clip_start[tt.clip]).clamp(min=0)
        return self.clip_first[tt.clip] + torch.minimum(j, self.clip_frames[tt.clip] - 1)

    def new_episode(self, w: torch.Tensor) -> None:
        """Per-episode draws on worlds ``w`` (bool), as Ladder's reset: drifts, nominal pose, no last action."""
        c, i = self.c, w.nonzero().squeeze(1)
        k = len(i)
        if k == 0:
            return
        self.xy_drift[i] = self._sym((k, 2), c.xy_drift)
        self.z_drift[i] = self._sym((k, 1), c.z_drift)
        self.nominal[i] = self.q0[self.jidx] + self._sym((k, self.nj), c.nominal_jitter)
        self.prev_action[i] = 0.0

    def observe(self, s: Any) -> torch.Tensor:
        """L1's full observation (actor, then critic columns), the command following the clip as Pretrain's."""
        c, f = self.c, self.ref_frame()
        self.assert_clips(f)
        self.heading_target[:] = self.ex_psi_f[f]
        self.cmd[:, 0], self.cmd[:, 1] = self.ex_speed_f[f], 0.0
        psi = _heading(s.body_quat[:, self.root])
        self.cmd[:, 2] = (c.heading_stiffness * _wrap(self.heading_target - psi)).clamp(*c.ang_z)
        return self.observation(s)

    def assert_clips(self, f: torch.Tensor) -> None:
        if bool((self.clip_first[self.tt.clip] < 0).any()):
            raise ValueError("a recipe clip is not among the student's clip tables (LadderConfig.style_modes)")


def student_net(view: StudentView, env: Any, cfg: Any, obs_dim: int) -> ActorCritic:
    """The network stage 1's PPO builds for the ladder task (PPO.__init__'s, all muscles)."""
    lo, hi = env_action_bounds(env, cfg.excitation_min)
    return ActorCritic(obs_dim, env.action_dim, cfg, action_low=lo, action_high=hi,
                       initial_mean=[cfg.initial_action_mean] * env.action_dim,
                       actor_cols=np.arange(view.n_actor, dtype=np.int64))


def student_loc(net: ActorCritic, x_actor: torch.Tensor) -> torch.Tensor:
    """Pre-squash mean from raw actor columns (the normaliser's actor slice, as ``net.loc(net.norm(obs))``)."""
    cols = net.actor_cols
    z = ((x_actor - net.norm.mean[cols]) / (net.norm.var[cols] + 1e-8).sqrt()).clamp(-net.norm.clip, net.norm.clip)
    return net.loc_limit * torch.tanh(net.pi(z) / net.loc_limit)


def evaluate(env: Any, task: Any, view: StudentView, act: Any, steps: int) -> dict[str, float]:
    """Every world driven by ``act(teacher_obs, student_obs) -> excitation`` for ``steps`` control steps from fresh
    recipe starts: ended episodes' length and fall share, joint error over all steps."""
    obs = env.torch(env.reset())
    view.new_episode(torch.ones(env.num_worlds, dtype=torch.bool, device=view.dev))
    so = view.observe(env.state)
    n_ep = fell = 0
    length: list[float] = []
    err = []
    t_ep = torch.zeros(env.num_worlds, device=view.dev)
    with torch.no_grad():
        for _ in range(steps):
            a = act(obs, so)
            obs, _, term, trunc, _ = env.step(a)
            obs = env.torch(obs)
            err.append(task.metrics["joint_error_deg"].mean().item())
            done = env.torch(term).bool() | env.torch(trunc).bool()
            t_ep += 1
            view.prev_action = env.state.excitation.clone()
            if bool(done.any()):
                n_ep += int(done.sum())
                fell += int(task._fell[done].sum())
                length += t_ep[done].tolist()
                t_ep[done] = 0
                view.new_episode(done)
            so = view.observe(env.state)
    return dict(episodes=n_ep, fell_share=fell / max(n_ep, 1), mean_length_steps=float(np.mean(length)) if length else float("nan"),
                joint_error_deg=float(np.mean(err)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", required=True, help="a train.py --task baseline run (its checkpoint.pt)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--dataset", default="ab01")
    ap.add_argument("--trials", default="", help="only these trials' windows, comma-separated (the teacher's)")
    ap.add_argument("--modes", default="walk,start_stop,stand", help="only these modes' windows (the student's clip tables hold these)")
    ap.add_argument("--foot-model", default="models/fpm_ab01.pt")
    ap.add_argument("--minutes", type=float, default=30)
    ap.add_argument("--worlds", type=int, default=2048)
    ap.add_argument("--rollout", type=int, default=24, help="control steps per iteration")
    ap.add_argument("--norm-iters", type=int, default=10,
                    help="first iterations: the teacher drives and the student's normaliser is set from their states "
                         "(a cumulative average; PPO's EMA at decay 0.999999 would keep its first batch, where every "
                         "last action is 0); no fitting yet")
    ap.add_argument("--beta-iters", type=int, default=150, help="iterations over which the teacher's share goes 1 -> 0")
    ap.add_argument("--buffer", type=int, default=1_000_000, help="labelled states kept (ring)")
    ap.add_argument("--updates", type=int, default=8, help="gradient steps per iteration")
    ap.add_argument("--batch", type=int, default=8192)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--eval-steps", type=int, default=800)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True)                       # a fresh folder per run, never overwrite one
    torch.manual_seed(a.seed)
    # the teacher's environment and its snapshot (a running teacher rewrites checkpoint.pt every 10 min)
    D = load_clips(a.dataset, a.trials, a.modes)
    env, task = build_recipe("baseline", D, a.worlds, a.seed, a.device)
    shutil.copy2(Path(a.teacher) / "checkpoint.pt", out / "teacher.pt")
    tcfg = default_ppo()
    tcfg = replace(tcfg, symmetry_coef=tcfg.symmetry_coef if task.symmetric else 0.0)
    teacher = PPO(env, tcfg, device="cuda" if env.device.is_cuda else "cpu")
    teacher.load(str(out / "teacher.pt"))
    T = teacher.net.eval()
    # the student: L1's inputs from the same body, stage 1's network and config
    view = StudentView(None, D, replace(PretrainConfig(), teacher=False, warp=0, bootstrap_iters=0, refit=False),
                       seed=a.seed, foot_model=a.foot_model)
    view.bind(env.model)
    view.attach(env, task)
    obs_dim = sum(b[1] for b in view.blocks)
    scfg = replace(default_ppo(), rollout_steps=view.c.steps_per_iter)
    S = student_net(view, env, scfg, obs_dim).to(view.dev)
    with torch.no_grad():
        S.raw_std.copy_(T.raw_std)
    opt = torch.optim.Adam(S.pi.parameters(), lr=a.lr)
    seed_norm = RunningNorm(obs_dim, ema_decay=None).to(view.dev)
    na, dev = view.n_actor, view.dev
    bx = torch.zeros(a.buffer, na, device=dev)
    by = torch.zeros(a.buffer, env.action_dim, device=dev)
    fill = head = 0
    git = subprocess.run(["git", "rev-parse", "HEAD"], cwd=Path(__file__).parent, capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain"], cwd=Path(__file__).parent, capture_output=True,
                                text=True).stdout.strip())
    record = dict(args=vars(a), gait_planner_git=git, dirty=dirty, sprint=sprint.__version__, clips=[str(n) for n in D["name"]],
                  teacher_env_steps=int(teacher.env_steps), student_inputs=na, student_obs_dim=obs_dim,
                  view=asdict(view.c), started=time.strftime("%Y-%m-%d %H:%M:%S"))
    (out / "record.json").write_text(json.dumps(record, indent=1, default=str))
    print(f"teacher {a.teacher} at {teacher.env_steps / 1e6:.1f}M steps; student {na} inputs ({obs_dim} with the critic's)",
          flush=True)

    obs = env.torch(env.reset())
    view.new_episode(torch.ones(a.worlds, dtype=torch.bool, device=dev))
    so = view.observe(env.state)
    by_teacher = torch.ones(a.worlds, dtype=torch.bool, device=dev)
    t_ep = torch.zeros(a.worlds, device=dev)
    f = (out / "progress.csv").open("w", buffering=1)
    w: csv.DictWriter | None = None
    start, it = time.time(), 0
    deadline = start + a.minutes * 60
    while time.time() < deadline:
        beta = min(1.0, max(0.0, 1.0 - (it - a.norm_iters) / max(a.beta_iters, 1)))
        ended = {"s_ep": 0, "s_fell": 0, "s_len": 0.0}
        s_err, t_err, loss_sum = [], [], 0.0
        with torch.no_grad():
            for _ in range(a.rollout):
                if it < a.norm_iters:
                    seed_norm.update(so)
                y = T.actor(obs)                                     # the teacher's pre-squash mean: the label
                x = so[:, :na]
                bi = (head + torch.arange(a.worlds, device=dev)) % a.buffer
                bx[bi], by[bi] = x, y
                head, fill = int((head + a.worlds) % a.buffer), min(fill + a.worlds, a.buffer)
                act = torch.where(by_teacher[:, None], T._squash(y), S._squash(student_loc(S, x)))
                obs, _, term, trunc, _ = env.step(act)
                obs = env.torch(obs)
                je = task.metrics["joint_error_deg"]
                if bool((~by_teacher).any()):
                    s_err.append(je[~by_teacher].mean().item())
                if bool(by_teacher.any()):
                    t_err.append(je[by_teacher].mean().item())
                done = env.torch(term).bool() | env.torch(trunc).bool()
                t_ep += 1
                view.prev_action = env.state.excitation.clone()
                if bool(done.any()):
                    sd = done & ~by_teacher
                    ended["s_ep"] += int(sd.sum())
                    ended["s_fell"] += int(task._fell[sd].sum())
                    ended["s_len"] += float(t_ep[sd].sum())
                    t_ep[done] = 0
                    view.new_episode(done)
                    by_teacher[done] = torch.rand(int(done.sum()), device=dev) < beta
                so = view.observe(env.state)
        if it + 1 == a.norm_iters:
            S.norm.mean.copy_(seed_norm.mean)
            S.norm.var.copy_(seed_norm.var)
            S.norm.count.copy_(seed_norm.count)
        for _ in range(a.updates if it >= a.norm_iters else 0):
            j = torch.randint(fill, (a.batch,), device=dev)
            loss = (student_loc(S, bx[j]) - by[j]).pow(2).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(S.pi.parameters(), 1.0)
            opt.step()
            loss_sum += loss.item()
        it += 1
        row = dict(iteration=it, seconds=round(time.time() - start), beta=round(beta, 3), loss=loss_sum / a.updates,
                   teacher_share=float(by_teacher.float().mean()),
                   student_joint_error_deg=float(np.mean(s_err)) if s_err else math.nan,
                   teacher_joint_error_deg=float(np.mean(t_err)) if t_err else math.nan,
                   student_episodes=ended["s_ep"], student_fell_share=ended["s_fell"] / max(ended["s_ep"], 1),
                   student_mean_length=ended["s_len"] / max(ended["s_ep"], 1), labels=fill)
        if w is None:
            w = csv.DictWriter(f, fieldnames=list(row))
            w.writeheader()
        w.writerow(row)
        if it % 10 == 1:
            print("  ".join(f"{k} {v:.4g}" if isinstance(v, float) else f"{k} {v}" for k, v in row.items()), flush=True)
    # save as stage 1 loads it (train_wys.py --init-from)
    torch.save(dict(net=S.state_dict(), distilled_from=str(a.teacher), iterations=it), out / "checkpoint.pt")
    torch.save(dict(fpm=view.fpm.state_dict(), fpm_ready=True), out / "task_state.pt")
    S.eval()
    res = dict(student=evaluate(env, task, view, lambda o, s: S._squash(student_loc(S, s[:, :na])), a.eval_steps),
               teacher=evaluate(env, task, view, lambda o, s: T.actor(o, squashed=True), a.eval_steps))
    (out / "eval.json").write_text(json.dumps(res, indent=1))
    record.update(finished=time.strftime("%Y-%m-%d %H:%M:%S"), iterations=it, eval=res)
    (out / "record.json").write_text(json.dumps(record, indent=1, default=str))
    print(json.dumps(res, indent=1), flush=True)


if __name__ == "__main__":
    main()
