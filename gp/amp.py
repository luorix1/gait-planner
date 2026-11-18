"""Adversarial Motion Prior: a discriminator (the style judge) whose score is the style reward. Copied from GAIT_amp
(``gait_amp/amp.py``, itself Walk_AMP's after Peng et al. 2021): least-squares loss (human pairs -> +1, the policy's
-> -1), gradient penalty on human inputs (5.0), logit-weight regulariser (0.01), hidden-weight decay (1e-4), Adam at
5e-5, grad-norm clip 4, a cumulative input normaliser clipped at +-5, a replay of the policy's pairs, mirrored human
pairs half the time, and style reward ``max(0, 1 - 0.25 (D - 1)^2)`` from the judge as it was before this iteration's
update.

One change from GAIT_amp: the human side is a table of whole pairs (``gp.ladder`` builds a pair from two poses plus
the root's motion between them), sampled uniformly over clips and then over a clip's pairs; the mirror map is the
pair's.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

__all__ = ["AMPConfig", "AMP"]


@dataclass
class AMPConfig:
    hidden: tuple[int, ...] = (512, 512)
    lr: float = 5e-5
    grad_penalty: float = 5.0
    logit_reg: float = 0.01
    weight_decay: float = 1e-4
    max_grad_norm: float = 4.0
    replay_size: int = 500_000
    #: Per judge step: this many fresh + this many replayed policy pairs, against twice as many human ones.
    batch: int = 1024
    #: Judge steps per PPO iteration: GAIT_amp's rule (one per PPO minibatch) at default_ppo's 2 epochs x 40.
    steps: int = 80
    mirror_expert: bool = True
    norm_clip: float = 5.0
    #: False freezes the judge and its normaliser (the style reward is then a fixed function).
    train_disc: bool = True


class WelfordNorm(nn.Module):
    """Cumulative mean/variance normaliser."""

    def __init__(self, dim: int, clip: float, min_std: float = 1e-4):
        super().__init__()
        self.clip, self.min_std = clip, min_std
        self.register_buffer("mean", torch.zeros(dim))
        self.register_buffer("m2", torch.zeros(dim))
        self.register_buffer("count", torch.zeros(()))

    @torch.no_grad()
    def update(self, x: torch.Tensor) -> None:
        n = x.shape[0]
        mu, var = x.mean(0), x.var(0, unbiased=False)
        tot = self.count + n
        d = mu - self.mean
        self.mean += d * n / tot
        self.m2 += var * n + d.pow(2) * self.count * n / tot
        self.count.copy_(tot)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        std = (self.m2 / self.count.clamp(min=1)).sqrt().clamp(min=self.min_std)
        return ((x - self.mean) / std).clamp(-self.clip, self.clip)


class AMP:
    """``pairs`` ``[P, dim]``: every human pair, clip by clip; ``clip_start`` ``[nclip + 1]``: each clip's first pair."""

    def __init__(self, cfg: AMPConfig, pairs: torch.Tensor, clip_start: np.ndarray,
                 mirror: tuple[np.ndarray, np.ndarray], *, device: torch.device):
        self.cfg, self.dev = cfg, device
        self.pairs = pairs.to(device, torch.float32)
        cs = np.asarray(clip_start, np.int64)
        self.start = torch.as_tensor(cs[:-1], device=device)
        self.count = torch.as_tensor(np.diff(cs), device=device)
        self.mperm = torch.as_tensor(mirror[0], dtype=torch.long, device=device)
        self.msign = torch.as_tensor(mirror[1], dtype=torch.float32, device=device)
        self.dim = int(self.pairs.shape[1])
        layers, k = [], self.dim
        for h in cfg.hidden:
            layers += [nn.Linear(k, h), nn.ReLU()]
            k = h
        self.body = nn.Sequential(*layers).to(device)
        self.head = nn.Linear(k, 1).to(device)
        nn.init.uniform_(self.head.weight, -1.0, 1.0)     # Walk_AMP's logit init
        nn.init.zeros_(self.head.bias)
        self.norm = WelfordNorm(self.dim, cfg.norm_clip).to(device)
        self.opt = torch.optim.Adam(list(self.body.parameters()) + list(self.head.parameters()), lr=cfg.lr)
        self.replay = torch.zeros(cfg.replay_size, self.dim, device=device)
        self.rfill = self.rptr = 0

    def expert(self, n: int) -> torch.Tensor:
        """``n`` human pairs: a uniform clip, then a uniform pair in it; mirrored half the time."""
        c = torch.randint(len(self.start), (n,), device=self.dev)
        x = self.pairs[self.start[c] + (torch.rand(n, device=self.dev) * self.count[c]).long()]
        if self.cfg.mirror_expert:
            m = torch.rand(n, 1, device=self.dev) < 0.5
            x = torch.where(m, x[:, self.mperm] * self.msign, x)
        return x

    def _push(self, x: torch.Tensor) -> None:
        cap = self.cfg.replay_size
        x = x[torch.randperm(x.shape[0], device=self.dev)[:cap]]
        idx = (self.rptr + torch.arange(x.shape[0], device=self.dev)) % cap
        self.replay[idx] = x
        self.rptr = int((self.rptr + x.shape[0]) % cap)
        self.rfill = min(self.rfill + x.shape[0], cap)

    def logit(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.body(x)).squeeze(-1)

    @torch.no_grad()
    def reward(self, pairs: torch.Tensor) -> torch.Tensor:
        d = self.logit(self.norm(pairs))
        return (1.0 - 0.25 * (d - 1.0).pow(2)).clamp(min=0.0)

    @torch.enable_grad()                 # called from the task's reward, inside the PPO's no-grad rollout
    def train(self, agent: torch.Tensor) -> dict[str, float]:
        """``cfg.steps`` judge updates against this iteration's policy pairs ``agent``."""
        c = self.cfg
        self.norm.update(torch.cat([agent, self.expert(agent.shape[0])]))
        params = list(self.body.parameters()) + list(self.head.parameters())
        stats = torch.zeros(5, device=self.dev)
        for _ in range(c.steps):
            fresh = agent[torch.randint(agent.shape[0], (c.batch,), device=self.dev)]
            if self.rfill:
                fresh = torch.cat([fresh, self.replay[torch.randint(self.rfill, (c.batch,), device=self.dev)]])
            pol = self.norm(fresh)
            exp = self.norm(self.expert(pol.shape[0])).requires_grad_(True)
            d_e, d_p = self.logit(exp), self.logit(pol)
            l_ls = 0.5 * ((d_e - 1.0).pow(2).mean() + (d_p + 1.0).pow(2).mean())
            g = torch.autograd.grad(d_e.sum(), exp, create_graph=True)[0]
            l_gp = g.pow(2).sum(-1).mean()
            l_wd = sum(m.weight.pow(2).sum() for m in self.body if isinstance(m, nn.Linear))
            loss = l_ls + c.grad_penalty * l_gp + c.logit_reg * self.head.weight.pow(2).sum() + c.weight_decay * l_wd
            self.opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(params, c.max_grad_norm)
            self.opt.step()
            with torch.no_grad():
                stats += torch.stack([l_ls.detach(), l_gp.detach(), d_e.mean(), d_p.mean(),
                                      0.5 * ((d_e > 0).float().mean() + (d_p < 0).float().mean())])
        self._push(agent)
        s = (stats / max(c.steps, 1)).tolist()
        return {"amp_loss": s[0], "amp_grad_pen": s[1], "amp_d_expert": s[2], "amp_d_policy": s[3], "amp_accuracy": s[4]}

    def state_dict(self) -> dict:
        return {"body": self.body.state_dict(), "head": self.head.state_dict(), "norm": self.norm.state_dict(),
                "opt": self.opt.state_dict()}

    def load_state_dict(self, sd: dict) -> None:
        self.body.load_state_dict(sd["body"])
        self.head.load_state_dict(sd["head"])
        self.norm.load_state_dict(sd["norm"])
        self.opt.load_state_dict(sd["opt"])
