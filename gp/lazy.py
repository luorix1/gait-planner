"""Event-triggered ("lazy") replanning: keep playing the plan already made and generate a new one only when something
says it is stale, instead of every 0.2 s (export_pose2 ``--keep 4``) or every 8 control steps and every touchdown
(gp.planner.Replanner).

After MotionBricks (2026: replan "whenever the control signals are changed or the future motion buffer does not have
enough frames left"; 10 Hz or on a command change on the G1) and RDM (Zhou et al. 2023) / DVAC (2026), which also
replan where the plan stops being trustworthy. Fewer generations also means fewer seams (scoreboard.py ``seam``) and
less switching between plans that each would have worked.

Reasons, bits of the ``why`` mask (per world):

- ``FRESH``   a new episode.
- ``LOW``     fewer than ``min_left_s`` of trusted plan left: the buffer floor, the old fixed rate made much slower.
- ``GOAL``    the intent changed: the goal's offset from the body (the target minus the pelvis, in the world) moved more
              than ``goal_tol`` m since the plan was made: a new command, a turn, a replay drifting off its clip.
- ``TERRAIN`` the ground under the plan changed: the heights the plan's patch saw, read again now, differ by more than
              ``terrain_tol`` m (a perception update; never fires on static terrain).
- ``DRIFT``   (closed loop) the measured body left the plan: centre of mass xz more than ``drift_tol`` m from the
              plan's at this frame.
- ``STEP``    (closed loop) a touchdown more than ``step_tol`` m from its planned foothold, or a foot down for
              ``early_s`` where the plan has it swinging ("the first look at the ground was wrong").

No replan within ``min_gap_s`` of the last one (hysteresis against thrashing), except ``FRESH`` and an empty buffer.

Commit horizon (optional, ``samples`` > 1 at generation): the plan is trusted up to the first frame where ``samples``
draws disagree by more than ``spread_tol`` m (pelvis and foot points), at least ``min_commit_s``; ``LOW`` counts from
there, so an uncertain plan is replaced sooner. Costs ``samples`` x the generation batch.

Pure torch (no SPRINT), so the trigger logic tests on any CPU (tests/test_lazy.py). Users: gp.planner.LazyReplanner
(closed loop, physics) and export_pose2.lazy_rollout (open loop, kinematic).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch

FRESH, LOW, GOAL, TERRAIN, DRIFT, STEP = 1, 2, 4, 8, 16, 32
REASONS = {"fresh": FRESH, "low": LOW, "goal": GOAL, "terrain": TERRAIN, "drift": DRIFT, "step": STEP}


@dataclass
class Triggers:
    """Thresholds (seconds and metres). Defaults are starting points, not tuned."""

    min_left_s: float = 0.4      # MotionBricks' buffer floor; with a 1.2 s plan, ~0.8 s is played before LOW
    min_gap_s: float = 0.1
    goal_tol: float = 0.25       # m of goal offset over 1.2 s: ~0.2 m/s of speed, or ~12 deg of heading at 1.2 m/s
    terrain_tol: float = 0.03
    drift_tol: float = 0.08
    step_tol: float = 0.06
    early_s: float = 0.05
    spread_tol: float = 0.05
    min_commit_s: float = 0.3

    def frames(self, s: float, hz: float) -> int:
        return max(1, int(round(s * hz)))

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


# --- distances ------------------------------------------------------------------------------------------------------

def goal_shift(off_now: torch.Tensor, off_plan: torch.Tensor) -> torch.Tensor:
    """``[B]`` m: how far the goal's offset from the body (``[B, 2]`` world xz) moved since the plan was made."""
    return (off_now - off_plan).norm(dim=-1)


def terrain_shift(h_now: torch.Tensor, h_plan: torch.Tensor) -> torch.Tensor:
    """``[B]`` m: the largest change of the ground at the points the plan saw (``[B, ...]`` heights; NaN = unseen)."""
    d = (h_now - h_plan).abs().flatten(1)
    return torch.nan_to_num(d, nan=0.0).amax(1)


def drift(xz_meas: torch.Tensor, xz_plan: torch.Tensor) -> torch.Tensor:
    """``[B]`` m: the measured body (``[B, 2]``) from the plan at this frame."""
    return (xz_meas - xz_plan).norm(dim=-1)


def step_miss(touchdown: torch.Tensor, foot_xz: torch.Tensor, planned_xz: torch.Tensor) -> torch.Tensor:
    """``[B]`` m: the largest miss of a foot that touched down this frame (``touchdown`` ``[B, 2]``; footholds
    ``[B, 2, 2]`` xz), 0 where none did."""
    miss = (foot_xz - planned_xz).norm(dim=-1)
    return torch.where(touchdown, miss, torch.zeros_like(miss)).amax(1)


def commit_frames(samples: torch.Tensor, tol: float, min_commit: int) -> torch.Tensor:
    """``[B]`` frames of a plan to trust: up to the first frame where draws ``[S, B, K, P, D]`` (S >= 2 samples of P
    points) spread more than ``tol`` m (the largest per-point standard deviation's norm), at least ``min_commit``."""
    S, B, K = samples.shape[:3]
    if S < 2:
        return torch.full((B,), K, dtype=torch.long, device=samples.device)
    spread = samples.std(0).norm(dim=-1).amax(-1)                          # [B, K]
    over = spread > tol
    first = torch.where(over.any(1), over.float().argmax(1), torch.full((B,), K, device=samples.device))
    return first.long().clamp(min=min(min_commit, K), max=K)


# --- the decision ---------------------------------------------------------------------------------------------------

def decide(T: Triggers, hz: float, *, since: torch.Tensor, left: torch.Tensor, fresh: torch.Tensor,
           goal: torch.Tensor | None = None, terrain: torch.Tensor | None = None, drift_m: torch.Tensor | None = None,
           step: torch.Tensor | None = None, early: torch.Tensor | None = None, min_left: int | None = None
           ) -> tuple[torch.Tensor, torch.Tensor]:
    """(``due`` ``[B]`` bool, ``why`` ``[B]`` reason bits) from frames since the last generation, trusted frames left
    and the distances above (any may be None: not measured). ``early``: ``[B]`` bool, a foot down for ``early_s``
    where the plan has it swinging. ``min_left``: the buffer floor in frames (default ``min_left_s``)."""
    why = torch.zeros_like(since)
    why |= torch.where(fresh, FRESH, 0)
    why |= torch.where(left < (T.frames(T.min_left_s, hz) if min_left is None else min_left), LOW, 0)
    for v, tol, bit in ((goal, T.goal_tol, GOAL), (terrain, T.terrain_tol, TERRAIN), (drift_m, T.drift_tol, DRIFT),
                        (step, T.step_tol, STEP)):
        if v is not None:
            why |= torch.where(v > tol, bit, 0)
    if early is not None:
        why |= torch.where(early, STEP, 0)
    why = torch.where(fresh, FRESH, why)              # a new episode's first plan is only that (no stale baselines)
    due = fresh | (left <= 0) | ((why != 0) & (since >= T.frames(T.min_gap_s, hz)))
    return due, torch.where(due, why, torch.zeros_like(why))


class Log:
    """Generations per reason (a generation with several reasons counts under each) and in all."""

    def __init__(self) -> None:
        self.counts = dict.fromkeys(REASONS, 0)
        self.replans = 0

    def add(self, why: torch.Tensor) -> None:
        w = why[why != 0]
        self.replans += len(w)
        for k, bit in REASONS.items():
            self.counts[k] += int(((w & bit) != 0).sum())

    def summary(self, seconds: float, worlds: int) -> dict[str, float]:
        """Replans per second per world, and the share of generations each reason fired in."""
        n = max(self.replans, 1)
        return dict(replans_per_s=round(self.replans / max(seconds * worlds, 1e-9), 2),
                    **{f"share_{k}": round(c / n, 3) for k, c in self.counts.items()})
