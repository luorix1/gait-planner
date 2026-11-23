"""gp.lazy's trigger logic on the CPU, torch only (no SPRINT, no data): each reason fires on its own and only past its
threshold, hysteresis holds back all but a fresh episode and an empty buffer, the commit horizon stops where samples
disagree, and the log counts reasons."""

import torch

from gp import lazy

HZ = 20.0


def _decide(**kw):
    B = 4
    base = dict(since=torch.full((B,), 10), left=torch.full((B,), 20), fresh=torch.zeros(B, dtype=torch.bool))
    base.update(kw)
    return lazy.decide(lazy.Triggers(), HZ, **base)


def test_quiet_when_nothing_happens() -> None:
    due, why = _decide(goal=torch.zeros(4), terrain=torch.zeros(4), drift_m=torch.zeros(4), step=torch.zeros(4),
                       early=torch.zeros(4, dtype=torch.bool))
    assert not due.any() and (why == 0).all()


def test_each_reason_alone() -> None:
    T = lazy.Triggers()
    over = torch.tensor([0.0, 1.0, 0.0, 0.0])
    for key, tol, bit in (("goal", T.goal_tol, lazy.GOAL), ("terrain", T.terrain_tol, lazy.TERRAIN),
                          ("drift_m", T.drift_tol, lazy.DRIFT), ("step", T.step_tol, lazy.STEP)):
        due, why = _decide(**{key: over * (tol * 1.01)})
        assert due.tolist() == [False, True, False, False], key
        assert int(why[1]) == bit, key
        due, _ = _decide(**{key: over * (tol * 0.99)})          # just under: nothing
        assert not due.any(), key
    due, why = _decide(early=over.bool())
    assert due.tolist() == [False, True, False, False] and int(why[1]) == lazy.STEP


def test_low_buffer_and_floor_override() -> None:
    floor = lazy.Triggers().frames(lazy.Triggers.min_left_s, HZ)          # 8 frames at 20 Hz
    left = torch.tensor([floor - 1, floor, 30, 30])
    due, why = _decide(left=left)
    assert due.tolist() == [True, False, False, False] and int(why[0]) == lazy.LOW
    due, _ = _decide(left=left, min_left=1)
    assert not due.any()


def test_hysteresis() -> None:
    since = torch.tensor([0, 0, 0, 5])
    due, why = _decide(since=since, goal=torch.ones(4), fresh=torch.tensor([True, False, False, False]),
                       left=torch.tensor([20, 0, 20, 20]))
    # fresh and an empty buffer go through at once; an event waits for min_gap (2 frames at 20 Hz)
    assert due.tolist() == [True, True, False, True]
    assert int(why[0]) == lazy.FRESH and int(why[1]) & lazy.LOW and int(why[2]) == 0 and int(why[3]) == lazy.GOAL


def test_distances() -> None:
    assert torch.allclose(lazy.goal_shift(torch.tensor([[1.0, 0.0]]), torch.tensor([[0.0, 0.0]])), torch.tensor([1.0]))
    h0 = torch.zeros(2, 3, 3)
    h1 = h0.clone()
    h1[1, 2, 2] = -0.1
    h1[0, 0, 0] = float("nan")                                             # unseen: ignored
    assert torch.allclose(lazy.terrain_shift(h1, h0), torch.tensor([0.0, 0.1]))
    td = torch.tensor([[True, False], [False, False]])
    foot = torch.tensor([[[0.1, 0.0], [5.0, 5.0]], [[9.0, 9.0], [9.0, 9.0]]])
    assert torch.allclose(lazy.step_miss(td, foot, torch.zeros(2, 2, 2)), torch.tensor([0.1, 0.0]))


def test_commit_frames() -> None:
    S, B, K = 4, 3, 24
    x = torch.zeros(S, B, K, 5, 3)
    noise = torch.tensor([-1.0, 1.0, -1.0, 1.0])[:, None, None]
    x[:, 0, 10:, :, 0] = noise * 0.2              # episode 0: samples disagree from frame 10
    x[:, 1, 2:, :, 0] = noise * 0.2               # episode 1: from frame 2, but at least min_commit (6)
    got = lazy.commit_frames(x, tol=0.05, min_commit=6)
    assert got.tolist() == [10, 6, K]             # episode 2: they agree throughout
    assert lazy.commit_frames(x[:1], 0.05, 6).tolist() == [K] * B


def test_log() -> None:
    log = lazy.Log()
    log.add(torch.tensor([lazy.FRESH | lazy.LOW, 0, lazy.GOAL, 0]))
    log.add(torch.tensor([0, lazy.GOAL, 0, 0]))
    s = log.summary(seconds=1.0, worlds=4)
    assert log.replans == 3 and s["replans_per_s"] == 0.75
    assert s["share_goal"] == round(2 / 3, 3) and s["share_low"] == round(1 / 3, 3) and s["share_drift"] == 0.0
