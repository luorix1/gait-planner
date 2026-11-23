"""gp.takeoff on synthetic straight routes (numpy only): what the scan calls a feature, the ballistic takeoff numbers,
the speed schedule reaching the takeoff speed at the takeoff point, refusing jumps out of reach, and the goal source
re-solving only on events (a walker that follows the run-up, and one that ignores it)."""

import math

import numpy as np

from gp import takeoff as tk

DS = 0.05


def route(*parts: tuple[float, float]) -> dict[str, np.ndarray]:
    """A straight route along +x from (length, height) pieces."""
    s = np.arange(0.0, sum(L for L, _ in parts), DS)
    edges = np.cumsum([0.0] + [L for L, _ in parts])
    h = np.array([float(parts[min(np.searchsorted(edges, v, side="right") - 1, len(parts) - 1)][1]) for v in s])
    return dict(s=s, x=s.copy(), z=np.zeros_like(s), h=h)


def first(r: dict[str, np.ndarray], s0: float = 0.0) -> tk.Feature | None:
    m = r["s"] >= s0
    return tk.scan(r["s"][m], r["h"][m])


def test_scan() -> None:
    assert first(route((10, 0))) is None
    stairs = [(3, 0)] + [(0.3, 0.17 * k) for k in range(1, 7)] + [(3, 1.02)]
    assert first(route(*stairs)) is None                                   # stair rises pass
    assert first(route((3, 0), (0.3, -1), (3, 0))) is None                 # a gap a step clears
    f = first(route((3, 0), (1.0, -1), (3, 0)))
    assert f.kind == "gap" and abs(f.s_near - 3.0) < DS and abs(f.size - 1.0) < 2 * DS
    f = first(route((3, 0), (2, 0.4), (3, 0.4)))
    assert f.kind == "up" and abs(f.size - 0.4) < 1e-6 and abs(f.s_near - 3.0) < DS
    assert first(route((3, 0), (5, -0.4))) is None                         # a step down to new ground
    assert first(route((3, 0), (5, -1.5))).kind == "edge"                  # a drop that does not come back


def test_solve_numbers() -> None:
    far = 0.0
    v = {w: tk.solve(tk.Feature("gap", 6.0, 6.0 + w, w), far, 1.2, 1.2, 20.0).v_takeoff for w in (0.6, 1.0, 1.4)}
    assert 1.8 < v[0.6] < 2.3 and 2.8 < v[1.0] < 3.2 and 3.5 < v[1.4] < 4.0, v
    up = tk.solve(tk.Feature("up", 6.0, 6.0, 0.6), far, 1.2, 1.2, 20.0)
    assert up.kind == "up" and 0.3 < up.crouch_m < 0.4 and up.s_takeoff < 6.0 < up.s_land
    assert tk.solve(tk.Feature("up", 6.0, 6.0, 1.0), far, 1.2, 1.2, 20.0).kind == "stop"     # crouch past the max
    assert tk.solve(tk.Feature("gap", 6.0, 8.5, 2.5), far, 1.2, 1.2, 20.0).kind == "stop"    # faster than max_speed
    late = tk.solve(tk.Feature("gap", 6.0, 7.0, 1.0), 5.0, 1.2, 1.2, 20.0)                   # seen 1 m before the edge
    assert late.kind == "stop" and late.s_end <= 6.0 - 0.4 + 1e-9 and "left" in late.note
    assert tk.solve(None, 0.0, 1.2, 1.2, 20.0).kind == "cruise"


def test_schedule_hits_takeoff_speed() -> None:
    sk = tk.solve(tk.Feature("gap", 8.0, 9.0, 1.0), 0.0, 1.2, 1.2, 30.0)
    assert sk.s_runup > 0.0 and sk.v_at(0.0) == 1.2 and sk.v_at(sk.s_runup - 0.1) == 1.2
    assert abs(sk.v_at(sk.s_takeoff - 1e-6) - sk.v_takeoff) < 1e-3
    vs = [sk.v_at(s) for s in np.arange(sk.s_runup, sk.s_takeoff, 0.1)]
    assert all(b >= a - 1e-9 for a, b in zip(vs, vs[1:], strict=False))                  # speeds up, never down, in the run-up
    assert sk.v_at(sk.s_land + 5) == 1.2                                  # back to cruising
    assert sk.v_at(sk.s_end) == 0.0 and sk.advance(sk.s_end - 0.01, 5.0) <= sk.s_end + 1e-6   # stops at the end
    stop = tk.solve(tk.Feature("gap", 6.0, 8.5, 2.5), 0.0, 1.2, 1.2, 20.0)
    assert stop.advance(0.0, 30.0) <= 6.0 - 0.4 + 1e-6                     # stops short of the edge


def _walk(r: dict[str, np.ndarray], follow: bool, goal_h: bool = False, seconds: float = 9.0, hz: float = 20.0):
    g = tk.TakeoffGoals([r], [1.2], [1.0], hz, goal_h=goal_h)
    s, hors = 1.0, []
    for _ in range(int(seconds * hz)):
        goals, hor = g.world_goals(np.array([[s, 0.0]]))
        hors.append(float(hor[0]))
        sk = g.skills[0]
        s += (sk.v_at(s) if follow else min(1.2, sk.v_at(s))) / hz        # ignoring it: never faster than cruise
    return g, s, hors


def test_goals_follow_the_run_up() -> None:
    r = route((8, 0), (1.0, -1), (8, 0))
    g, s, hors = _walk(r, follow=True, goal_h=True)
    assert [k for _, k, _ in g.history[0]] == ["gap", "cruise"], g.history  # solved once, then passed
    assert g.log["feature"] == 1 and g.log["passed"] == 1 and g.log["off"] == 0
    assert s > 9.5                                                         # across
    assert min(hors) < 1.2 and min(hors) >= 0.4                            # the landing goal at its own time


def test_goals_refuse_when_the_body_does_not_speed_up() -> None:
    r = route((8, 0), (1.0, -1), (8, 0))
    g, s, _ = _walk(r, follow=False)
    kinds = [k for _, k, _ in g.history[0]]
    assert kinds[0] == "gap" and kinds[-1] == "stop" and g.log["off"] >= 1, g.history
    assert s < 8.0 - 0.4 + 0.05                                            # stopped before the edge
    assert not math.isinf(g.skills[0].s_end)


def test_stepping_over_a_small_gap_is_not_a_feature() -> None:
    g, s, _ = _walk(route((6, 0), (0.3, -1), (6, 0)), follow=True)
    assert [k for _, k, _ in g.history[0]] == ["cruise"] and g.log["off"] == 0, g.history
    assert s > 7.0
