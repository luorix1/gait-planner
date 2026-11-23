"""The skill layer above the planner: jumps that need preparation, a run-up before a gap or a crouch before a platform.

The planner (gp.pose_planner) sees 3 m of ground and 1.2 s ahead, and its goal (where the pelvis is 1.2 s on) sets its
speed, so it cannot decide on its own to speed up 3 s before a gap, and a run-up half built is dropped at the next
replan. Here the run-up is planned once, when the feature comes into view, and the goals carry it out:

1. **Scan** the ground along the route ``view_m`` ahead (on a robot, its perception range; here the course profile)
   for the first thing an ordinary step cannot take: a gap wider than ``step_gap_m`` (ground drops away and comes
   back), or a rise taller than ``step_rise_m`` (a platform). Stairs, ramps and small gaps pass.
2. **Solve** the takeoff from point-mass ballistics (MotionBricks' smart primitives and Kimodo's keyframes play this
   role there: a target the generator in-betweens to):
   - gap of width ``w``: the centre of mass flies ``w + takeoff_back_m + landing_past_m - stance_m`` (the stances
     before and after cover the rest), at takeoff angle ``angle_deg``: horizontal takeoff speed
     ``sqrt(g d / (2 tan angle))``; 0.6 m -> 2.1 m/s, 1.0 m -> 3.0, 1.4 m -> 3.7 (walking: 1.2);
   - platform of height ``h``: vertical takeoff speed ``sqrt(2 g (h + apex_clear_m))``, takeoff ``0.4 + 0.5 h`` m
     before the riser, horizontal speed to land ``land_on_m`` past its edge on the way down;
     0.2 / 0.4 / 0.6 m -> ~2 m/s and a 15 / 25 / 35 cm crouch;
   - crouch depth from the vertical speed: ``vz^2 / (2 push_g g)`` (a countermovement pushing at ``push_g``);
   - out of reach (faster than ``max_speed``, deeper than ``crouch_max_m``, or too little room left to build the
     speed): **stop** ``stop_back_m`` before the edge instead.
3. **Schedule** speed along the route: accelerate as late as possible at ``accel`` (or brake at ``decel``) to reach
   the takeoff speed at the takeoff point, hold it through the flight, return to the cruise speed after landing, and
   stop before the route's end. The planner's goal is where this schedule puts the body ``horizon_s`` on; with a
   ``goal_h`` planner, the landing point at its own time once the landing is within ``horizon_s``.

Re-solved only on events (the same idea as gp.lazy, one level up), bits of ``why``: ``FEATURE`` (the first feature
ahead appeared, went, or moved more than ``feature_tol_m``: the first look was wrong), ``OFF`` (the body is
``speed_tol`` off the schedule before takeoff: re-solved from the speed it really has, which may turn a jump into a
stop), ``PASSED`` (landed: back to cruising).

The planner has no crouch input, so the crouch depth and the takeoff speed are targets to check the rollout against
(export_jumps.py), and the keyframes a keyframe-conditioned planner would take (Kimodo, MotionBricks).

Numbers here are first guesses; check them against SEED's jumps before trusting them. Pure numpy (tests/test_takeoff.py).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

G_ACC = 9.81
FEATURE, OFF, PASSED = 1, 2, 4
REASONS = {"feature": FEATURE, "off": OFF, "passed": PASSED}


@dataclass(frozen=True)
class JumpSpec:
    view_m: float = 8.0
    ds: float = 0.05
    step_gap_m: float = 0.35       # a gap an ordinary step clears (gp.terrain_gen.CourseSpec's gentle gaps end here)
    step_rise_m: float = 0.20      # a rise a stair step takes (stair rises go to 0.19 m)
    hole_m: float = 0.25           # ground this far below the walker's counts as a hole
    drop_ok_m: float = 0.7         # a drop to level ground this deep is walked or hopped down, deeper is an edge
    takeoff_back_m: float = 0.10
    landing_past_m: float = 0.25
    stance_m: float = 0.6
    angle_deg: float = 22.0
    apex_clear_m: float = 0.10
    land_on_m: float = 0.30
    push_g: float = 2.0
    crouch_max_m: float = 0.40
    accel: float = 1.5             # m/s^2
    decel: float = 2.0
    max_speed: float = 4.0         # the fastest run-up the planner's data holds; past it, stop
    stop_back_m: float = 0.4
    end_pad_m: float = 1.0
    feature_tol_m: float = 0.2
    speed_tol: float = 0.3
    speed_window_s: float = 0.2    # the measured speed: route travel over this window
    horizon_s: float = 1.2
    min_horizon_s: float = 0.4     # gp.pose_planner.GOAL_H0 at 20 Hz


@dataclass
class Feature:
    kind: str                      # "gap" | "up" | "edge" (ground drops away and does not come back in view)
    s_near: float
    s_far: float
    size: float                    # gap width or rise (m)


@dataclass
class Skill:
    kind: str                      # "cruise" | "gap" | "up" | "stop"
    v_cruise: float
    s_end: float                   # stop here (the route's end, or before an edge out of reach)
    s_takeoff: float = math.inf
    s_land: float = math.inf
    v_takeoff: float = 0.0
    t_flight: float = 0.0
    crouch_m: float = 0.0
    s_runup: float = math.inf      # where the speed starts to change for the takeoff
    feature: Feature | None = None
    note: str = ""
    accel: float = JumpSpec.accel
    decel: float = JumpSpec.decel

    # --- the speed schedule ---------------------------------------------------------------------------------------
    def v_at(self, s: float) -> float:
        """Planned horizontal speed at route position ``s``."""
        v = self.v_cruise
        if self.kind in ("gap", "up"):
            vt = self.v_takeoff
            if s < self.s_takeoff:
                d = self.s_takeoff - s
                v = (max(self.v_cruise, math.sqrt(max(vt ** 2 - 2 * self.accel * d, 0.0))) if vt >= self.v_cruise
                     else min(self.v_cruise, math.sqrt(vt ** 2 + 2 * self.decel * d)))
            elif s < self.s_land:
                v = vt
            else:                                                  # back to cruising after the landing
                d = s - self.s_land
                v = (min(self.v_cruise, math.sqrt(vt ** 2 + 2 * self.accel * d)) if vt < self.v_cruise
                     else max(self.v_cruise, math.sqrt(max(vt ** 2 - 2 * self.decel * d, 0.0))))
        return min(v, math.sqrt(2 * self.decel * max(self.s_end - s, 0.0)))

    def advance(self, s: float, dt: float, step: float = 0.05) -> float:
        """Route position ``dt`` s after ``s`` on the schedule."""
        n = max(1, int(math.ceil(dt / step)))
        for _ in range(n):
            s = min(s + self.v_at(s) * dt / n, max(s, self.s_end))      # never stepped past a stop
        return s

    def time_to(self, s0: float, s1: float, step: float = 0.05, cap: float = 10.0) -> float:
        """Seconds the schedule takes from ``s0`` to ``s1`` (``cap`` if it never gets there)."""
        t, s = 0.0, s0
        while s < s1 and t < cap:
            s += self.v_at(s) * step
            t += step
        return t


def scan(s: np.ndarray, h: np.ndarray, J: JumpSpec = JumpSpec(), ref: float | None = None) -> Feature | None:
    """The first feature on a route profile (positions ``s`` from the walker on, ground heights ``h``) that an
    ordinary step cannot take. ``ref``: the height of the ground the walker stands on (default ``h[0]``; not the
    bottom of a gap it is stepping over)."""
    if len(h) < 2:
        return None
    ref = float(h[0]) if ref is None else ref
    low = h < ref - J.hole_m
    i = 0
    while i < len(h):
        if low[i]:
            j = i
            while j < len(h) and low[j]:
                j += 1
            if j < len(h):                                         # the ground comes back: a gap
                if s[j] - s[i] > J.step_gap_m:
                    return Feature("gap", float(s[i]), float(s[j]), float(s[j] - s[i]))
                i = j
                continue
            if ref - float(np.median(h[i:])) > J.drop_ok_m:       # it does not, in view: an edge
                return Feature("edge", float(s[i]), math.inf, math.inf)
            ref = h[i]                                             # a step down to new ground: walk on from there
            low = h < ref - J.hole_m
        if i and not low[i - 1] and h[i] - h[i - 1] > J.step_rise_m:     # (not climbing out of a gap just passed)
            return Feature("up", float(s[i]), float(s[i]), float(h[i] - h[i - 1]))
        i += 1
    return None


def solve(f: Feature | None, s_now: float, v_now: float, v_cruise: float, s_route_end: float,
          J: JumpSpec = JumpSpec()) -> Skill:
    """The skill for the first feature ahead, from where the walker is and how fast it goes."""
    s_end = s_route_end - J.end_pad_m
    if f is None:
        return Skill("cruise", v_cruise, s_end, accel=J.accel, decel=J.decel)
    stop = Skill("stop", v_cruise, min(s_end, f.s_near - J.stop_back_m), feature=f, accel=J.accel, decel=J.decel)
    if f.kind == "edge":
        stop.note = "ground drops away out of view"
        return stop
    if f.kind == "gap":
        d = max(f.size + J.takeoff_back_m + J.landing_past_m - J.stance_m, 0.1)
        vt = max(math.sqrt(G_ACC * d / (2 * math.tan(math.radians(J.angle_deg)))), 0.0)
        vt = max(vt, min(v_cruise, J.max_speed))                   # never slow down for a gap
        vz = vt * math.tan(math.radians(J.angle_deg))
        s_take, s_land, tf = f.s_near - J.takeoff_back_m, f.s_far + J.landing_past_m, 2 * vz / G_ACC
    else:
        vz = math.sqrt(2 * G_ACC * (f.size + J.apex_clear_m))
        back = 0.4 + 0.5 * f.size
        tf = (vz + math.sqrt(max(vz ** 2 - 2 * G_ACC * f.size, 0.0))) / G_ACC   # up to its height, on the way down
        vt = (back + J.land_on_m) / tf
        s_take, s_land = f.s_near - back, f.s_near + J.land_on_m
    crouch = vz ** 2 / (2 * J.push_g * G_ACC)
    if vt > J.max_speed:
        stop.note = f"needs {vt:.1f} m/s, past max_speed"
        return stop
    if crouch > J.crouch_max_m:
        stop.note = f"needs a {100 * crouch:.0f} cm crouch, past crouch_max_m"
        return stop
    room = s_take - s_now
    need = (vt ** 2 - v_now ** 2) / (2 * J.accel) if vt > v_now else (v_now ** 2 - vt ** 2) / (2 * J.decel)
    if room < need - 1e-9:                         # too close to build (or shed) the speed: refuse, as a person would
        stop.note = f"{room:.1f} m left to go from {v_now:.1f} to {vt:.1f} m/s (needs {need:.1f})"
        return stop
    ramp = abs(vt ** 2 - v_cruise ** 2) / (2 * (J.accel if vt > v_cruise else J.decel))
    return Skill(f.kind, v_cruise, s_end, s_takeoff=s_take, s_land=s_land, v_takeoff=vt, t_flight=tf, crouch_m=crouch,
                 s_runup=s_take - ramp, feature=f, accel=J.accel, decel=J.decel)


class TakeoffGoals:
    """Goals for a rollout (export_pose2.rollout / lazy_rollout ``goal_of``) from the skill layer, per episode on its
    own route: ``paths[b]`` with ``s``, ``x``, ``z`` and the ground profile ``h`` along it (gp.terrain_gen.centreline,
    as course tiles carry), cruising at ``speeds[b]``. ``goal_h``: the planner takes a goal horizon (``[B, 3]`` goals).
    ``self.log``: re-solves by reason; ``self.skills``: the current skill per episode."""

    def __init__(self, paths: list[dict[str, np.ndarray]], speeds: list[float], s0: list[float], hz: float,
                 goal_h: bool = False, J: JumpSpec = JumpSpec()) -> None:
        self.paths, self.speeds, self.hz, self.goal_h, self.J = paths, list(speeds), hz, goal_h, J
        self.s = np.asarray(s0, float)
        self.k = max(1, int(round(J.speed_window_s * hz)))
        self.past: list[list[float]] = [[] for _ in s0]            # route positions over the last speed window
        self.ground = [float(np.interp(v, p["s"], p["h"])) for v, p in zip(s0, paths, strict=True)]   # last support
        self.skills: list[Skill | None] = [None] * len(paths)
        self.log = dict.fromkeys(REASONS, 0)
        self.history: list[list[tuple[float, str, str]]] = [[] for _ in paths]   # (s, kind, note) per re-solve

    def _locate(self, b: int, xz: np.ndarray) -> float:
        p = self.paths[b]
        """Route position of ``xz``: the nearest route point near the last one, refined onto the segment after it
        (route points are 5 cm apart, too coarse for a speed over 0.2 s)."""
        win = np.flatnonzero((p["s"] >= self.s[b] - 0.5) & (p["s"] <= self.s[b] + 2.0))
        i = int(win[np.argmin((p["x"][win] - xz[0]) ** 2 + (p["z"][win] - xz[1]) ** 2)])
        i = min(i, len(p["s"]) - 2)
        d = np.array([p["x"][i + 1] - p["x"][i], p["z"][i + 1] - p["z"][i]])
        u = float(np.dot(xz - [p["x"][i], p["z"][i]], d) / max(float(d @ d), 1e-12))
        return float(p["s"][i] + np.clip(u, -1.0, 1.0) * (p["s"][i + 1] - p["s"][i]))

    def world_goals(self, o: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Goal xz ``[B, 2]`` (world) and horizon (s) per episode, for pelvis positions ``o`` ``[B, 2]``; call once
        per plan frame."""
        J, dt = self.J, 1.0 / self.hz
        goals, hor = np.zeros((len(self.paths), 2)), np.full(len(self.paths), J.horizon_s)
        for b, p in enumerate(self.paths):
            self.s[b] = s_now = self._locate(b, o[b])
            past = self.past[b] = (self.past[b] + [s_now])[-(self.k + 1):]
            v_meas = (past[-1] - past[0]) / (dt * (len(past) - 1)) if len(past) > 1 else self.speeds[b]
            s_mid = 0.5 * (past[0] + past[-1])          # a mean speed over the window belongs to its middle
            h_here = float(np.interp(s_now, p["s"], p["h"]))
            if h_here >= self.ground[b] - J.hole_m:        # on ground, not over a hole: it is what the walker stands on
                self.ground[b] = h_here
            ahead = (p["s"] >= s_now) & (p["s"] <= s_now + J.view_m)
            f = scan(p["s"][ahead], p["h"][ahead], J, ref=self.ground[b])
            sk, why = self.skills[b], 0
            airborne = sk is not None and sk.kind in ("gap", "up") and sk.s_takeoff <= s_now < sk.s_land
            if airborne:
                pass                                   # committed: no change of mind between takeoff and landing
            elif sk is None:
                why = FEATURE
            else:
                old = sk.feature
                if (f is None) != (old is None) or (f and old and (f.kind != old.kind
                                                                     or abs(f.s_near - old.s_near) > J.feature_tol_m)):
                    why |= PASSED if old and s_now >= sk.s_land else FEATURE
                if (sk.kind in ("gap", "up") and s_now < sk.s_takeoff and len(past) > self.k
                        and abs(v_meas - sk.v_at(s_mid)) > J.speed_tol):
                    why |= OFF
            if why:
                v0 = self.speeds[b] if sk is None else float(v_meas)
                sk = self.skills[b] = solve(f, s_now, v0, self.speeds[b], float(p["s"][-1]), J)
                for k, bit in REASONS.items():
                    self.log[k] += bool(why & bit)
                self.history[b].append((round(s_now, 2), sk.kind, sk.note))
            s_goal = sk.advance(s_now, J.horizon_s)
            if self.goal_h and sk.kind in ("gap", "up") and s_now < sk.s_land:
                t_land = sk.time_to(s_now, sk.s_land)
                if J.min_horizon_s <= t_land <= J.horizon_s:     # the landing point, at its own time
                    s_goal, hor[b] = sk.s_land, t_land
            goals[b] = np.interp(s_goal, p["s"], p["x"]), np.interp(s_goal, p["s"], p["z"])
        return goals, hor

    def __call__(self, f: int, o, psi):
        """export_pose2's ``goal_of``: anchor-frame goals ``[B, 2]`` (``[B, 3]`` with the horizon share for a
        ``goal_h`` planner)."""
        import torch

        from .planner import to_frame
        g, hor = self.world_goals(o.detach().cpu().numpy())
        gt = to_frame(torch.as_tensor(g, dtype=torch.float32, device=o.device)[:, None], o, psi)[:, 0]
        if self.goal_h:
            gt = torch.cat([gt, torch.as_tensor(hor / self.J.horizon_s, dtype=torch.float32, device=o.device)[:, None]], 1)
        return gt
