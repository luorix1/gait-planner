"""Maximal joint torque of the training model, for comparison with human dynamometry (2026-11-22 strength question).

For each test (a coordinate swept over its range with the neighbouring joints in a dynamometer posture), at every
angle: every muscle whose moment arm about that coordinate is at least 1 mm in the test direction at full activation,
every other muscle at its minimum activation (so antagonists add only passive force, as in a maximal effort), the
muscles equilibrated (tendon compliance included), and the net muscle moment sum F_tendon x r about the coordinate.
Isometric only: setting the coordinate's speed before OpenSim's equilibrateMuscles left the forces unchanged (its
equilibrium ignores the MTU speed), so concentric capacity is left to jump_test.py.

Right leg; the model's own conventions: hip_flexion + = flexion, knee_angle + = flexion (Rajagopal), ankle_angle + =
dorsiflexion.

Run (OpenSim 4.6, bolt env): ~/miniconda3/envs/bolt/bin/python strength_check.py [--model PATH] [--out runs/strength_check]
Out: <out>/strength.json (per test and direction: angles, isometric and concentric torques, the peak and its angle,
N m and N m/kg), <out>/strength.png
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import opensim as osim

MODEL = Path.home() / "Jinwoo/neural_test/models/Rajagopal_lower_collision_v3.osim"
SPEEDS = (0.0,)                                 # deg/s; see the docstring
# (label, coordinate, sweep deg, posture deg, (name of + direction, name of - direction))
TESTS = (
    ("hip, knee 0", "hip_flexion_r", np.arange(-20, 111, 10), {"knee_angle_r": 0}, ("flexion", "extension")),
    ("hip, knee 90", "hip_flexion_r", np.arange(-20, 111, 10), {"knee_angle_r": 90}, ("flexion", "extension")),
    ("knee, seated (hip 90)", "knee_angle_r", np.arange(0, 121, 10), {"hip_flexion_r": 90}, ("flexion", "extension")),
    ("knee, supine (hip 0)", "knee_angle_r", np.arange(0, 121, 10), {"hip_flexion_r": 0}, ("flexion", "extension")),
    ("ankle, knee 0", "ankle_angle_r", np.arange(-40, 26, 5), {"knee_angle_r": 0}, ("dorsiflexion", "plantarflexion")),
    ("ankle, knee 90", "ankle_angle_r", np.arange(-40, 26, 5), {"knee_angle_r": 90, "hip_flexion_r": 90},
     ("dorsiflexion", "plantarflexion")),
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(MODEL))
    ap.add_argument("--out", default="runs/strength_check")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    model = osim.Model(a.model)
    s = model.initSystem()
    mass = model.getTotalMass(s)
    cs, ms = model.getCoordinateSet(), model.getMuscles()
    muscles = [ms.get(i) for i in range(ms.getSize())]
    amin = [osim.Millard2012EquilibriumMuscle.safeDownCast(m).getMinimumActivation() for m in muscles]
    res: dict = dict(model=a.model, mass_kg=round(mass, 2), speeds_dps=list(SPEEDS), tests=[])

    def pose(coord: str, deg: float, posture: dict, speed_dps: float) -> None:
        for i in range(cs.getSize()):
            c = cs.get(i)
            if not c.get_locked():
                c.setValue(s, 0.0, False)
            c.setSpeedValue(s, 0.0)
        for name, v in {**posture, coord: deg}.items():
            cs.get(name).setValue(s, math.radians(v), False)
        model.assemble(s)                                         # patella and knee couplings
        cs.get(coord).setSpeedValue(s, math.radians(speed_dps))
        model.realizeVelocity(s)

    for label, coord, sweep, posture, dirs in TESTS:
        c = cs.get(coord)
        for sign, dname in ((1.0, dirs[0]), (-1.0, dirs[1])):
            curves = {}
            for v in SPEEDS:
                torque = []
                for deg in sweep:
                    pose(coord, float(deg), posture, sign * v)
                    r = np.array([m.computeMomentArm(s, c) for m in muscles])
                    for m, ri, lo in zip(muscles, r, amin, strict=True):
                        m.setActivation(s, 1.0 if sign * ri >= 0.001 else lo)
                    model.equilibrateMuscles(s)
                    model.realizeDynamics(s)
                    torque.append(float(sum(m.getTendonForce(s) * ri for m, ri in zip(muscles, r, strict=True))))
                curves[str(int(v))] = [round(x, 1) for x in torque]
            iso = np.array(curves["0"]) * sign
            k = int(np.argmax(iso))
            res["tests"].append(dict(test=label, coordinate=coord, direction=dname, posture=posture,
                                     angles_deg=[int(x) for x in sweep], torque_nm=curves,
                                     peak_nm=round(float(iso[k]), 1), peak_nm_per_kg=round(float(iso[k] / mass), 2),
                                     peak_angle_deg=int(sweep[k])))
            print(f"{label:<24} {dname:<15} peak {iso[k]:6.1f} N m ({iso[k] / mass:.2f} N m/kg) at {sweep[k]} deg",
                  flush=True)
    (out / "strength.json").write_text(json.dumps(res, indent=1))
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        for ax, (label, *_rest) in zip(axes.flat, TESTS, strict=True):
            for t in res["tests"]:
                if t["test"] != label:
                    continue
                for v, ls in zip(SPEEDS, ("-", "--", "-.", ":")[:len(SPEEDS)], strict=True):
                    ax.plot(t["angles_deg"], np.abs(t["torque_nm"][str(int(v))]), ls,
                            label=f"{t['direction']} {int(v)} deg/s")
            ax.set_title(label)
            ax.set_xlabel(f"{TESTS[[x[0] for x in TESTS].index(label)][1]} (deg)")
            ax.set_ylabel("N m")
            ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(out / "strength.png", dpi=110)
    except ImportError:
        pass
    print(f"-> {out}/strength.json")


if __name__ == "__main__":
    main()
