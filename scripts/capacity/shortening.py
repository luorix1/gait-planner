"""Muscle shortening speeds the hard clips' reference kinematics demand (OpenSim 4.6, bolt env)."""
import math
import sys
import numpy as np
import opensim as osim

Q = np.load(sys.argv[1])
names = [str(n) for n in Q["coord_names"]]
hz = float(Q["hz"])
model = osim.Model(sys.argv[2])
s = model.initSystem()
cs, ms = model.getCoordinateSet(), model.getMuscles()
MUS = ["gasmed_r", "gaslat_r", "soleus_r", "vasmed_r", "vaslat_r", "recfem_r", "glmax2_r", "bflh_r",
       "gasmed_l", "soleus_l", "vaslat_l", "glmax2_l"]
mus = {m: osim.Millard2012EquilibriumMuscle.safeDownCast(ms.get(m)) for m in MUS}
prop = {}
for m, mm in mus.items():
    tfl = mm.getTendonForceLengthCurve()
    prop[m] = dict(lopt=mm.getOptimalFiberLength(), lts=mm.getTendonSlackLength(),
                   alpha=mm.getPennationAngleAtOptimalFiberLength(), vmax=mm.getMaxContractionVelocity(),
                   strain=tfl.getStrainAtOneNormForce(), rigid=mm.get_ignore_tendon_compliance())
clips = [k for k in Q.files if k not in ("coord_names", "hz")]
print(f"{'muscle':<9} {'Lopt cm':>7} {'Lts cm':>6} {'strain@F0':>9} {'vmax Lopt/s':>11} rigid")
for m, p in prop.items():
    print(f"{m:<9} {100*p['lopt']:7.1f} {100*p['lts']:6.1f} {p['strain']:9.3f} {p['vmax']:11.1f} {p['rigid']}")
for clip in clips:
    q = Q[clip]
    L = np.zeros((len(q), len(MUS)))
    for t in range(len(q)):
        for j, n in enumerate(names):
            c = cs.get(n)
            if not c.get_locked():
                c.setValue(s, float(q[t, j]), False)
        model.assemble(s)
        model.realizePosition(s)
        L[t] = [mus[m].getLength(s) for m in MUS]
    v = np.gradient(L, 1.0 / hz, axis=0)                      # MTU speed, m/s (+ = lengthening)
    print(f"\n== {clip} ({len(q) / hz:.1f} s)")
    for i, m in enumerate(MUS):
        p = prop[m]
        w = p["lopt"] * math.sin(p["alpha"])
        lmc = np.maximum(L[:, i] - p["lts"], 1e-4)
        lf = np.sqrt(w * w + lmc * lmc)
        vf = lmc / lf * v[:, i] / p["lopt"]                    # rigid-tendon fiber speed, Lopt/s
        k = int(np.argmin(vf))
        # what tendon recoil could cover: full stretch at max force released over the 0.15 s before the peak
        recoil = p["strain"] * p["lts"] / 0.15 / p["lopt"]
        print(f"  {m:<9} peak MTU shortening {-v[k, i]:5.2f} m/s at {k / hz:4.2f} s -> rigid-tendon fibre {-vf[k]:5.1f} Lopt/s "
              f"({-vf[k] / p['vmax']:4.0%} of vmax); tendon recoil could cover <= {recoil:4.1f} Lopt/s")
