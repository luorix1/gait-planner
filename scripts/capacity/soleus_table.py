"""Soleus muscle-tendon length over (ankle, subtalar) for each side, from OpenSim (bolt env)."""
import math, sys
import numpy as np
import opensim as osim
model = osim.Model(sys.argv[1]); s = model.initSystem()
cs, ms = model.getCoordinateSet(), model.getMuscles()
A = np.arange(-50.0, 40.01, 1.0); S = np.arange(-30.0, 30.01, 2.0)
out = dict(ankle_deg=A, subtalar_deg=S)
for side in "rl":
    m = osim.Millard2012EquilibriumMuscle.safeDownCast(ms.get(f"soleus_{side}"))
    L = np.zeros((len(A), len(S)))
    for i, a in enumerate(A):
        for j, b in enumerate(S):
            cs.get(f"ankle_angle_{side}").setValue(s, math.radians(a), False)
            cs.get(f"subtalar_angle_{side}").setValue(s, math.radians(b), False)
            model.assemble(s); model.realizePosition(s)
            L[i, j] = m.getLength(s)
    out[f"L_{side}"] = L
    out[f"lopt_{side}"], out[f"lts_{side}"] = m.getOptimalFiberLength(), m.getTendonSlackLength()
    out[f"alpha_{side}"], out[f"vmax_{side}"] = m.getPennationAngleAtOptimalFiberLength(), m.getMaxContractionVelocity()
    print(side, "length range", L.min(), L.max())
np.savez(sys.argv[2], **out)
