"""Peak soleus fibre-speed demand (rigid tendon, % of vmax) of every exported seed-library clip."""
import csv, json, re, sys, urllib.request
import numpy as np
from scipy.interpolate import RegularGridInterpolator
sys.path.insert(0, ".")
from gp import data

T = np.load(sys.argv[1]); out_csv = sys.argv[2]
rows = json.load(urllib.request.urlopen("http://127.0.0.1:8765/api/index", timeout=300))
cols = rows["cols"]; ix = {c: i for i, c in enumerate(cols)}
interp = {s: RegularGridInterpolator((T["ankle_deg"], T["subtalar_deg"]), T[f"L_{s}"], bounds_error=False, fill_value=None) for s in "rl"}
res = []
for r in rows["rows"]:
    if not r[ix["exported"]]:
        continue
    d = np.load(data.SEED / r[ix["file"]])
    cn = [str(c) for c in d["coord_names"]]
    cols4 = [cn.index(c) for c in ("ankle_angle_r", "subtalar_angle_r", "ankle_angle_l", "subtalar_angle_l")]
    q = data.resample(d["q"][:, cols4].astype(np.float64))[0]           # low-pass, 40 Hz grid, as gp.data.build
    peak = {}
    for k, s in enumerate("rl"):
        L = interp[s](np.degrees(q[:, 2 * k:2 * k + 2]))
        v = np.gradient(L, 1.0 / data.HZ)
        lopt, lts, al, vmax = (float(T[f"{n}_{s}"]) for n in ("lopt", "lts", "alpha", "vmax"))
        lmc = np.maximum(L - lts, 1e-4); lf = np.sqrt((lopt * np.sin(al)) ** 2 + lmc ** 2)
        vf = lmc / lf * v / lopt
        peak[s] = float(-vf.min() / vmax) if len(vf) else 0.0
    name = r[ix["name"]]
    m = re.match(r"seed_(.+?)_\d{3}(__|_R_)", name) or re.match(r"(parc_[a-z]+_[a-z]+)", name) or re.match(r"([a-z0-9]+)_", name)
    res.append(dict(name=name, source=r[ix["source"]], mode=r[ix["mode"]], terrain=r[ix["terrain_class"]],
                    family=m.group(1) if m else name, not_used=r[ix["not_used"]] or "", soleus_r=round(peak["r"], 3),
                    soleus_l=round(peak["l"], 3), soleus=round(max(peak.values()), 3), seconds=r[ix["seconds"]]))
with open(out_csv, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(res[0])); w.writeheader(); w.writerows(res)
print(len(res), "clips ->", out_csv)
