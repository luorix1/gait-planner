"""Local debug viewer for the combined seed library (the Rajagopal v3 export: AB01, PARC, Camargo, Bones SEED).

Pick a clip and see the body as SPRINT poses it (forward kinematics of the training model, its foot contact spheres
coloured by the clip's contact labels) on the clip's own terrain. Beside it are the manifest's QA numbers and three
traces:
- one coordinate against its range;
- the lowest foot sphere's height above the terrain (below zero = inside the ground);
- the fastest joint (rad/s, raw 30 Hz differences; gp.data.build drops clips above 20 rad/s after its low-pass).

Every clip carries why it is not used, if it is not: the dataset build's own drops (gp.data.build: rejected by the
library's gates, uses its hands, excluded by name for the active dataset (gp.data.excluded: EXCLUDE, and PARC's
physics-tracker replays from v3 on), a joint faster than MAX_JOINT_SPEED after the build's low-pass), plus the viewer's
flags. GP_DATASET picks the dataset whose rules apply. The page can hide them all. Which sources a dataset takes (gp.data.DATASETS) is
left to the source checkboxes. The joint-speed check reads every clip once; results are cached in
seed_library/data/viewer_cache.json by file modification time.

Clips can be flagged; flags go to seed_library/data/viewer_flags.json. SEED is confidential: the server binds to
127.0.0.1 only, reads the library in place and copies nothing.

Run:  cd ~/Jinwoo/gait_planner && [GP_DATASET=v3] .venv/bin/python seed_viewer/server.py [--port 8765]
      then open http://127.0.0.1:8765 (VS Code forwards the port from a remote machine)
"""

from __future__ import annotations

import argparse
import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gp import data  # noqa: E402

HERE = Path(__file__).resolve().parent
LIB = data.SEED                                            # seed_library/data/osim_export
FLAGS = LIB.parent / "viewer_flags.json"
CACHE = LIB.parent / "viewer_cache.json"
THREE = Path.home() / "Jinwoo/GaitStudio/sprint_studio/web/vendor/three"
SEGMENTS = [("pelvis", "femur_l"), ("pelvis", "femur_r"), ("femur_l", "tibia_l"), ("femur_r", "tibia_r"),
            ("tibia_l", "talus_l"), ("tibia_r", "talus_r"), ("talus_l", "calcn_l"), ("talus_r", "calcn_r"),
            ("calcn_l", "toes_l"), ("calcn_r", "toes_r"), ("pelvis", "torso"), ("torso", "head")]
COLS = ["name", "source", "subject", "mode", "mode_src", "terrain_class", "seconds", "exported", "not_used", "file",
        "ik_cm", "skate_cms", "worst_sole_cm", "out_of_range_pct", "hip_excess_deg", "toe_stuck_pct", "speed_mps",
        "grade_deg", "uses_hands", "joint_speed"]


def _num(v, k=1.0, nd=2):
    return None if v is None or not np.isfinite(float(v)) else round(float(v) * k, nd)


def _finite(o):
    """JSON has no NaN: non-finite numbers in metadata become null."""
    if isinstance(o, float):
        return o if np.isfinite(o) else None
    if isinstance(o, dict):
        return {k: _finite(v) for k, v in o.items()}
    if isinstance(o, list):
        return [_finite(v) for v in o]
    return o


def joint_speeds(files: list[str], names: list[str]) -> dict[str, float | str]:
    """Each exported clip's fastest joint after the dataset build's resampling (gp.data.resample), cached."""
    cache = json.loads(CACHE.read_text()) if CACHE.exists() else {}
    rot = [k for k, c in enumerate(names) if not c.startswith("pelvis_t")]
    out, todo = {}, 0
    for n, f in enumerate(files):
        mt = (LIB / f).stat().st_mtime
        if f in cache and cache[f][0] == mt:
            out[f] = cache[f][1]
            continue
        try:
            d = np.load(LIB / f)
            cn = list(d["coord_names"])
            qvel = data.resample(d["q"][:, [cn.index(c) for c in names]].astype(np.float64))[1]
            out[f] = round(float(np.abs(qvel[:, rot]).max()), 2)
        except Exception as e:  # noqa: BLE001 - an unreadable clip is a reason not to use it
            out[f] = f"unreadable: {type(e).__name__}: {e}"[:120]
        cache[f] = [mt, out[f]]
        todo += 1
        if todo % 2000 == 0:
            print(f"joint speeds: {n + 1}/{len(files)}", flush=True)
    if todo:
        CACHE.write_text(json.dumps(cache))
    return out


def not_used(r: dict, speed: float | str | None) -> str:
    """Why gp.data.build would leave this clip out (empty if it would take it, its source permitting)."""
    if not r.get("exported"):
        return "rejected: " + "; ".join(r.get("rejected") or ["not exported"])
    if r.get("uses_hands"):
        return "uses its hands"
    hit = data.excluded(r["name"])
    if hit:
        return f"excluded: {hit}"
    if isinstance(speed, str):
        return speed
    if speed is not None and speed > data.MAX_JOINT_SPEED:
        return f"joint speed {speed:.1f} > {data.MAX_JOINT_SPEED:g} rad/s"
    return ""


def index(names: list[str]) -> list[list]:
    man = [r for f in sorted(LIB.glob("manifest_*.json")) for r in json.loads(f.read_text())]
    speed = joint_speeds([r["file"] for r in man if r.get("exported") and r.get("file")], names)
    rows = []
    for r in man:
        rows.append([r["name"], r.get("source"), r.get("subject"), r.get("mode"), r.get("mode_src"),
                     r.get("terrain_class"), _num(r.get("seconds")), bool(r.get("exported")),
                     not_used(r, speed.get(r.get("file"))), r.get("file"),
                     _num(r.get("ik_rms_m"), 100), _num(r.get("skate"), 100), _num(r.get("model_worst_contact_sole_m"), 100),
                     _num(r.get("coord_out_of_range_frac"), 100, 1), _num(r.get("hip_rotation_excess_deg"), 1, 1),
                     _num(r.get("toe_stuck_frac"), 100, 1), _num(r.get("speed_path_mps")), _num(r.get("grade_deg"), 1, 1),
                     bool(r.get("uses_hands")), sp if isinstance(sp := speed.get(r.get("file")), float) else None])
    return rows


class Clips:
    """Forward kinematics of exported clips with the training model (gp.data.load_model)."""

    def __init__(self) -> None:
        from sprint.recipes.motion import model_layout
        self.m = data.load_model()
        h, L = self.m.host, model_layout(self.m)
        self.names = list(self.m.coord_names)
        self.bodies = list(self.m.body_names)
        self.sph = np.asarray(L.contact_spheres)
        self.sph_body = np.asarray(h["geom_body"])[self.sph].astype(int)
        self.sph_pos = np.asarray(h["geom_pos"], np.float64)[self.sph]
        self.sph_r = np.asarray(h["geom_size"], np.float64)[self.sph, 0]
        self.trans = np.asarray(h["coord_is_trans"]).astype(bool)[:len(self.names)]
        lo, hi = np.asarray(h["coord_lo"], np.float64), np.asarray(h["coord_hi"], np.float64)
        self.range = [(None, None) if t else (float(np.degrees(a)), float(np.degrees(b)))
                      for t, a, b in zip(self.trans, lo, hi, strict=False)]

    def clip(self, file: str) -> dict:
        from sprint.recipes.motion import _rotate, forward_kinematics_np
        d = np.load(LIB / file, allow_pickle=True)
        cn = list(d["coord_names"])
        q = d["q"][:, [cn.index(c) for c in self.names]].astype(np.float64)
        bad = [n for n, ok in (("coordinates", np.isfinite(q).all()), ("terrain", np.isfinite(d["hf"]).all())) if not ok]
        if bad:
            raise ValueError(f"non-finite values in the clip's {' and '.join(bad)}")
        fps = float(d["fps"])
        xp, xq = (a.astype(np.float64) for a in forward_kinematics_np(self.m, q.astype(np.float32)))
        T = len(q)
        bi = {b: i for i, b in enumerate(self.bodies)}
        torso = bi["torso"]
        head = xp[:, torso] + _rotate(xq[:, torso], np.broadcast_to([0.0, 0.5, 0.0], (T, 3)))
        pts = np.concatenate([xp, head[:, None]], 1)                         # bodies + a head point
        ctr = xp[:, self.sph_body] + _rotate(xq[:, self.sph_body], np.broadcast_to(self.sph_pos, (T, len(self.sph), 3)))
        hf, hf_min, dx = np.asarray(d["hf"], np.float64), np.asarray(d["hf_min"], np.float64), float(d["dx"])
        # terrain under each sphere: canonical (x, y) = (osim x, -osim z), cell (i, j) at hf_min + (i, j) dx (nearest)
        ij = np.rint((np.stack([ctr[..., 0], -ctr[..., 2]], -1) - hf_min) / dx).astype(int)
        g = hf[np.clip(ij[..., 0], 0, hf.shape[0] - 1), np.clip(ij[..., 1], 0, hf.shape[1] - 1)]
        gap = (ctr[..., 1] - self.sph_r - g).min(1)
        rot = ~self.trans
        speed = np.r_[0.0, np.abs(np.diff(q[:, rot], axis=0)).max(1) * fps]
        coords = {n: (np.degrees(q[:, k]) if not self.trans[k] else q[:, k]).round(3).tolist() for k, n in enumerate(self.names)}
        side = np.array([self.bodies[b][-1] for b in self.sph_body])
        return dict(file=file, fps=fps, frames=T, meta=_finite(json.loads(str(d["meta"]))), terrain=_finite(json.loads(str(d["terrain"]))),
                    bodies=[*self.bodies, "head"], segments=[[bi.get(a, len(self.bodies)), bi.get(b, len(self.bodies))] for a, b in SEGMENTS],
                    pts=pts.round(4).tolist(), spheres=ctr.round(4).tolist(), radius=self.sph_r.round(4).tolist(),
                    sphere_side=side.tolist(), contact=np.asarray(d["contact"]).astype(int).tolist(),
                    hf=hf.round(4).tolist(), hf_min=hf_min.tolist(), dx=dx,
                    gap_cm=(gap * 100).round(2).tolist(), speed=speed.round(2).tolist(),
                    coords=coords, coord_range={n: self.range[k] for k, n in enumerate(self.names)})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    a = ap.parse_args()
    clips = Clips()
    rows = index(clips.names)
    files = {r[COLS.index("file")] for r in rows if r[COLS.index("exported")] and r[COLS.index("file")]}
    print(f"{len(rows)} clips ({len(files)} exported) from {LIB}", flush=True)

    class H(BaseHTTPRequestHandler):
        def _send(self, body: bytes, ctype: str, code: int = 200) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _json(self, obj, code: int = 200) -> None:
            self._send(json.dumps(obj, separators=(",", ":")).encode(), "application/json", code)

        def do_GET(self) -> None:
            u = urlparse(self.path)
            if u.path in ("/", "/index.html"):
                return self._send((HERE / "index.html").read_bytes(), "text/html; charset=utf-8")
            if u.path.startswith("/vendor/three/"):
                p = (THREE / u.path[len("/vendor/three/"):]).resolve()
                if THREE.resolve() in p.parents and p.is_file():
                    return self._send(p.read_bytes(), "text/javascript")
                return self._json({"error": "not found"}, 404)
            if u.path == "/api/index":
                flags = json.loads(FLAGS.read_text()) if FLAGS.exists() else {}
                return self._json(dict(cols=COLS, rows=rows, flags=flags))
            if u.path == "/api/clip":
                f = parse_qs(u.query).get("file", [""])[0]
                if f not in files:                         # only exported clips listed in a manifest: no path tricks
                    return self._json({"error": f"not an exported clip: {f}"}, 404)
                try:
                    return self._json(clips.clip(f))
                except Exception as e:  # noqa: BLE001 - a broken clip is something to see, not a crash
                    return self._json({"error": f"{type(e).__name__}: {e}"}, 500)
            return self._json({"error": "not found"}, 404)

        def do_POST(self) -> None:
            if urlparse(self.path).path != "/api/flag":
                return self._json({"error": "not found"}, 404)
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
            flags = json.loads(FLAGS.read_text()) if FLAGS.exists() else {}
            name, note = str(body.get("name", "")), str(body.get("note", ""))
            if body.get("flag"):
                flags[name] = note
            else:
                flags.pop(name, None)
            FLAGS.write_text(json.dumps(flags, indent=1, sort_keys=True))
            self._json(flags)

        def log_message(self, *args) -> None:
            pass

    print(f"open http://127.0.0.1:{a.port}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", a.port), H).serve_forever()


if __name__ == "__main__":
    main()
