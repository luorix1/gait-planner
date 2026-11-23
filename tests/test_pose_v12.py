"""v12 smoothness options (GPU): root steps round-trip; per-frame normalisation has per-frame stats; every option
combination trains forward/backward and samples finite frames; v11 checkpoints load and score unchanged; the seam
crossfade and smoothed hold run in a rollout, and the smoothed hold steps less at the seam than the plain one.

Run (bolt env): PYTHONPATH=. python tests/test_pose_v12.py
"""

import torch

from export_pose2 import NS, circmean, frame_features, hold_feet, start
from gp import data, footfk
from gp.pose_planner import FUT, HIST, PosePlanner, Windows, patch, root_from_vel, root_to_vel

V11 = data.ROOT / "runs" / "pose2_planner_v11" / "ckpt_030000.pt"


def main() -> None:
    dev = "cuda"
    torch.manual_seed(0)
    D = data.load(("test_trial", "test_terrain"))
    fk = footfk.fit(D, dev, data.ROOT / "data" / "foot_fk.pt")

    x = torch.randn(4, HIST + FUT, 45, device=dev)
    e = root_to_vel(x)
    back = root_from_vel(e[:, HIST:], x[:, HIST - 1, 8:10])
    assert torch.allclose(back, x[:, HIST:], atol=1e-5), "root steps must integrate back to the path"

    base = dict(layers=2, heads=4, head_dim=32, contact_w=0.3, fk_w=0.1, pen_w=0.5, qk_norm=True, qk_gain=False,
                segments=True, hist_noise=0.1, patch_res=0.05, contact_surface=True, rand_terrain=0.1, dropout=0.1,
                sample_steps=10)
    W = None
    for opts in (dict(frame_norm=True), dict(vel_w=1.0), dict(root_vel=True),
                 dict(frame_norm=True, vel_w=1.0, root_vel=True)):
        P = PosePlanner(**base, **opts).to(dev).attach(fk)
        if W is None:
            W = Windows(D, dev, fk, grid=P.grid, sfk=P.sfk)
            xb, pb, gb, ctx = W.batch(512, mirrored=True)
        P.set_normaliser(xb, pb, gb)
        if opts.get("frame_norm"):
            assert P.x_std.shape == (HIST + FUT, P.c)
            if not opts.get("root_vel"):               # (steps don't spread along the window)
                assert P.x_std[HIST, 8] < 0.5 * P.x_std[-1, 8], "pelvis path std must grow along the window"
        P.train()
        loss = P.loss(xb, pb, gb, ctx)
        loss.backward()
        assert torch.isfinite(loss) and all(torch.isfinite(q.grad).all() for q in P.net.parameters() if q.grad is not None)
        assert ("vel" in P.parts) == bool(opts.get("vel_w"))
        P.eval()
        out = P.generate(xb[:8, :HIST], pb[:8], gb[:8])
        assert out.shape == (8, FUT, P.c) and torch.isfinite(out).all()
        print(opts, "loss", round(float(loss.detach()), 3), {k: round(v, 3) for k, v in P.parts.items()})

    if V11.exists():                                   # old checkpoints: shared stats, no new buffers
        P = PosePlanner.load(V11, dev).attach(fk)
        assert P.x_std.dim() == 1 and not P.cfg.get("frame_norm")
        Wv = Windows(D, dev, fk, grid=P.grid, sfk=P.sfk)
        Wv.starts, Wv.lengths = D["start"], D["frames"]
        a = torch.as_tensor([int(s) + 2 * (HIST - 1) for s in D["start"][:16]], device=dev)
        s, ph = start(Wv, a)
        o, a_psi = s["pcom"][:, -1, [0, 2]], circmean(ph[:, -NS:])
        g0 = Wv.fr.G(Wv.fr.clip[a], s["pcom"][:, -1:, 0], s["pcom"][:, -1:, 2])[:, 0]
        hist = P.with_segments(frame_features(s, o, a_psi, g0))
        fut = P.generate(hist, patch(Wv.fr.G, Wv.fr.clip[a], o, a_psi, g0, P.grid), torch.zeros(16, 2, device=dev))[:, :4]
        with torch.enable_grad():
            plain = hold_feet(P, hist, fut, 50)
            smooth = hold_feet(P, hist, fut, 50, smooth=1.0)
        step = lambda f: (f[:, 0] - fut[:, 0]).abs().mean()          # noqa: E731  change at the first frame
        print(f"hold change at the seam frame: plain {float(step(plain)):.4f}, smooth {float(step(smooth)):.4f}")
        assert step(smooth) < step(plain)
    print("ok")


if __name__ == "__main__":
    main()
