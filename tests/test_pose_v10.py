"""v10 full-pose planner options on held-out data (GPU): v9 checkpoints load unchanged; batches carry segments, a 5 cm
patch and a terrain context; mirrored windows look up the exact terrain correctly; body-mode history noise stays one
body; the loss runs forward/backward and blended 10-step sampling returns the model's channels.

Run: GP_DATASET=v5 PYTHONPATH=. .venv/bin/python tests/test_pose_v10.py
"""

import torch

from gp import data, footfk
from gp.pose_planner import DEFAULT_GRID, HIST, PosePlanner, Windows, exact_ground, geometry, seg_channels

V9 = data.ROOT / "runs" / "pose2_planner_v9" / "planner_best.pt"


def main() -> None:
    dev = "cuda"
    torch.manual_seed(0)
    D = data.load(("test_trial", "test_terrain"))
    fk = footfk.fit(D, dev, data.ROOT / "data" / "foot_fk.pt")
    if V9.exists():
        P9 = PosePlanner.load(V9, dev).attach(fk)
        assert P9.c == 45 and P9.grid == DEFAULT_GRID and P9.sfk is None
    P = PosePlanner(layers=4, heads=8, head_dim=32, contact_w=0.3, fk_w=0.1, pen_w=0.5, qk_norm=True, qk_gain=False,
                    segments=True, hist_drop=0.15, hist_noise=1.0, hist_noise_mode="body", goal_h=True, patch_res=0.05,
                    contact_surface=True, rand_terrain=0.1, dropout=0.1, sample_steps=10).to(dev).attach(fk)
    W = Windows(D, dev, fk, grid=P.grid, sfk=P.sfk)
    x, p, g, ctx = W.batch(512, mirrored=True)
    assert x.shape[-1] == P.c == 126 and tuple(p.shape[1:]) == (73, 33), (x.shape, p.shape)
    assert torch.allclose(x[..., 45:], seg_channels(P.sfk, x), atol=1e-5)
    P.set_normaliser(x, p, g)

    # a planted foot point sits on its own terrain, mirrored or not (exact lookup flips the lateral back)
    ch = geometry(fk, x)[0]
    clear = ch[..., 1] - exact_ground(ctx, ch[..., 0], ch[..., 2])
    on = x[..., :4] > 0
    f = ctx["flip"][:, None, None].expand_as(on)
    m_flip, m_keep = clear[on & f].median(), clear[on & ~f].median()
    print(f"planted point over exact terrain: median {100 * m_keep:.1f} cm (as is), {100 * m_flip:.1f} cm (mirrored)")
    assert abs(m_flip - m_keep) < 0.005 and 0.0 < m_keep < 0.06

    xb = P.body_frames(x[:, :HIST], torch.ones(len(x), device=dev))
    chb, kinb, _ = geometry(fk, xb)
    assert (chb - kinb).abs().max() < 1e-4, "body-mode noise must keep foot points on the joints"
    assert (xb[..., 45:] - seg_channels(P.sfk, xb)).abs().max() < 1e-5

    P.train()
    P.cfg["rand_terrain"] = 0.5                     # both terrain paths in one batch
    loss = P.loss(x, p, g, ctx)
    loss.backward()
    assert torch.isfinite(loss) and all(torch.isfinite(q.grad).all() for q in P.net.parameters() if q.grad is not None)
    print("loss", round(float(loss.detach()), 4), P.parts)
    P.eval()
    out = P.generate(x[:8, :HIST], p[:8], g[:8])
    assert out.shape == (8, 24, P.c) and torch.isfinite(out).all()
    assert torch.allclose(out[..., 45:], seg_channels(P.sfk, out), atol=1e-4)
    print("ok")


if __name__ == "__main__":
    main()
