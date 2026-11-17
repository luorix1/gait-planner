"""Checks for the Watch Your Step replication's pure parts: the terrain map and the predictor's labels."""

import numpy as np
import torch

from gp import fpm, wys_terrain


def test_terrain_layout() -> None:
    types = wys_terrain.column_types()
    counts = {t: types.count(t) for t in set(types)}
    # IsaacLab's assignment of 20 columns from the paper's proportions (flat's 2.5 % rounds to none)
    assert counts == {"pyramid_stairs": 3, "pyramid_stairs_inv": 3, "boxes": 3, "random_rough": 1,
                      "hf_pyramid_slope": 2, "hf_pyramid_slope_inv": 1, "stepping_stones": 6, "pit": 1}
    T = wys_terrain.build()
    assert T.heights.shape == (T.block.shape[0] + 1, T.block.shape[1] + 1)
    assert np.array_equal(np.isnan(T.block), np.isnan(wys_terrain.build().block))     # deterministic in the seed
    s = wys_terrain.summary(T)
    assert s["pyramid_stairs"]["level9"][1] > s["pyramid_stairs"]["level0"][1] > 0       # stairs rise with level
    assert s["stepping_stones"]["level9"][0] < s["stepping_stones"]["level0"][0] < 0     # deeper holes with level


def test_foothold_labels() -> None:
    # one world, one foot walking +x: down (t 0-1), air (2-3), down at x=1 (4-5), air (6), down at x=2 (7-8)
    down = torch.tensor([1, 1, 0, 0, 1, 1, 0, 1, 1], dtype=torch.float32)[None, :, None]
    x = torch.tensor([0, 0, 0.3, 0.7, 1, 1, 1.5, 2, 2])
    foot = torch.stack([x, torch.zeros(9), torch.zeros(9)], -1)[None, :, None]
    root = torch.zeros(1, 9, 3)
    quat = torch.tensor([0.0, 0.0, 0.0, 1.0]).expand(1, 9, 4)
    p = fpm.future_contacts(down, root, quat, foot, torch.zeros(1, 9))[0, :, 0, 0]
    # in contact: the landing after the coming lift-off; in the air: the landing it is about to make
    assert torch.allclose(p, torch.tensor([1, 1, 1, 1, 2, 2, 2, 2, 2], dtype=torch.float32))
