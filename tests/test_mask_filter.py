from types import SimpleNamespace

import numpy as np

from graspnt_vlm_rm.mask_filter import filter_candidates_by_mask


def test_filter_candidates_by_mask_keeps_candidates_inside_mask():
    intrinsics = SimpleNamespace(fx=100, fy=100, cx=50, cy=50, width=100, height=100)
    mask = np.zeros((100, 100), dtype=bool)
    mask[40:61, 40:61] = True
    inside = SimpleNamespace(translation=[0, 0, 1])
    outside = SimpleNamespace(translation=[0.4, 0.4, 1])
    invalid = SimpleNamespace(translation=[0, 0, -1])

    kept, report = filter_candidates_by_mask(
        [inside, outside, invalid],
        intrinsics,
        mask,
        mask_id=3,
    )

    assert kept == [inside]
    assert report == {
        "input_count": 3,
        "kept_count": 1,
        "outside_count": 1,
        "invalid_projection_count": 1,
        "mask_id": 3,
        "mask_area": 441,
    }


def test_filter_candidates_by_mask_rejects_out_of_bounds_projection():
    intrinsics = SimpleNamespace(fx=100, fy=100, cx=50, cy=50, width=100, height=100)
    mask = np.ones((100, 100), dtype=bool)
    out_of_bounds = SimpleNamespace(translation=[2.0, 0, 1])

    kept, report = filter_candidates_by_mask(
        [out_of_bounds],
        intrinsics,
        mask,
        mask_id=1,
    )

    assert kept == []
    assert report["outside_count"] == 1
