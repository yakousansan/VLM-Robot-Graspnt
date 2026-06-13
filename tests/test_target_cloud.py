import numpy as np

from graspnt_vlm_rm.graspnet_infer import (
    build_depth_range_mask,
    build_target_inference_mask,
)


def test_build_depth_range_mask_uses_metric_depth_bounds():
    depth = np.array(
        [
            [0, 100, 300],
            [500, 900, 1300],
        ],
        dtype=np.uint16,
    )

    mask = build_depth_range_mask(depth, scale=1000.0, depth_min=0.2, depth_max=1.0)

    assert mask.tolist() == [
        [False, False, True],
        [True, True, False],
    ]


def test_build_target_inference_mask_combines_dilated_target_workspace_and_depth():
    depth = np.full((7, 7), 500, dtype=np.uint16)
    depth[3, 4] = 0
    target_mask = np.zeros((7, 7), dtype=bool)
    target_mask[3, 3] = True
    scene_mask = np.ones((7, 7), dtype=bool)
    scene_mask[:, 0] = False

    candidate_mask, collision_mask, report = build_target_inference_mask(
        depth,
        target_mask,
        scene_mask,
        scale=1000.0,
        config={"dilate_px": 1, "depth_min": 0.2, "depth_max": 1.0, "min_points": 4},
    )

    assert collision_mask.sum() == 41
    assert candidate_mask[3, 3] is np.True_
    assert candidate_mask[3, 4] is np.False_
    assert candidate_mask[:, 0].sum() == 0
    assert report["target_mask_points"] == 1
    assert report["candidate_mask_points"] >= 4
    assert report["collision_mask_points"] == 41


def test_build_target_inference_mask_reports_too_few_candidate_points():
    depth = np.full((5, 5), 500, dtype=np.uint16)
    target_mask = np.zeros((5, 5), dtype=bool)
    target_mask[2, 2] = True
    scene_mask = np.ones((5, 5), dtype=bool)

    candidate_mask, _collision_mask, report = build_target_inference_mask(
        depth,
        target_mask,
        scene_mask,
        scale=1000.0,
        config={"dilate_px": 0, "depth_min": 0.2, "depth_max": 1.0, "min_points": 4},
    )

    assert candidate_mask.sum() == 1
    assert report["candidate_mask_has_enough_points"] is False
