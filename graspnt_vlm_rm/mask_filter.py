from __future__ import annotations

import math
from typing import Any

import numpy as np

from graspnt_vlm_rm.visualization import project_point_to_pixel


def _is_finite_translation(translation: Any) -> bool:
    try:
        x, y, z = translation
    except (TypeError, ValueError):
        return False

    try:
        return all(math.isfinite(float(value)) for value in (x, y, z))
    except (TypeError, ValueError, OverflowError):
        return False


def _project_candidate_translation(candidate: Any, intrinsics: Any) -> tuple[int, int] | None:
    try:
        translation = candidate.translation
    except AttributeError:
        return None

    if not _is_finite_translation(translation):
        return None

    try:
        return project_point_to_pixel(translation, intrinsics)
    except Exception:
        return None


def _pixel_inside_mask(pixel: tuple[int, int], mask: np.ndarray) -> bool:
    u, v = pixel
    height, width = mask.shape[:2]
    if u < 0 or u >= width or v < 0 or v >= height:
        return False
    return bool(mask[v, u])


def filter_candidates_by_mask(
    candidates: list[Any],
    intrinsics: Any,
    mask: np.ndarray,
    mask_id: int,
) -> tuple[list[Any], dict[str, Any]]:
    mask_bool = np.asarray(mask, dtype=bool)
    if mask_bool.ndim != 2:
        raise ValueError("target mask must be a 2D array")

    kept = []
    outside_count = 0
    invalid_projection_count = 0
    for candidate in candidates:
        pixel = _project_candidate_translation(candidate, intrinsics)
        if pixel is None:
            invalid_projection_count += 1
            continue
        if _pixel_inside_mask(pixel, mask_bool):
            kept.append(candidate)
        else:
            outside_count += 1

    report = {
        "input_count": len(candidates),
        "kept_count": len(kept),
        "outside_count": outside_count,
        "invalid_projection_count": invalid_projection_count,
        "mask_id": int(mask_id),
        "mask_area": int(mask_bool.sum()),
    }
    return kept, report
