from __future__ import annotations

import math
from numbers import Real


def _is_real_number(value: object) -> bool:
    return isinstance(value, Real) and not isinstance(value, bool)


def validate_pose_shape(pose: list[float]) -> None:
    if len(pose) != 6:
        raise ValueError("pose must contain 6 values")
    if not all(_is_real_number(value) for value in pose):
        raise ValueError("pose must contain real numeric values")
    if not all(math.isfinite(value) for value in pose):
        raise ValueError("pose contains non-finite values")


def validate_workspace_bounds(
    pose: list[float],
    workspace_bounds: dict[str, list[float]] | None,
) -> None:
    if not workspace_bounds:
        return
    for axis, index in {"x": 0, "y": 1, "z": 2}.items():
        if axis not in workspace_bounds:
            continue
        bounds = workspace_bounds[axis]
        if len(bounds) != 2:
            raise ValueError(f"workspace bounds for {axis} must contain min and max")
        lower, upper = bounds
        value = pose[index]
        if value < lower or value > upper:
            raise ValueError(
                f"pose {axis}={value:.4f} is outside workspace bounds "
                f"[{lower:.4f}, {upper:.4f}]"
            )


def validate_grasp_pose(
    pose: list[float],
    min_grasp_z: float,
    workspace_bounds: dict[str, list[float]] | None = None,
) -> None:
    validate_pose_shape(pose)
    if pose[2] < min_grasp_z:
        raise ValueError(f"grasp z={pose[2]:.4f} is below min_grasp_z={min_grasp_z:.4f}")
    validate_workspace_bounds(pose, workspace_bounds)


def validate_motion_plan(plan: dict, safety_config: dict) -> None:
    for label in ("pre_grasp_pose", "grasp_pose", "lift_pose"):
        try:
            validate_grasp_pose(
                plan[label],
                float(safety_config["min_grasp_z"]),
                safety_config.get("workspace_bounds"),
            )
        except ValueError as exc:
            raise ValueError(f"{label}: {exc}") from exc


def format_pose(label: str, pose: list[float]) -> str:
    values = ", ".join(f"{float(value):.6f}" for value in pose)
    return f"{label}: [{values}]"
