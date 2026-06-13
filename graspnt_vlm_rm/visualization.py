from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


def _import_cv2():
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError(
            "opencv-python is required for 2D visualization and PNG debug output. "
            "Install it with `python -m pip install opencv-python`."
        ) from exc
    return cv2


def _import_open3d():
    try:
        import open3d as o3d
    except ImportError as exc:
        raise RuntimeError(
            "open3d is required for 3D grasp visualization. "
            "Install it in the GRASPNT runtime environment."
        ) from exc
    return o3d


def project_point_to_pixel(point: Any, intrinsics: Any) -> tuple[int, int] | None:
    point_array = np.asarray(point, dtype=float)
    if point_array.shape != (3,):
        raise ValueError("point must contain 3 camera-frame coordinates")
    x, y, z = point_array
    if z <= 0:
        return None

    u = int(round(float(intrinsics.fx) * x / z + float(intrinsics.cx)))
    v = int(round(float(intrinsics.fy) * y / z + float(intrinsics.cy)))
    return u, v


def build_workspace_overlay(
    color: np.ndarray,
    workspace_mask: np.ndarray | None,
    outside_tint: tuple[int, int, int] = (0, 0, 180),
    alpha: float = 0.35,
) -> np.ndarray:
    image = np.asarray(color, dtype=np.uint8).copy()
    if workspace_mask is None:
        return image
    mask = np.asarray(workspace_mask, dtype=bool)
    if mask.shape != image.shape[:2]:
        raise ValueError("workspace_mask shape must match the color image size")

    tint = np.asarray(outside_tint, dtype=np.float32)
    outside = ~mask
    image_float = image.astype(np.float32)
    image_float[outside] = image_float[outside] * (1.0 - alpha) + tint * alpha
    return image_float.astype(np.uint8)


def _draw_square_marker(
    image: np.ndarray,
    center: tuple[int, int],
    color: tuple[int, int, int],
    radius: int = 6,
) -> None:
    u, v = center
    height, width = image.shape[:2]
    if u < 0 or u >= width or v < 0 or v >= height:
        return
    x0 = max(0, u - radius)
    x1 = min(width, u + radius + 1)
    y0 = max(0, v - radius)
    y1 = min(height, v + radius + 1)
    image[y0:y1, x0:x1] = color


def build_rgb_debug_image(
    color: np.ndarray,
    intrinsics: Any,
    candidates: list[Any],
    workspace_mask: np.ndarray | None,
) -> np.ndarray:
    image = build_workspace_overlay(color, workspace_mask)
    for index, candidate in enumerate(candidates):
        pixel = project_point_to_pixel(candidate.translation, intrinsics)
        if pixel is None:
            continue
        marker_color = (0, 255, 0) if index == 0 else (160, 160, 160)
        radius = 7 if index == 0 else 3
        _draw_square_marker(image, pixel, marker_color, radius=radius)
    return image


def _depth_to_color(depth: np.ndarray) -> np.ndarray:
    cv2 = _import_cv2()
    depth_array = np.asarray(depth)
    valid = depth_array[depth_array > 0]
    if valid.size == 0:
        normalized = np.zeros(depth_array.shape, dtype=np.uint8)
    else:
        min_depth = float(valid.min())
        max_depth = float(valid.max())
        if max_depth <= min_depth:
            normalized = np.where(depth_array > 0, 255, 0).astype(np.uint8)
        else:
            clipped = np.clip(depth_array.astype(np.float32), min_depth, max_depth)
            normalized = ((clipped - min_depth) / (max_depth - min_depth) * 255).astype(
                np.uint8
            )
            normalized[depth_array <= 0] = 0
    return cv2.applyColorMap(normalized, cv2.COLORMAP_JET)


def _show_2d(
    color: np.ndarray,
    depth: np.ndarray,
    intrinsics: Any,
    candidates: list[Any],
    workspace_mask: np.ndarray | None,
    config: dict[str, Any],
) -> None:
    cv2 = _import_cv2()
    top_n = int(config.get("top_n", 20))
    rgb_debug = build_rgb_debug_image(color, intrinsics, candidates[:top_n], workspace_mask)
    depth_debug = _depth_to_color(depth)
    panel = np.concatenate([rgb_debug, depth_debug], axis=1)
    cv2.imshow("GRASPNT RGB / Depth Debug", panel)
    cv2.waitKey(int(config.get("wait_key_ms", 0)))
    if config.get("destroy_windows", False):
        cv2.destroyAllWindows()


def _key_code(value: Any, default: str) -> int:
    text = str(value or default)
    if text.lower() == "space":
        return ord(" ")
    if text.lower() in {"esc", "escape"}:
        return 27
    return ord(text[0])


def preview_workspace(
    camera: Any,
    workspace_config: dict[str, Any],
    config: dict[str, Any] | None,
) -> Any | None:
    config = config or {}
    if not config.get("enabled", False):
        return camera.capture()

    from graspnt_vlm_rm.graspnet_infer import build_workspace_mask

    cv2 = _import_cv2()
    window_name = str(config.get("window_name", "GRASPNT Workspace Preview"))
    continue_key = _key_code(config.get("wait_key_continue", "space"), "space")
    quit_key = _key_code(config.get("wait_key_quit", "q"), "q")
    show_depth = bool(config.get("show_depth", True))
    wait_ms = int(config.get("wait_key_ms", 30))

    while True:
        frame = camera.capture(warmup_frames=0)
        workspace_mask = build_workspace_mask(frame.depth, workspace_config)
        preview = build_workspace_overlay(frame.color, workspace_mask)
        if show_depth:
            preview = np.concatenate([preview, _depth_to_color(frame.depth)], axis=1)
        cv2.imshow(window_name, preview)

        key = cv2.waitKey(wait_ms) & 0xFF
        if key == continue_key:
            return frame
        if key in {quit_key, 27}:
            if config.get("destroy_windows", False):
                cv2.destroyAllWindows()
            return None


def _create_point_cloud(o3d: Any, debug_data: Any, color_order: str):
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(
        np.asarray(debug_data.cloud_points, dtype=np.float64)
    )
    colors = np.asarray(debug_data.cloud_colors, dtype=np.float64)
    if colors.size:
        if colors.max() > 1.0:
            colors = colors / 255.0
        if color_order.lower() == "bgr":
            colors = colors[:, ::-1]
        point_cloud.colors = o3d.utility.Vector3dVector(np.clip(colors, 0.0, 1.0))
    return point_cloud


def _create_gripper_lines(
    o3d: Any,
    candidate: Any,
    color: tuple[float, float, float],
    finger_length: float,
    approach_length: float,
):
    center = np.asarray(candidate.translation, dtype=float)
    rotation = np.asarray(candidate.rotation_matrix, dtype=float)
    width = max(float(candidate.width), 0.01)

    approach_axis = rotation[:, 0]
    closing_axis = rotation[:, 1]
    half_width = width / 2.0

    left_tip = center + closing_axis * half_width
    right_tip = center - closing_axis * half_width
    left_base = left_tip - approach_axis * finger_length
    right_base = right_tip - approach_axis * finger_length
    handle = (left_base + right_base) / 2.0 - approach_axis * approach_length

    points = np.vstack([left_tip, right_tip, left_base, right_base, handle])
    lines = np.array([[0, 2], [1, 3], [2, 3], [4, 2], [4, 3]], dtype=np.int32)
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector(np.tile(color, (len(lines), 1)))
    return line_set


def _show_3d(candidates: list[Any], debug_data: Any, config: dict[str, Any]) -> None:
    if debug_data is None:
        raise RuntimeError("3D visualization requires GraspNet inference debug data")
    o3d = _import_open3d()
    geometries = [
        _create_point_cloud(o3d, debug_data, str(config.get("color_order", "bgr")))
    ]
    top_n = int(config.get("top_n", 20))
    finger_length = float(config.get("gripper_finger_length", 0.06))
    approach_length = float(config.get("gripper_approach_length", 0.04))
    for index, candidate in enumerate(candidates[:top_n]):
        color = (0.0, 1.0, 0.0) if index == 0 else (0.65, 0.65, 0.65)
        geometries.append(
            _create_gripper_lines(
                o3d,
                candidate,
                color=color,
                finger_length=finger_length,
                approach_length=approach_length,
            )
        )
    o3d.visualization.draw_geometries(
        geometries,
        window_name="GRASPNT 3D Grasp Debug",
        width=int(config.get("window_width", 1280)),
        height=int(config.get("window_height", 720)),
    )


def save_debug_artifacts(
    frame: Any,
    candidates: list[Any],
    debug_data: Any,
    plan: dict[str, Any],
    config: dict[str, Any],
) -> list[str]:
    cv2 = _import_cv2()
    debug_dir = Path(config.get("debug_dir", "debug_outputs"))
    debug_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    top_n = int(config.get("top_n", 20))
    workspace_mask = getattr(debug_data, "workspace_mask", None) if debug_data else None

    rgb_debug = build_rgb_debug_image(
        frame.color,
        frame.intrinsics,
        candidates[:top_n],
        workspace_mask,
    )
    depth_debug = _depth_to_color(frame.depth)

    rgb_path = debug_dir / f"{timestamp}_rgb_grasp.png"
    depth_path = debug_dir / f"{timestamp}_depth.png"
    cv2.imwrite(str(rgb_path), rgb_debug)
    cv2.imwrite(str(depth_path), depth_debug)

    saved_files = [str(rgb_path), str(depth_path)]
    return saved_files


def visualize_debug(
    frame: Any,
    candidates: list[Any],
    debug_data: Any,
    plan: dict[str, Any],
    config: dict[str, Any] | None,
) -> dict[str, Any]:
    config = config or {}
    if not config.get("enabled", False):
        return {"enabled": False, "saved_files": []}

    mode = str(config.get("mode", "both")).lower()
    workspace_mask = getattr(debug_data, "workspace_mask", None) if debug_data else None
    result: dict[str, Any] = {"enabled": True, "mode": mode, "saved_files": []}

    if config.get("save_debug", False) or mode == "save_only":
        result["saved_files"] = save_debug_artifacts(
            frame,
            candidates,
            debug_data,
            plan,
            config,
        )

    if mode in ("2d", "both"):
        _show_2d(frame.color, frame.depth, frame.intrinsics, candidates, workspace_mask, config)
        result["shown_2d"] = True

    if mode in ("3d", "both"):
        _show_3d(candidates, debug_data, config)
        result["shown_3d"] = True

    return result
