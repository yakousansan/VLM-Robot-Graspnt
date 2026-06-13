from __future__ import annotations

import importlib
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from graspnt_vlm_rm.camera_realsense import CameraIntrinsics


GRASPNET_API_SOURCE_ROOT = Path(__file__).resolve().parents[1] / "graspnetAPI"


@dataclass(frozen=True)
class GraspCandidate:
    translation: np.ndarray
    rotation_matrix: np.ndarray
    score: float
    width: float


@dataclass(frozen=True)
class InferenceDebugData:
    workspace_mask: np.ndarray
    cloud_points: np.ndarray
    cloud_colors: np.ndarray
    candidate_mask: np.ndarray | None = None
    collision_mask: np.ndarray | None = None


def filter_grasp_candidates(
    candidates: list[GraspCandidate],
    config: dict[str, Any],
) -> list[GraspCandidate]:
    min_score = float(config.get("min_score", 0.0))
    score_filtered = [candidate for candidate in candidates if candidate.score >= min_score]
    if not score_filtered:
        return []

    if "top_down_angle_deg" not in config:
        return score_filtered

    axis_index = int(config.get("approach_axis", 0))
    vertical = np.array([0.0, 0.0, 1.0])
    threshold = np.deg2rad(float(config["top_down_angle_deg"]))
    top_down = []
    for candidate in score_filtered:
        approach = np.asarray(candidate.rotation_matrix, dtype=float)[:, axis_index]
        norm = np.linalg.norm(approach)
        if norm == 0:
            continue
        cos_angle = np.clip(np.dot(approach / norm, vertical), -1.0, 1.0)
        if np.arccos(cos_angle) <= threshold:
            top_down.append(candidate)
    return top_down if top_down else score_filtered


def build_workspace_mask(depth: np.ndarray, workspace_config: dict[str, Any]) -> np.ndarray:
    mask = np.asarray(depth) > 0
    if workspace_config.get("mode", "center") == "center":
        height, width = mask.shape
        x0 = int(width * float(workspace_config["x_min_ratio"]))
        x1 = int(width * float(workspace_config["x_max_ratio"]))
        y0 = int(height * float(workspace_config["y_min_ratio"]))
        y1 = int(height * float(workspace_config["y_max_ratio"]))

        center_mask = np.zeros_like(mask, dtype=bool)
        center_mask[y0:y1, x0:x1] = True
        mask &= center_mask
    return mask


def build_depth_range_mask(
    depth: np.ndarray,
    scale: float,
    depth_min: float | None = None,
    depth_max: float | None = None,
) -> np.ndarray:
    depth_array = np.asarray(depth)
    mask = depth_array > 0
    scale_value = float(scale)
    if depth_min is not None:
        mask &= depth_array >= float(depth_min) * scale_value
    if depth_max is not None:
        mask &= depth_array <= float(depth_max) * scale_value
    return np.asarray(mask, dtype=bool)


def _dilate_mask(mask: np.ndarray, radius_px: int) -> np.ndarray:
    mask_bool = np.asarray(mask, dtype=bool)
    radius = max(0, int(radius_px))
    if radius == 0:
        return mask_bool
    try:
        import cv2

        kernel_size = radius * 2 + 1
        kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
        return cv2.dilate(mask_bool.astype(np.uint8), kernel, iterations=1).astype(bool)
    except ImportError:
        padded = np.pad(mask_bool, radius, mode="constant", constant_values=False)
        output = np.zeros_like(mask_bool, dtype=bool)
        for y_offset in range(radius * 2 + 1):
            for x_offset in range(radius * 2 + 1):
                output |= padded[
                    y_offset : y_offset + mask_bool.shape[0],
                    x_offset : x_offset + mask_bool.shape[1],
                ]
        return output


def build_target_inference_mask(
    depth: np.ndarray,
    target_mask: np.ndarray,
    scene_mask: np.ndarray,
    scale: float,
    config: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, dict[str, int | bool]]:
    depth_mask = build_depth_range_mask(
        depth,
        scale=scale,
        depth_min=config.get("depth_min"),
        depth_max=config.get("depth_max"),
    )
    scene_mask_bool = np.asarray(scene_mask, dtype=bool) & depth_mask
    target_mask_bool = np.asarray(target_mask, dtype=bool)
    if target_mask_bool.shape != scene_mask_bool.shape:
        raise ValueError("target_mask shape must match depth image shape")

    target_valid_mask = target_mask_bool & scene_mask_bool
    dilated_target_mask = _dilate_mask(target_mask_bool, int(config.get("dilate_px", 0)))
    candidate_mask = dilated_target_mask & scene_mask_bool
    collision_mask = scene_mask_bool
    min_points = int(config.get("min_points", 1))
    report = {
        "target_mask_points": int(target_valid_mask.sum()),
        "candidate_mask_points": int(candidate_mask.sum()),
        "collision_mask_points": int(collision_mask.sum()),
        "candidate_mask_has_enough_points": bool(candidate_mask.sum() >= min_points),
        "dilate_px": int(config.get("dilate_px", 0)),
    }
    return candidate_mask, collision_mask, report


def _import_grasp_group() -> type:
    try:
        graspnet_api = importlib.import_module("graspnetAPI")
    except ImportError as exc:
        raise RuntimeError(
            "missing GraspNet runtime dependency: graspnetAPI. "
            "Install graspnetAPI before creating GraspNetRunner."
        ) from exc

    grasp_group = getattr(graspnet_api, "GraspGroup", None)
    if grasp_group is not None:
        return grasp_group

    source_root = GRASPNET_API_SOURCE_ROOT
    if source_root.exists():
        source_root_text = str(source_root)
        if source_root_text in sys.path:
            sys.path.remove(source_root_text)
        sys.path.insert(0, source_root_text)
        importlib.invalidate_caches()

        module = sys.modules.get("graspnetAPI")
        if module is not None and getattr(module, "__spec__", None) is not None:
            try:
                graspnet_api = importlib.reload(module)
            except ImportError:
                sys.modules.pop("graspnetAPI", None)
                try:
                    graspnet_api = importlib.import_module("graspnetAPI")
                except ImportError as exc:
                    raise RuntimeError(
                        "missing GraspNet runtime dependency: "
                        "graspnetAPI.GraspGroup. Ensure graspnetAPI is installed "
                        f"or that the source checkout path {source_root} is importable."
                    ) from exc
        else:
            sys.modules.pop("graspnetAPI", None)
            try:
                graspnet_api = importlib.import_module("graspnetAPI")
            except ImportError as exc:
                raise RuntimeError(
                    "missing GraspNet runtime dependency: "
                    "graspnetAPI.GraspGroup. Ensure graspnetAPI is installed "
                    f"or that the source checkout path {source_root} is importable."
                ) from exc

        grasp_group = getattr(graspnet_api, "GraspGroup", None)
        if grasp_group is not None:
            return grasp_group

    raise RuntimeError(
        "missing GraspNet runtime dependency: graspnetAPI.GraspGroup. "
        "Ensure graspnetAPI is installed or that the source checkout path "
        f"{source_root} is importable."
    )


class GraspNetRunner:
    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.last_debug: InferenceDebugData | None = None
        root = Path(config["root"])
        for relative_path in ("models", "dataset", "utils"):
            runtime_path = str(root / relative_path)
            if runtime_path not in sys.path:
                sys.path.append(runtime_path)

        try:
            torch = importlib.import_module("torch")
            o3d = importlib.import_module("open3d")
            GraspGroup = _import_grasp_group()
            collision_detector = importlib.import_module("collision_detector")
            data_utils = importlib.import_module("data_utils")
            graspnet = importlib.import_module("graspnet")
        except ImportError as exc:
            missing = exc.name or str(exc)
            raise RuntimeError(
                "missing GraspNet runtime dependency: "
                f"{missing}. Install torch, open3d, graspnetAPI, and the "
                "graspnet-baseline runtime modules before creating GraspNetRunner."
            ) from exc

        self._torch = torch
        self._o3d = o3d
        self.GraspGroup = GraspGroup
        self.CameraInfo = data_utils.CameraInfo
        self.ModelFreeCollisionDetector = collision_detector.ModelFreeCollisionDetector
        self.create_point_cloud_from_depth_image = (
            data_utils.create_point_cloud_from_depth_image
        )
        self.pred_decode = graspnet.pred_decode

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.net = graspnet.GraspNet(
            input_feature_dim=0,
            num_view=int(config["num_view"]),
            num_angle=12,
            num_depth=4,
            cylinder_radius=0.05,
            hmin=-0.02,
            hmax_list=[0.01, 0.02, 0.03, 0.04],
            is_training=False,
        ).to(self.device)

        checkpoint = torch.load(config["checkpoint"], map_location=self.device)
        self.net.load_state_dict(checkpoint["model_state_dict"])
        self.net.eval()

    def build_workspace_mask(
        self,
        depth: np.ndarray,
        workspace_config: dict[str, Any],
    ) -> np.ndarray:
        return build_workspace_mask(depth, workspace_config)

    def infer(
        self,
        color: np.ndarray,
        depth: np.ndarray,
        intrinsics: CameraIntrinsics,
        workspace_config: dict[str, Any],
        candidate_mask: np.ndarray | None = None,
        collision_mask: np.ndarray | None = None,
    ) -> tuple[list[GraspCandidate], dict[str, int]]:
        camera = self.CameraInfo(
            intrinsics.width,
            intrinsics.height,
            intrinsics.fx,
            intrinsics.fy,
            intrinsics.cx,
            intrinsics.cy,
            intrinsics.scale,
        )
        cloud = self.create_point_cloud_from_depth_image(depth, camera, organized=True)
        color_float = np.asarray(color, dtype=np.float32) / 255.0
        workspace_mask = self.build_workspace_mask(depth, workspace_config)
        if candidate_mask is None:
            candidate_mask_bool = workspace_mask
        else:
            candidate_mask_bool = np.asarray(candidate_mask, dtype=bool)
            if candidate_mask_bool.shape != workspace_mask.shape:
                raise ValueError("candidate_mask shape must match depth image shape")

        if collision_mask is None:
            collision_mask_bool = workspace_mask
        else:
            collision_mask_bool = np.asarray(collision_mask, dtype=bool)
            if collision_mask_bool.shape != workspace_mask.shape:
                raise ValueError("collision_mask shape must match depth image shape")

        candidate_cloud = cloud[candidate_mask_bool]
        candidate_colors = color_float[candidate_mask_bool]
        collision_cloud = cloud[collision_mask_bool]
        collision_colors = color_float[collision_mask_bool]
        if len(candidate_cloud) == 0:
            self.last_debug = None
            raise RuntimeError("candidate mask produced zero valid depth points")
        self.last_debug = InferenceDebugData(
            workspace_mask=np.asarray(candidate_mask_bool, dtype=bool),
            cloud_points=np.asarray(collision_cloud, dtype=np.float32),
            cloud_colors=np.asarray(collision_colors, dtype=np.float32),
            candidate_mask=np.asarray(candidate_mask_bool, dtype=bool),
            collision_mask=np.asarray(collision_mask_bool, dtype=bool),
        )

        torch = self._torch
        o3d = self._o3d

        num_point = int(self.config["num_point"])
        if len(candidate_cloud) >= num_point:
            idxs = np.random.choice(len(candidate_cloud), num_point, replace=False)
        else:
            idxs_keep = np.arange(len(candidate_cloud))
            idxs_extra = np.random.choice(
                len(candidate_cloud),
                num_point - len(candidate_cloud),
                replace=True,
            )
            idxs = np.concatenate([idxs_keep, idxs_extra], axis=0)

        cloud_sampled = torch.from_numpy(
            candidate_cloud[idxs][np.newaxis].astype(np.float32)
        ).to(self.device)
        end_points = {
            "point_clouds": cloud_sampled,
            "cloud_colors": candidate_colors[idxs],
        }

        with torch.no_grad():
            end_points = self.net(end_points)
            grasp_preds = self.pred_decode(end_points)

        gg = self.GraspGroup(grasp_preds[0].detach().cpu().numpy())

        cloud_o3d = o3d.geometry.PointCloud()
        cloud_o3d.points = o3d.utility.Vector3dVector(collision_cloud.astype(np.float32))
        cloud_o3d.colors = o3d.utility.Vector3dVector(collision_colors.astype(np.float32))

        collision_thresh = float(self.config.get("collision_thresh", 0.0))
        if collision_thresh > 0 and len(collision_cloud) > 0:
            detector = self.ModelFreeCollisionDetector(
                np.asarray(cloud_o3d.points),
                voxel_size=float(self.config.get("voxel_size", 0.01)),
            )
            collision_mask = detector.detect(
                gg,
                approach_dist=0.05,
                collision_thresh=collision_thresh,
            )
            gg = gg[~collision_mask]

        gg = gg.nms()
        gg = gg.sort_by_score()

        candidates = [
            GraspCandidate(
                translation=np.asarray(grasp.translation, dtype=float),
                rotation_matrix=np.asarray(grasp.rotation_matrix, dtype=float),
                score=float(grasp.score),
                width=float(grasp.width),
            )
            for grasp in gg
        ]
        candidates = filter_grasp_candidates(candidates, self.config)
        report = {
            "valid_workspace_points": int(len(candidate_cloud)),
            "candidate_source_points": int(len(candidate_cloud)),
            "collision_scene_points": int(len(collision_cloud)),
            "candidate_count": int(len(candidates)),
        }
        return candidates, report
