from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from scipy.spatial.transform import Rotation


DEFAULT_GRASPNET_TO_GRIPPER = np.array(
    [
        [0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0],
        [-1.0, 0.0, 0.0],
    ],
    dtype=float,
)


def build_transform(
    rotation: Sequence[float] | np.ndarray,
    translation: Sequence[float] | np.ndarray,
) -> np.ndarray:
    transform = np.eye(4)
    rotation_array = np.asarray(rotation, dtype=float)
    if rotation_array.shape == (9,):
        rotation_array = rotation_array.reshape(3, 3)
    if rotation_array.shape != (3, 3):
        raise ValueError("rotation must be a 3x3 matrix or flat 9-value sequence")

    translation_array = np.asarray(translation, dtype=float)
    if translation_array.shape != (3,):
        raise ValueError("translation must contain 3 values")

    transform[:3, :3] = rotation_array
    transform[:3, 3] = translation_array
    return transform


def pose_to_transform(pose: Sequence[float] | np.ndarray) -> np.ndarray:
    pose_array = np.asarray(pose, dtype=float)
    if pose_array.shape != (6,):
        raise ValueError("pose must contain [x, y, z, rx, ry, rz]")

    rotation = Rotation.from_euler("xyz", pose_array[3:]).as_matrix()
    return build_transform(rotation, pose_array[:3])


def transform_to_pose(transform: Sequence[Sequence[float]] | np.ndarray) -> list[float]:
    transform_array = np.asarray(transform, dtype=float)
    if transform_array.shape != (4, 4):
        raise ValueError("transform must be a 4x4 matrix")

    euler = Rotation.from_matrix(transform_array[:3, :3]).as_euler("xyz")
    return [
        float(transform_array[0, 3]),
        float(transform_array[1, 3]),
        float(transform_array[2, 3]),
        float(euler[0]),
        float(euler[1]),
        float(euler[2]),
    ]


def hand_eye_transform(
    rotation: Sequence[float] | np.ndarray,
    translation: Sequence[float] | np.ndarray,
) -> np.ndarray:
    return build_transform(rotation, translation)


def graspnet_to_camera_transform(
    translation: Sequence[float] | np.ndarray,
    rotation_matrix: Sequence[Sequence[float]] | np.ndarray,
    gripper_length: float = 0.0,
    align_rotation: Sequence[Sequence[float]] | np.ndarray = DEFAULT_GRASPNET_TO_GRIPPER,
) -> np.ndarray:
    camera_from_grasp = build_transform(rotation_matrix, translation)
    align_rotation_array = np.asarray(align_rotation, dtype=float)
    gripper_offset_in_grasp = align_rotation_array @ np.array(
        [0.0, 0.0, -float(gripper_length)],
        dtype=float,
    )
    grasp_from_gripper = build_transform(align_rotation_array, gripper_offset_in_grasp)
    return camera_from_grasp @ grasp_from_gripper


def camera_grasp_to_base_pose(
    grasp_translation: Sequence[float] | np.ndarray,
    grasp_rotation: Sequence[Sequence[float]] | np.ndarray,
    current_end_pose: Sequence[float] | np.ndarray,
    hand_eye_rotation: Sequence[float] | np.ndarray,
    hand_eye_translation: Sequence[float] | np.ndarray,
    gripper_length: float = 0.0,
) -> list[float]:
    base_from_end = pose_to_transform(current_end_pose)
    end_from_camera = hand_eye_transform(hand_eye_rotation, hand_eye_translation)
    camera_from_gripper = graspnet_to_camera_transform(
        grasp_translation,
        grasp_rotation,
        gripper_length=gripper_length,
    )

    return transform_to_pose(base_from_end @ end_from_camera @ camera_from_gripper)


def offset_pose_along_local_z(
    pose: Sequence[float] | np.ndarray,
    offset: float,
) -> list[float]:
    transform = pose_to_transform(pose)
    pose_array = np.asarray(pose, dtype=float).copy()
    pose_array[:3] += transform[:3, 2] * float(offset)
    return pose_array.tolist()


def offset_pose_along_base_z(
    pose: Sequence[float] | np.ndarray,
    offset: float,
) -> list[float]:
    pose_array = np.asarray(pose, dtype=float).copy()
    if pose_array.shape != (6,):
        raise ValueError("pose must contain [x, y, z, rx, ry, rz]")
    pose_array[2] += float(offset)
    return pose_array.tolist()
