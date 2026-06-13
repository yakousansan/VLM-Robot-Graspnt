from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np

from graspnt_vlm_rm.camera_realsense import RealSenseCamera
from graspnt_vlm_rm.config import load_config, validate_runtime_config
from graspnt_vlm_rm.graspnet_infer import GraspNetRunner
from graspnt_vlm_rm.safety import (
    format_pose,
    validate_motion_plan,
)
from graspnt_vlm_rm.transform import (
    camera_grasp_to_base_pose,
    offset_pose_along_base_z,
    offset_pose_along_local_z,
)
from graspnt_vlm_rm.udp_client import UdpRobotClient, extract_current_end_pose
from graspnt_vlm_rm.visualization import preview_workspace, visualize_debug


def candidate_to_dict(candidate: Any) -> dict[str, Any]:
    if isinstance(candidate, dict):
        translation = candidate["translation"]
        rotation_matrix = candidate["rotation_matrix"]
        score = candidate["score"]
        width = candidate["width"]
    else:
        translation = candidate.translation
        rotation_matrix = candidate.rotation_matrix
        score = candidate.score
        width = candidate.width

    return {
        "translation": np.asarray(translation, dtype=float).tolist(),
        "rotation_matrix": np.asarray(rotation_matrix, dtype=float).tolist(),
        "score": float(score),
        "width": float(width),
    }


def build_plan(
    config: dict[str, Any],
    candidate: Any,
    current_end_pose: list[float],
    validate: bool = True,
) -> dict[str, Any]:
    safety_config = config["safety"]
    candidate_data = candidate_to_dict(candidate)
    grasp_pose = camera_grasp_to_base_pose(
        candidate_data["translation"],
        candidate_data["rotation_matrix"],
        current_end_pose,
        config["hand_eye"]["rotation"],
        config["hand_eye"]["translation"],
        gripper_length=float(safety_config["gripper_length"]),
    )

    pre_grasp_pose = offset_pose_along_local_z(
        grasp_pose,
        -float(safety_config.get("pre_grasp_offset", 0.0)),
    )
    lift_pose = offset_pose_along_base_z(
        grasp_pose,
        float(safety_config.get("lift_offset", 0.0)),
    )
    plan = {
        "score": candidate_data["score"],
        "width": candidate_data["width"],
        "grasp_pose": grasp_pose,
        "pre_grasp_pose": pre_grasp_pose,
        "lift_pose": lift_pose,
    }
    if validate:
        validate_motion_plan(plan, safety_config)
    return plan


def print_report(
    frame_report: dict[str, Any],
    grasp_report: dict[str, Any],
    robot_state: Any,
    plan: dict[str, Any],
) -> None:
    print(f"frame: {frame_report}")
    print(f"grasp: {grasp_report}")
    print(f"robot_state: {robot_state}")
    print(f"score: {plan['score']:.6f}")
    print(f"width: {plan['width']:.6f}")
    print(format_pose("pre_grasp_pose", plan["pre_grasp_pose"]))
    print(format_pose("grasp_pose", plan["grasp_pose"]))
    print(format_pose("lift_pose", plan["lift_pose"]))


def _make_udp_client(execution_config: dict[str, Any]) -> UdpRobotClient:
    return UdpRobotClient(
        host=str(execution_config.get("udp_host", "127.0.0.1")),
        port=int(execution_config.get("udp_port", 6556)),
        ack_timeout_sec=float(execution_config.get("ack_timeout_sec", 1.0)),
        result_timeout_sec=float(execution_config.get("result_timeout_sec", 60.0)),
        max_retries=int(execution_config.get("max_retries", 3)),
    )


def run(config_path: str | Path) -> dict[str, Any]:
    config = load_config(config_path)
    validate_runtime_config(config)

    camera = None
    udp_client = None
    try:
        camera_config = config.get("camera", {})
        camera = RealSenseCamera(
            width=int(camera_config.get("width", 640)),
            height=int(camera_config.get("height", 480)),
            fps=int(camera_config.get("fps", 30)),
            serial=camera_config.get("serial"),
        )
        camera.start()

        frame = preview_workspace(
            camera,
            config.get("workspace", {}),
            config.get("camera_preview", {}),
        )
        if frame is None:
            raise RuntimeError("workspace preview cancelled")

        execution_config = config.get("execution", {})
        backend = str(execution_config.get("backend", "udp_cpp"))
        if backend != "udp_cpp":
            raise ValueError("only execution.backend='udp_cpp' is supported")
        udp_client = _make_udp_client(execution_config)
        robot_state = udp_client.request_pose()
        current_end_pose = extract_current_end_pose(robot_state)

        runner = GraspNetRunner(config["graspnet"])
        candidates, grasp_report = runner.infer(
            frame.color,
            frame.depth,
            frame.intrinsics,
            config.get("workspace", {}),
        )
        if not candidates:
            raise RuntimeError("GraspNet returned zero grasp candidates")

        plan = build_plan(config, candidates[0], current_end_pose, validate=False)
        visualization = visualize_debug(
            frame,
            candidates,
            getattr(runner, "last_debug", None),
            plan,
            config.get("visualization", {}),
        )
        frame_report = {
            "color_shape": tuple(frame.color.shape),
            "depth_shape": tuple(frame.depth.shape),
        }
        print_report(frame_report, grasp_report, robot_state, plan)
        validate_motion_plan(plan, config["safety"])
        execution_result = udp_client.execute_grasp(plan)
        print(f"execution_result: {execution_result}")

        return {
            "frame_report": frame_report,
            "grasp_report": grasp_report,
            "candidate": candidate_to_dict(candidates[0]),
            "robot_state": robot_state,
            "plan": plan,
            "visualization": visualization,
            "execution_result": execution_result,
        }
    finally:
        if camera is not None:
            camera.stop()
        if udp_client is not None:
            udp_client.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one basic GRASPNT grasp.")
    parser.add_argument(
        "config",
        nargs="?",
        default=Path(__file__).with_name("config.yaml"),
        help="Path to runtime YAML config.",
    )
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
