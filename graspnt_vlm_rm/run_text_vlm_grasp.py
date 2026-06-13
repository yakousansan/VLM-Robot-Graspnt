from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from graspnt_vlm_rm.camera_realsense import RealSenseCamera
from graspnt_vlm_rm.config import load_config, validate_runtime_config
from graspnt_vlm_rm.graspnet_infer import (
    GraspNetRunner,
    build_target_inference_mask,
)
from graspnt_vlm_rm.run_basic_grasp import (
    _make_udp_client,
    build_plan,
    candidate_to_dict,
    print_report,
)
from graspnt_vlm_rm.mask_filter import filter_candidates_by_mask
from graspnt_vlm_rm.mask_target import (
    MaskTargetSelection,
    build_mask_selection_overlay,
    generate_sam_mask_proposals,
    get_proposal_by_id,
    save_mask_debug_images,
    save_mask_overlay_image,
    select_mask_target,
)
from graspnt_vlm_rm.safety import validate_motion_plan
from graspnt_vlm_rm.udp_client import extract_current_end_pose
from graspnt_vlm_rm.visualization import (
    preview_workspace,
    visualize_debug,
)
from graspnt_vlm_rm.video_recorder import record_grasp_video


def ask_text_command() -> str:
    command = input("Text grasp command: ").strip()
    if not command:
        raise RuntimeError("empty text command")
    return command


def print_target_report(
    command: str,
    target: MaskTargetSelection,
    target_filter_report: dict[str, Any],
) -> None:
    print(f"user_command: {command}")
    print(f"target_name: {target.target_name}")
    print(f"target_mask_id: {target.mask_id}")
    print(f"target_confidence: {target.confidence:.3f}")
    print(f"target_reason: {target.reason}")
    if target.safety_note:
        print(f"target_safety_note: {target.safety_note}")
    print(f"target_filter: {target_filter_report}")


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

        command = ask_text_command()

        runner = GraspNetRunner(config["graspnet"])
        target_filter_config = config.get("target_filter", {})
        visualization_config = config.get("visualization", {})
        visualization_extra: dict[str, Any] = {}

        proposals = generate_sam_mask_proposals(frame.color, config.get("sam", {}))
        if not proposals:
            raise RuntimeError("SAM returned zero usable mask proposals")
        mask_overlay = build_mask_selection_overlay(frame.color, proposals)
        visualization_extra["sam_masks"] = save_mask_overlay_image(
            mask_overlay,
            visualization_config,
        )
        target = select_mask_target(
            mask_overlay,
            command,
            proposals,
            config["vlm"],
        )
        selected_proposal = get_proposal_by_id(proposals, target.mask_id)
        visualization_extra.update(
            save_mask_debug_images(
                frame.color,
                mask_overlay,
                selected_proposal,
                visualization_config,
            )
        )
        target_debug_path = visualization_extra["selected_mask"]

        scene_mask = runner.build_workspace_mask(frame.depth, config.get("workspace", {}))
        candidate_mask, collision_mask, target_cloud_report = build_target_inference_mask(
            frame.depth,
            selected_proposal.mask,
            scene_mask,
            scale=frame.intrinsics.scale,
            config=config.get("target_cloud", {}),
        )
        target_filter_report = {
            "mode": "sam_target_cloud",
            "mask_id": int(target.mask_id),
            "selected_mask_area": int(selected_proposal.mask.sum()),
            "target_cloud": target_cloud_report,
        }
        if not target_cloud_report["candidate_mask_has_enough_points"]:
            print_target_report(command, target, target_filter_report)
            print(f"target_debug_image: {target_debug_path}")
            raise RuntimeError(
                "Selected target mask has too few valid depth points for "
                "GraspNet candidate generation."
            )

        candidates, grasp_report = runner.infer(
            frame.color,
            frame.depth,
            frame.intrinsics,
            config.get("workspace", {}),
            candidate_mask=candidate_mask,
            collision_mask=collision_mask,
        )
        if not candidates:
            raise RuntimeError("GraspNet returned zero target-specific grasp candidates")

        projected_candidates, projection_report = filter_candidates_by_mask(
            candidates,
            frame.intrinsics,
            candidate_mask,
            mask_id=target.mask_id,
        )
        target_filter_report["projection_check"] = projection_report
        if projected_candidates:
            filtered_candidates = projected_candidates
        else:
            target_filter_report["projection_filter_fallback_used"] = True
            filtered_candidates = candidates

        if not filtered_candidates:
            print_target_report(command, target, target_filter_report)
            print(f"target_debug_image: {target_debug_path}")
            if bool(target_filter_config.get("require_target_candidate", True)):
                raise RuntimeError(
                    "No valid GraspNet candidate projected inside the selected target region."
                )
            filtered_candidates = candidates

        execution_config = config.get("execution", {})
        backend = str(execution_config.get("backend", "udp_cpp"))
        if backend != "udp_cpp":
            raise ValueError("only execution.backend='udp_cpp' is supported")
        udp_client = _make_udp_client(execution_config)
        robot_state = udp_client.request_pose()
        current_end_pose = extract_current_end_pose(robot_state)

        plan = build_plan(config, filtered_candidates[0], current_end_pose, validate=False)
        visualization = visualize_debug(
            frame,
            filtered_candidates,
            getattr(runner, "last_debug", None),
            plan,
            visualization_config,
        )
        visualization["target_debug_image"] = target_debug_path
        visualization.update(visualization_extra)

        frame_report = {
            "color_shape": tuple(frame.color.shape),
            "depth_shape": tuple(frame.depth.shape),
        }
        print_target_report(command, target, target_filter_report)
        print_report(frame_report, grasp_report, robot_state, plan)
        validate_motion_plan(plan, config["safety"])

        recorder = record_grasp_video(
            camera,
            config.get("video_recording", {}),
            initial_frame=frame,
        )
        if recorder.path is not None:
            visualization["grasp_video"] = recorder.path
            print(f"grasp_video: {recorder.path}")
        try:
            recorder.start()
            execution_result = udp_client.execute_grasp(plan)
        finally:
            recorder.stop()
        print(f"execution_result: {execution_result}")
        return {
            "frame_report": frame_report,
            "grasp_report": grasp_report,
            "target": target,
            "target_filter_report": target_filter_report,
            "candidate": candidate_to_dict(filtered_candidates[0]),
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
    parser = argparse.ArgumentParser(description="Run one text-command VLM GRASPNT grasp.")
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
