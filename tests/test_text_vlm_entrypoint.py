from pathlib import Path


def test_text_vlm_entrypoint_has_no_bbox_target_mode():
    source = Path("graspnt_vlm_rm/run_text_vlm_grasp.py").read_text(encoding="utf-8")

    assert "filter_candidates_by_bbox" not in source
    assert "select_target" not in source
    assert "target_mode" not in source
    assert "target_filter.mode must be" not in source


def test_text_vlm_entrypoint_has_no_python_execution_confirmation():
    source = Path("graspnt_vlm_rm/run_text_vlm_grasp.py").read_text(encoding="utf-8")

    assert "confirm_execution" not in source
    assert "Execute grasp?" not in source
    assert "skipped_by_user" not in source


def test_text_vlm_entrypoint_records_video_around_udp_execution():
    source = Path("graspnt_vlm_rm/run_text_vlm_grasp.py").read_text(encoding="utf-8")

    recorder_index = source.index("record_grasp_video(")
    start_index = source.index("recorder.start()")
    execute_index = source.index("udp_client.execute_grasp(plan)")
    stop_index = source.index("recorder.stop()")

    assert recorder_index < start_index < execute_index < stop_index
    assert "video_recording" in source
    assert "grasp_video" in source
