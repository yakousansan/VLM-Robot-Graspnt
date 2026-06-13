import sys

import numpy as np
import pytest

import graspnt_vlm_rm.mask_target as mask_target
from graspnt_vlm_rm.mask_target import (
    MaskProposal,
    MaskTargetSelection,
    build_mask_selection_prompt,
    normalize_sam_annotations,
    parse_mask_target_selection,
    select_mask_target,
)


def test_normalize_sam_annotations_filters_and_assigns_ids():
    mask_a = np.zeros((10, 10), dtype=bool)
    mask_a[1:4, 1:4] = True
    mask_b = np.zeros((10, 10), dtype=bool)
    mask_b[5:9, 5:9] = True
    tiny = np.zeros((10, 10), dtype=bool)
    tiny[0, 0] = True

    proposals = normalize_sam_annotations(
        [
            {"segmentation": tiny, "bbox": [0, 0, 1, 1], "area": 1},
            {"segmentation": mask_b, "bbox": [5, 5, 4, 4], "area": 16},
            {"segmentation": mask_a, "bbox": [1, 1, 3, 3], "area": 9},
        ],
        image_shape=(10, 10),
        min_area=4,
        max_area_ratio=0.5,
        top_n=2,
    )

    assert [proposal.mask_id for proposal in proposals] == [1, 2]
    assert [proposal.area for proposal in proposals] == [16, 9]
    assert proposals[0].bbox == (5, 5, 8, 8)
    assert proposals[1].bbox == (1, 1, 3, 3)


def test_parse_mask_target_selection_accepts_known_mask_id():
    proposal = MaskProposal(
        mask_id=2,
        mask=np.ones((4, 4), dtype=bool),
        bbox=(0, 0, 3, 3),
        area=16,
        score=0.9,
    )
    raw = (
        '{"action":"pick","target_name":"green cap","mask_id":2,'
        '"confidence":0.85,"needs_clarification":false,"reason":"right green object",'
        '"safety_note":"clear"}'
    )

    selection = parse_mask_target_selection(raw, [proposal], min_confidence=0.4)

    assert selection == MaskTargetSelection(
        action="pick",
        target_name="green cap",
        mask_id=2,
        confidence=0.85,
        needs_clarification=False,
        clarification_question="",
        reason="right green object",
        safety_note="clear",
        raw_response=raw,
    )


def test_parse_mask_target_selection_rejects_unknown_mask_id():
    raw = (
        '{"action":"pick","target_name":"green cap","mask_id":99,'
        '"confidence":0.85,"needs_clarification":false}'
    )

    with pytest.raises(ValueError, match="mask_id"):
        parse_mask_target_selection(raw, [], min_confidence=0.4)


def test_build_mask_selection_prompt_lists_candidate_ids():
    prompt = build_mask_selection_prompt([1, 3, 5])
    assert "mask_id" in prompt
    assert "1, 3, 5" in prompt


def test_select_mask_target_posts_overlay_image_and_returns_selection(monkeypatch):
    captured = {}
    proposal = MaskProposal(
        mask_id=1,
        mask=np.ones((8, 8), dtype=bool),
        bbox=(0, 0, 7, 7),
        area=64,
        score=0.8,
    )
    raw_content = (
        '{"action":"pick","target_name":"cap","mask_id":1,'
        '"confidence":0.9,"needs_clarification":false,"reason":"matched",'
        '"safety_note":"clear"}'
    )

    class FakeResponse:
        def raise_for_status(self):
            captured["raise_for_status_called"] = True

        def json(self):
            return {"choices": [{"message": {"content": raw_content}}]}

    class FakeRequests:
        @staticmethod
        def post(endpoint, headers, json, timeout):
            captured["endpoint"] = endpoint
            captured["headers"] = headers
            captured["json"] = json
            captured["timeout"] = timeout
            return FakeResponse()

    monkeypatch.setitem(sys.modules, "requests", FakeRequests)
    monkeypatch.setattr(
        mask_target,
        "encode_bgr_image_as_data_url",
        lambda image: "data:image/jpeg;base64,overlay",
    )

    result = select_mask_target(
        overlay_bgr=np.zeros((8, 8, 3), dtype=np.uint8),
        user_command="pick the cap",
        proposals=[proposal],
        config={
            "endpoint": "http://localhost:8000/v1/chat/completions",
            "model": "model",
            "api_token": "",
            "timeout_sec": 60,
            "min_confidence": 0.4,
        },
    )

    assert result.mask_id == 1
    assert captured["raise_for_status_called"] is True
    assert captured["endpoint"] == "http://localhost:8000/v1/chat/completions"
    assert captured["json"]["model"] == "model"
    assert captured["json"]["temperature"] == 0
    assert captured["json"]["messages"][1]["content"][0]["image_url"]["url"].endswith(
        "overlay"
    )
