from __future__ import annotations

import base64
import json
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class MaskProposal:
    mask_id: int
    mask: np.ndarray
    bbox: tuple[int, int, int, int]
    area: int
    score: float


@dataclass(frozen=True)
class MaskTargetSelection:
    action: str
    target_name: str
    mask_id: int
    confidence: float
    needs_clarification: bool
    clarification_question: str
    reason: str
    safety_note: str
    raw_response: str


def _import_cv2():
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("opencv-python is required for mask target visualization") from exc
    return cv2


def build_headers(api_token: str | None) -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    token = str(api_token or "").strip()
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def extract_json_object(content: str) -> dict[str, Any]:
    text = str(content).strip()
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end < start:
        raise ValueError("VLM response does not contain a JSON object")
    try:
        value = json.loads(text[start : end + 1])
    except json.JSONDecodeError as exc:
        raise ValueError("VLM response JSON is invalid") from exc
    if not isinstance(value, dict):
        raise ValueError("VLM response JSON must be an object")
    return value


def extract_chat_content(response_payload: dict[str, Any]) -> str:
    message = "VLM service returned an unexpected response shape"
    if not isinstance(response_payload, dict):
        raise ValueError(message)

    choices = response_payload.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ValueError(message)

    first_choice = choices[0]
    if not isinstance(first_choice, dict):
        raise ValueError(message)

    chat_message = first_choice.get("message")
    if not isinstance(chat_message, dict):
        raise ValueError(message)

    content = chat_message.get("content")
    if not isinstance(content, str):
        raise ValueError(message)

    return content


def _annotation_bbox_to_xyxy(annotation: dict[str, Any], mask: np.ndarray) -> tuple[int, int, int, int]:
    bbox = annotation.get("bbox")
    if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
        x, y, width, height = [int(round(float(value))) for value in bbox]
        return x, y, max(x, x + width - 1), max(y, y + height - 1)

    ys, xs = np.where(mask)
    if xs.size == 0 or ys.size == 0:
        raise ValueError("mask has zero area")
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def normalize_sam_annotations(
    annotations: list[dict[str, Any]],
    image_shape: tuple[int, int],
    min_area: int,
    max_area_ratio: float,
    top_n: int,
) -> list[MaskProposal]:
    image_height, image_width = image_shape
    max_area = int(image_height * image_width * float(max_area_ratio))
    proposals = []
    for annotation in annotations:
        mask = np.asarray(annotation.get("segmentation"), dtype=bool)
        if mask.shape != (image_height, image_width):
            continue
        area = int(annotation.get("area", int(mask.sum())))
        if area < int(min_area) or area > max_area:
            continue
        try:
            bbox = _annotation_bbox_to_xyxy(annotation, mask)
        except (TypeError, ValueError):
            continue
        score = float(
            annotation.get(
                "predicted_iou",
                annotation.get("stability_score", 0.0),
            )
            or 0.0
        )
        proposals.append(
            MaskProposal(
                mask_id=0,
                mask=mask,
                bbox=bbox,
                area=area,
                score=score,
            )
        )

    proposals.sort(key=lambda proposal: (proposal.area, proposal.score), reverse=True)
    limited = proposals[: max(0, int(top_n))]
    return [
        MaskProposal(
            mask_id=index + 1,
            mask=proposal.mask,
            bbox=proposal.bbox,
            area=proposal.area,
            score=proposal.score,
        )
        for index, proposal in enumerate(limited)
    ]


def build_mask_selection_overlay(
    color_bgr: np.ndarray,
    proposals: list[MaskProposal],
    alpha: float = 0.45,
) -> np.ndarray:
    cv2 = _import_cv2()
    output = np.asarray(color_bgr, dtype=np.uint8).copy()
    palette = [
        (0, 255, 255),
        (0, 128, 255),
        (255, 128, 0),
        (255, 0, 255),
        (0, 255, 0),
        (255, 255, 0),
        (128, 0, 255),
        (0, 0, 255),
    ]
    for proposal in proposals:
        color = np.asarray(palette[(proposal.mask_id - 1) % len(palette)], dtype=np.float32)
        mask = np.asarray(proposal.mask, dtype=bool)
        output_float = output.astype(np.float32)
        output_float[mask] = output_float[mask] * (1.0 - alpha) + color * alpha
        output = output_float.astype(np.uint8)
        x1, y1, x2, y2 = proposal.bbox
        cv2.rectangle(output, (x1, y1), (x2, y2), tuple(int(v) for v in color), 2)
        label = str(proposal.mask_id)
        cv2.putText(
            output,
            label,
            (max(0, x1), max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            output,
            label,
            (max(0, x1), max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )
    return output


def build_selected_mask_debug_image(
    color_bgr: np.ndarray,
    proposal: MaskProposal,
    alpha: float = 0.5,
) -> np.ndarray:
    cv2 = _import_cv2()
    output = np.asarray(color_bgr, dtype=np.uint8).copy()
    mask = np.asarray(proposal.mask, dtype=bool)
    color = np.asarray((0, 255, 255), dtype=np.float32)
    output_float = output.astype(np.float32)
    output_float[mask] = output_float[mask] * (1.0 - alpha) + color * alpha
    output = output_float.astype(np.uint8)
    x1, y1, x2, y2 = proposal.bbox
    cv2.rectangle(output, (x1, y1), (x2, y2), (0, 255, 255), 2)
    cv2.putText(
        output,
        f"mask {proposal.mask_id}",
        (max(0, x1), max(20, y1 - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 0),
        3,
        cv2.LINE_AA,
    )
    cv2.putText(
        output,
        f"mask {proposal.mask_id}",
        (max(0, x1), max(20, y1 - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return output


def save_mask_debug_images(
    color_bgr: np.ndarray,
    overlay_bgr: np.ndarray,
    selected_proposal: MaskProposal,
    config: dict[str, Any],
) -> dict[str, str]:
    cv2 = _import_cv2()
    debug_dir = Path(config.get("debug_dir", "debug_outputs"))
    debug_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    overlay_path = debug_dir / f"{timestamp}_sam_masks.png"
    selected_path = debug_dir / f"{timestamp}_selected_mask.png"
    selected_image = build_selected_mask_debug_image(color_bgr, selected_proposal)
    cv2.imwrite(str(overlay_path), overlay_bgr)
    cv2.imwrite(str(selected_path), selected_image)
    return {"sam_masks": str(overlay_path), "selected_mask": str(selected_path)}


def save_mask_overlay_image(
    overlay_bgr: np.ndarray,
    config: dict[str, Any],
) -> str:
    cv2 = _import_cv2()
    debug_dir = Path(config.get("debug_dir", "debug_outputs"))
    debug_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    overlay_path = debug_dir / f"{timestamp}_sam_masks.png"
    cv2.imwrite(str(overlay_path), np.asarray(overlay_bgr, dtype=np.uint8))
    return str(overlay_path)


def encode_bgr_image_as_data_url(image: np.ndarray) -> str:
    cv2 = _import_cv2()
    ok, buffer = cv2.imencode(".jpg", np.asarray(image, dtype=np.uint8))
    if not ok:
        raise ValueError("failed to encode mask overlay image as JPEG")
    encoded = base64.b64encode(buffer).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def build_mask_selection_prompt(mask_ids: list[int]) -> str:
    ids = ", ".join(str(mask_id) for mask_id in mask_ids)
    return (
        "You are the target selection module for a robot grasping system. "
        "The image contains colored instance masks with visible numeric labels. "
        "Select exactly one mask id that best matches the user command. "
        f"Allowed mask ids: {ids}. "
        "Return exactly one JSON object and no Markdown. "
        'Schema: {"action":"pick","target_name":"string","mask_id":1,'
        '"confidence":0.0,"needs_clarification":false,'
        '"clarification_question":"","reason":"string","safety_note":"string"}. '
        "If no numbered mask matches the command, set needs_clarification to true."
    )


def parse_mask_target_selection(
    raw_response: str,
    proposals: list[MaskProposal],
    min_confidence: float,
) -> MaskTargetSelection:
    payload = extract_json_object(raw_response)
    action = str(payload.get("action", "")).strip().lower()
    if action != "pick":
        raise ValueError(f"unsupported action: {action or 'empty'}")

    needs_clarification = bool(payload.get("needs_clarification", False))
    clarification_question = str(payload.get("clarification_question", "") or "")
    if needs_clarification:
        raise ValueError(f"VLM requested clarification: {clarification_question}")

    confidence = float(payload.get("confidence", 0.0))
    if not math.isfinite(confidence) or confidence < 0.0 or confidence > 1.0:
        raise ValueError("VLM confidence must be finite and within [0.0, 1.0]")
    min_confidence_value = float(min_confidence)
    if confidence < min_confidence_value:
        raise ValueError(
            f"VLM confidence {confidence:.3f} is below minimum {min_confidence_value:.3f}"
        )

    try:
        mask_id = int(payload.get("mask_id"))
    except (TypeError, ValueError) as exc:
        raise ValueError("mask_id must be an integer") from exc
    allowed_ids = {proposal.mask_id for proposal in proposals}
    if mask_id not in allowed_ids:
        raise ValueError(f"mask_id {mask_id} is not in available mask ids {sorted(allowed_ids)}")

    target_name = str(payload.get("target_name", "") or "").strip() or f"mask_{mask_id}"
    return MaskTargetSelection(
        action=action,
        target_name=target_name,
        mask_id=mask_id,
        confidence=confidence,
        needs_clarification=needs_clarification,
        clarification_question=clarification_question,
        reason=str(payload.get("reason", "") or ""),
        safety_note=str(payload.get("safety_note", "") or ""),
        raw_response=raw_response,
    )


def select_mask_target(
    overlay_bgr: np.ndarray,
    user_command: str,
    proposals: list[MaskProposal],
    config: dict[str, Any],
) -> MaskTargetSelection:
    try:
        import requests
    except ImportError as exc:
        raise RuntimeError(
            "requests is required for VLM HTTP calls. Install it with "
            "`python -m pip install requests` in the runtime environment."
        ) from exc

    if not proposals:
        raise ValueError("at least one mask proposal is required")
    endpoint = str(config["endpoint"])
    model = str(config["model"])
    timeout_sec = float(config.get("timeout_sec", 20))
    min_confidence = float(config.get("min_confidence", 0.4))
    data_url = encode_bgr_image_as_data_url(overlay_bgr)
    mask_ids = [proposal.mask_id for proposal in proposals]

    messages = [
        {"role": "system", "content": build_mask_selection_prompt(mask_ids)},
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": data_url}},
                {"type": "text", "text": str(user_command)},
            ],
        },
    ]
    request_payload = {"model": model, "messages": messages, "temperature": 0}
    response = requests.post(
        endpoint,
        headers=build_headers(config.get("api_token", "")),
        json=request_payload,
        timeout=timeout_sec,
    )
    response.raise_for_status()
    try:
        response_payload = response.json()
    except ValueError as exc:
        raise ValueError("VLM service returned an unexpected response shape") from exc
    content = extract_chat_content(response_payload)
    return parse_mask_target_selection(content, proposals, min_confidence)


def get_proposal_by_id(proposals: list[MaskProposal], mask_id: int) -> MaskProposal:
    for proposal in proposals:
        if proposal.mask_id == mask_id:
            return proposal
    raise ValueError(f"mask_id {mask_id} is not available")


def generate_sam_mask_proposals(
    color_bgr: np.ndarray,
    config: dict[str, Any],
) -> list[MaskProposal]:
    try:
        import torch
        from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
    except ImportError as exc:
        raise RuntimeError(
            "SAM mode requires `segment-anything` and torch. Install them in the "
            "graspnet environment before using text-command SAM grasping."
        ) from exc

    cv2 = _import_cv2()
    model_type = str(config.get("model_type", "vit_b"))
    checkpoint = str(config["checkpoint"])
    device = str(config.get("device", "cuda"))
    if model_type not in sam_model_registry:
        raise ValueError(f"unsupported SAM model_type: {model_type}")

    sam = sam_model_registry[model_type](checkpoint=checkpoint)
    if device:
        sam.to(device=device)
    generator = SamAutomaticMaskGenerator(
        sam,
        points_per_side=int(config.get("points_per_side", 16)),
        pred_iou_thresh=float(config.get("pred_iou_thresh", 0.88)),
        stability_score_thresh=float(config.get("stability_score_thresh", 0.9)),
        min_mask_region_area=int(config.get("min_mask_region_area", 100)),
    )
    rgb = cv2.cvtColor(np.asarray(color_bgr, dtype=np.uint8), cv2.COLOR_BGR2RGB)
    annotations = generator.generate(rgb)
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()
    return normalize_sam_annotations(
        annotations,
        image_shape=rgb.shape[:2],
        min_area=int(config.get("min_area", 200)),
        max_area_ratio=float(config.get("max_area_ratio", 0.35)),
        top_n=int(config.get("top_n", 12)),
    )
