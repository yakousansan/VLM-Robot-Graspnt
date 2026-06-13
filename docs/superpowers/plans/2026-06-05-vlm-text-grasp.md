# VLM Text Grasp Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an isolated `graspnt_vlm_rm` Python project that uses a local Qwen VL service to select a target bbox from a text command, filters GraspNet candidates by that bbox, and requires manual confirmation before UDP robot execution.

**Architecture:** Copy the stable `graspnt_rm` package into `graspnt_vlm_rm`, then add VLM-specific modules inside the copied package only. The C++ executor and the original `graspnt_rm` package remain unchanged. The first version uses OpenAI-compatible HTTP calls to a local Qwen VL service and bbox-based candidate filtering, not SAM.

**Tech Stack:** Python, NumPy, PyYAML, requests, OpenCV for visualization, existing RealSense/GraspNet/Open3D runtime, existing UDP JSON executor.

---

## File Structure

Create or modify only these paths:

```text
graspnt_vlm_rm/                         # New copied package
graspnt_vlm_rm/config.yaml              # Copied config plus VLM sections
graspnt_vlm_rm/config.py                # Copied config validation plus new optional keys
graspnt_vlm_rm/vlm_target.py            # New VLM target selection client/parser
graspnt_vlm_rm/target_filter.py         # New bbox candidate filtering
graspnt_vlm_rm/visualization.py         # Copied visualization plus bbox overlay support
graspnt_vlm_rm/run_text_vlm_grasp.py    # New text-command entrypoint
tests/test_vlm_target.py                # New lightweight parser/validation tests
tests/test_target_filter.py             # New lightweight bbox filtering tests
```

Do not modify:

```text
graspnt_rm/
graspnt_robot_executor/
```

The repository root is not currently a valid Git repository, so commit steps are included as optional. If the user later provides a real Git repository root, execute the commit steps there.

---

### Task 1: Copy Stable Python Package

**Files:**
- Create directory: `graspnt_vlm_rm/`
- Copy from: `graspnt_rm/`
- Do not modify: `graspnt_rm/`

- [ ] **Step 1: Copy the directory**

Run:

```bash
cp -a graspnt_rm graspnt_vlm_rm
```

Expected: a new `graspnt_vlm_rm/` directory exists with the same Python files as `graspnt_rm/`.

- [ ] **Step 2: Verify copied files**

Run:

```bash
find graspnt_vlm_rm -maxdepth 1 -type f | sort
```

Expected output includes:

```text
graspnt_vlm_rm/__init__.py
graspnt_vlm_rm/camera_realsense.py
graspnt_vlm_rm/config.py
graspnt_vlm_rm/config.yaml
graspnt_vlm_rm/graspnet_infer.py
graspnt_vlm_rm/run_basic_grasp.py
graspnt_vlm_rm/safety.py
graspnt_vlm_rm/transform.py
graspnt_vlm_rm/udp_client.py
graspnt_vlm_rm/visualization.py
```

- [ ] **Step 3: Verify original package was not changed**

Run:

```bash
find graspnt_rm -maxdepth 1 -type f | sort
```

Expected: original files are still present.

- [ ] **Step 4: Optional commit**

Skip if root is still not a Git repository. If a valid repo is available:

```bash
git add graspnt_vlm_rm
git commit -m "chore: copy graspnt package for vlm prototype"
```

---

### Task 2: Add VLM Configuration

**Files:**
- Modify: `graspnt_vlm_rm/config.yaml`
- Modify: `graspnt_vlm_rm/config.py`

- [ ] **Step 1: Extend `config.yaml`**

Add these sections to the bottom of `graspnt_vlm_rm/config.yaml`:

```yaml
# 本地多模态模型服务配置
vlm:
  # OpenAI-compatible chat completions endpoint, for example vLLM.
  endpoint: "http://127.0.0.1:8000/v1/chat/completions"
  # Must match the served model id/name accepted by the VLM service.
  model: "Qwen/Qwen3-VL-2B-Instruct"
  # Empty for local vLLM. Set only when using a protected remote service.
  api_token: ""
  # HTTP request timeout in seconds.
  timeout_sec: 20
  # Below this confidence, do not execute.
  min_confidence: 0.4

# 目标过滤配置：第一版只使用 bbox，不使用 SAM。
target_filter:
  mode: "bbox"
  # Expand VLM bbox to tolerate small localization errors.
  bbox_margin_px: 8
  # If no candidate falls inside target bbox, stop instead of grabbing another object.
  require_target_candidate: true

# 人机交互安全配置
interaction:
  # First version must require explicit y before UDP execution.
  require_confirmation: true
```

- [ ] **Step 2: Extend config validation**

In `graspnt_vlm_rm/config.py`, update `validate_runtime_config` to require the new first-version keys:

```python
def validate_runtime_config(config: dict[str, Any]) -> None:
    require_keys(
        config,
        [
            "graspnet.root",
            "graspnet.checkpoint",
            "hand_eye.rotation",
            "hand_eye.translation",
            "safety.gripper_length",
            "safety.min_grasp_z",
            "vlm.endpoint",
            "vlm.model",
            "target_filter.mode",
            "interaction.require_confirmation",
        ],
    )
```

- [ ] **Step 3: Verify YAML loads**

Run:

```bash
python - <<'PY'
from graspnt_vlm_rm.config import load_config, validate_runtime_config
cfg = load_config("graspnt_vlm_rm/config.yaml")
validate_runtime_config(cfg)
print(cfg["vlm"]["endpoint"])
print(cfg["target_filter"]["mode"])
print(cfg["interaction"]["require_confirmation"])
PY
```

Expected:

```text
http://127.0.0.1:8000/v1/chat/completions
bbox
True
```

- [ ] **Step 4: Optional commit**

Skip if root is still not a Git repository. If available:

```bash
git add graspnt_vlm_rm/config.yaml graspnt_vlm_rm/config.py
git commit -m "feat: add vlm grasp configuration"
```

---

### Task 3: Implement VLM Target Selection Parser and Client

**Files:**
- Create: `graspnt_vlm_rm/vlm_target.py`
- Create: `tests/test_vlm_target.py`

- [ ] **Step 1: Write tests for JSON parsing and validation**

Create `tests/test_vlm_target.py`:

```python
import pytest

from graspnt_vlm_rm.vlm_target import (
    TargetSelection,
    build_headers,
    extract_json_object,
    parse_target_selection,
)


def test_extract_json_object_from_plain_json():
    raw = '{"action":"pick","target_name":"cup","bbox":[1,2,30,40],"confidence":0.8,"needs_clarification":false}'
    assert extract_json_object(raw)["target_name"] == "cup"


def test_extract_json_object_from_text_wrapped_response():
    raw = 'I choose the cup.\n{"action":"pick","target_name":"cup","bbox":[1,2,30,40],"confidence":0.8,"needs_clarification":false}'
    assert extract_json_object(raw)["bbox"] == [1, 2, 30, 40]


def test_parse_target_selection_clips_bbox():
    raw = '{"action":"pick","target_name":"cup","bbox":[-5,2,700,480],"confidence":0.8,"needs_clarification":false}'
    result = parse_target_selection(raw, image_width=640, image_height=480, min_confidence=0.4)
    assert result == TargetSelection(
        action="pick",
        target_name="cup",
        bbox=(0, 2, 639, 479),
        confidence=0.8,
        needs_clarification=False,
        clarification_question="",
        reason="",
        safety_note="",
        raw_response=raw,
    )


def test_parse_target_selection_rejects_low_confidence():
    raw = '{"action":"pick","target_name":"cup","bbox":[1,2,30,40],"confidence":0.1,"needs_clarification":false}'
    with pytest.raises(ValueError, match="confidence"):
        parse_target_selection(raw, image_width=640, image_height=480, min_confidence=0.4)


def test_parse_target_selection_rejects_clarification():
    raw = '{"action":"pick","target_name":"cup","bbox":[1,2,30,40],"confidence":0.8,"needs_clarification":true,"clarification_question":"Which cup?"}'
    with pytest.raises(ValueError, match="clarification"):
        parse_target_selection(raw, image_width=640, image_height=480, min_confidence=0.4)


def test_parse_target_selection_rejects_invalid_bbox():
    raw = '{"action":"pick","target_name":"cup","bbox":[30,40,1,2],"confidence":0.8,"needs_clarification":false}'
    with pytest.raises(ValueError, match="bbox"):
        parse_target_selection(raw, image_width=640, image_height=480, min_confidence=0.4)


def test_build_headers_without_token():
    assert build_headers("") == {"Content-Type": "application/json"}


def test_build_headers_with_token():
    assert build_headers("abc") == {
        "Content-Type": "application/json",
        "Authorization": "Bearer abc",
    }
```

- [ ] **Step 2: Run tests and verify they fail before implementation**

Run:

```bash
pytest tests/test_vlm_target.py -q
```

Expected: FAIL because `graspnt_vlm_rm.vlm_target` does not exist.

- [ ] **Step 3: Create `vlm_target.py`**

Create `graspnt_vlm_rm/vlm_target.py`:

```python
from __future__ import annotations

import base64
import json
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class TargetSelection:
    action: str
    target_name: str
    bbox: tuple[int, int, int, int]
    confidence: float
    needs_clarification: bool
    clarification_question: str
    reason: str
    safety_note: str
    raw_response: str


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


def _clip_int(value: Any, lower: int, upper: int) -> int:
    number = int(round(float(value)))
    return max(lower, min(upper, number))


def _parse_bbox(value: Any, image_width: int, image_height: int) -> tuple[int, int, int, int]:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise ValueError("bbox must contain four values")
    x1 = _clip_int(value[0], 0, image_width - 1)
    y1 = _clip_int(value[1], 0, image_height - 1)
    x2 = _clip_int(value[2], 0, image_width - 1)
    y2 = _clip_int(value[3], 0, image_height - 1)
    if x1 >= x2 or y1 >= y2:
        raise ValueError("bbox is invalid after clipping")
    return x1, y1, x2, y2


def parse_target_selection(
    raw_response: str,
    image_width: int,
    image_height: int,
    min_confidence: float,
) -> TargetSelection:
    payload = extract_json_object(raw_response)
    action = str(payload.get("action", "")).strip().lower()
    if action != "pick":
        raise ValueError(f"unsupported action: {action or 'empty'}")

    needs_clarification = bool(payload.get("needs_clarification", False))
    clarification_question = str(payload.get("clarification_question", "") or "")
    if needs_clarification:
        raise ValueError(f"VLM requested clarification: {clarification_question}")

    confidence = float(payload.get("confidence", 0.0))
    if confidence < float(min_confidence):
        raise ValueError(
            f"VLM confidence {confidence:.3f} is below minimum {float(min_confidence):.3f}"
        )

    bbox = _parse_bbox(payload.get("bbox"), image_width, image_height)
    target_name = str(payload.get("target_name", "") or "").strip()
    if not target_name:
        target_name = "unknown"

    return TargetSelection(
        action=action,
        target_name=target_name,
        bbox=bbox,
        confidence=confidence,
        needs_clarification=needs_clarification,
        clarification_question=clarification_question,
        reason=str(payload.get("reason", "") or ""),
        safety_note=str(payload.get("safety_note", "") or ""),
        raw_response=raw_response,
    )


def encode_bgr_image_as_data_url(image: np.ndarray) -> str:
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("opencv-python is required to encode VLM image input") from exc

    ok, buffer = cv2.imencode(".jpg", np.asarray(image, dtype=np.uint8))
    if not ok:
        raise ValueError("failed to encode image as JPEG")
    encoded = base64.b64encode(buffer).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def build_vlm_prompt(image_width: int, image_height: int) -> str:
    return (
        "You are the target selection module for a robot grasping system. "
        "Select exactly one visible object that best matches the user command. "
        "You only choose the target object; you do not generate robot motion. "
        "Return exactly one JSON object and no Markdown. "
        f"Coordinates must be integer pixel coordinates in the original image size "
        f"width={image_width}, height={image_height}. "
        'Schema: {"action":"pick","target_name":"string","bbox":[x1,y1,x2,y2],'
        '"confidence":0.0,"needs_clarification":false,'
        '"clarification_question":"","reason":"string","safety_note":"string"}. '
        "If the command is ambiguous or the target is not visible, set "
        "needs_clarification to true."
    )


def select_target(
    image_bgr: np.ndarray,
    user_command: str,
    config: dict[str, Any],
) -> TargetSelection:
    try:
        import requests
    except ImportError as exc:
        raise RuntimeError(
            "requests is required for VLM HTTP calls. Install it with "
            "`python -m pip install requests` in the runtime environment."
        ) from exc

    height, width = np.asarray(image_bgr).shape[:2]
    endpoint = str(config["endpoint"])
    model = str(config["model"])
    timeout_sec = float(config.get("timeout_sec", 20))
    min_confidence = float(config.get("min_confidence", 0.4))
    data_url = encode_bgr_image_as_data_url(image_bgr)

    messages = [
        {"role": "system", "content": build_vlm_prompt(width, height)},
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
    content = response.json()["choices"][0]["message"]["content"]
    return parse_target_selection(
        content,
        image_width=width,
        image_height=height,
        min_confidence=min_confidence,
    )
```

- [ ] **Step 4: Run tests and verify they pass**

Run:

```bash
pytest tests/test_vlm_target.py -q
```

Expected: all tests pass.

- [ ] **Step 5: Optional commit**

Skip if root is still not a Git repository. If available:

```bash
git add graspnt_vlm_rm/vlm_target.py tests/test_vlm_target.py
git commit -m "feat: add vlm target selection parser"
```

---

### Task 4: Implement Bbox Candidate Filtering

**Files:**
- Create: `graspnt_vlm_rm/target_filter.py`
- Create: `tests/test_target_filter.py`

- [ ] **Step 1: Write tests**

Create `tests/test_target_filter.py`:

```python
from dataclasses import dataclass

import numpy as np

from graspnt_vlm_rm.target_filter import (
    expand_bbox,
    filter_candidates_by_bbox,
    point_inside_bbox,
)


@dataclass(frozen=True)
class FakeIntrinsics:
    fx: float = 100.0
    fy: float = 100.0
    cx: float = 50.0
    cy: float = 50.0
    width: int = 100
    height: int = 100


@dataclass(frozen=True)
class FakeCandidate:
    translation: np.ndarray
    score: float = 1.0


def test_expand_bbox_clips_to_image():
    assert expand_bbox((5, 6, 20, 30), margin_px=10, width=40, height=50) == (0, 0, 30, 40)


def test_point_inside_bbox():
    assert point_inside_bbox((10, 10), (0, 0, 20, 20))
    assert not point_inside_bbox((21, 10), (0, 0, 20, 20))


def test_filter_candidates_by_bbox_keeps_projected_inside_candidate():
    candidates = [
        FakeCandidate(np.array([0.0, 0.0, 1.0])),   # pixel 50,50
        FakeCandidate(np.array([0.4, 0.4, 1.0])),   # pixel 90,90
        FakeCandidate(np.array([0.0, 0.0, -1.0])),  # invalid z
    ]
    kept, report = filter_candidates_by_bbox(
        candidates,
        intrinsics=FakeIntrinsics(),
        bbox=(40, 40, 60, 60),
        margin_px=0,
    )
    assert kept == [candidates[0]]
    assert report["input_count"] == 3
    assert report["kept_count"] == 1
    assert report["invalid_projection_count"] == 1


def test_filter_candidates_by_bbox_margin_can_keep_nearby_candidate():
    candidates = [FakeCandidate(np.array([0.25, 0.0, 1.0]))]  # pixel 75,50
    kept, report = filter_candidates_by_bbox(
        candidates,
        intrinsics=FakeIntrinsics(),
        bbox=(40, 40, 60, 60),
        margin_px=20,
    )
    assert kept == candidates
    assert report["expanded_bbox"] == (20, 20, 80, 80)
```

- [ ] **Step 2: Run tests and verify they fail before implementation**

Run:

```bash
pytest tests/test_target_filter.py -q
```

Expected: FAIL because `graspnt_vlm_rm.target_filter` does not exist.

- [ ] **Step 3: Create `target_filter.py`**

Create `graspnt_vlm_rm/target_filter.py`:

```python
from __future__ import annotations

from typing import Any

from graspnt_vlm_rm.visualization import project_point_to_pixel

BBox = tuple[int, int, int, int]


def expand_bbox(bbox: BBox, margin_px: int, width: int, height: int) -> BBox:
    x1, y1, x2, y2 = bbox
    margin = max(int(margin_px), 0)
    return (
        max(0, int(x1) - margin),
        max(0, int(y1) - margin),
        min(int(width) - 1, int(x2) + margin),
        min(int(height) - 1, int(y2) + margin),
    )


def point_inside_bbox(point: tuple[int, int], bbox: BBox) -> bool:
    u, v = point
    x1, y1, x2, y2 = bbox
    return x1 <= u <= x2 and y1 <= v <= y2


def filter_candidates_by_bbox(
    candidates: list[Any],
    intrinsics: Any,
    bbox: BBox,
    margin_px: int = 0,
) -> tuple[list[Any], dict[str, Any]]:
    expanded_bbox = expand_bbox(
        bbox,
        margin_px=margin_px,
        width=int(intrinsics.width),
        height=int(intrinsics.height),
    )
    kept = []
    invalid_projection_count = 0
    outside_count = 0
    for candidate in candidates:
        pixel = project_point_to_pixel(candidate.translation, intrinsics)
        if pixel is None:
            invalid_projection_count += 1
            continue
        if point_inside_bbox(pixel, expanded_bbox):
            kept.append(candidate)
        else:
            outside_count += 1
    report = {
        "input_count": int(len(candidates)),
        "kept_count": int(len(kept)),
        "outside_count": int(outside_count),
        "invalid_projection_count": int(invalid_projection_count),
        "bbox": tuple(int(value) for value in bbox),
        "expanded_bbox": expanded_bbox,
    }
    return kept, report
```

- [ ] **Step 4: Run tests and verify they pass**

Run:

```bash
pytest tests/test_target_filter.py -q
```

Expected: all tests pass.

- [ ] **Step 5: Optional commit**

Skip if root is still not a Git repository. If available:

```bash
git add graspnt_vlm_rm/target_filter.py tests/test_target_filter.py
git commit -m "feat: filter grasp candidates by vlm bbox"
```

---

### Task 5: Extend Copied Visualization with Target Overlay

**Files:**
- Modify: `graspnt_vlm_rm/visualization.py`

- [ ] **Step 1: Add bbox drawing helpers**

In `graspnt_vlm_rm/visualization.py`, add these functions near the existing 2D drawing helpers:

```python
def draw_bbox(
    image: np.ndarray,
    bbox: tuple[int, int, int, int],
    color: tuple[int, int, int] = (0, 255, 255),
    thickness: int = 2,
) -> np.ndarray:
    cv2 = _import_cv2()
    output = np.asarray(image, dtype=np.uint8).copy()
    x1, y1, x2, y2 = [int(value) for value in bbox]
    cv2.rectangle(output, (x1, y1), (x2, y2), color, int(thickness))
    return output


def build_target_debug_image(
    color: np.ndarray,
    intrinsics: Any,
    candidates: list[Any],
    filtered_candidates: list[Any],
    workspace_mask: np.ndarray | None,
    bbox: tuple[int, int, int, int],
) -> np.ndarray:
    image = build_rgb_debug_image(color, intrinsics, candidates, workspace_mask)
    image = draw_bbox(image, bbox)
    for index, candidate in enumerate(filtered_candidates):
        pixel = project_point_to_pixel(candidate.translation, intrinsics)
        if pixel is None:
            continue
        marker_color = (0, 255, 255) if index > 0 else (0, 128, 255)
        radius = 9 if index == 0 else 5
        _draw_square_marker(image, pixel, marker_color, radius=radius)
    return image
```

- [ ] **Step 2: Add a save helper for target debug images**

Add this function near `save_debug_artifacts`:

```python
def save_target_debug_image(
    frame: Any,
    all_candidates: list[Any],
    filtered_candidates: list[Any],
    debug_data: Any,
    bbox: tuple[int, int, int, int],
    config: dict[str, Any],
) -> str:
    cv2 = _import_cv2()
    debug_dir = Path(config.get("debug_dir", "debug_outputs"))
    debug_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    top_n = int(config.get("top_n", 20))
    workspace_mask = getattr(debug_data, "workspace_mask", None) if debug_data else None
    image = build_target_debug_image(
        frame.color,
        frame.intrinsics,
        all_candidates[:top_n],
        filtered_candidates[:top_n],
        workspace_mask,
        bbox,
    )
    output_path = debug_dir / f"{timestamp}_vlm_target_bbox.png"
    cv2.imwrite(str(output_path), image)
    return str(output_path)
```

- [ ] **Step 3: Verify import succeeds**

Run:

```bash
python - <<'PY'
from graspnt_vlm_rm.visualization import build_target_debug_image, draw_bbox, save_target_debug_image
print(draw_bbox.__name__)
print(build_target_debug_image.__name__)
print(save_target_debug_image.__name__)
PY
```

Expected:

```text
draw_bbox
build_target_debug_image
save_target_debug_image
```

- [ ] **Step 4: Optional commit**

Skip if root is still not a Git repository. If available:

```bash
git add graspnt_vlm_rm/visualization.py
git commit -m "feat: add vlm target debug visualization"
```

---

### Task 6: Add Text VLM Grasp Entrypoint

**Files:**
- Create: `graspnt_vlm_rm/run_text_vlm_grasp.py`

- [ ] **Step 1: Create the entrypoint**

Create `graspnt_vlm_rm/run_text_vlm_grasp.py`:

```python
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from graspnt_vlm_rm.camera_realsense import RealSenseCamera
from graspnt_vlm_rm.config import load_config, validate_runtime_config
from graspnt_vlm_rm.graspnet_infer import GraspNetRunner
from graspnt_vlm_rm.run_basic_grasp import (
    _make_udp_client,
    build_plan,
    candidate_to_dict,
    print_report,
)
from graspnt_vlm_rm.safety import validate_motion_plan
from graspnt_vlm_rm.target_filter import filter_candidates_by_bbox
from graspnt_vlm_rm.udp_client import extract_current_end_pose
from graspnt_vlm_rm.visualization import (
    preview_workspace,
    save_target_debug_image,
    visualize_debug,
)
from graspnt_vlm_rm.vlm_target import TargetSelection, select_target


def ask_text_command() -> str:
    command = input("Text grasp command: ").strip()
    if not command:
        raise RuntimeError("empty text command")
    return command


def confirm_execution(config: dict[str, Any]) -> bool:
    if not bool(config.get("interaction", {}).get("require_confirmation", True)):
        return True
    answer = input("Execute grasp? [y/N]: ").strip().lower()
    return answer == "y"


def print_target_report(
    command: str,
    target: TargetSelection,
    target_filter_report: dict[str, Any],
) -> None:
    print(f"user_command: {command}")
    print(f"target_name: {target.target_name}")
    print(f"target_bbox: {target.bbox}")
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
        target = select_target(frame.color, command, config["vlm"])

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

        target_filter_config = config.get("target_filter", {})
        filtered_candidates, target_filter_report = filter_candidates_by_bbox(
            candidates,
            frame.intrinsics,
            target.bbox,
            margin_px=int(target_filter_config.get("bbox_margin_px", 0)),
        )
        if not filtered_candidates:
            if bool(target_filter_config.get("require_target_candidate", True)):
                raise RuntimeError(
                    "No valid GraspNet candidate projected inside the selected target bbox."
                )
            filtered_candidates = candidates

        plan = build_plan(config, filtered_candidates[0], current_end_pose, validate=False)
        visualization = visualize_debug(
            frame,
            filtered_candidates,
            getattr(runner, "last_debug", None),
            plan,
            config.get("visualization", {}),
        )
        target_debug_path = save_target_debug_image(
            frame,
            candidates,
            filtered_candidates,
            getattr(runner, "last_debug", None),
            target.bbox,
            config.get("visualization", {}),
        )
        visualization["target_debug_image"] = target_debug_path

        frame_report = {
            "color_shape": tuple(frame.color.shape),
            "depth_shape": tuple(frame.depth.shape),
        }
        print_target_report(command, target, target_filter_report)
        print_report(frame_report, grasp_report, robot_state, plan)
        validate_motion_plan(plan, config["safety"])

        if not confirm_execution(config):
            print("execution_result: skipped_by_user")
            return {
                "frame_report": frame_report,
                "grasp_report": grasp_report,
                "target": target,
                "target_filter_report": target_filter_report,
                "candidate": candidate_to_dict(filtered_candidates[0]),
                "robot_state": robot_state,
                "plan": plan,
                "visualization": visualization,
                "execution_result": {"status": "skipped_by_user"},
            }

        execution_result = udp_client.execute_grasp(plan)
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
```

- [ ] **Step 2: Verify entrypoint imports**

Run:

```bash
python - <<'PY'
from graspnt_vlm_rm.run_text_vlm_grasp import ask_text_command, confirm_execution, run
print(run.__name__)
PY
```

Expected:

```text
run
```

- [ ] **Step 3: Optional commit**

Skip if root is still not a Git repository. If available:

```bash
git add graspnt_vlm_rm/run_text_vlm_grasp.py
git commit -m "feat: add text vlm grasp entrypoint"
```

---

### Task 7: Run Lightweight Test Suite

**Files:**
- Test only

- [ ] **Step 1: Run parser tests**

Run:

```bash
pytest tests/test_vlm_target.py -q
```

Expected: all tests pass.

- [ ] **Step 2: Run filter tests**

Run:

```bash
pytest tests/test_target_filter.py -q
```

Expected: all tests pass.

- [ ] **Step 3: Run all lightweight tests**

Run:

```bash
pytest tests/test_vlm_target.py tests/test_target_filter.py -q
```

Expected: all tests pass.

- [ ] **Step 4: Optional commit**

Skip if root is still not a Git repository. If available:

```bash
git add tests
git commit -m "test: cover vlm target parsing and bbox filtering"
```

---

### Task 8: Manual Mock-Bbox Dry Run

**Files:**
- Temporarily inspect only; do not commit test hacks

- [ ] **Step 1: Add a temporary mock branch only if the VLM service is not ready**

If the Qwen service is unavailable, temporarily replace the `target = select_target(...)` line in `run_text_vlm_grasp.py` during local testing only with:

```python
from graspnt_vlm_rm.vlm_target import TargetSelection

target = TargetSelection(
    action="pick",
    target_name="mock target",
    bbox=(100, 100, 400, 400),
    confidence=1.0,
    needs_clarification=False,
    clarification_question="",
    reason="manual mock bbox for dry run",
    safety_note="",
    raw_response="mock",
)
```

Do not commit this temporary change.

- [ ] **Step 2: Run the new entrypoint and decline execution**

Run in the Windows GraspNet runtime environment:

```bash
python -m graspnt_vlm_rm.run_text_vlm_grasp graspnt_vlm_rm/config.yaml
```

Expected:

- Camera preview opens.
- After pressing Space, text command prompt appears.
- GraspNet runs.
- Candidate counts print.
- Debug files are saved.
- When prompted `Execute grasp? [y/N]:`, entering `n` exits without robot motion.

- [ ] **Step 3: Remove temporary mock branch**

Restore the real `select_target(...)` line before continuing.

---

### Task 9: Manual Real-VLM Dry Run

**Files:**
- Runtime only

- [ ] **Step 1: Start local Qwen VL service**

Example with vLLM:

```bash
vllm serve "Qwen/Qwen3-VL-2B-Instruct"
```

Expected: service starts and listens on port `8000`.

- [ ] **Step 2: Check model list**

Run from the environment that will call the service:

```bash
curl http://127.0.0.1:8000/v1/models
```

Expected: response includes a model id accepted by the service. If the id differs from `Qwen/Qwen3-VL-2B-Instruct`, update `graspnt_vlm_rm/config.yaml`.

- [ ] **Step 3: Run the new entrypoint and decline execution**

Run:

```bash
python -m graspnt_vlm_rm.run_text_vlm_grasp graspnt_vlm_rm/config.yaml
```

Expected:

- VLM raw response parses into bbox.
- Debug target image shows the bbox on the intended object.
- Candidate filtering count is non-zero for reachable scenes.
- Entering `n` at confirmation skips execution.

---

### Task 10: Manual Full Execution

**Files:**
- Runtime only

- [ ] **Step 1: Start C++ executor**

Run the existing `graspnt_robot_executor` exactly as in the stable Windows setup.

Expected: C++ UDP server is listening on the configured port, usually `6556`.

- [ ] **Step 2: Start local Qwen VL service**

Run the selected vLLM service command.

Expected: `/v1/chat/completions` is reachable from Windows Python.

- [ ] **Step 3: Run text VLM grasp**

Run:

```bash
python -m graspnt_vlm_rm.run_text_vlm_grasp graspnt_vlm_rm/config.yaml
```

Expected before execution:

- Target bbox is correct.
- Filtered candidate count is greater than zero.
- `pre_grasp_pose`, `grasp_pose`, and `lift_pose` pass safety validation.
- Debug image confirms selected candidate is on the requested object.

- [ ] **Step 4: Execute only after explicit confirmation**

At:

```text
Execute grasp? [y/N]:
```

enter:

```text
y
```

Expected: UDP command is accepted by the C++ executor and the result is printed.

---

## Plan Self-Review Checklist

- Spec coverage: The plan covers project copy, VLM service call, bbox parsing, bbox filtering, visualization, manual confirmation, tests, and manual hardware integration.
- Placeholder scan: The plan contains concrete file paths, commands, and code snippets instead of deferred work notes.
- Type consistency: `TargetSelection`, bbox tuple types, config keys, and function names are consistent across tasks.
- Scope: The plan does not add speech, SAM, C++ changes, or changes to original `graspnt_rm/`.
