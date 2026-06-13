# VLM Text-Command Grasp Design

## Purpose

Build a first interactive grasping version that uses a local Qwen-family vision-language model to understand a text command and select the target object in the RGB image. The system will keep the existing GraspNet, hand-eye transform, safety validation, and UDP robot execution architecture, while isolating all new work in a copied Python project directory.

The first version intentionally uses text input only. It does not add speech recognition, TTS, SAM, Grounded-SAM, or C++ protocol changes.

## Current Context

The existing stable Python project is `graspnt_rm/`. It already supports:

- RealSense capture and workspace preview.
- GraspNet inference through `GraspNetRunner`.
- Candidate filtering by score and top-down approach angle.
- Camera-frame grasp to robot base-frame pose conversion.
- Motion plan generation for `pre_grasp_pose`, `grasp_pose`, and `lift_pose`.
- Safety validation.
- UDP communication with the C++ executor.
- 2D/3D debugging visualization.

The existing C++ executor is `graspnt_robot_executor/`. It accepts the current UDP JSON grasp command and executes the robot motion. This design keeps that C++ side unchanged.

## Goals

- Copy `graspnt_rm/` to a new isolated Python project named `graspnt_vlm_rm/`.
- Add a new text-command entrypoint in the copied project.
- Call a local OpenAI-compatible Qwen VL service, such as a vLLM service, to obtain a target bbox from an RGB image and user text.
- Use the returned bbox to filter GraspNet candidates by projecting candidate centers back into the image.
- Require manual `y` confirmation before sending any grasp command to the C++ executor.
- Preserve the existing `graspnt_rm/` project as the known-good baseline.
- Preserve the existing UDP protocol and C++ robot executor.

## Non-Goals

- No speech input or TTS in the first version.
- No SAM or Grounded-SAM in the first version.
- No direct in-process loading of Qwen in the grasping Python program.
- No C++ changes.
- No changes to the stable `graspnt_rm/` directory after the copy is made.
- No automatic fallback to the global best GraspNet candidate when the target bbox contains zero candidates.

## Proposed Project Layout

Create:

```text
graspnt_vlm_rm/
```

by copying the current contents of:

```text
graspnt_rm/
```

The copied project will contain the original modules plus new VLM-specific modules:

```text
graspnt_vlm_rm/
  __init__.py
  camera_realsense.py
  config.py
  config.yaml
  graspnet_infer.py
  run_basic_grasp.py
  run_text_vlm_grasp.py
  safety.py
  target_filter.py
  transform.py
  udp_client.py
  visualization.py
  vlm_target.py
```

Responsibilities:

- `run_text_vlm_grasp.py`: new first-version entrypoint; orchestrates camera capture, text input, VLM target selection, GraspNet inference, bbox candidate filtering, visualization, manual confirmation, and UDP execution.
- `vlm_target.py`: calls the Qwen VL HTTP service and parses a strict target-selection JSON response.
- `target_filter.py`: projects GraspNet candidate centers into image pixels and filters candidates inside the VLM bbox.
- `visualization.py`: in the copied project only, may be extended to draw the VLM bbox and distinguish all candidates from bbox-filtered candidates.
- `config.yaml`: copied from the stable project, then extended with `vlm`, `target_filter`, and `interaction` sections.

## VLM Service Model

The Qwen model should run as a separate local HTTP service rather than being loaded directly inside the grasping program.

Recommended first setup:

```text
Qwen3-VL / Qwen2.5-VL local service
  <--- HTTP / OpenAI-compatible API --->
graspnt_vlm_rm.run_text_vlm_grasp
```

This keeps the GraspNet robot runtime separate from the VLM runtime. It also avoids mixing heavy model dependencies and GPU memory usage in the same Python process.

Example local endpoint:

```text
http://127.0.0.1:8000/v1/chat/completions
```

If vLLM is used with:

```bash
vllm serve "Qwen/Qwen3-VL-2B-Instruct"
```

then the default OpenAI-compatible endpoint is expected to be under:

```text
http://127.0.0.1:8000/v1
```

If the service is started with a custom port or served model name, `config.yaml` will be adjusted accordingly.

## Configuration

The copied `graspnt_vlm_rm/config.yaml` will keep all existing GraspNet, camera, hand-eye, workspace, visualization, execution, and safety settings, and add:

```yaml
vlm:
  endpoint: "http://127.0.0.1:8000/v1/chat/completions"
  model: "Qwen/Qwen3-VL-2B-Instruct"
  api_token: ""
  timeout_sec: 20
  min_confidence: 0.4

target_filter:
  mode: "bbox"
  bbox_margin_px: 8
  require_target_candidate: true

interaction:
  require_confirmation: true
```

`api_token` is optional. For a local vLLM service, it is normally empty. If a remote or protected service is used later, `vlm_target.py` can include:

```text
Authorization: Bearer <token>
```

only when `api_token` is non-empty.

## Target Selection Contract

`vlm_target.py` will ask the VLM to output one strict JSON object, without Markdown:

```json
{
  "action": "pick",
  "target_name": "red cup",
  "bbox": [100, 120, 260, 330],
  "confidence": 0.82,
  "needs_clarification": false,
  "clarification_question": "",
  "reason": "The user asked for the red cup, and the red cup is visible in the image.",
  "safety_note": ""
}
```

Validation rules:

- `action` must be `pick` for this first version.
- `bbox` must contain four numeric values.
- `x1 < x2` and `y1 < y2`.
- bbox values are clipped to the captured image dimensions.
- bbox area must be non-zero after clipping.
- `confidence` must be greater than or equal to `vlm.min_confidence`.
- If `needs_clarification` is true, the system must not execute a grasp.

If the VLM response is invalid, the entrypoint prints the raw model response and stops before GraspNet execution or robot execution.

## Candidate Filtering

GraspNet inference remains scene-level and workspace-level. It should not be restricted to the target bbox during point-cloud generation in the first version.

The filtering step is:

```text
for candidate in candidates:
    pixel = project candidate.translation with camera intrinsics
    if pixel is inside expanded bbox:
        keep candidate
```

The bbox can be expanded by `target_filter.bbox_margin_px` to tolerate small VLM localization errors.

If no candidate remains inside the target bbox and `require_target_candidate` is true, the program stops and reports:

```text
No valid GraspNet candidate projected inside the selected target bbox.
```

The first version must not silently fall back to `candidates[0]`, because that would likely grab a different object.

## Runtime Flow

The new entrypoint flow is:

```text
load config
validate config
start RealSense camera
preview workspace and capture frame
prompt user for text command
call VLM with frame.color and command
validate target selection
request current robot pose over UDP
run GraspNet inference on full workspace
filter candidates by VLM bbox
build motion plan from best filtered candidate
show or save debug visualization with bbox and candidates
print command, target, bbox, candidate counts, and poses
ask Execute grasp? [y/N]
if user enters y:
    validate motion plan
    send UDP grasp command
else:
    exit without robot motion
stop camera and close UDP socket
```

Robot execution remains exactly the same as the stable project after the selected candidate has been converted into a motion plan.

## Debug Output

The copied project should save or show enough information to diagnose failures:

- User text command.
- Raw VLM response.
- Parsed target selection.
- Captured RGB/depth shapes.
- VLM bbox.
- GraspNet candidate count before bbox filtering.
- Candidate count after bbox filtering.
- Selected candidate score and width.
- `pre_grasp_pose`, `grasp_pose`, and `lift_pose`.
- Debug image with workspace overlay, VLM bbox, candidate points, and the selected candidate.

## Error Handling

The system should stop before robot motion for these cases:

- Camera preview is cancelled.
- Empty text command.
- VLM service request times out or returns invalid JSON.
- VLM returns `needs_clarification=true`.
- VLM confidence is below threshold.
- bbox is invalid after clipping.
- GraspNet returns zero candidates.
- bbox filtering returns zero candidates.
- Motion plan fails safety validation.
- User does not explicitly enter `y`.

The system may print diagnostic details, but it should not continue to UDP execution in these cases.

## Testing Strategy

Implementation should include lightweight tests that do not require RealSense, GraspNet, Qwen, or the robot:

- bbox clipping and validation.
- VLM JSON parsing from valid and invalid model responses.
- optional authorization header construction when `api_token` is set.
- projection-based candidate filtering using fake candidates and fake camera intrinsics.
- confirmation gate behavior through a small pure function if practical.

Integration tests with actual hardware and models are manual:

1. Run with a mock VLM bbox and no UDP execution.
2. Run with real VLM service and visualization only.
3. Run with UDP pose request but decline execution at the confirmation prompt.
4. Run full flow and input `y` only after bbox and final pose look correct.

## Implementation Order

1. Copy `graspnt_rm/` to `graspnt_vlm_rm/`.
2. Add VLM target-selection dataclasses and JSON parsing.
3. Add bbox validation and clipping.
4. Add projection-based candidate filtering.
5. Add debug visualization for bbox and filtered candidates.
6. Add `run_text_vlm_grasp.py`.
7. Add config extensions.
8. Add lightweight tests.
9. Manually test with a mock bbox.
10. Manually test with a local Qwen VL service.
11. Manually test with UDP execution only after explicit `y` confirmation.

## Open Operational Notes

- The Windows GraspNet environment is the primary runtime environment.
- `requests` should be installed in that Windows Python environment.
- If WSL is used only for lightweight tests, dependencies should be installed in a conda-managed environment.
- The actual Qwen endpoint, port, and served model name will be adjusted to match the model service command used on the machine.
- `Qwen/Qwen3-VL-2B-Instruct` is acceptable for the first prototype. If bbox quality is poor, a larger Qwen VL model or a detector/SAM stage can be added later.
