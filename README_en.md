# GRASPNT — VLM-Guided Interactive Robotic Grasping

<p align="center">
  <img src="https://img.shields.io/badge/platform-Windows%20%7C%20Linux-lightgrey" alt="Platform">
  <img src="https://img.shields.io/badge/python-3.8%2B-blue" alt="Python">
  <img src="https://img.shields.io/badge/C%2B%2B-17-blue" alt="C++">
  <img src="https://img.shields.io/badge/PyTorch-2.3%2B-red" alt="PyTorch">
</p>

<p align="center">
  <a href="README.md">简体中文</a> | <b>English</b>
</p>

---

### Overview

GRASPNT is a multimodal interactive robotic grasping system that enables a robot arm to understand natural language commands — "grasp the green object on the right" — and execute the corresponding 6-DoF grasp. It combines a Vision-Language Model (VLM), Segment Anything (SAM), GraspNet, and RGB-D point cloud processing to bridge semantic understanding and geometric reasoning.

### Table of Contents

- [Overview](#overview)
- [Pipeline](#pipeline)
- [Architecture](#architecture)
- [Features](#features)
- [Demo](#demo)
- [Output Gallery](#output-gallery)
- [Dependencies](#dependencies)
- [Environment Setup](#environment-setup)
- [Installation](#installation)
- [Before You Run](#before-you-run)
- [Usage](#usage)
- [Execution Logs](#execution-logs)
- [Project Structure](#project-structure)
- [Configuration](#configuration)
- [Troubleshooting](#troubleshooting)
- [License](#license)

---

### Pipeline

```
User: "grasp the green object on the right"
           │
           ▼
   ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
   │  RealSense   │────▶│     SAM      │────▶│  VLM (Qwen)  │
   │  RGB-D Frame │     │  Mask Proposals   │  Select mask_id  │
   └──────────────┘     └──────────────┘     └──────┬───────┘
                                                     │
                    ┌────────────────────────────────┘
                    ▼
   ┌─────────────────────────────────────────────────────┐
   │              Point Cloud Separation                  │
   │  Target mask → candidate point cloud (for GraspNet)  │
   │  Full scene  → collision point cloud (for safety)    │
   └────────────────────┬────────────────────────────────┘
                        ▼
   ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
   │   GraspNet   │────▶│  Collision   │────▶│  Hand-Eye    │
   │  6D Candidates    │  Detection   │     │  Transform   │
   └──────────────┘     └──────────────┘     └──────┬───────┘
                                                     │
                                                     ▼
   ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
   │  C++ Robot   │◀────│  UDP Client  │◀────│   Safety     │
   │  Executor    │     │  (Python)    │     │  Validation  │
   └──────────────┘     └──────────────┘     └──────────────┘
```

### Architecture

#### Python Side (Perception & Planning)

| Module | Role | Key Technology |
|--------|------|---------------|
| **VLM Target Selection** | Understand natural language, pick target from numbered mask overlay | [Qwen3-VL-2B-Instruct-FP8](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct-FP8) (via vLLM) |
| **SAM Instance Segmentation** | Generate pixel-level candidate masks of all visible objects | SAM (vit_h) |
| **GraspNet 6-DoF Grasping** | Generate grasp poses from target point cloud only | GraspNet-baseline |
| **Collision Detection** | Filter unsafe grasps using full scene point cloud | Model-Free Collision Detector |
| **Hand-Eye Calibration** | Transform camera-frame grasps to robot base coordinates | `transform.py` |
| **Safety Validation** | Workspace bounds, minimum height, pre-grasp/lift offsets | `safety.py` |
| **Video Recording** | Capture the full grasp execution for debugging | OpenCV VideoWriter |

#### C++ Executor (Motion Control)

```
Start → Connect robot arm → Home → UDP listen loop
                                          │
                    ┌─────────────────────┘
                    ▼
          Receive UDP command
          ╔═══════════════════╗
          ║  pose_request     ║──→ Return current end-effector pose + joint angles
          ║  grasp_execute    ║──→ Print plan → IK reachability check
          ╚═══════════════════╝         │
                                        ▼
                              "Execute this grasp? [y/N]"
                                   │            │
                                   y            N
                                   ▼            ▼
                    pre_grasp → grasp →     Send "rejected"
                    close gripper → lift →
                    home → Send "done"
```

| Component | Role | Key Technology |
|-----------|------|---------------|
| **Robot Driver** | Arm connection, motion control (joint/linear) | RealMan SDK (`api_cpp`) |
| **UDP Server** | Receive commands from Python, return results | Winsock2 (Windows) |
| **Protocol Parser** | JSON command parsing and serialization | nlohmann/json |
| **Safety Checker** | C++-side IK reachability and robot state validation | RealMan SDK IK |
| **Grasp Executor** | Orchestrates pre_grasp → grasp → lift → home sequence | — |

> **Dual-layer safety**: Python validates geometry and workspace boundaries; C++ validates IK reachability and robot state. C++ also requires a human operator to type `y` before any motion — this is the final safety gate.

### Features

- **Natural Language Grounding**: Describe what to grasp in plain language — "the cup", "the leftmost object", "the green box on the right"
- **Mask-Based Target Selection**: SAM generates numbered instance masks; VLM only picks a number — more stable and accurate than direct bbox output
- **Target-Scene Point Cloud Separation**: GraspNet generates candidates exclusively on the target region; collision detection uses the full workspace — prevents grabbing wrong objects
- **6-DoF Grasp Pose Estimation**: GraspNet-based inference with collision detection, NMS, top-down angle filter, and score sorting
- **Python/C++ Separation**: Python handles perception and planning; C++ handles real-time motion execution — communicating via UDP
- **Dual-Layer Safety**: Python checks workspace bounds, min height, pre-grasp/lift offsets; C++ checks IK reachability and robot state
- **Human-in-the-Loop**: C++ executor requires local console `[y/N]` confirmation before any motion — operator has final authority
- **Automatic Video Recording**: Records the entire grasp execution (approach → close → lift) for offline analysis
- **Rich Debug Visualization**: 2D RGB/depth overlay, 3D point cloud with gripper wireframes, SAM mask overlay, selected mask highlight, debug file export (PNG, MP4)

### Demo

https://github.com/user-attachments/assets/ec0aad5c-70b2-47q6-9d14-07d6c1aa174d

### Output Gallery

Each text-command grasp produces the following debug outputs:

| Stage | Output | Description |
|-------|--------|-------------|
| 1 | `*_depth.png` | Depth map visualized as JET colormap |
| 2 | `*_rgb_grasp.png` | Final grasp visualization — workspace overlay + grasp pose markers (green = best candidate) |
| 3 | `*_sam_masks.png` | SAM candidate mask overlay — all proposals with colored regions and numbered labels (sent to VLM) |
| 4 | `*_selected_mask.png` | VLM-selected target mask — highlighted with bounding box and "mask N" label |
| 5 | `*_grasp.mp4` | Full execution video recording |

### Dependencies

#### Python

| Dependency | Purpose |
|-----------|---------|
| Python 3.8+ (3.10 recommended) | |
| PyTorch 2.3.1+ | GraspNet model inference |
| Open3D | Point cloud processing, 3D visualization |
| OpenCV | Image processing, video recording |
| NumPy, SciPy | Numerical computation, rotation transforms |
| pyrealsense2 | Intel RealSense D435i driver |
| segment-anything | SAM instance segmentation |
| graspnet-baseline | GraspNet model, dataset utilities, collision detection |
| graspnetAPI | GraspNet data structures (GraspGroup) |
| requests | HTTP client for VLM API |
| vLLM (or compatible) | OpenAI-compatible VLM inference endpoint |

#### C++

| Dependency | Purpose |
|-----------|---------|
| C++17 | |
| CMake 3.16+ | Build system |
| RealMan SDK (`api_cpp`) | ECO65-6F arm control (motion, gripper, IK) |
| nlohmann/json | UDP protocol JSON parsing |
| Winsock2 | UDP socket communication (Windows) |

### Environment Setup

```bash
# Create and activate conda environment
conda create -n graspnt python=3.10
conda activate graspnt

# Install PyTorch (adjust CUDA version as needed)
conda install pytorch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 pytorch-cuda=12.1 -c pytorch -c nvidia

# Install core dependencies
pip install opencv-python open3d pyrealsense2 numpy scipy requests

# Install SAM
pip install git+https://github.com/facebookresearch/segment-anything.git
```

### Installation

#### 1. GraspNet-baseline

> For a detailed step-by-step Windows installation guide (Chinese), see: [GraspNet Baseline Windows 复现指南](https://blog.csdn.net/SWORDHOLDER/article/details/159793585)

```bash
# Clone GraspNet-baseline
git clone https://github.com/graspnet/graspnet-baseline.git
cd graspnet-baseline
# Comment out torch in requirements.txt first, then install other deps
pip install -r requirements.txt

# Compile pointnet2 (REQUIRES VS2019 — VS2022/VS2026 do NOT work)
cd pointnet2
python setup.py install
cd ..

# Compile knn
cd knn
# IMPORTANT: replace all "long" with "int64_t" and add #include <cstdint> before compiling
python setup.py install
cd ..
```

Key notes:
- **VS2019 is required** — multiple VS versions can coexist on the same machine.
- **knn compile fix**: Replace all `long` → `int64_t` and add `#include <cstdint>` to resolve `LNK2001`.
- **PyTorch 1.9+**: Replace `torch._six.container_abcs` → `collections.abc`.

**Verify GraspNet installation:**

```bash
cd graspnet-baseline
python demo.py --checkpoint_path checkpoint-rs.tar
```

A 3D point cloud window with grasp gripper wireframes should appear — GraspNet is working.

#### 2. graspnetAPI

```bash
git clone https://github.com/graspnet/graspnetAPI.git
cd graspnetAPI
# Edit setup.py: change "sklearn" to "scikit-learn", then install
pip install .
cd ..
```

#### 3. Model Weights

Download checkpoint weights and place under `graspnet-baseline/`:

| Checkpoint | Camera |
|-----------|--------|
| `checkpoint-rs.tar` | RealSense D435i |
| `checkpoint-kn.tar` | Kinect |

Download link (Google Drive / Baidu Pan) — see [graspnet-baseline README](https://github.com/graspnet/graspnet-baseline).

#### 4. SAM

Download the SAM checkpoint (e.g., `sam_vit_h_4b8939.pth`) and update the `sam.checkpoint` path in `config.yaml`.

#### 5. VLM Service

Start the VLM inference service with [Qwen3-VL-2B-Instruct-FP8](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct-FP8) via vLLM:

```bash
vllm serve path/to/Qwen3-VL-2B-Instruct-FP8 \
  --served-model-name Qwen/Qwen3-VL-2B-Instruct-FP8 \
  --port 8000
```

#### 6. C++ Robot Executor

```bash
cd graspnt_robot_executor
mkdir build && cd build
cmake .. -DROBOTIC_ARM_DIR=path/to/Robotic_Arm
cmake --build . --config Release
```

The CMake expects this **3rdparty directory structure**:

```
graspnt_robot_executor/
└── 3rdparty/
    └── Robotic_Arm/          # RealMan SDK root
        ├── include/
        │   └── rm_service.h
        └── lib/
            ├── api_cpp.lib
            └── api_cpp.dll
```

CMake handles:
- Finding `api_cpp` library and header
- Linking nlohmann/json (or using header-only fallback)
- Linking Winsock2 on Windows
- Auto-copying `api_cpp.dll` to output directory

### Before You Run

Update these items in `graspnt_vlm_rm/config.yaml` before your first run:

- [ ] `graspnet.root` → path to your `graspnet-baseline` directory
- [ ] `graspnet.checkpoint` → path to your pretrained weights (`checkpoint-rs.tar`)
- [ ] `sam.checkpoint` → path to your SAM weights (`sam_vit_h_4b8939.pth`)
- [ ] `camera.serial` → your RealSense D435i serial number
- [ ] `hand_eye.rotation` / `hand_eye.translation` → your hand-eye calibration results
- [ ] `safety.gripper_length` → TCP-to-contact-point distance for your gripper
- [ ] `vlm.endpoint` → your VLM inference service URL
- [ ] `vlm.model` → model name matching your VLM service
- [ ] `execution.udp_host` / `execution.udp_port` → must match C++ executor

In `graspnt_robot_executor/src/main.cpp`:

- [ ] `robot_ip` → RealMan controller IP
- [ ] `robot_port` → RealMan controller port (default: `8080`)
- [ ] `udp_port` → UDP listen port (default: `6556`)

### Usage

1. **Start the C++ executor** (connects robot, homes, listens on UDP):

   ```
   .\graspnt_robot_executor.exe
   ```

2. **Run the text-command grasping script:**

   ```bash
   python -m graspnt_vlm_rm.run_text_vlm_grasp config.yaml
   ```

3. **Workflow:**

   - Python opens live camera preview with workspace mask overlay
   - Press **Space** to confirm the scene and capture a frame
   - Python requests current robot pose from C++ via UDP (`pose_request`)
   - Python runs SAM → VLM target selection → GraspNet inference
   - Python displays 2D/3D debug visualizations
   - Python sends `grasp_execute` (pre_grasp / grasp / lift poses) to C++ via UDP
   - C++ prints the plan and asks **`Execute this grasp? [y/N]`**
   - Type **y** — robot executes: pre_grasp → grasp → close gripper → lift → home
   - C++ returns `done`; video recording stops

4. **Enter a natural language command**, for example:

   ```
   grasp the green object on the right
   the cup on the left
   the largest box in the center
   ```

### Execution Logs

#### Python Side

```
frame: {'color_shape': (480, 640, 3), 'depth_shape': (480, 640)}
grasp: {'valid_workspace_points': 101364, 'candidate_count': 9}
target_cloud: {
  "target_mask_points": 4280,
  "candidate_mask_points": 10837,
  "collision_mask_points": 98245,
  "candidate_mask_has_enough_points": true,
  "dilate_px": 20
}
user_command: 抓取右边绿色物体
target_name: green object
target_mask_id: 3
target_confidence: 0.950
target_reason: The green object on the right has mask id 3
score: 0.225615
width: 0.067467
pre_grasp_pose: [-0.315860, 0.099039, 0.380548, -2.894357, -0.048547, -0.063852]
grasp_pose:     [-0.309602, 0.123161, 0.283703, -2.894357, -0.048547, -0.063852]
lift_pose:      [-0.309602, 0.123161, 0.383703, -2.894357, -0.048547, -0.063852]
execution_result: done
grasp_video: debug_outputs/20260611_150714_grasp.mp4
```

#### C++ Executor Side

```
[UDP] Listening on port 6556...
[UDP] Received: pose_request
[UDP] Sent pose_response
[UDP] Received: grasp_execute
--- Grasp Plan ---
score: 0.225615
width: 0.067467
pre_grasp_pose: [-0.315860, 0.099039, 0.380548, -2.894357, -0.048547, -0.063852]
grasp_pose:     [-0.309602, 0.123161, 0.283703, -2.894357, -0.048547, -0.063852]
lift_pose:      [-0.309602, 0.123161, 0.383703, -2.894357, -0.048547, -0.063852]
[SAFETY] IK check passed
Execute this grasp? [y/N]: y
[EXEC] Moving to pre_grasp...
[EXEC] Moving to grasp...
[EXEC] Closing gripper...
[EXEC] Moving to lift...
[EXEC] Moving to home...
[UDP] Sent: done
```

### Project Structure

```
├── graspnt_vlm_rm/                # Core Python module
│   ├── run_text_vlm_grasp.py      # Main entry: text-command VLM grasp
│   ├── run_basic_grasp.py         # Basic GraspNet grasp (no VLM)
│   ├── mask_target.py             # SAM proposals + VLM target selection
│   ├── mask_filter.py             # Grasp candidate projection filtering
│   ├── graspnet_infer.py          # GraspNet inference, point cloud ops
│   ├── camera_realsense.py        # RealSense D435i driver
│   ├── transform.py               # Hand-eye calibration transforms
│   ├── safety.py                  # Python-side safety validation
│   ├── udp_client.py              # UDP robot communication client
│   ├── visualization.py           # 2D/3D debug visualization
│   ├── video_recorder.py          # Execution video recording
│   ├── config.py                  # Configuration loader
│   └── config.yaml                # Runtime configuration
├── graspnt_robot_executor/        # C++ robot motion executor
│   ├── include/                   # Header files
│   ├── src/
│   │   ├── main.cpp               # Entry point, UDP listen loop
│   │   ├── grasp_executor.cpp     # Grasp execution sequence
│   │   ├── protocol.cpp           # JSON protocol parsing
│   │   ├── robot_driver.cpp       # RealMan SDK wrapper
│   │   ├── safety_checker.cpp     # C++-side IK safety validation
│   │   └── udp_server.cpp         # UDP server
│   ├── 3rdparty/                  # Third-party SDKs
│   │   └── Robotic_Arm/           # RealMan SDK (api_cpp)
│   └── CMakeLists.txt
├── graspnet-baseline/             # [Dependency] GraspNet baseline model
├── graspnetAPI/                   # [Dependency] GraspNet data & evaluation API
├── tests/                         # Unit tests
└── docs/                          # Documentation & visual outputs
```

### Configuration

All runtime parameters are in `graspnt_vlm_rm/config.yaml`:

| Section | Purpose | Key Parameters |
|---------|---------|---------------|
| `graspnet` | Model checkpoint, inference, collision | `root`, `checkpoint`, `num_point`, `num_view`, `collision_thresh`, `min_score`, `top_down_angle_deg` |
| `camera` | RealSense stream settings | `serial`, `width`, `height`, `fps` |
| `camera_preview` | Pre-inference workspace preview | `enabled`, `show_depth`, `wait_key_continue` |
| `hand_eye` | Extrinsic calibration | `direction`, `rotation` (3×3), `translation` |
| `workspace` | Coarse workspace mask | `mode`, `x_min_ratio`, `x_max_ratio`, `y_min_ratio`, `y_max_ratio` |
| `target_cloud` | Target mask → point cloud | `dilate_px`, `min_points`, `depth_min`, `depth_max` |
| `vlm` | VLM inference endpoint | `endpoint`, `model`, `timeout_sec`, `min_confidence` |
| `sam` | SAM automatic mask generation | `model_type`, `checkpoint`, `points_per_side`, `pred_iou_thresh`, `top_n` |
| `target_filter` | Candidate projection guard | `require_target_candidate` |
| `safety` | Grasp safety constraints | `gripper_length`, `min_grasp_z`, `pre_grasp_offset`, `lift_offset`, `workspace_bounds` |
| `execution` | Robot communication | `backend`, `udp_host`, `udp_port`, `ack_timeout_sec`, `result_timeout_sec`, `max_retries` |
| `visualization` | Debug display and export | `enabled`, `mode`, `save_debug`, `debug_dir`, `top_n` |
| `video_recording` | Execution recording | `enabled`, `output_dir`, `fps`, `codec`, `extension` |

> **Important**: Configuration values like IP addresses, file paths, camera serials, and hand-eye calibration matrices are specific to a particular hardware setup. See [Before You Run](#before-you-run) for the full checklist.

### Troubleshooting

| Symptom | Likely Cause | Fix |
|---------|-------------|-----|
| VLM picks wrong mask | SAM didn't segment target as separate mask | Lower `sam.min_area`, increase `points_per_side` |
| "zero target-specific grasp candidates" | Target point cloud too sparse | Increase `target_cloud.dilate_px`, reduce `depth_min` |
| Robot grabs wrong side | Hand-eye calibration direction reversed | Verify `hand_eye.rotation` and camera-to-base coordinate transform |
| No candidates after collision filter | `collision_thresh` too strict | Temporarily lower to `0.005` for diagnosis |
| VLM returns 404 | Model name mismatch with vLLM | Use `--served-model-name` flag or match exact snapshot path |
| Video not generated | Codec unavailable | Switch to `"XVID"` codec with `.avi` extension |
| pointnet2 compile error with VS2022 | VS version too new | Use VS2019 (multiple VS versions can coexist) |
| knn `LNK2001` linker error | `long` type mismatch | Replace `long` → `int64_t` and add `#include <cstdint>` |
| `cannot import name 'container_abcs'` | PyTorch ≥1.9 deprecated `torch._six` | Replace with `collections.abc` |
| C++ cmake can't find api_cpp | Wrong SDK path | Set `-DROBOTIC_ARM_DIR=path/to/Robotic_Arm` |
| C++ executor no response | IP/port mismatch | Verify `udp_host`/`udp_port` match between Python config and C++ main.cpp |

### License

This project is for research purposes. See LICENSE file for details. See also the respective licenses of GraspNet, graspnetAPI, SAM, and RealMan SDK.
