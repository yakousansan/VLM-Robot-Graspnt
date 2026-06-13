# README Gap Analysis — GRASPNT-1 vs graspnet-robot-arm

**Date:** 2026-06-13
**Comparison:** [graspnet-robot-arm](https://github.com/yakousansan/graspnet-robot-arm) (no VLM) vs GRASPNT-1 (with VLM+SAM)

## Summary of Differences

| Aspect | graspnet-robot-arm | GRASPNT-1 (current) |
|--------|-------------------|---------------------|
| Target selection | GraspNet on full workspace | VLM + SAM mask → target point cloud |
| Python module | `graspnt_rm` | `graspnt_vlm_rm` (adds VLM/SAM pipeline) |
| Config | `graspnt_rm/config.yaml` | `graspnt_vlm_rm/config.yaml` |
| C++ executor | Same (shared) | Same (shared) |

## Gaps Identified in Current README

### 1. C++ Build Details (THIN)
Current just says `cmake .. && make`. Missing:
- **RealMan SDK** (`api_cpp.lib` / `api_cpp.dll`) — `-DROBOTIC_ARM_DIR=path/to/Robotic_Arm`
- **nlohmann/json** — C++ JSON protocol parsing
- **Winsock2** — Windows UDP socket
- **3rdparty directory structure** expected by CMake
- **Visual Studio** build steps (Windows)

### 2. C++ Executor Runtime Flow
Not described. Should mention:
- Connects robot arm, homes, starts UDP listen loop
- Receives `pose_request` → returns current end-effector pose
- Receives `grasp_execute` → prints plan, runs safety checks (IK reachability)
- Prompts `Execute this grasp? [y/N]` — human-in-the-loop confirmation
- Executes pre_grasp → grasp → close gripper → lift → home

### 3. Execution Logs Example
The other README shows sample Python and C++ logs. Adding this helps users verify their setup is working correctly.

### 4. "Must Update Before Running" Checklist
A clear checklist of config items users MUST change. Currently scattered in prose.

### 5. Dependencies Table Incomplete
Missing: `SciPy`, `nlohmann/json`, `RealMan SDK (api_cpp)`, `Winsock2`

### 6. GraspNet Verification Step
Missing: `python demo.py --checkpoint_path checkpoint-rs.tar` to verify GraspNet install.

### 7. Video/GIF Asset
`20260611_150714_grasp.mp4` (2.8M) and `20260611_150714_grasp.gif` (6.4M) exist in root but not referenced.

### 8. Human-in-the-loop Confirmation
The C++ executor prompts `[y/N]` — an important safety feature not mentioned. Current README says "Python no longer confirms" but doesn't explain C++ side confirmation.

### 9. Star History
The other project has it. Current project has no git remote, so skip for now.

## Recommended Updates

1. Expand **Installation → C++ Executor** with RealMan SDK, cmake flags, and 3rdparty layout
2. Add **C++ Executor Flow** subsection under Architecture
3. Add **Execution Logs** section with sample output
4. Add **Before You Run** checklist
5. Complete **Dependencies** table
6. Add **Verification** step for GraspNet
7. Add **Demo Video** reference to existing mp4/gif
8. Mention C++ side `[y/N]` human-in-the-loop confirmation
