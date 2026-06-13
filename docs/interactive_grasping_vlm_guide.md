# 多模态模型理解人类意图并进行交互式抓取：当前实现梳理

本文结合当前工作区里的 `graspnt_vlm_rm` 最新代码，说明如何用视觉语言模型理解文本命令，并通过 SAM、GraspNet、RGB-D 点云、手眼标定和 C++ 执行端完成交互式抓取。

当前实现已经从早期“VLM 输出 bbox，再过滤 GraspNet 候选”的方案，收敛为：

```text
文本命令
  -> SAM 自动分割全图候选 mask
  -> VLM 在带编号 mask 图上选择目标 mask_id
  -> 目标 mask + 深度图 + 粗 workspace 生成目标候选点云
  -> GraspNet 只在目标点云上生成候选
  -> 完整粗工作区点云做碰撞检测
  -> 选出目标专用安全候选
  -> 手眼标定转换到机器人 base 坐标
  -> 安全校验
  -> UDP 发给 C++ 执行端
  -> Python 录制抓取阶段视频
```

核心思想是：VLM 负责理解人类意图和选择目标；SAM 负责提供像素级目标区域；GraspNet 负责 3D 抓取位姿；完整场景点云负责碰撞检测；C++ 执行端负责机械臂真实运动和最终确认。

## 1. 当前项目关系

- `Dehao-Zhou/利用多模态模型Qwen2.5-VL理解人类意图，进行交互式抓取/`：参考项目。它展示了 VLM、SAM、GraspNet、机械臂控制串联的思路，但不能直接照搬。
- `graspnet-baseline/` 和 `graspnetAPI/`：官方 GraspNet 相关代码。负责 6D grasp 候选预测和候选格式处理。
- `graspnt_rm/`：你原始的 Python 抓取项目。主要是基础 GraspNet 抓取和 UDP 执行链路。
- `graspnt_vlm_rm/`：当前 VLM 交互抓取项目。它是在 `graspnt_rm` 基础上分离出来的新工程，已加入 SAM mask 目标选择、VLM 意图理解、目标点云 GraspNet、抓取阶段视频记录。
- `graspnt_robot_executor/`：C++ 执行端。它接收 Python 通过 UDP 发来的抓取计划，并执行机械臂动作。当前 Python 端不再做 `Execute grasp? [y/N]` 二次确认，因为确认已经交给 C++。

## 2. 当前实现的准确流程

入口文件是：

```text
graspnt_vlm_rm/run_text_vlm_grasp.py
```

主要流程如下。

### 2.1 相机预览与文本命令

Python 先启动 RealSense，打开 workspace 预览：

```text
preview_workspace(...)
```

用户按空格确认当前画面后，输入文本命令：

```text
Text grasp command: 抓取右边绿色物体
```

当前版本只做文本命令，不做语音输入。

### 2.2 SAM 生成候选 mask

代码调用：

```python
generate_sam_mask_proposals(frame.color, config.get("sam", {}))
```

SAM 会在整张 RGB 图上自动生成多个候选实例 mask。随后代码把这些 mask 画成带数字编号的 overlay：

```python
build_mask_selection_overlay(frame.color, proposals)
```

调试输出包括：

```text
*_sam_masks.png
*_selected_mask.png
```

`*_sam_masks.png` 给 VLM 看，也给人检查。`*_selected_mask.png` 是 VLM 最终选中的目标 mask。

### 2.3 VLM 选择目标 mask_id

当前不是让 VLM 直接输出 bbox。VLM 看到的是带编号 mask 的图片，然后只需要选择一个 `mask_id`。

相关代码：

```python
select_mask_target(mask_overlay, command, proposals, config["vlm"])
```

VLM 被要求返回结构化 JSON，核心字段是：

```json
{
  "action": "pick",
  "target_name": "green object",
  "mask_id": 5,
  "confidence": 0.95,
  "needs_clarification": false,
  "clarification_question": "",
  "reason": "...",
  "safety_note": "..."
}
```

如果 `confidence` 低于配置阈值，或者 `needs_clarification=true`，代码不会执行抓取。

这一版不再保留旧的 bbox 模式。原因是实测发现 VLM 直接输出 bbox 容易出现坐标不准、左右错位、框到邻近物体等问题；而 SAM mask 编号选择把任务简化成“从可见分割区域中选一个”，稳定性更好。

### 2.4 目标 mask 点云与完整场景点云分离

这是当前架构最关键的地方。

早期方案是：

```text
完整工作区点云 -> GraspNet 生成候选 -> 用 bbox/mask 过滤候选
```

这个方案在小目标或目标难抓时容易失败：GraspNet 会优先给几何上更容易抓的物体生成高分候选，目标物体上可能没有候选。

当前方案改成两阶段分离：

```text
目标 mask 点云  -> GraspNet 候选生成  -> 目标专用候选
完整场景点云    -> 碰撞检测           -> 安全候选
```

代码对应：

```python
scene_mask = runner.build_workspace_mask(frame.depth, config.get("workspace", {}))

candidate_mask, collision_mask, target_cloud_report = build_target_inference_mask(
    frame.depth,
    selected_proposal.mask,
    scene_mask,
    scale=frame.intrinsics.scale,
    config=config.get("target_cloud", {}),
)
```

其中：

- `candidate_mask`：SAM 选中目标 mask 膨胀后，再叠加有效深度和粗 workspace。GraspNet 只从这里采样点云并生成候选。
- `collision_mask`：完整粗 workspace 加深度范围过滤。碰撞检测看的是它，避免忽略桌面和周围障碍物。
- `workspace`：现在只是粗约束，不再主导目标选择。它主要排除画面边缘、无效深度和明显不可能区域。

当前配置：

```yaml
target_cloud:
  dilate_px: 20
  min_points: 300
  depth_min: 0.15
  depth_max: 1.20
```

含义：

- `dilate_px`：对 SAM mask 做像素膨胀。小目标如果只用原始 mask，点云可能太少；适当膨胀能给 GraspNet 更多上下文。
- `min_points`：目标有效点云太少就停止，不强行执行。
- `depth_min/depth_max`：过滤 0 深度、过近、过远噪声。

运行日志里重点看：

```text
target_cloud: {
  "target_mask_points": 4280,
  "candidate_mask_points": 10837,
  "collision_mask_points": 98245,
  "candidate_mask_has_enough_points": true,
  "dilate_px": 20
}
grasp: {
  "candidate_source_points": 10837,
  "collision_scene_points": 98245,
  "candidate_count": 9
}
```

如果 `candidate_source_points` 接近目标点云数量，而 `collision_scene_points` 远大于它，就说明“两阶段分离”生效了。

### 2.5 GraspNet 生成目标专用候选

当前 `GraspNetRunner.infer(...)` 支持：

```python
runner.infer(
    color,
    depth,
    intrinsics,
    workspace_config,
    candidate_mask=candidate_mask,
    collision_mask=collision_mask,
)
```

内部逻辑是：

```text
RGB-D -> organized point cloud
candidate_mask -> candidate_cloud -> GraspNet 网络输入
collision_mask -> collision_cloud -> ModelFreeCollisionDetector
GraspGroup -> collision filter -> NMS -> sort_by_score -> top-down/min-score filter
```

这比“全场景 GraspNet + 后过滤”更符合你的当前目标：用户说抓哪个物体，就优先在那个物体附近生成抓取候选。

### 2.6 候选投影检查

GraspNet 输出的抓取候选是相机坐标系 3D 位姿。代码会把候选中心投影回图像，并检查是否落在 `candidate_mask` 内：

```python
filter_candidates_by_mask(candidates, frame.intrinsics, candidate_mask, mask_id=target.mask_id)
```

这一步现在是“保护和调试”，不是唯一目标绑定手段。因为候选生成阶段已经使用了目标点云，理论上候选应主要来自目标区域。

日志示例：

```text
projection_check: {
  "input_count": 9,
  "kept_count": 9,
  "outside_count": 0,
  "invalid_projection_count": 0
}
```

这说明 9 个候选投影都在目标候选区域内。

### 2.7 坐标转换、安全校验与 UDP 执行

通过 `build_plan(...)` 把相机坐标系下的抓取候选转换到机械臂 base 坐标：

```text
T_base_grasp =
  T_base_end
  * T_end_camera
  * T_camera_grasp
  * gripper/tool offset
```

相关文件：

```text
graspnt_vlm_rm/transform.py
graspnt_vlm_rm/run_basic_grasp.py
```

然后进行安全校验：

```python
validate_motion_plan(plan, config["safety"])
```

最后通过 UDP 发给 C++：

```python
udp_client.execute_grasp(plan)
```

当前 Python 端已经删除：

```text
Execute grasp? [y/N]:
```

因为 C++ 执行端已经承担最终确认，这样不会出现 Python 和 C++ 两层重复确认。

### 2.8 抓取阶段视频记录

当前新增了：

```text
graspnt_vlm_rm/video_recorder.py
```

Python 会在发送 UDP 执行命令前开始后台录制，在 C++ 返回执行结果后停止：

```text
record_grasp_video(...)
recorder.start()
udp_client.execute_grasp(plan)
recorder.stop()
```

配置：

```yaml
video_recording:
  enabled: true
  output_dir: "debug_outputs"
  fps: 30
  codec: "mp4v"
  extension: ".mp4"
```

运行时会打印：

```text
grasp_video: debug_outputs/20260611_XXXXXX_grasp.mp4
```

当前不再保存：

- `*_grasps.json`
- `*_scene.ply`

保留的主要调试文件是：

- `*_sam_masks.png`
- `*_selected_mask.png`
- `*_rgb_grasp.png`
- `*_depth.png`
- `*_grasp.mp4`

## 3. VLM 在系统中的真实作用

多模态模型不应该直接输出机械臂位姿。它在当前系统中的职责是：

```text
自然语言 + 带编号 mask 图 -> 选择一个目标 mask_id
```

它不负责：

- 估计 3D 抓取点。
- 判断夹爪宽度是否合适。
- 判断机械臂是否可达。
- 做碰撞检测。
- 生成 pre-grasp/grasp/lift 轨迹。

这些仍由 GraspNet、点云碰撞检测、坐标转换、安全校验和 C++ 执行端完成。

这样拆分的好处是：

- VLM 只处理语义和目标选择，任务更简单。
- SAM 提供像素级 mask，避免 VLM bbox 不准。
- GraspNet 仍基于真实 RGB-D 点云生成 6D grasp。
- 完整场景点云仍参与碰撞检测，不会只看目标而忽略周围障碍。

## 4. 为什么不再用 bbox 模式

之前尝试过让 VLM 直接返回：

```json
{
  "bbox": [x1, y1, x2, y2]
}
```

实测问题包括：

- VLM bbox 坐标可能明显偏移。
- 有时用户说右侧，模型框到左侧或中心。
- bbox 是矩形，会包含背景和邻近物体。
- bbox 坐标可能受图片缩放、模型内部坐标系、提示词影响。
- 对小物体尤其不稳定。

当前改为：

```text
SAM 先给出可见候选实例 -> VLM 只选编号
```

这降低了 VLM 的定位负担。VLM 不需要精确画框，只需要判断“哪个编号的 mask 符合用户命令”。

因此当前代码已删除：

```text
vlm_target.py
target_filter.py
bbox target mode
```

## 5. SAM 在当前系统中的角色

SAM 不是用来生成抓取姿态的。它承担的是“语义目标到像素区域”的桥接。

当前角色是：

```text
RGB 图像 -> SAM 自动实例 mask
带编号 overlay -> VLM 选择 mask_id
mask_id -> 目标点云 candidate_mask
```

它解决的问题是：GraspNet 只看几何，不知道用户说的“绿色物体”“右边物体”“那个盖子”是哪一个。SAM mask 把用户意图绑定到图像区域，再通过深度图变成目标点云。

如果去掉 SAM，会退化成：

```text
VLM bbox 或全场景 GraspNet
```

这对单物体场景可能可用，但多物体交互式抓取会明显不稳。

## 6. workspace 在当前系统中的角色

旧理解里，workspace 可能是主过滤器：GraspNet 在 workspace 里找最优抓取。

当前理解应该改成：

```text
workspace 是粗约束，不是目标选择依据
```

它主要用于：

- 排除图像边缘。
- 排除无效深度。
- 限制相机视野中明显不可能的区域。
- 为 collision_mask 定义完整场景范围。

真正决定“抓哪个物体”的是：

```text
SAM mask + VLM mask_id
```

## 7. 坐标系与左右方向问题

你已经通过测试确认：当前出现过“图像右侧目标，机械臂抓到左侧”的现象，不是 VLM/SAM 目标选择失败，而是相机画面左右与机器人 base 坐标理解相反。

判断方法：

- 如果 `*_selected_mask.png` 中选中的 mask 是正确目标。
- 如果 `*_rgb_grasp.png` 中最佳抓取点也落在该目标上。
- 但机械臂实际去另一侧。

那么问题在后半段：

```text
camera grasp -> base pose
```

重点检查：

- 手眼标定外参方向是否正确。
- `hand_eye.direction` 语义是否和矩阵实际方向一致。
- 相机安装姿态导致的图像左右和机器人 base 左右是否相反。
- 机器人 base 坐标系 X/Y 正方向和人的观察方向是否一致。
- `current_end_pose` 的姿态单位和欧拉角顺序是否与 C++ 端一致。

不要再从 bbox 或 mask 逻辑上修这个问题。mask 选对、2D 抓取点选对时，应该查坐标变换。

## 8. 当前配置要点

### 8.1 VLM 服务

当前使用 OpenAI-compatible vLLM endpoint：

```yaml
vlm:
  endpoint: "http://10.16.45.54:8000/v1/chat/completions"
  model: "/home/p/.cache/huggingface/hub/models--Qwen--Qwen3-VL-2B-Instruct-FP8/snapshots/..."
  api_token: ""
  timeout_sec: 60
  min_confidence: 0.4
```

注意：`model` 必须和 `/v1/models` 返回的 id 一致。之前出现过：

```text
404 Not Found: The model `Qwen/Qwen3-VL-2B-Instruct-FP8` does not exist.
```

根因是 vLLM 服务暴露的是本地 snapshot 路径，而不是短模型名。更稳的做法是在启动 vLLM 时加：

```bash
--served-model-name Qwen/Qwen3-VL-2B-Instruct-FP8
```

这样配置里就能使用短模型名。

### 8.2 SAM 配置

当前使用 SAM 自动 mask：

```yaml
sam:
  model_type: "vit_h"
  checkpoint: "D:/ArmProject/GraspNet/segment-anything/sam_vit_h_4b8939.pth"
  device: "cuda"
  points_per_side: 16
  pred_iou_thresh: 0.88
  stability_score_thresh: 0.90
  min_mask_region_area: 100
  min_area: 300
  max_area_ratio: 0.35
  top_n: 12
```

调参建议：

- mask 太碎：增大 `min_area`，或提高 `pred_iou_thresh/stability_score_thresh`。
- 漏掉目标：降低 `min_area`，提高 `points_per_side`。
- VLM 看到太多候选：减小 `top_n`。
- 太慢：降低 `points_per_side`，或换 `vit_b/vit_l`。

### 8.3 目标点云配置

```yaml
target_cloud:
  dilate_px: 20
  min_points: 300
  depth_min: 0.15
  depth_max: 1.20
```

调参建议：

- 目标点云太少：增大 `dilate_px`，例如 30 或 40。
- 候选混到邻近物体：减小 `dilate_px`。
- 近处噪声多：增大 `depth_min`。
- 远处桌面/背景混入：减小 `depth_max`。

### 8.4 视频记录

```yaml
video_recording:
  enabled: true
  output_dir: "debug_outputs"
  fps: 30
  codec: "mp4v"
  extension: ".mp4"
```

录制范围是：

```text
Python 发送 UDP 执行命令前
  -> C++ 执行端确认/执行
  -> C++ 返回 result
```

所以视频会覆盖机械臂真实抓取过程。

## 9. 对 Dehao-Zhou 参考项目的审查

参考项目的总体思路是对的：

```text
人类命令 -> VLM 理解目标 -> SAM 分割 -> GraspNet 抓取 -> 机械臂执行
```

但不建议直接照搬。

可借鉴点：

- 用 VLM 做自然语言目标选择。
- 用 SAM 把目标区域变成 mask。
- 用 mask 将语义目标和 GraspNet 候选关联起来。
- 保留可视化，方便调试 grounding 和 grasp 是否一致。

需要警惕的问题：

- 服务地址、模型名、checkpoint、相机参数等硬编码多。
- VLM 输出 JSON 的解析方式不够稳健。
- 直接依赖 VLM bbox，缺少坐标范围校验和歧义处理。
- SAM mask 选择和错误处理不够完整。
- 抓取候选为空时容易崩溃。
- 坐标转换和夹爪补偿分散，不如当前 `transform.py` 集中。
- 执行逻辑和推理逻辑耦合较强，不如你现在 Python 推理、C++ 执行的分层清楚。

当前 `graspnt_vlm_rm` 已经吸收了它有价值的部分，但做了工程化修正：

- 不让 VLM 直接输出 bbox。
- 不用 bbox 作为目标绑定方式。
- SAM mask 先行。
- GraspNet 目标点云候选生成。
- 完整场景点云碰撞检测。
- UDP 与 C++ 执行端解耦。
- 执行阶段视频记录。

## 10. 常见失败模式与排查

### 10.1 VLM 选错 mask

现象：

```text
target_mask_id 与用户目标不一致
```

排查：

- 看 `*_sam_masks.png`，确认目标是否有正确编号。
- 看 VLM reason 是否理解了命令。
- 如果目标没有被 SAM 分出独立 mask，问题不在 VLM，而在 SAM 候选。

改进：

- 调整 SAM 参数。
- 降低 `min_area`。
- 增大 `points_per_side`。
- 简化命令，例如先用“抓取编号 5 的物体”验证链路。

### 10.2 SAM 选中对，但 GraspNet 没候选

现象：

```text
candidate_mask_has_enough_points: false
```

或：

```text
GraspNet returned zero target-specific grasp candidates
```

排查：

- `target_mask_points` 是否太少。
- `candidate_mask_points` 是否太少。
- 深度图里目标是否有有效深度。
- `depth_min/depth_max` 是否过滤过度。

改进：

- 增大 `target_cloud.dilate_px`。
- 降低 `target_cloud.min_points`，但不要过低。
- 调整 RealSense 视角和距离。
- 清理目标表面的反光/透明问题。

### 10.3 2D 目标正确，但机械臂抓偏

现象：

- `*_selected_mask.png` 正确。
- `*_rgb_grasp.png` 抓取点正确。
- 机械臂实际位置偏到另一侧。

排查：

- 手眼标定。
- 相机坐标系到机器人 base 坐标系方向。
- gripper length 补偿方向。
- C++ 端对姿态向量 `[rx, ry, rz]` 的解释。

这是坐标链路问题，不是 VLM/SAM 问题。

### 10.4 碰撞检测过严或过松

配置：

```yaml
graspnet:
  collision_thresh: 0.01
  voxel_size: 0.01
```

如果候选经常被过滤光，可以临时降低或关闭 `collision_thresh` 做对比；但真实抓取不要长期关闭碰撞检测。

### 10.5 视频文件没有生成

检查：

- `video_recording.enabled` 是否为 true。
- `output_dir` 是否有写权限。
- OpenCV 是否支持 `mp4v` 编码。
- C++ 是否很快返回导致视频很短。

如果 mp4 写不出来，可尝试：

```yaml
codec: "XVID"
extension: ".avi"
```

## 11. 推荐实验顺序

### 实验 1：只检查 SAM/VLM 选目标

运行完整入口，但先观察：

- `*_sam_masks.png`
- `*_selected_mask.png`
- 控制台 `target_mask_id`

目标：确认自然语言能选中正确 mask。

### 实验 2：检查目标点云数量

看日志：

```text
target_mask_points
candidate_mask_points
collision_mask_points
candidate_source_points
collision_scene_points
```

目标：确认 GraspNet 的候选来源是目标点云，碰撞检测来源是完整场景点云。

### 实验 3：检查 2D 抓取点

看：

```text
*_rgb_grasp.png
```

目标：确认最佳抓取候选投影在目标上。

### 实验 4：只观察 C++ 确认前状态

Python 已不再二次确认。执行确认在 C++ 侧完成。你可以在 C++ 确认阶段观察机械臂即将执行的计划。

目标：避免重复确认，同时保留真实执行前的人为保护。

### 实验 5：完整抓取并回看视频

看：

```text
*_grasp.mp4
```

目标：分析机械臂接近、闭合、抬升过程，判断失败是目标选择、抓取姿态、手眼标定还是执行动作导致。

## 12. 当前系统边界

当前版本已经适合做：

- 单轮文本命令抓取。
- 多物体场景下的语义目标选择。
- SAM mask 目标绑定。
- 目标点云 GraspNet 候选生成。
- 完整场景点云碰撞过滤。
- UDP 下发给 C++ 执行端。
- 抓取阶段视频记录。

当前还没有做：

- 语音输入。
- 多轮对话澄清。
- 放置任务。
- 主动换视角。
- 失败后自动重试。
- 基于力/夹爪反馈判断是否抓稳。

后续如果继续扩展，优先级建议是：

1. 先修正/标定相机坐标到 robot base 的左右方向理解问题。
2. 增加失败原因记录，例如目标选择失败、目标点云不足、无候选、坐标越界、C++ 执行失败。
3. 增加多轮澄清：当 VLM 不确定或多个 mask 符合时，让用户指定编号。
4. 增加抓取后视频与日志的统一命名，方便把一次抓取的图像、日志、视频对应起来。
