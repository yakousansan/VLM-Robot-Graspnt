# GRASPNT — 基于视觉语言模型的交互式机器人抓取系统

<p align="center">
  <img src="https://img.shields.io/badge/platform-Windows%20%7C%20Linux-lightgrey" alt="平台">
  <img src="https://img.shields.io/badge/python-3.8%2B-blue" alt="Python">
  <img src="https://img.shields.io/badge/C%2B%2B-17-blue" alt="C++">
  <img src="https://img.shields.io/badge/PyTorch-2.3%2B-red" alt="PyTorch">
</p>

<p align="center">
  <b>简体中文</b> | <a href="README_en.md">English</a>
</p>

---

### 项目简介

GRASPNT 是一个多模态交互式机器人抓取系统。它通过视觉语言模型（VLM）理解人类的自然语言指令（如"抓取右边绿色物体"），结合 SAM 实例分割、GraspNet 6-DoF 抓取估计和 RGB-D 点云处理，实现"说抓什么就抓什么"的机械臂智能抓取。

### 目录

- [项目简介](#项目简介)
- [系统流程](#系统流程)
- [系统架构](#系统架构)
- [功能特性](#功能特性)
- [演示](#演示)
- [输出结果展示](#输出结果展示)
- [依赖项](#依赖项)
- [环境配置](#环境配置)
- [安装说明](#安装说明)
- [运行前必改清单](#运行前必改清单)
- [使用方法](#使用方法)
- [执行日志示例](#执行日志示例)
- [项目结构](#项目结构)
- [配置参数](#配置参数)
- [常见问题](#常见问题)
- [许可证](#许可证)

---

### 系统流程

```
用户输入："抓取右边绿色物体"
           │
           ▼
   ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
   │  RealSense   │────▶│     SAM      │────▶│  VLM (Qwen)  │
   │  RGB-D 采集  │     │  自动实例分割    │  选择 mask_id  │
   └──────────────┘     └──────────────┘     └──────┬───────┘
                                                     │
                    ┌────────────────────────────────┘
                    ▼
   ┌─────────────────────────────────────────────────────┐
   │               点云分离策略                            │
   │  目标 mask → 候选点云（供 GraspNet 生成抓取）          │
   │  完整场景  → 碰撞点云（供碰撞检测）                     │
   └────────────────────┬────────────────────────────────┘
                        ▼
   ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
   │   GraspNet   │────▶│   碰撞检测    │────▶│   手眼标定   │
   │  6D 抓取候选  │     │  (完整场景)   │     │   坐标变换   │
   └──────────────┘     └──────────────┘     └──────┬───────┘
                                                     │
                                                     ▼
   ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
   │  C++ 执行端  │◀────│  UDP 通信    │◀────│   安全校验   │
   │  机械臂控制  │     │  (Python)    │     │   多级保护   │
   └──────────────┘     └──────────────┘     └──────────────┘
```

### 系统架构

#### Python 端（感知与规划）

| 模块 | 职责 | 核心技术 |
|------|------|---------|
| **VLM 目标选择** | 理解自然语言意图，从编号 mask 叠加图中选择目标 | [Qwen3-VL-2B-Instruct-FP8](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct-FP8)（通过 vLLM 部署） |
| **SAM 实例分割** | 像素级实例分割，生成所有可见物体的候选 mask | SAM (vit_h) |
| **GraspNet 抓取生成** | 仅在目标区域点云上生成 6-DoF 抓取位姿 | GraspNet-baseline |
| **碰撞检测** | 基于完整场景点云过滤危险抓取 | Model-Free Collision Detector |
| **手眼标定** | 将相机坐标系抓取位姿转换到机器人基座坐标系 | `transform.py` |
| **安全校验** | 工作空间边界、最低高度、预抓取/抬升偏移 | `safety.py` |
| **视频录制** | 全程录制抓取执行过程，便于调试回放 | OpenCV VideoWriter |

#### C++ 执行端（运动控制）

```
启动 → 连接机械臂 → 回零 → UDP 监听循环
                              │
            ┌─────────────────┘
            ▼
      收到 UDP 命令
      ╔═══════════════════╗
      ║  pose_request     ║──→ 返回当前末端位姿 + 关节角
      ║  grasp_execute    ║──→ 打印计划 → IK 可达性检查
      ╚═══════════════════╝         │
                                    ▼
                         "Execute this grasp? [y/N]"
                              │            │
                              y            N
                              ▼            ▼
               pre_grasp → grasp →    返回 "rejected"
               夹爪闭合 → lift →
               home → 返回 "done"
```

| 组件 | 职责 | 核心技术 |
|------|------|---------|
| **Robot Driver** | 机械臂连接、运动控制（关节/直线） | RealMan SDK (`api_cpp`) |
| **UDP Server** | 接收 Python 命令，返回执行结果 | Winsock2 (Windows) |
| **Protocol Parser** | JSON 命令解析与序列化 | nlohmann/json |
| **Safety Checker** | C++ 端 IK 可达性及机器人状态校验 | RealMan SDK IK |
| **Grasp Executor** | 编排 pre_grasp → grasp → lift → home 执行序列 | — |

> **双层安全机制**：Python 端校验几何边界和工作空间；C++ 端校验 IK 可达性和机器人状态。C++ 端还要求操作员在控制台输入 `y` 确认后才会运动——这是最后一道安全闸门。

### 功能特性

- **自然语言目标定位**：用日常语言描述抓取目标——"那个杯子"、"最左边的物体"、"右边的绿色盒子"
- **Mask 编号选择机制**：SAM 生成带编号的实例 mask 叠加图，VLM 只需选择一个编号——比直接输出 bbox 更稳定准确
- **目标/场景点云分离**：GraspNet 仅在目标区域生成候选抓取，碰撞检测使用完整场景——避免误抓其他物体
- **6-DoF 抓取位姿估计**：基于 GraspNet 推理，结合碰撞检测、NMS、自上而下角度过滤和分数排序
- **Python/C++ 模块分离**：Python 负责感知与规划，C++ 负责实时运动执行，通过 UDP 解耦
- **双层安全保护**：Python 检查工作空间边界、最低高度、预抓取/抬升偏移；C++ 检查 IK 可达性和机器人状态
- **人工最终确认**：C++ 执行端在执行前要求控制台输入 `[y/N]` ——操作员拥有最终决定权
- **自动视频录制**：全程录制抓取执行过程（接近→闭合→抬升），支持离线分析
- **丰富的调试可视化**：2D RGB/深度叠加、3D 点云夹爪线框、SAM mask 叠加图、选中 mask 高亮、调试文件导出（PNG、MP4）

### 演示

<p align="center">
  <img src="demo.gif" width="640" alt="抓取演示">
</p>

### 输出结果展示

每次文本命令抓取会生成以下调试输出：

| 阶段 | 输出文件 | 说明 |
|------|---------|------|
| 1 | `*_depth.png` | 深度图 JET 伪彩色可视化 |
| 2 | `*_rgb_grasp.png` | 最终抓取可视化——工作区叠加 + 抓取位姿标记（绿色=最优候选） |
| 3 | `*_sam_masks.png` | SAM 候选 mask 叠加图——所有候选区域带彩色覆盖和数字编号（发送给 VLM） |
| 4 | `*_selected_mask.png` | VLM 选中的目标 mask——高亮显示 + 边界框 + "mask N" 标签 |
| 5 | `*_grasp.mp4` | 完整抓取执行过程视频 |

### 依赖项

#### Python

| 依赖 | 用途 |
|------|------|
| Python 3.8+（推荐 3.10） | |
| PyTorch 2.3.1+ | GraspNet 模型推理 |
| Open3D | 点云处理、3D 可视化 |
| OpenCV | 图像处理、视频录制 |
| NumPy, SciPy | 数值计算、旋转变换 |
| pyrealsense2 | Intel RealSense D435i 驱动 |
| segment-anything | SAM 实例分割 |
| graspnet-baseline | GraspNet 模型、数据工具、碰撞检测 |
| graspnetAPI | GraspNet 数据结构（GraspGroup） |
| requests | VLM API HTTP 调用 |
| vLLM（或兼容服务） | OpenAI 兼容的 VLM 推理端点 |

#### C++

| 依赖 | 用途 |
|------|------|
| C++17 | |
| CMake 3.16+ | 构建系统 |
| RealMan SDK (`api_cpp`) | ECO65-6F 机械臂控制（运动、夹爪、IK） |
| nlohmann/json | UDP 协议 JSON 解析 |
| Winsock2 | UDP socket 通信（Windows） |

### 环境配置

```bash
# 创建并激活 conda 环境
conda create -n graspnt python=3.10
conda activate graspnt

# 安装 PyTorch（根据 CUDA 版本调整）
conda install pytorch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 pytorch-cuda=12.1 -c pytorch -c nvidia

# 安装核心依赖
pip install opencv-python open3d pyrealsense2 numpy scipy requests

# 安装 SAM
pip install git+https://github.com/facebookresearch/segment-anything.git
```

### 安装说明

#### 1. GraspNet-baseline

> 详细的 Windows 安装避坑指南参见：[GraspNet Baseline Windows 复现指南](https://blog.csdn.net/SWORDHOLDER/article/details/159793585)

```bash
# 克隆 GraspNet-baseline
git clone https://github.com/graspnet/graspnet-baseline.git
cd graspnet-baseline
# 先注释掉 requirements.txt 中的 torch，再安装其他依赖
pip install -r requirements.txt

# 编译 pointnet2（必须使用 VS2019！VS2022/VS2026 均不可用）
cd pointnet2
python setup.py install
cd ..

# 编译 knn
cd knn
# 重要：编译前将所有 "long" 替换为 "int64_t"，并添加 #include <cstdint>
python setup.py install
cd ..
```

关键注意事项：
- **必须使用 VS2019** 编译 `pointnet2`，更高版本的 VS 会报错。多版本 VS 可以共存。
- **knn 编译修复**：将 knn 目录下所有 `long` → `int64_t`，并添加 `#include <cstdint>`，否则出现 `LNK2001` 链接错误。
- PyTorch 1.9+：`torch._six.container_abcs` 需改为 `collections.abc`。

**验证 GraspNet 安装：**

```bash
cd graspnet-baseline
python demo.py --checkpoint_path checkpoint-rs.tar
```

如果弹出 3D 点云窗口并显示夹爪线框，说明 GraspNet 安装成功。

#### 2. graspnetAPI

```bash
git clone https://github.com/graspnet/graspnetAPI.git
cd graspnetAPI
# 修改 setup.py：将 "sklearn" 改为 "scikit-learn"，再安装
pip install .
cd ..
```

#### 3. 模型权重

下载检查点权重文件，放入 `graspnet-baseline/` 目录：

| 检查点 | 适用相机 |
|--------|---------|
| `checkpoint-rs.tar` | RealSense D435i |
| `checkpoint-kn.tar` | Kinect |

下载链接（Google Drive / 百度网盘）见 [graspnet-baseline README](https://github.com/graspnet/graspnet-baseline)。

#### 4. SAM

下载 SAM 检查点（如 `sam_vit_h_4b8939.pth`），并在 `config.yaml` 中更新 `sam.checkpoint` 路径。

#### 5. VLM 推理服务

启动 VLM 推理服务，使用 [Qwen3-VL-2B-Instruct-FP8](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct-FP8) 模型，通过 vLLM 部署：

```bash
vllm serve path/to/Qwen3-VL-2B-Instruct-FP8 \
  --served-model-name Qwen/Qwen3-VL-2B-Instruct-FP8 \
  --port 8000
```

#### 6. C++ 机械臂执行端

```bash
cd graspnt_robot_executor
mkdir build && cd build
cmake .. -DROBOTIC_ARM_DIR=path/to/Robotic_Arm
cmake --build . --config Release
```

CMake 期望以下 **3rdparty 目录结构**：

```
graspnt_robot_executor/
└── 3rdparty/
    └── Robotic_Arm/          # RealMan SDK 根目录
        ├── include/
        │   └── rm_service.h
        └── lib/
            ├── api_cpp.lib
            └── api_cpp.dll
```

CMake 自动处理：
- 查找 `api_cpp` 库和头文件
- 链接 nlohmann/json（或使用 header-only 回退）
- Windows 下链接 Winsock2
- 自动复制 `api_cpp.dll` 到输出目录

### 运行前必改清单

首次运行前，务必修改 `graspnt_vlm_rm/config.yaml` 中的以下配置：

- [ ] `graspnet.root` → 你的 `graspnet-baseline` 目录路径
- [ ] `graspnet.checkpoint` → 你的预训练权重路径（`checkpoint-rs.tar`）
- [ ] `sam.checkpoint` → 你的 SAM 权重路径（`sam_vit_h_4b8939.pth`）
- [ ] `camera.serial` → 你的 RealSense D435i 序列号
- [ ] `hand_eye.rotation` / `hand_eye.translation` → 你的手眼标定结果
- [ ] `safety.gripper_length` → 夹爪 TCP 到接触点的距离
- [ ] `vlm.endpoint` → 你的 VLM 推理服务地址
- [ ] `vlm.model` → 与 VLM 服务匹配的模型名
- [ ] `execution.udp_host` / `execution.udp_port` → 必须与 C++ 执行端一致

在 `graspnt_robot_executor/src/main.cpp` 中：

- [ ] `robot_ip` → RealMan 控制器 IP
- [ ] `robot_port` → RealMan 控制器端口（默认：`8080`）
- [ ] `udp_port` → UDP 监听端口（默认：`6556`）

### 使用方法

1. **启动 C++ 执行端**（连接机械臂、回零、开始 UDP 监听）：

   ```
   .\graspnt_robot_executor.exe
   ```

2. **运行文本命令抓取脚本：**

   ```bash
   python -m graspnt_vlm_rm.run_text_vlm_grasp config.yaml
   ```

3. **完整工作流程：**

   - Python 打开实时相机预览，叠加工作区 mask
   - 按**空格键**确认场景并捕获帧
   - Python 通过 UDP 向 C++ 请求当前机器人位姿（`pose_request`）
   - Python 依次执行 SAM → VLM 目标选择 → GraspNet 推理
   - Python 显示 2D/3D 调试可视化
   - Python 通过 UDP 发送 `grasp_execute`（pre_grasp / grasp / lift 位姿）给 C++
   - C++ 打印抓取计划，询问 **`Execute this grasp? [y/N]`**
   - 输入 **y** — 机械臂执行：pre_grasp → grasp → 夹爪闭合 → lift → home
   - C++ 返回 `done`；视频录制停止

4. **输入自然语言命令**，例如：

   ```
   抓取右边绿色物体
   左边的杯子
   中间最大的盒子
   ```

### 执行日志示例

#### Python 端

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

#### C++ 执行端

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

### 项目结构

```
├── graspnt_vlm_rm/                # 核心 Python 模块
│   ├── run_text_vlm_grasp.py      # 主入口：文本命令 VLM 交互抓取
│   ├── run_basic_grasp.py         # 基础 GraspNet 抓取（不使用 VLM）
│   ├── mask_target.py             # SAM mask 生成 + VLM 目标选择
│   ├── mask_filter.py             # 抓取候选投影过滤
│   ├── graspnet_infer.py          # GraspNet 推理与点云操作
│   ├── camera_realsense.py        # RealSense D435i 相机驱动
│   ├── transform.py               # 手眼标定坐标变换
│   ├── safety.py                  # Python 端安全校验
│   ├── udp_client.py              # UDP 机器人通信客户端
│   ├── visualization.py           # 2D/3D 调试可视化
│   ├── video_recorder.py          # 执行过程视频录制
│   ├── config.py                  # 配置加载器
│   └── config.yaml                # 运行时配置
├── graspnt_robot_executor/        # C++ 机械臂运动执行端
│   ├── include/                   # 头文件
│   ├── src/
│   │   ├── main.cpp               # 入口，UDP 监听循环
│   │   ├── grasp_executor.cpp     # 抓取执行序列
│   │   ├── protocol.cpp           # JSON 协议解析
│   │   ├── robot_driver.cpp       # RealMan SDK 封装
│   │   ├── safety_checker.cpp     # C++ 端 IK 安全校验
│   │   └── udp_server.cpp         # UDP 服务器
│   ├── 3rdparty/                  # 第三方 SDK
│   │   └── Robotic_Arm/           # RealMan SDK (api_cpp)
│   └── CMakeLists.txt
├── graspnet-baseline/             # [依赖] GraspNet 基线模型
├── graspnetAPI/                   # [依赖] GraspNet 数据与评估 API
├── tests/                         # 单元测试
└── docs/                          # 文档与可视化输出
```

### 配置参数

所有运行时参数在 `graspnt_vlm_rm/config.yaml` 中：

| 配置节 | 用途 | 关键参数 |
|--------|------|---------|
| `graspnet` | 模型推理与碰撞检测 | `root`, `checkpoint`, `num_point`, `num_view`, `collision_thresh`, `min_score`, `top_down_angle_deg` |
| `camera` | RealSense 图像流 | `serial`, `width`, `height`, `fps` |
| `camera_preview` | 推理前工作区预览 | `enabled`, `show_depth`, `wait_key_continue` |
| `hand_eye` | 手眼标定外参 | `direction`, `rotation`（3×3 旋转矩阵）, `translation` |
| `workspace` | 粗工作区 mask | `mode`, `x_min_ratio`, `x_max_ratio`, `y_min_ratio`, `y_max_ratio` |
| `target_cloud` | 目标 mask 转点云 | `dilate_px`（膨胀像素）, `min_points`, `depth_min`, `depth_max` |
| `vlm` | VLM 推理服务 | `endpoint`, `model`, `timeout_sec`, `min_confidence` |
| `sam` | SAM 自动 mask 生成 | `model_type`, `checkpoint`, `points_per_side`, `pred_iou_thresh`, `top_n` |
| `target_filter` | 候选投影保护 | `require_target_candidate` |
| `safety` | 抓取安全约束 | `gripper_length`, `min_grasp_z`, `pre_grasp_offset`, `lift_offset`, `workspace_bounds` |
| `execution` | 机器人通信 | `backend`, `udp_host`, `udp_port`, `ack_timeout_sec`, `result_timeout_sec`, `max_retries` |
| `visualization` | 调试显示与导出 | `enabled`, `mode`, `save_debug`, `debug_dir`, `top_n` |
| `video_recording` | 执行录制 | `enabled`, `output_dir`, `fps`, `codec`, `extension` |

> **注意**：配置文件中的 IP 地址、文件路径、相机序列号和手眼标定参数属于特定硬件环境。详见[运行前必改清单](#运行前必改清单)。

### 常见问题

| 现象 | 可能原因 | 解决方法 |
|------|---------|---------|
| VLM 选错 mask | SAM 未将目标分割为独立 mask | 降低 `sam.min_area`，提高 `points_per_side` |
| 提示"目标抓取候选为零" | 目标点云太少 | 增大 `target_cloud.dilate_px`，减小 `depth_min` |
| 机械臂抓取位置偏到另一侧 | 手眼标定方向有误 | 检查 `hand_eye.rotation` 和相机到基座坐标变换 |
| 碰撞检测后无候选 | `collision_thresh` 过严 | 临时降至 `0.005` 排查 |
| VLM 返回 404 错误 | 模型名与 vLLM 服务不匹配 | 使用 `--served-model-name` 或匹配完整 snapshot 路径 |
| 视频文件未生成 | 编码器不可用 | 换用 `"XVID"` 编码 + `.avi` 扩展名 |
| pointnet2 VS2022 编译报错 | VS 版本过高 | 改用 VS2019（多版本 VS 可共存） |
| knn 出现 `LNK2001` 错误 | `long` 类型不匹配 | 将 `long` → `int64_t` 并添加 `#include <cstdint>` |
| `cannot import name 'container_abcs'` | PyTorch ≥1.9 废弃 `torch._six` | 改为 `collections.abc` |
| C++ cmake 找不到 api_cpp | SDK 路径不对 | 设置 `-DROBOTIC_ARM_DIR=path/to/Robotic_Arm` |
| C++ 执行端无响应 | IP/端口不匹配 | 检查 Python config 与 C++ main.cpp 中的 `udp_host`/`udp_port` 是否一致 |

### 许可证

本项目用于研究目的。详见 LICENSE 文件。同时请遵循 GraspNet、graspnetAPI、SAM 和 RealMan SDK 各自的许可证。
