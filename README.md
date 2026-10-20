# OpenCV Multi-Object Tracking

[![CI](https://github.com/CGmoke/opencv-multi-object-tracking/actions/workflows/ci.yml/badge.svg)](https://github.com/CGmoke/opencv-multi-object-tracking/actions/workflows/ci.yml)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)](pyproject.toml)
[![OpenCV](https://img.shields.io/badge/opencv-4.5.1%2B-green.svg)](pyproject.toml)

一个**纯 OpenCV** 的多目标跟踪框架：为每个目标各持有一个 CSRT / KCF 跟踪器，用 IoU 做帧间关联来维持唯一 ID，目标丢失超过容忍帧数即销毁 ID，新目标自动分配新 ID，并绘制包围框、ID 与运动轨迹，轨迹点可导出 CSV。

仓库定位是**教学与二次开发骨架**：依赖少、分层清楚、核心逻辑可脱离 OpenCV 单测。它不是生产级 MOT 系统（没有 ReID、卡尔曼滤波，也没有内置检测器驱动）。

> 状态：`v0.2.0`（alpha）。仓库框架、参考实现与测试已就位；真实摄像头、GUI 交互与扩展方向见 [已知限制](#已知限制)。

---

## 目录

- [特性](#特性)
- [目录结构](#目录结构)
- [快速开始](#快速开始)
- [命令行](#命令行)
- [交互按键](#交互按键)
- [配置文件](#配置文件)
- [轨迹 CSV](#轨迹-csv)
- [架构总览](#架构总览)
- [测试与 CI](#测试与-ci)
- [已验证环境](#已验证环境)
- [扩展方向](#扩展方向)
- [已知限制](#已知限制)
- [安全与隐私](#安全与隐私)
- [参与贡献](#参与贡献)
- [许可证](#许可证)

## 特性

| 能力 | 说明 |
| --- | --- |
| 视频 / 摄像头输入 | `--source 0` 读摄像头，`--source path/to/video.mp4` 读文件；`--loop` 循环播放 |
| 跟踪器初始化 | 启动时用 `--seed-box x,y,w,h` 初始化目标，或运行时用鼠标框选新增 |
| 帧间跟踪 + IoU 关联 | 每个目标一个跟踪器；检测框与已有目标按 IoU 关联，支持 `greedy` / `hungarian` 两种匹配 |
| ID 生命周期 | 目标丢失后保留 ID `max_missing` 帧等待恢复；超时销毁；无主检测自动分配新 ID |
| 轨迹绘制与导出 | 绘制包围框、ID、带衰减的历史轨迹；轨迹点写入 CSV（含中心点、时间戳、标签） |
| 多种跟踪后端 | `csrt`（默认，精度优先）、`kcf`（速度优先）、`mil`、`mosse` |
| 无显示运行 | `--no-display` 用于服务器 / CI / 批量处理 |
| 录像 | `--record out.mp4` 保存带标注的视频 |
| 安全默认值 | 拒绝符号链接写入、CSV 公式注入转义、配置项严格类型与上界校验、`--append` 显式追加 |

## 目录结构

```text
.
├── configs/
│   └── default.yaml            # 完整配置示例（含注释）
├── src/mot/
│   ├── __init__.py             # 公开 API
│   ├── __main__.py             # python -m mot 入口
│   ├── app.py                  # CLI、主循环、OpenCV 窗口与渲染编排
│   ├── config.py               # YAML 配置加载与严格校验
│   ├── interactive.py          # 交互状态机（不依赖 cv2，可无头测试）
│   ├── iou.py                  # IoU / 重叠矩阵
│   ├── matching.py             # 贪心与匈牙利关联
│   ├── models.py               # BBox、Detection、TrackedObject、IdAllocator
│   ├── pipeline.py             # 检测 → 去重 → 跟踪 的单帧管道
│   ├── track_manager.py        # ID 生命周期核心（不导入 cv2）
│   ├── trackers.py             # OpenCV 跟踪器后端的兼容封装
│   ├── trajectory.py           # CSV 导出（原子写、注入防护）
│   ├── utils.py                # FPS 计、日志、文件名清洗
│   ├── video_source.py         # 摄像头/文件输入、录像输出
│   ├── visualization.py        # 绘制函数
│   └── detectors/
│       ├── __init__.py
│       └── yolo.py             # 可选 YOLO 检测器适配器（实验性，未接入 CLI）
├── tests/                      # pytest 用例（核心逻辑无需 OpenCV）
├── .github/                    # CI、Issue/PR 模板
├── pyproject.toml
├── requirements.txt
├── requirements-dev.txt
├── Makefile
└── CONTRIBUTING.md · SECURITY.md · CODE_OF_CONDUCT.md · CHANGELOG.md · LICENSE
```

## 快速开始

要求 Python 3.9+，以及带 tracking 模块的 OpenCV（`opencv-contrib-python`）。

```bash
git clone https://github.com/CGmoke/opencv-multi-object-tracking.git
cd opencv-multi-object-tracking

python -m venv <venv-dir>
source <venv-dir>/bin/activate   # Windows: <venv-dir>\Scripts\activate
pip install -e ".[dev]"          # 或：pip install -r requirements-dev.txt
```

确认当前环境有哪些跟踪后端：

```bash
mot-track --list-trackers        # 等价于 python -m mot --list-trackers
```

用摄像头跑（需要图形界面）：

```bash
mot-track --source 0
```

用视频文件跑，并在第一帧用给定框初始化一个目标：

```bash
mot-track --source demo.mp4 --seed-box 120,80,60,140 --csv trails.csv
```

无界面 / 服务器上运行，只看统计结果：

```bash
mot-track --source demo.mp4 --seed-box 120,80,60,140 \
          --no-display --csv trails.csv --max-frames 200
```

查看合并 CLI 覆盖后的最终配置：

```bash
mot-track --dump-config --iou-threshold 0.4 --tracker kcf
```

## 命令行

| 参数 | 默认 | 说明 |
| --- | --- | --- |
| `--source, -s` | `0` | 摄像头索引或视频路径 |
| `--config, -c` | 无 | YAML 配置文件，见 [`configs/default.yaml`](configs/default.yaml) |
| `--tracker, -t` | `csrt` | `csrt` / `kcf` / `mil` / `mosse` |
| `--iou-threshold` | `0.3` | 保持 ID 的最小 IoU，取值 `(0, 1]` |
| `--max-missing` | `15` | 丢失目标最多保留 ID 的帧数 |
| `--trail-length` | `120` | 每个目标保留的轨迹点上限 |
| `--max-tracks` | `256` | 同时存活 ID 的硬上限 |
| `--match-method` | `greedy` | `greedy` 或 `hungarian`（后者需要 scipy） |
| `--csv` | `outputs/trajectories.csv` | 轨迹输出路径，传 `none` 关闭 |
| `--append` | 关 | 追加写而不是截断已有 CSV |
| `--record` | 无 | 保存带标注视频的路径 |
| `--seed-box x,y,w,h` | 无 | 启动时初始化目标，可重复 |
| `--max-frames` | 无 | 处理 N 帧后停止 |
| `--loop` | 关 | 视频结束后循环（配合 `--max-frames` 使用） |
| `--no-display` | 关 | 不创建窗口 |
| `--list-trackers` | — | 打印可用后端并退出 |
| `--dump-config` | — | 打印最终配置并退出 |
| `--verbose, -v` | 关 | 调试日志 |

退出码：`0` 正常，`1` 未读取到任何帧，`2` 参数 / 配置 / 跟踪器 / 输入输出错误，`130` 被 `Ctrl-C` 中断。

轨迹 CSV、录像与截图等运行时产物默认写入仓库内的产物目录（已被 `.gitignore` 忽略，不会上传），上表中出现的默认路径只描述程序行为，不是仓库内容。

## 交互按键

| 按键 | 作用 |
| --- | --- |
| `q` / `Esc` | 退出 |
| `空格` | 暂停 / 继续 |
| `n` | 进入新增模式，鼠标拖拽画框，松开后创建目标 |
| `x` | 进入删除模式，点击目标将其删除（命中面积最小的目标） |
| `c` | 清空所有轨迹线（保留目标） |
| `r` | 重置会话（销毁所有目标，ID 计数继续） |
| `s` | 保存当前帧截图（写入默认输出目录，同名自动加序号） |
| `h` | 显示 / 隐藏帮助 |

新增与删除在暂停状态下同样可用，方便逐帧标注。

## 配置文件

完整示例见 [`configs/default.yaml`](configs/default.yaml)：

```yaml
source:
  source: "0"          # 摄像头索引或视频路径
  max_frames: null
  loop: false
  seed_boxes: []       # [[x, y, w, h], ...]

tracking:
  backend: csrt
  iou_threshold: 0.3
  max_missing: 15
  trail_length: 120
  match_method: greedy
  max_tracks: 256

display:
  enabled: true
  show_trails: true
  show_labels: true
  show_hud: true
  trail_fade: true
  window_name: "OpenCV MOT"

output:
  csv_path: outputs/trajectories.csv
  video_path: null
  append: false
```

校验是严格的：未知字段、类型不符、越界（如 `iou_threshold > 1`、`max_missing < 0`）都会以退出码 `2` 报错，而不是静默截断。CLI 参数优先于配置文件。

## 轨迹 CSV

每帧为每个处于 `ACTIVE` 状态的目标写一行：

```csv
track_id,frame,timestamp,x,y,w,h,cx,cy,label
1,1,0.033333,120.000,80.000,60.000,140.000,150.000,150.000,object
1,2,0.066667,121.500,80.000,60.000,140.000,151.500,150.000,object
```

| 列 | 含义 |
| --- | --- |
| `track_id` | 会话内唯一的自增 ID |
| `frame` | 会话内帧号（`r` 重置后从 1 重新计数，ID 继续） |
| `timestamp` | 秒，按视频 FPS 估算（假定恒定帧率） |
| `x, y, w, h` | 包围框，左上角 + 宽高 |
| `cx, cy` | 包围框中心 |
| `label` | 标签（手工框为 `object`，检测器输出沿用其类别名） |

默认覆盖写；需要追加时显式加 `--append`。以 `= + - @` 开头的标签会被加上 `'` 前缀，防止表格软件把轨迹文件当公式执行。

## 架构总览

```text
            ┌────────────┐   frame    ┌──────────────┐  detections ┌──────────────────┐
 video/ ───►│ VideoSource├───────────►│  MOTPipeline ├────────────►│  TrackManager    │
 camera     └────────────┘            │  (可选 detector)            │  IoU 关联 + ID   │
                                      └──────┬───────┘             │  生命周期 + trail │
                                             │  FrameResult         └───────┬──────────┘
                                             ▼                             │ tracker 工厂注入
                                    ┌────────────────┐                     ▼
                                    │  app.py 主循环  │            OpenCV CSRT/KCF/...
                                    │  绘制 / 交互 /  │
                                    │  CSV / 录像     │
                                    └────────────────┘
```

关键设计：

- **核心层不依赖 OpenCV**：`models` / `iou` / `matching` / `track_manager` / `pipeline` / `trajectory` 都可以在无 cv2 的环境导入，跟踪器通过零参工厂注入，因此可用假跟踪器完整单测 ID 生命周期。
- **单帧管道是纯函数式的**：`MOTPipeline.step(frame)` 返回 `FrameResult`（目标、新建、销毁、匹配、检测），便于替换 UI。
- **ID 语义**：ID 单调自增、会话内不复用；目标丢失后保留 `max_missing` 帧供 IoU 恢复；同一位置的新目标拿到新 ID。
- **同帧去重**：手工框与检测框使用同一个 IoU 阈值合并，避免“一物两 ID”。

细节（状态机、匹配算法、设计取舍、性能热点）见源码与 `tests/` 下的对应用例。

## 测试与 CI

```bash
make test      # python -m pytest
make lint      # ruff check src tests
```

- 核心逻辑（IoU、匹配、ID 生命周期、管道、CSV、配置）用假跟踪器测试，不需要 OpenCV。
- 依赖 cv2 的用例在模块级 `pytest.importorskip("cv2")`，无 OpenCV 环境自动跳过。
- 当前：**279 个用例**，覆盖 CLI 端到端（合成视频 → 跟踪 → CSV / 录像 / 错误路径）。

CI（[`.github/workflows/ci.yml`](.github/workflows/ci.yml)）在 Ubuntu 上跑 Python 3.9 / 3.11 / 3.12 矩阵：ruff 静态检查、pytest、`python -m build` + `twine check` 打包校验。

## 已验证环境

| 组件 | 已验证 | 声明支持 |
| --- | --- | --- |
| Python | 3.12.3 | >= 3.9 |
| OpenCV | 4.6.0（contrib，含 tracking） | >= 4.5.1 |
| NumPy | 1.26.4 | >= 1.21 |
| 平台 | Linux（无显示器，`--no-display`） | 跨平台，macOS/Windows 未实测 |

跟踪器构造按 `cv2.TrackerX_create` → `cv2.legacy.TrackerX_create` → `cv2.TrackerX.create()` 的顺序回退，以兼容新旧 OpenCV 的 API 变动。GUI、真实摄像头与高版本 OpenCV 尚未在 CI 中验证。

## 扩展方向

- **结合 YOLO 做检测 + 跟踪**：`MOTPipeline` 已支持注入 `detector`（`frame -> Sequence[Detection]`），`mot.detectors.YoloDetector` 提供适配器，但**尚未接入 config / CLI**——目前需要在 Python 里显式构造，属于实验性代码，没有自动化测试。
- **换匹配算法**：`--match-method hungarian` 改用全局最优匹配（需要 `pip install scipy`；缺失时自动回退 greedy 并告警）。
- **替换跟踪器**：实现 `init(frame, bbox) -> bool` / `update(frame) -> (bool, bbox)` 的对象即可通过 `TrackManager(tracker_factory=...)` 注入。
- **机器人 / ROS 复用**：`TrackManager` 无 cv2、无全局状态，可复用；`app.py` 的主循环需要改造（时间戳随帧传入、事件化输出）。

## 已知限制

- 摄像头输入、真实视频与 GUI 交互未在 CI 中验证；本仓库的自动化证据来自合成视频。
- `r` 重置后 `frame` 列从 1 重新计数，而 `track_id` 继续递增；把同一 CSV 按 `frame` 关联视频时需要留意。
- 时间戳基于 FPS 估算，假定恒定帧率；`--loop` 回绕后与真实时间脱钩。
- `matched` 字段返回的是“当帧目标列表下标对”，不是 `track_id`。
- `--loop` 且不指定 `--max-frames` 时会无限运行。
- YOLO 适配器未接入 CLI，且加载不可信 `.pt` 权重存在代码执行风险，见 [SECURITY.md](SECURITY.md)。
- 检测间隔 `detection_interval` 的语义是“第 interval 帧起每 interval 帧检测一次”，短片段可能一次都不检测。

## 安全与隐私

- 轨迹 CSV 可能包含人员位置等敏感数据，默认写入 `outputs/`（已被 `.gitignore` 忽略），发布演示前请脱敏。
- CSV 标签做了公式注入转义（含 BOM / 零宽字符绕过）；配置文件限制 1 MB 并拒绝 FIFO / 设备文件。
- 写入轨迹与录像前会拒绝符号链接，避免通过预置链接覆盖任意文件。
- 详细威胁模型与报告方式见 [SECURITY.md](SECURITY.md)。

## 参与贡献

欢迎 Issue / PR。开发环境、代码风格、测试要求与提交规范见 [CONTRIBUTING.md](CONTRIBUTING.md)。参与即表示同意遵守 [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md)。

## 许可证

[Apache License 2.0](LICENSE) © 2026 CGmoke
