# 项目架构设计与模块规划（目标框架 v0.2.0）

> 对象仓库：`opencv-multi-object-tracking`（纯 OpenCV 多目标跟踪）
> 基线：commit `495762a`（`v0.2.0`，即 [ARCHITECTURE.md](file:///home/moke/opencv-multi-object-tracking/docs/ARCHITECTURE.md) 描述的实现）
> 定位：本文是**目标架构设计 + 模块实现指引**，用于后续功能落地。它是演进式重构方案，不是推倒重来。
> 前置：需求与技术债清单见 [DESIGN_ANALYSIS.md](file:///home/moke/opencv-multi-object-tracking/docs/DESIGN_ANALYSIS.md)（本报告的 `G*` 编号沿用该文）。
> 注：分析时工作区 `src/mot/**`、`tests/**` 处于未提交删除状态，内容取自 git HEAD；恢复命令 `git restore src tests`。

---

## 0. 文档定位

| 项 | 说明 |
| --- | --- |
| 目标 | 建立分层清晰、职责单一、可扩展、可测试、低技术债的目标架构 |
| 方法 | 端口-适配器（Ports & Adapters）+ 严格分层 + 显式契约 |
| 原则 | 领域层零框架依赖；依赖倒置；无全局可变状态；显式数据流 |
| 交付 | 目录规范、依赖规则、模块接口、通信/错误/配置规范、质量基线、实施步骤 |
| 兼容 | 保持包名 `mot`、公开 API 与 CLI 参数向后兼容；按阶段迁移 |

---

## 1. 现有框架能力、约束与技术栈边界

### 1.1 核心能力清单

| 能力 | 承载模块 | 说明 |
| --- | --- | --- |
| 视频/摄像头输入 | `video_source.VideoSource` | 索引或文件、`loop`、fps 回退与上限 |
| 每目标一跟踪器 | `trackers` + `track_manager` | CSRT/KCF/MIL/MOSSE，后端兼容链 |
| IoU 关联 | `iou` + `matching` | greedy / hungarian，包含式阈值 |
| ID 生命周期 | `track_manager` + `models` | 单调不复用 ID、丢失容忍、容量上限 |
| 轨迹绘制与 CSV | `visualization` + `trajectory` | 带衰减轨迹、注入转义、拒绝符号链接 |
| 交互控制 | `interactive` + `app` | 暂停/新增/删除/重置/清轨迹/截图/帮助 |
| 配置系统 | `config` | 严格 YAML 校验 + CLI 覆盖 |
| 检测器扩展点 | `pipeline` + `detectors/yolo` | `Callable[[Frame], Sequence[Detection]]` |

### 1.2 约束条件

- **运行约束**：单进程、单线程、逐帧同步；每帧每个目标一次跟踪器 `update`（线性成本）。
- **资源约束**：`max_tracks=256`、`MAX_DETECTIONS=256`、`trail_length=120`、配置 ≤1 MB、`MAX_SEED_BOXES=100`、`MAX_FRAMES=1e7`。
- **时间约束**：`timestamp = frame / fps`（恒定帧率假设，摄像头漂移、`--loop` 失准）。
- **安全约束**：写盘前拒绝符号链接（CWE-59）；CSV 公式注入转义；配置类型/上界校验。
- **可测试约束**：领域层不得 `import cv2`（核心逻辑用假跟踪器单测，279 用例）。

### 1.3 技术栈边界

| 层 | 必须 | 可选 | 禁止 |
| --- | --- | --- | --- |
| 领域层 | `numpy`（仅向量化几何/关联） | — | `cv2`、`yaml`、`csv`、`os`、`logging` 之外的一切 I/O |
| 应用层 | 领域层 API | — | `cv2`、GUI、文件系统细节 |
| 适配层 | `cv2`（contrib）、`numpy`、`csv`、`os` | `scipy`（hungarian）、`ultralytics`（YOLO） | 领域层内部符号（只经 `ports` 交互） |
| 接口层 | `argparse`、`logging` | ROS2 / gRPC（未来） | 直接操作跟踪器内部状态 |

**技术栈基线**：Python ≥3.9、OpenCV（contrib）≥4.5.1、numpy ≥1.21、PyYAML ≥5.4；`scipy` 与 `ultralytics` 为可选 extra。

### 1.4 现有架构技术债（驱动本次设计）

| 编号 | 技术债 | 目标架构中的对策 |
| --- | --- | --- |
| **G1** | config 与 domain 对 `iou_threshold` 边界不一致 → 未捕获异常 | 约束集中到 `domain/limits.py`，config 与 domain 共用同一校验器 |
| **G2/A1** | `reset` 后 CSV 帧号回绕 | 引入 `Clock` 端口 + 会话全局帧号来源 |
| **G3/A5** | `_refresh` 异常穿透 `step` | 领域层内统一"非致命失败"处理（返回状态而非抛异常） |
| **G6** | 时间戳基于 FPS 估算 | `Clock` 端口，时间戳随帧传入 |
| **G7** | `matched` 为下标对 | `FrameResult` 增 `matched_ids` 便捷视图 |
| **G8** | 死代码 / 未接线 API | 明确分层归属：数据类入 domain，写入器入 adapter；未用 API 标注 `experimental` |
| **G9/A8** | `detection_interval` 首检语义 | `Detector` 端口 + 首次检测即触发 |
| **A10** | `app.run` 约 200 行编排 | 拆为 application 层用例 + interfaces 层薄壳 |

---

## 2. 目标架构总览

### 2.1 架构风格

**端口-适配器（六边形）+ 严格分层**。核心（领域）定义**端口（Protocol）**，外部框架以**适配器**实现端口；应用层编排用例；接口层是唯一的"组合根"（composition root），负责装配所有适配器。

```text
                     ┌──────────────────────── interfaces ───────────────────────┐
                     │  cli.py（组合根：装配 adapters + 调 application 用例）      │
                     │  ui.py / 未来 ros.py                                        │
                     └───────────────▲───────────────────────────────▲──────────┘
                                     │ 调用用例                       │ 注入端口实现
                     ┌───────────────┴───────────────────────────────┴──────────┐
                     │  application：pipeline / session / config / errors       │
                     └───────────────▲───────────────────────────────▲──────────┘
                                     │ 使用领域 API                   │ 依赖端口(Protocol)
     ┌───────────────────────────────┴───────────────┐   ┌──────────┴────────────────────┐
     │  domain（纯逻辑，零框架依赖）                 │◄──┤  adapters：sources/trackers/  │
     │  models · geometry · association · lifecycle  │   │  detectors/sinks/rendering/ui │
     │  ports · limits · records                     │   │  （实现 ports 中的 Protocol） │
     └───────────────────────────────────────────────┘   └───────────────────────────────┘
```

### 2.2 分层职责

| 层 | 包 | 职责 | 可依赖 |
| --- | --- | --- | --- |
| 领域层 | `domain/` | 数据模型、几何度量、关联算法、ID 生命周期、端口定义、约束常量 | stdlib + numpy |
| 应用层 | `application/` | 单帧用例编排、会话状态、配置解析与校验、错误翻译 | `domain` |
| 适配层 | `adapters/` | 实现 ports：采集、跟踪器、检测器、轨迹/视频落盘、渲染、交互 | `domain`（仅经 ports/models） |
| 接口层 | `interfaces/` | CLI、主循环、组合根、未来 ROS/gRPC | `application` + `adapters` |
| 公共 | `utils/` | 无状态工具：FPS 计、日志、文件名清洗 | stdlib |

### 2.3 目录结构规范

```text
src/mot/
├── __init__.py                # 稳定公开 API（再导出）
├── __main__.py                # python -m mot → interfaces.cli:main
├── py.typed
├── domain/                    # 纯逻辑：禁止 import cv2 / yaml / csv / os
│   ├── __init__.py
│   ├── limits.py              # 约束常量单一真源（MIN_BOX_SIDE、MAX_TRACKS、阈值范围…）
│   ├── errors.py              # ValidationError(ValueError) 等领域级异常
│   ├── models.py              # BBox/Detection/TrackedObject/TrackState/StepResult/IdAllocator
│   ├── records.py             # TrackPoint 等纯数据记录（原 trajectory 中数据部分）
│   ├── geometry.py            # box_iou / iou_matrix / pairwise_centroid_distance（原 iou）
│   ├── association.py         # greedy/hungarian match（原 matching）
│   ├── lifecycle.py           # TrackManager（原 track_manager）
│   └── ports.py               # Protocol: FrameSource/Tracker/Detector/TrajectorySink/EventSink/Clock
├── application/               # 用例编排：禁止 import cv2
│   ├── __init__.py
│   ├── pipeline.py            # MOTPipeline（单帧用例）
│   ├── session.py             # TrackingSession：把 app.run 的编排拆为可测用例
│   ├── config.py              # AppConfig：加载/校验/describe（引用 domain.limits）
│   └── errors.py              # MotError 基类 + exit_code 映射
├── adapters/                  # 实现 ports：唯一允许 import cv2 的一层
│   ├── __init__.py
│   ├── sources.py             # VideoSource（实现 FrameSource）+ resolve_source
│   ├── trackers.py            # OpenCV 跟踪器工厂/兼容链（实现 Tracker）
│   ├── detectors/
│   │   ├── __init__.py
│   │   ├── base.py            # 端口说明与假检测器（测试用）
│   │   └── yolo.py            # YoloDetector（可选 ultralytics）
│   ├── sinks/
│   │   ├── __init__.py
│   │   ├── csv.py             # CsvTrajectoryWriter（实现 TrajectorySink）
│   │   └── video.py           # VideoWriterSink
│   ├── rendering.py           # draw_*（纯函数，frame→frame）
│   └── ui.py                  # InteractiveSession（原 interactive，仅用 cv2 事件常量）
├── interfaces/
│   ├── __init__.py
│   ├── cli.py                 # build_parser/build_config/run/main（组合根，薄壳）
│   └── ui_window.py           # 窗口回调、鼠标绑定（把 ui 状态接到 cv2）
└── utils/
    ├── __init__.py
    ├── fps.py                 # FpsMeter
    ├── logging.py             # setup_logging / rate-limited logger
    └── paths.py               # sanitize_filename / ensure_parent_dir
```

### 2.4 模块划分原则

1. **单一职责**：一个模块解决一类问题；单文件目标 ≤ 300 行，函数 ≤ 50 行。
2. **依赖倒置**：高层（应用）不依赖 OpenCV；通过 `ports` 中的 `Protocol` 依赖抽象。
3. **框架无关的核心**：domain 只依赖 stdlib + numpy，可在无 cv2 环境导入。
4. **显式数据流**：模块间只经函数入参与返回值传递；禁止模块级可变全局状态（logger 例外）。
5. **契约优先**：跨层边界以 `Protocol` 定义，适配器必须通过"端口契约测试"。
6. **可替换**：任何适配器可被同端口的另一实现替换（如 CSV sink ↔ Parquet sink、VideoSource ↔ ROS image source）。

### 2.5 依赖关系规则（必须强制）

**允许方向**（实线可依赖，虚线禁止）：

```text
interfaces ──► application ──► domain
     │               │            ▲
     └───────────────┴──► adapters ┘   （adapters 只允许依赖 domain.models / domain.ports / domain.limits）
domain ──✗──► adapters / application / interfaces
domain ──✗──► cv2 / yaml / csv / os / argparse
application ──✗──► cv2 / argparse / GUI
adapters ──✗──► application（适配器不得反向调用用例）
```

**强制手段**（纳入 CI，作为架构守卫）：

```python
# tests/test_architecture.py
FORBIDDEN = {
    "mot.domain": ("cv2", "yaml", "csv", "os", "mot.adapters", "mot.application", "mot.interfaces"),
    "mot.application": ("cv2", "argparse", "mot.adapters", "mot.interfaces"),
    "mot.adapters": ("mot.application", "mot.interfaces"),
}

def test_import_directions():
    import ast, pathlib
    root = pathlib.Path("src/mot")
    for pkg, banned in FORBIDDEN.items():
        for file in (root / pkg.split(".")[-1]).rglob("*.py"):
            tree = ast.parse(file.read_text(encoding="utf-8"))
            mods = [n.module or "" for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)]
            mods += [a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names]
            for mod in mods:
                assert not any(mod == b or mod.startswith(b + ".") for b in banned), \
                    f"{file} violates architecture: imports {mod}"
```

### 2.6 现有文件 → 目标模块 迁移映射

| 现有文件 | 目标模块 | 迁移动作 | 风险 |
| --- | --- | --- | --- |
| `models.py` | `domain/models.py` + `domain/limits.py` | 常量抽到 `limits.py` | 低 |
| `iou.py` | `domain/geometry.py` | 移动，无改 | 低 |
| `matching.py` | `domain/association.py` | 移动，无改 | 低 |
| `track_manager.py` | `domain/lifecycle.py` | 移动 + 修 G3（`_refresh` 容错） | 中 |
| `trajectory.py` | `domain/records.py`（`TrackPoint`）+ `adapters/sinks/csv.py`（writer） | 拆分数据与 I/O | 中 |
| `pipeline.py` | `application/pipeline.py` | 移动 + 修 G9 | 中 |
| `config.py` | `application/config.py` | 引用 `domain/limits`，修 G1 | 中 |
| `video_source.py` | `adapters/sources.py` | 移动，实现 `FrameSource` | 低 |
| `trackers.py` | `adapters/trackers.py` | 移动，实现 `Tracker` 端口 | 低 |
| `detectors/yolo.py` | `adapters/detectors/yolo.py` | 移动 + 接线（见 §6） | 中 |
| `visualization.py` | `adapters/rendering.py` | 移动 | 低 |
| `interactive.py` | `adapters/ui.py` | 移动，去除 cv2 事件常量依赖到 `interfaces/ui_window.py` | 中 |
| `app.py` | `interfaces/cli.py` + `application/session.py` | 拆编排为用例（修 A10） | 高 |
| `utils.py` | `utils/{fps,logging,paths}.py` | 拆分 | 低 |

---

## 3. 模块逐一设计

### 3.1 领域层（`domain/`）

#### `domain/limits.py` — 约束单一真源
集中所有边界常量与校验器，供 config 与 domain 共用，杜绝 G1 类不一致。

```python
MIN_BOX_SIDE = 1.0
MAX_MISSING_LIMIT = 1_000
MAX_TRAIL_LENGTH = 10_000
MAX_TRACKS = 10_000
MAX_DETECTIONS = 256
MAX_SEED_BOXES = 100
MAX_FRAMES = 10_000_000
IOU_OPEN_LOW = True          # 阈值下界为开区间 (0, 1]

def validate_iou_threshold(value: float) -> float:
    if not (0.0 < float(value) <= 1.0):
        raise ValidationError("iou_threshold must be within (0, 1]")
    return float(value)
```

#### `domain/models.py` — 数据模型与基础校验
保持现有 `BBox/Detection/TrackedObject/TrackState/StepResult/IdAllocator`，新增 `matched_ids` 便捷视图与 `FrameResult` 迁出到应用层（见 §3.2）。

| 符号 | 职责 | 输入 | 输出 |
| --- | --- | --- | --- |
| `normalize_bbox` | 校验并归一 bbox | `(x,y,w,h)` | `BBox`；违规抛 `ValidationError` |
| `clip_bbox` | 裁剪到画面 | `bbox, w, h` | `BBox` 或 `None` |
| `Detection` | 单次观测 | `bbox, conf, label` | 不可变对象 |
| `TrackedObject` | 存活目标（含 tracker/bbox/trail/计数/state） | — | 可变对象 |
| `IdAllocator` | 单调 ID | `start` | `int` |
| `StepResult.matched_ids` | **G7 修复**：由 `(track_idx, det_idx)` 映射出 `(track_id, det_label)` | `tracks` | 只读视图 |

#### `domain/ports.py` — 端口契约（Protocol）

```python
from typing import Protocol, Sequence, Tuple, Optional, Any, runtime_checkable

Frame = Any  # 领域层视为不透明对象

@runtime_checkable
class SelfHostedTracker(Protocol):
    def init(self, frame: Frame, bbox: Sequence[float]) -> bool: ...
    def update(self, frame: Frame) -> Tuple[bool, Sequence[float]]: ...

TrackerFactory = Callable[[], SelfHostedTracker]

@runtime_checkable
class FrameSource(Protocol):
    @property
    def fps(self) -> float: ...
    @property
    def name(self) -> str: ...
    def read(self) -> Tuple[bool, Optional[Frame]]: ...
    def release(self) -> None: ...

@runtime_checkable
class Detector(Protocol):
    def __call__(self, frame: Frame, timestamp: float) -> Sequence[Detection]: ...

@runtime_checkable
class TrajectorySink(Protocol):
    def write_frame(self, frame_index: int, timestamp: float, tracks: Sequence[TrackedObject]) -> int: ...
    def close(self) -> None: ...

@runtime_checkable
class EventSink(Protocol):
    def on_frame(self, result: "FrameResult") -> None: ...

@runtime_checkable
class Clock(Protocol):
    def now(self) -> float: ...   # 单调秒；由适配器实现（wall/monotonic）
```

#### `domain/lifecycle.py` — ID 生命周期（原 `track_manager`）
保持现有语义，修复 G3：把"重新 init 失败"从异常改为**领域内可处理状态**。

```python
def _refresh(self, track, detection, frame) -> bool:
    """重新锚定观测；失败则降级为一次 miss，不抛异常。"""
    try:
        init_tracker(track.tracker, frame, detection.bbox)
    except Exception as exc:      # 仅此处允许宽捕获，且必须记录
        LOGGER.warning("re-init failed for id %s: %s", track.track_id, exc)
        track.misses += 1
        track.state = TrackState.LOST
        return False
    track.bbox, track.label, track.confidence = detection.bbox, detection.label, detection.confidence
    track.push_trail(bbox_center(detection.bbox), self.trail_length)
    return True
```

> 注意：`init_tracker`/`update_tracker` 是**适配器**能力。领域层不 import cv2，故这两个函数应作为 `TrackerFactory` 之外注入，或迁至 `ports` 中的 `tracker_ops`。推荐做法：`TrackManager` 构造时注入 `init_fn`/`update_fn`（默认由 `adapters.trackers` 提供），保持 domain 零 cv2。

#### `domain/geometry.py` / `domain/association.py`
保持现有实现与向量化；`association` 的 `greedy` 排序确定性、`hungarian` 无 scipy 回退行为保留。

### 3.2 应用层（`application/`）

#### `application/pipeline.py` — 单帧用例
`MOTPipeline` 保持 `step(frame, timestamp, extra_detections) -> FrameResult`；修复 G9（首帧即检测）：

```python
def _should_run_detector(self) -> bool:
    self._since_detection += 1
    if self.frame_count == 1:                 # 首帧立即检测
        self._since_detection = 0
        return True
    if self._since_detection >= self._detection_interval:
        self._since_detection = 0
        return True
    return False
```

`FrameResult`（应用层返回体，也是与表现层的"帧消息"）字段：

```python
@dataclass
class FrameResult:
    frame_index: int
    timestamp: float
    tracks: Tuple[TrackedObject, ...]
    created: Tuple[int, ...]
    removed: Tuple[int, ...]
    matched: Tuple[Tuple[int, int], ...]
    matched_ids: Tuple[Tuple[int, str], ...]   # 新：可直接消费的 (track_id, label)
    detections: Tuple[Detection, ...]
    spawn_failures: int
```

#### `application/session.py` — 会话编排（拆自 `app.run`，修 A10）
把"读帧→种子→交互→step→写 sink→渲染"编排为可无 GUI 测试的用例类，**不 import cv2**：

```python
class TrackingSession:
    def __init__(self, pipeline, source, clock, sinks=(), renderer=None, max_frames=None): ...
    def run(self) -> SessionStats:
        """纯编排：每帧 read→apply actions→step→persist→render→emit events。"""
    def handle_event(self, event: str) -> None:      # quit/pause/reset/clear_trails/save_frame
    def queue_box(self, bbox) -> None
    def queue_delete(self, point) -> None
```

接口规范：

| 方法 | 输入 | 输出 | 说明 |
| --- | --- | --- | --- |
| `run()` | — | `SessionStats` | 主循环；无 cv2 亦可运行（renderer=None） |
| `handle_event` | 事件名 | — | 与 `adapters.ui` 的事件命名一致 |
| `queue_box/queue_delete` | bbox/point | — | 由 UI 适配器入队 |

#### `application/config.py` — 配置（修 G1）
改为**引用 `domain/limits` 的校验器**，取消重复且不一致的范围定义；新增 detector 配置段（为 §6 接线预留）。

```yaml
# configs/default.yaml（目标）
source:   { source: "0", max_frames: null, loop: false, seed_boxes: [] }
tracking: { backend: csrt, iou_threshold: 0.3, max_missing: 15, trail_length: 120,
            match_method: greedy, max_tracks: 256 }
detector: { backend: none, model: null, confidence: 0.35, classes: [], interval: 1 }
output:   { csv_path: outputs/trajectories.csv, video_path: null, append: false, emit_events: false }
display:  { enabled: true, show_trails: true, show_labels: true, show_hud: true,
            trail_fade: true, window_name: "OpenCV MOT" }
```

优先级：**defaults < 配置文件 < 环境变量（`MOT_*`）< CLI**。未知键报错；越界报错；`--dump-config` 打印合并结果。

#### `application/errors.py` — 错误基类与退出码映射

```python
class MotError(Exception):
    exit_code = 2

class ConfigError(MotError): ...              # 配置/CLI 错误
class SourceError(MotError): ...              # 输入打开失败
class SinkError(MotError): ...                # 输出打开/写入失败
class TrackerSetupError(MotError): ...        # 后端不可用/构造失败
```

### 3.3 适配层（`adapters/`）

| 模块 | 实现端口 | 关键接口 | 依赖 |
| --- | --- | --- | --- |
| `sources.py` `VideoSource` | `FrameSource` | `read()`、`fps`、`name`、`release()`；`resolve_source` 校验路径 | cv2 |
| `trackers.py` | `SelfHostedTracker` + `TrackerFactory` | `create_tracker`、`make_tracker_factory`、`init_tracker`、`update_tracker` | cv2 |
| `detectors/yolo.py` `YoloDetector` | `Detector` | `__call__(frame, timestamp)`；`yolo_available()` | ultralytics（可选） |
| `sinks/csv.py` `CsvTrajectoryWriter` | `TrajectorySink` | `write_frame`、`flush`、`close`；`sanitize_label` | csv/os |
| `sinks/video.py` `VideoWriterSink` | —（输出适配器） | `write(canvas)`、`release()`、`codec` | cv2 |
| `rendering.py` | —（视图） | `draw_tracks/draw_hud/...` `frame→frame` 纯函数 | cv2 |
| `ui.py` `InteractiveSession` | —（交互状态） | `handle_key`、`handle_mouse`、`consume_*`、`drag_box` | 无（cv2 常量由 `interfaces/ui_window.py` 提供） |

**契约测试**：每个实现端口的适配器复用同一套 conformance 测试（见 §5.2）。

### 3.4 接口层（`interfaces/`）— 组合根

- `cli.py`：`build_parser` → `build_config` → **装配**（source/tracker factory/detector/sinks/renderer/clock）→ `TrackingSession(...).run()` → 退出码。
- `ui_window.py`：`namedWindow`、`setMouseCallback`、`waitKey`、`imshow`、`getWindowProperty`；把键/鼠标事件翻译为 `adapters.ui` 的调用与 `session.handle_event` 的事件名。

> **组合根职责**：只有接口层知道"用哪个适配器"，应用层只接收端口实例。这是保证可替换性的关键。

---

## 4. 模块间通信、错误处理、配置管理

### 4.1 通信机制

| 场景 | 机制 | 说明 |
| --- | --- | --- |
| 进程内主链路 | **同步函数调用** | 接口层驱动 `session.run()`，逐帧推进 |
| 帧级消息 | **`FrameResult` 数据对象** | 应用层 → 表现/落盘适配器的唯一"消息" |
| 事件输出 | **`EventSink` 端口** | `emit_events=true` 时把 `FrameResult` 推给消费者（日志/ROS/测试） |
| 时间来源 | **`Clock` 端口** | 修复 G6：`timestamp=clock.now()-t0`，不再依赖 fps 估算 |
| 未来异步 | 工作线程 + 有界队列 | 检测器独立线程，"最新帧优先"；跟踪循环保持实时（见 §6.2） |
| 跨进程（未来） | `EventSink` 的 ROS/gRPC 实现 | 适配器替换即可，领域层不动 |

**规则**：不得通过模块级全局变量传递状态；所有跨模块数据经入参/返回值；适配器不得反向调用应用层。

### 4.2 统一错误处理方案

| 层 | 异常类型 | 处理策略 |
| --- | --- | --- |
| domain | `ValidationError(ValueError)` | 立即抛出，携带字段名与非法值（校验边界） |
| application | `MotError` 子类 | 向上翻译；配置/输入错误 → `exit_code=2` |
| adapters | `SourceError`/`SinkError`/`TrackerSetupError` | 打开失败即抛；运行期可选失败**返回 None/False** |
| interfaces | 兜底 | `except MotError → 打印 + exit_code`；`KeyboardInterrupt → 130`；未处理异常 → 3 |

**分级处理原则**：

1. **致命**（启动期）：后端不可用、源/输出打不开、配置非法 → 快速失败并入退出码 `2`。
2. **非致命**（循环内）：单次跟踪器 `update` 失败、单检测框非法、检测器抛错 → **捕获 + 限频告警 + 降级继续**，不得中断主循环。
3. **禁止**：裸 `except: pass`；宽 `except Exception` 必须 `LOGGER.warning` 且限定在适配器/非致命点（现有 `_spawn`、detector、`update_tracker` 处的宽捕获保留并补日志）。
4. **退出码契约**：`0` 正常 / `1` 无帧 / `2` 配置或输入输出错误 / `130` 中断。
5. **G1 修复落点**：`config.validate` 使用 `domain.limits.validate_iou_threshold`，非法值在配置阶段即报 `ConfigError`（exit 2），不再穿透到 `TrackManager`。

### 4.3 配置管理规范

- **单一真源**：所有范围/常量定义在 `domain/limits.py`；schema 与校验器引用之，禁止复制。
- **分层覆盖**：defaults → file → env(`MOT_*`) → CLI；每层只做浅合并，最终一次 `validate()`。
- **严格校验**：未知段/键报错；类型不符/越界报错；不接受静默截断。
- **可观测**：`describe()` 输出启动摘要；`--dump-config` 输出合并结果（YAML）。
- **安全默认**：输出路径拒绝目录与符号链接；CSV 标签清洗；配置体积 ≤1 MB。
- **新增配置项 checklist**：`limits 常量 → schema 字段 → validate 规则 → CLI 参数 → default.yaml 注释 → 文档 → 测试`（七步齐全才算完成）。

---

## 5. 质量属性标准

### 5.1 可维护性

- 单文件 ≤ 300 行、单函数 ≤ 50 行、圈复杂度 ≤ 10；`app.run` 拆分（A10）。
- 命名规范：模块小写下划线；类 `PascalCase`；常量 `UPPER`；私有前缀 `_`。
- `ruff`（E/F/W/I/B）+ `import-linter`（或 §2.5 架构测试）；`pyproject` 保持 py39 兼容。
- 删除死代码或显式 `experimental` 标注（G8）：`TrajectoryStore`/`write_points_atomic` 归入 `adapters/sinks/csv.py` 或标注实验性。
- 文档与代码同步：`docs/ARCHITECTURE_DESIGN.md`（本文件）+ `ARCHITECTURE.md`（现状）+ `ROADMAP.md`（计划）职责分离。

### 5.2 可测试性

| 层级 | 测试方式 | 依赖 |
| --- | --- | --- |
| domain | `FakeTracker`/`FakeDetector` 驱动；纯 Python | 无 cv2 |
| application | `FakeSource`/`FakeClock`/内存 sink 驱动 `TrackingSession.run()` | 无 cv2、无 GUI |
| adapters | 契约测试（同一套 port conformance）+ `importorskip("cv2")` | cv2 可选 |
| interfaces | 端到端（合成视频 → 跟踪 → CSV/录像/错误路径退出码） | cv2 |

**KPI**：domain+application 覆盖率 ≥ 90%；每个端口 ≥1 套契约测试；无 GUI 逻辑混入用例层。

### 5.3 性能标准

- **预算**：定义每阶段耗时上限（采集/预处理/跟踪/关联/渲染/落盘），以 640×480 为基准；给出基准脚本与数据。
- **规则**：非显示/录像路径**不做整帧 copy**；CSV 批量 `flush`（默认每帧可配）；IoU/关联向量化（已满足）；关联矩阵与检测数受 `max_tracks`/`MAX_DETECTIONS` 上界约束。
- **优化手段**：ROI/降采样、灰度化预处理、`cv2.UMat`、检测异步化、必要时 `cv2.cuda`。
- **度量**：`FpsMeter` 滑动窗口有效 FPS + 单帧各阶段计时。

### 5.4 技术债防控（模块 DoD 检查单）

每个新模块/功能合并前必须满足：

- [ ] 归层正确且通过 §2.5 架构测试；
- [ ] 对外接口以 `Protocol` 或 dataclass 明确；
- [ ] 非致命失败返回状态、致命失败抛 `MotError` 子类；
- [ ] 配置项走 §4.3 七步 checklist；
- [ ] domain/application 无 cv2、可无头测试；
- [ ] 单测 + （若为适配器）端口契约测试通过；
- [ ] 无新增全局可变状态与裸宽捕获；
- [ ] 文档（模块 docstring）与 `docs/` 同步。

---

## 6. 实施指引（可直接落地）

### 6.1 分阶段迁移路径

| 阶段 | 目标 | 动作 | 验收 |
| --- | --- | --- | --- |
| P0 架构守卫 | 防新增债 | 落地 `domain/limits.py`、`ports.py`、`errors.py` 与 `tests/test_architecture.py` | 架构测试通过；`ruff` 通过 |
| P1 领域层搬迁 | 零风险 | `iou→geometry`、`matching→association`、`track_manager→lifecycle`（注入 `init_fn`/`update_fn`） | 现有 279 用例全绿（仅改 import） |
| P2 应用层拆分 | 修 A10/G6/G7/G9 | `trajectory` 拆 records/csv；`pipeline` 加 `timestamp` 与首帧检测；新增 `session.py`、`FrameResult.matched_ids` | application 层无 cv2 用例覆盖 |
| P3 适配层搬迁 | 对齐端口 | `video_source/trackers/visualization/interactive` 迁入 `adapters/`；`interactive` 的 cv2 常量解耦到 `interfaces/ui_window.py` | 端口契约测试通过 |
| P4 接口层重组 | 组合根 | `app.py` → `interfaces/cli.py`（薄壳）+ `ui_window.py`；装配逻辑集中 | 端到端与退出码测试通过 |
| P5 YOLO 接线 | 扩展 | config/CLI 暴露 `detector` 段；`YoloDetector.__call__(frame, timestamp)` 对齐端口 | smoke 测试（假 detector）通过 |

**兼容策略**：`mot/__init__.py` 保持现有公开符号再导出（`MOTPipeline`、`TrackManager`、`make_tracker_factory`、`Detection`…），新增符号追加，不破坏 `mot-track` CLI 参数与 CSV 契约。旧路径可在 P3/P4 之后保留一个版本期的 re-export 垫片，随后删除。

### 6.2 面向未来（YOLO / 机器人）的装配示例

```python
# interfaces/cli.py（组合根节选）
source   = VideoSource(config.source.source, loop=config.source.loop)      # FrameSource 适配器
factory  = make_tracker_factory(config.tracking.backend)                   # TrackerFactory
detector = build_detector(config.detector) or None                         # Detector 适配器（可选）
clock    = MonotonicClock()                                               # Clock 适配器
sinks    = [CsvTrajectoryWriter(config.output.csv_path, append=...)]       # TrajectorySink
pipeline = MOTPipeline(TrackManager(factory, **tracking_kwargs), detector=detector,
                       detection_interval=config.detector.interval)
session  = TrackingSession(pipeline, source, clock, sinks=sinks,
                           renderer=WindowRenderer(...) if display else None)
return session.run().exit_code
```

机器人侧的改造点（不改领域层即可完成）：把 `VideoSource` 换成 ROS2 image 适配器；新增 `EventSink` 实现发布 `/tracks`；检测器放入独立线程 + 有界队列；`Clock` 换成 ROS 时间源。

---

## 7. 附录

### 7.1 常量与参数（目标基线）

| 名称 | 值 | 来源 |
| --- | --- | --- |
| `MIN_BOX_SIDE` | 1.0 px | `domain/limits` |
| `MIN_DRAG_SIDE` | 8.0 px | `adapters/ui` |
| `MAX_TRACKS` / `MAX_DETECTIONS` | 256 / 256 | `domain/limits` |
| `MAX_MISSING_LIMIT` / `MAX_TRAIL_LENGTH` | 1000 / 10000 | `domain/limits` |
| `iou_threshold` 范围 | `(0, 1]`（单一真源） | `domain.limits.validate_iou_threshold` |
| 配置体积上限 | 1 MB | `application/config` |
| 退出码 | 0/1/2/130 | `application/errors` |

### 7.2 命名与分层速查

```text
domain      : 名词（模型/几何/生命周期），无动词 I/O
application : 用例（Pipeline/Session/Config），编排与校验
adapters    : 外部系统名（VideoSource/YoloDetector/CsvSink），实现 ports
interfaces  : 入口名（cli/ros/ui），组合根
utils       : 纯工具（FpsMeter/sanitize_filename）
```

### 7.3 与既有文档的关系

| 文档 | 定位 |
| --- | --- |
| `docs/DESIGN_ANALYSIS.md` | 现状需求与技术债审查（本次设计的输入） |
| `docs/ARCHITECTURE_DESIGN.md`（本文件） | 目标架构与实施指引（本次设计输出） |
| `docs/ARCHITECTURE.md` | 现状实现的分层与数据流说明（迁移完成后更新为"当前实现"） |
| `docs/ROADMAP.md` | 里程碑与问题跟踪（把 §1.4 的技术债映射为条目） |
