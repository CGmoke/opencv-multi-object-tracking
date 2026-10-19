# 项目设计分析报告：需求分析 + 模块架构设计

> 对象仓库：`opencv-multi-object-tracking`（纯 OpenCV 多目标跟踪）
> 分析基线：commit `495762a`（`v0.2.0`）
> 说明：分析时工作区存在**未提交的文件删除**（`src/mot/**`、`tests/**`），本报告内容取自 git HEAD 中的代码，恢复命令 `git restore src tests`。
> 结构：按用户要求覆盖 **需求完整性校验 / 模块划分合理性 / 核心流程逻辑 / 可扩展性** 四个维度。

---

## 0. 结论摘要

| 维度 | 结论 | 关键证据 |
| --- | --- | --- |
| 需求完整性 | **基本完备，存在 3 处一致性/边界缺口与若干未验证项** | 核心功能齐全，但 `iou_threshold` 边界在 config 与 manager 之间不一致（G1），`reset` 后 CSV 帧号回绕（G2/A1），`_refresh` 无异常保护（G3/A5） |
| 模块划分 | **分层清晰、依赖倒置正确，领域层零 cv2 依赖**，但存在少量未接线/死代码与一个约 200 行的过程式编排函数 | `TrackManager` 无 `import cv2`，跟踪器/检测器以工厂与 `Callable` 注入 |
| 核心流程 | **初始化→帧间跟踪→IoU 关联→ID 生命周期**闭环完整，状态机仅 `ACTIVE/LOST` 两态，语义明确 | `step()` 先推进跟踪器再吸收检测，命中即 `_refresh` 恢复 ID |
| 可扩展性 | **对 YOLO 接入适配性良好（依赖倒置已就位）**，但尚未接线到 config/CLI；对机器人场景需补齐"时间戳随帧传入、事件化输出、检测异步解耦、3D 坐标变换"四项 | `MOTPipeline(detector=...)` 已支持，`YoloDetector` 适配器存在但未接 CLI |

**最高优先级修复**：G1（配置 `iou_threshold: 0` 会以未捕获异常崩溃而非退出码 2）、G2/A1（CSV 帧号语义）、G3/A5（`_refresh` 异常穿透 `step`）。

---

## 1. 需求完整性校验

### 1.1 功能性需求矩阵

| 编号 | 功能需求 | 实现位置 | 状态 |
| --- | --- | --- | --- |
| F1 | 视频文件 / 摄像头输入，支持 `--loop` | [video_source.py](file:///home/moke/opencv-multi-object-tracking/src/mot/video_source.py) `VideoSource` / `resolve_source` | 已实现 |
| F2 | 每目标一个跟踪器（CSRT/KCF/MIL/MOSSE） | [trackers.py](file:///home/moke/opencv-multi-object-tracking/src/mot/trackers.py) `make_tracker_factory` + [track_manager.py](file:///home/moke/opencv-multi-object-tracking/src/mot/track_manager.py) | 已实现 |
| F3 | IoU 帧间关联（greedy / hungarian） | [iou.py](file:///home/moke/opencv-multi-object-tracking/src/mot/iou.py) + [matching.py](file:///home/moke/opencv-multi-object-tracking/src/mot/matching.py) | 已实现 |
| F4 | ID 生命周期（丢失容忍、超时销毁、新 ID 分配） | `TrackManager` + `models.IdAllocator` / `TrackState` | 已实现 |
| F5 | 轨迹绘制（带衰减）与 CSV 导出 | [visualization.py](file:///home/moke/opencv-multi-object-tracking/src/mot/visualization.py) + [trajectory.py](file:///home/moke/opencv-multi-object-tracking/src/mot/trajectory.py) | 已实现 |
| F6 | 交互控制（暂停/新增/删除/重置/清轨迹/截图/帮助） | [interactive.py](file:///home/moke/opencv-multi-object-tracking/src/mot/interactive.py) + [app.py](file:///home/moke/opencv-multi-object-tracking/src/mot/app.py) `apply_event` | 已实现 |
| F7 | 无显示运行、录像输出、`--max-frames` | `app.py` `_display_available` / `VideoWriterSink` | 已实现（CI 只验证前者） |
| F8 | 严格 YAML 配置加载与 CLI 覆盖 | [config.py](file:///home/moke/opencv-multi-object-tracking/src/mot/config.py) | 已实现 |
| F9 | 可选检测器注入（YOLO 适配器） | [pipeline.py](file:///home/moke/opencv-multi-object-tracking/src/mot/pipeline.py) + [detectors/yolo.py](file:///home/moke/opencv-multi-object-tracking/src/mot/detectors/yolo.py) | **部分实现**：接口就位，未接 config/CLI，无自动化测试 |
| F10 | 轨迹内存存储 + 原子写 | `TrajectoryStore` / `write_points_atomic` | **未接线**：内部未使用（死代码） |

### 1.2 非功能性需求矩阵

| 类别 | 需求 | 现状 | 缺口 |
| --- | --- | --- | --- |
| 性能 | 实时性 | 未定义目标 FPS；CSRT 640×480 约十几~几十 FPS，KCF 60+ FPS（docstring 声明，未基准） | 缺基准脚本与延迟预算 |
| 资源 | 有界内存 | `max_tracks=256`、`MAX_DETECTIONS=256`、`trail_length=120`、配置 ≤1 MB、`MAX_SEED_BOXES=100` | 充分 |
| 可靠性 | 退出码语义 | 0 正常 / 1 无帧 / 2 参数错误 / 130 Ctrl-C | 见 G1：`iou_threshold:0` 走未捕获异常 |
| 安全 | 符号链接拒绝、CSV 公式注入转义、配置体积/类型上界 | 已实现（CWE-59 防护） | 充分 |
| 可移植 | Python ≥3.9、OpenCV ≥4.5.1、跨平台 | 声明支持；仅 Linux 无显示环境实测 | 摄像头/GUI/高版本 OpenCV 未在 CI 验证 |
| 可测性 | 领域层可无 cv2 单测 | 279 用例，`FakeTracker` 驱动核心逻辑 | 充分 |

### 1.3 输入与边界规格（明确定义）

- **输入格式**：`--source` 接受摄像头索引（`int` 或数字字符串）或视频文件路径；路径必须存在且为普通文件（[video_source.py](file:///home/moke/opencv-multi-object-tracking/src/mot/video_source.py) `resolve_source`）。
- **编解码**：依赖 OpenCV 的 FFMPEG/GStreamer 后端，打开失败给出可诊断错误信息。
- **分辨率**：**代码未做约束**（帧对领域层不透明，仅 `frame.shape` 用于裁剪）；种子框/手工框越界即 `clip_bbox` 裁剪，完全出界则 `None` 跳过。→ 缺口 G4。
- **帧率**：`VideoSource.fps` 取 `CAP_PROP_FPS`，非有限或 ≤1e-3 时回退 **30.0**，上限 **240.0**。
- **IoU 阈值**：默认 **0.3**，语义为**包含式** `>=`；`TrackManager` 强制 `(0, 1]`（[track_manager.py](file:///home/moke/opencv-multi-object-tracking/src/mot/track_manager.py)），同时用于**关联**与**同帧去重**（手工框与检测框共用）。
- **ID 生命周期规则**：ID 由 `IdAllocator` **单调自增、会话内不复用**，起始 1；目标丢失 `misses > max_missing`（严格大于）时销毁；`reset()` 默认**保留计数器**，仅显式传 `start_id` 才重排；`max_tracks` 为硬上限。

### 1.4 需求缺口与一致性风险

| 编号 | 问题 | 证据（模块.符号） | 影响 | 建议 |
| --- | --- | --- | --- | --- |
| **G1** | `iou_threshold` 边界不一致 | `config.validate` 允许 `[0,1]`；`TrackManager.__init__` 要求 `(0,1]` 并抛 `ValueError`；`app.run` 仅捕获 `TrackerBackendError` | 配 `iou_threshold: 0` 会**未捕获异常崩溃**，不走退出码 2 | config 下限改开区间校验；或在 `app.run` 捕获 `ValueError` → 退出码 2 |
| **G2 / A1** | `reset` 后 `frame` 列从 1 重新计数 | `TrackManager.reset` 置 `frame_index=0` | 同一 CSV 出现两段帧号，按帧对齐视频会串帧 | 用会话全局帧号写 CSV，或 `r` 时轮转 CSV |
| **G3 / A5** | 关联命中后 `_refresh` 重初始化跟踪器无 try/except | `TrackManager._refresh` 调用 `init_tracker` 未保护 | 非法框会让异常穿出 `step` | 捕获异常 → `misses+=1` 转 `LOST` |
| G4 | 分辨率/帧率未作为需求约束校验 | 无最小/最大分辨率校验，无目标 FPS 指标 | 极端输入（超小/超大帧）行为未定义 | 增加分辨率/帧率范围校验与文档指标 |
| G5 | 摄像头与 GUI 未在 CI 验证 | README「已知限制」 | 真实场景证据不足 | 增手测清单与 CI 版本矩阵（A3） |
| G6 | 时间戳基于 FPS 估算 | `app.run` `timestamp = processed / fps_estimate` | 摄像头漂移、`--loop` 后与真实时间脱钩 | 时间戳随帧传入 |
| G7 | `FrameResult.matched` 返回**下标对**而非 `track_id` | `pipeline.py` / ARCHITECTURE §5 | 下游易误用为 ID | 提供 `track_id` 视图或便捷映射 |
| G8 | 死代码/未接线 API | `TrajectoryStore`、`write_points_atomic`、`pairwise_centroid_distance`、`DEDUPE_IOU` 常量、`YoloDetector` | 公共 API 面偏大、文档负担 | 标注 experimental 或接入 |
| G9 / A8 | `detection_interval` 语义：首个检测在第 `interval` 帧 | `MOTPipeline.step` 用 `frame_count % interval == 0` | 短片段可能一次都不检测 | 首帧即检测或明确文档 |
| G10 | `--loop` 无 `--max-frames` 时无限运行 | `app.run` | 误用风险 | 加警告/确认 |
| G11 | 调度为**单进程顺序**，多目标时跟踪器逐帧线性执行 | `TrackManager.step` 串行循环 | 目标数多时实时性下降，无并行/异步 | 见 §4.2 异步解耦 |

---

## 2. 模块架构设计

### 2.1 分层与模块清单

```text
┌──────────────────────────────────────────────────────────────┐
│ 表现层   app.py          CLI、主循环、窗口、事件分发           │
│          visualization.py 绘制（纯函数，frame->frame）         │
│          interactive.py   交互状态机（不依赖 cv2）            │
├──────────────────────────────────────────────────────────────┤
│ 应用层   pipeline.py      单帧管道：检测→去重→跟踪           │
├──────────────────────────────────────────────────────────────┤
│ 领域层   track_manager.py ID 生命周期与 IoU 关联              │
│          matching.py      贪心 / 匈牙利匹配                   │
│          iou.py           重叠与距离度量                      │
│          models.py        数据模型、校验、ID 分配             │
├──────────────────────────────────────────────────────────────┤
│ 基础层   trackers.py      OpenCV 跟踪器兼容封装（后端工厂）   │
│          video_source.py  视频输入 / 录像输出                 │
│          trajectory.py    CSV 轨迹导出                        │
│          config.py        YAML 配置校验                       │
│          utils.py         FPS、日志、文件名                   │
│          detectors/yolo.py 可选检测器适配器（扩展点）         │
└──────────────────────────────────────────────────────────────┘
```

**约束（已满足）**：领域层不 `import cv2`；跟踪器以零参工厂注入，检测器为 `Callable[[Frame], Sequence[Detection]]`。表现层之外只消费 `FrameResult`。

### 2.2 模块接口规范与数据流转

| 用户划分模块 | 承载代码 | 输入 | 输出 | 依赖 |
| --- | --- | --- | --- | --- |
| **输入采集** | `VideoSource` / `VideoWriterSink` | spec(索引/路径), loop | `read()->(bool, ndarray HxWx3 uint8 BGR)`；`width/height/fps/frame_count` | cv2 |
| **跟踪器管理（池）** | `trackers.py` + `TrackManager._tracks` | 后端名 → `TrackerFactory`；`frame, bbox` | `create_tracker/make_tracker_factory`；`init_tracker`；`update_tracker->Optional[BBox]` | cv2（仅 trackers.py） |
| **IOU 匹配关联** | `iou.py` + `matching.py` | `boxes_a, boxes_b`, `threshold, method` | `iou_matrix->(N,M) float64`；`match->(pairs, unmatched_rows, unmatched_cols)` | numpy |
| **ID 生命周期** | `TrackManager` + `models` | `frame, detections` | `StepResult(tracks, created, removed, matched, spawn_failures)`；`IdAllocator()->int` | numpy（经 iou/matching） |
| **轨迹绘制与存储** | `visualization.py` + `trajectory.py` | `frame, tracks`；`frame_index, timestamp, tracks` | `frame`；`CsvTrajectoryWriter.write_frame->行数`（`CSV_HEADER` 固定 10 列） | cv2（绘制）/ csv |
| **交互控制** | `interactive.py` + `app.apply_event` | `key:int`, `mouse(event,x,y,flags)` | `handle_key->event名`；`consume_new_boxes()->[BBox]`；`consume_delete_points()->[Point]` | cv2（仅鼠标常量） |
| **单帧编排** | `MOTPipeline` | `frame, extra_detections` | `FrameResult(frame_index, tracks, created, removed, matched, detections, spawn_failures)` | 领域层 |
| **配置** | `AppConfig` | YAML 路径 / CLI 覆盖 | 校验后的 `AppConfig`；`describe()->str` | PyYAML |

### 2.3 单帧数据流

```text
video/camera ─► VideoSource.read ─► frame
                                       │
   seed_boxes / session 新增框 ────────┼──► MOTPipeline.add_manual_box ─► TrackManager.add_or_refresh
                                       ▼
                              MOTPipeline.step(frame)
                                 │ 1 规范化检测
                                 │ 2 可选 detector(frame)（按 interval）
                                 │ 3 同帧去重（IoU ≥ iou_threshold，保留高置信）
                                 ▼
                          TrackManager.step(frame, detections)
                             1) 逐目标 update_tracker → 成功刷新 / 失败 LOST / 超时销毁
                             2) iou_matrix + match → 命中 _refresh 恢复 ID；未命中 _spawn 新 ID
                                 ▼
                             FrameResult
              ┌────────────────┼───────────────────┐
              ▼                ▼                    ▼
   visualization.draw_*   CsvTrajectoryWriter   VideoWriterSink / imshow
   （canvas）             .write_frame          （录像 / 预览）
```

### 2.4 跟踪器池管理机制与并行调度

- **池结构**：`TrackManager._tracks: Dict[int, TrackedObject]`，key 为单调 `track_id`；每个 `TrackedObject` 持有**独立 tracker 实例**、`bbox`、`trail`、`hits/misses/age/state`。
- **创建**：仅在 `_spawn` 中通过 `tracker_factory()` 新建并 `init_tracker`；先做 `max_tracks` 容量检查，再 `try/except` 包住构造与初始化（失败计入 `spawn_failures`，不中断循环）。
- **刷新/恢复**：关联命中已有 ID 时 `_refresh` 用**新观测框重新 `init`** 该目标跟踪器（保证 IoU 命中即"重新锚定"）。
- **调度现状**：`step()` 为**单进程顺序循环**，逐目标 `update` 后统一做一次关联矩阵。**不存在真正的多跟踪器并行**；并行度上界受 `max_tracks`、关联矩阵 `O(N·M)` 与 CSRT 单帧成本约束。真正的并行化需在 §4.2 的异步解耦中引入。

### 2.5 轨迹内存结构与 CSV 持久化

- **内存结构**：轨迹挂在 `TrackedObject.trail: Deque[Point]`，`push_trail` 去重相邻同点并 `deque` 截断至 `trail_length`（默认 120）。另有独立的 `TrajectoryStore`（`OrderedDict[int, list[TrackPoint]]`，每目标上限 10000），**当前内部未使用**。
- **CSV 契约**：列固定 `track_id,frame,timestamp,x,y,w,h,cx,cy,label`；仅写 `ACTIVE` 目标；`timestamp=frame/fps`（恒定帧率假设）；每帧 `flush`（`flush_every` 可调）；默认覆盖写，`--append` 追加；标签经 `sanitize_label`（去 BOM/零宽、`= + - @` 前缀加 `'`、限 64 字符）；写盘前**拒绝符号链接**。
- **原子写**：`write_points_atomic`（临时文件 + `os.replace`）提供整文件原子落盘，但**未接线**。

---

## 3. 核心逻辑设计说明

### 3.1 目标初始化

三条入口，最终都落到 `_spawn` 或 `add_or_refresh`：

1. **启动种子框** `--seed-box`：首帧 `clip_bbox` 裁剪 → `pipeline.add_manual_box`；出界跳过并告警。
2. **交互拖拽新增**（`n` 进入 add 模式，拖拽 ≥ `MIN_DRAG_SIDE=8px`）：`InteractiveSession` 入队 → 主循环 `consume_new_boxes()` → `add_manual_box`。
3. **检测器输出**：`MOTPipeline.step` 中未匹配的检测 → `TrackManager._spawn`。

统一校验：`normalize_bbox`（允许负宽高翻正、拒绝非有限值、边长 < `MIN_BOX_SIDE=1.0` 报错）→ `_spawn` 分配新 ID → `init_tracker`（框取整到整数像素）。

### 3.2 帧间跟踪 + IoU 关联（`TrackManager.step`）

```text
frame_index += 1
① 推进已有目标（含 LOST）：
   age += 1 → update_tracker(tracker, frame)
     ├─ 成功：clip 到画面 → _accept_box（更新 bbox/hits/misses/last_seen/trail）
     └─ 失败：misses += 1 → state = LOST；若 misses > max_missing → 删除并记入 removed
② 关联本帧检测（若有）：
   _associate：track_boxes = 所有存活目标 bbox（LOST 仍保留最后 bbox）
             → iou_matrix → match(threshold, method) → (pairs, unmatched_tracks, unmatched_dets)
   命中：_refresh（新框重 init 跟踪器）→ state=ACTIVE → misses=0 → 若本帧未更新过则 hits+1
   未命中：_spawn 新 ID（容量/异常由 _spawn 兜底，失败计入 spawn_failures）
返回 StepResult(tracks, created, removed, matched, spawn_failures)
```

**关联语义要点**：
- `iou_threshold` 包含式；`greedy` 按 IoU 降序贪心、平局按 `(row,col)` 定序（可复现）；`hungarian` 用 `linear_sum_assignment`，低于阈值以大代价（`1e6`）禁止配对，**无 scipy 时回退 greedy 并告警一次**。
- 关联对 `LOST` 目标同样参与，故 IoU 命中即"救回 ID"。
- **手工框与检测框同帧去重**采用同一阈值（`MOTPipeline._dedupe` 按置信度降序，与已保留框 IoU ≥ 阈值即丢弃），确保"一物一 ID"。
- `matched` 是**当帧 `tracks` 列表下标对**，非 `track_id`（G7）。

### 3.3 目标消失判定与 ID 生命周期状态机

```text
     新检测 / 手工框
          │  _spawn（容量检查 + init）
          ▼
   ┌───────────────────┐    tracker 更新成功      ┌────────────────────┐
   │   ACTIVE          │ ◄─── _accept_box ──────── │ hits+1, misses=0   │
   └─────────┬─────────┘                          └────────────────────┘
             │ tracker 更新失败 → misses += 1
             ▼
   ┌───────────────────┐   IoU 关联命中 → _refresh（重 init）→ 回 ACTIVE
   │   LOST            │
   └─────────┬─────────┘
             │ misses > max_missing（默认 15；0=首次丢失即销毁）
             ▼
        ID 销毁（记入 removed）；同一位置再现时分配**新 ID**
```

- `state` 仅 `ACTIVE / LOST` 两态；`reset()` 清空所有目标但**默认保留计数器**（避免 CSV 内同 ID 指向不同目标）。
- 独立于状态机的"交互删除" `remove()/delete_at()` 直接销毁目标（命中面积最小的目标）。

### 3.4 交互功能实现逻辑与状态转换

`InteractiveSession` 只保存**状态**，所有 cv2 窗口调用留在 `app.py`：

| 输入 | 事件 | 状态变化 | 应用层动作 |
| --- | --- | --- | --- |
| `q`/`Esc` | `quit` | `quit_requested=True` | 退出循环 |
| 空格 | `pause` | `paused` 取反 + notice | 暂停/恢复；暂停期仍可新增/删除 |
| `n` | `add_mode` | `mode: browse↔add` | 拖拽松手入队 → `consume_new_boxes` → `add_manual_box` |
| `x` | `delete_mode` | `mode: browse↔delete` | 点击入队 → `consume_delete_points` → `delete_at` |
| `c` | `clear_trails` | — | `pipeline.clear_trails()`（保留 ID 与跟踪器） |
| `r` | `reset` | — | `pipeline.reset()`（销毁全部目标，编号继续） |
| `s` | `save_frame` | — | `save_snapshot`（同名加序号、拒绝符号链接） |
| `h` | `toggle_help` | `help_visible` 取反 | 显示/隐藏帮助 |

**状态转换规则**：`mode` 为三态互斥（`browse/add/delete`），完成一次新增或删除后自动回到 `browse`；拖拽使用 `_drag_start/_drag_current` 区间，`drag_box` 仅用于预览；`notice` 带 TTL（`tick()` 递减）控制提示条淡出。鼠标坐标依赖 `WINDOW_AUTOSIZE`（1:1 像素）避免缩放映射问题。

---

## 4. 可扩展性评估

### 4.1 对后续结合 YOLO 检测模块的适配性

**结论：适配性良好（依赖倒置已就位），但尚需 4 处接线。**

已具备的支撑：

- `MOTPipeline(detector: Callable[[Frame], Sequence[Detection]], detection_interval)` 已存在，跟踪器覆盖检测间隔之间的帧。
- `YoloDetector` 适配器已实现：惰性导入 `ultralytics`（可选依赖），`xywh→xyxy` 并 `clip_bbox`，输出标准 `Detection(bbox, confidence, label)`。
- 容错与去重：检测器异常**捕获并告警一次**、不中断跟踪；同帧去重把检测框与手工框统一到 `iou_threshold`；`MAX_DETECTIONS=256` 按置信度截断，防止"检测风暴"。

待接线/待改：

1. **config/CLI 未暴露**：无 `detector` 配置段与 `--detector/--model/--detection-interval` 参数（ROADMAP A2/R6 已列）。
2. **`detection_interval` 语义待定**（G9/A8）：首次检测落在第 `interval` 帧。
3. **模型安全**：加载不可信 `.pt` 存在代码执行风险，应限制本地路径、默认不自动下载权重（SECURITY.md）。
4. **无自动化测试**：适配器未纳入 CI。

**建议接线路径**：`config.py` 增加 `detector: {backend: yolo, model, confidence, classes, interval}`（默认关闭）→ `app.build_config` 解析 → `app.run` 按 backend 构造 detector 传入 `MOTPipeline` → 补 smoke 测试（可用假 detector 保证 CI 可跑）。预处理/归一化/后处理契约由**适配器内部**承担，领域层只认 `Detection`——这一点当前设计已经正确。

### 4.2 针对机器人视觉追踪场景的架构适配建议

| 维度 | 现状 | 建议改造 | 优先级 |
| --- | --- | --- | --- |
| 时间基准 | `timestamp=frame/fps`（估算） | `step()` 增加 `timestamp` 参数，由采集端提供**单调时钟**；`--loop` 后与真实时间对齐 | P1 |
| 输出耦合 | 渲染/CSV 耦合在 `app.py` 主循环 | 基于 `FrameResult` 的 `created/removed/matched` **事件化输出**（如 ROS2 topic），与窗口解耦 | P1 |
| 检测 / 跟踪解耦 | 检测器与跟踪**同步串行** | 检测放独立工作线程 + 有界队列，"最新帧优先"；跟踪循环保持实时 | P1 |
| 坐标系统 | 仅像素 bbox/中心 | 新增相机模型模块（内参/畸变）+ 深度或地面单应 → **世界坐标**；`TrackedObject` 扩展世界位姿 | P2 |
| 运动模型 | 纯 IoU（快速运动不稳定） | 引入可选 **Kalman** 预测，与已有 `pairwise_centroid_distance` 融合为代价 | P2 |
| 多传感器一致性 | 单一时间戳 | 统一 TF/时间同步；多相机需扩展为跨镜头关联（当前非目标） | P2 |
| 嵌入式/边缘 | 未适配 | KCF/MOSSE 提速、降采样 ROI、`cv2.UMat`；Jetson 可考虑 `cv2.cuda` | P2 |
| 复用性 | `TrackManager` 无 cv2、无全局状态 | 可直接复用于 ROS 节点；`app.py` 主循环需改造（见上） | 已就绪 |

> 合规提醒：轨迹 CSV 含人员位置等敏感数据，机器人/安防场景须做脱敏、权限收紧与人工复核设计，遵守隐私与伦理要求。

---

## 5. 调试验证与性能部署要点

**功能验证指标**：初始化成功率、ID 切换次数、`created/removed` 计数一致性、CSV 行数与 `ACTIVE` 目标数一致、`spawn_failures` 计数。**精度（需 GT）**：IoU、IDF1、MOTA/MOTP。**性能**：有效 FPS（`FpsMeter` 滑动窗口）、单帧各阶段耗时（采集/预处理/跟踪/渲染/写盘）。

**调参顺序**：`iou_threshold`（0.3→0.4~0.5 减少 ID 交换；0.2~0.3 减少漏跟）→ `max_missing`（与遮挡时长匹配）→ `trail_length` → `backend`（KCF 提速 / CSRT 提准）→ `detection_interval`。

**性能热点（静态推断，未基准）**：① 每帧每目标一次 tracker `update`（线性于目标数）；② CSV 每帧 `flush`（可调 `flush_every`）；③ 显示/录像时整帧 `copy` 与绘制；④ 关联矩阵 `O(N·M)`（`max_tracks`/`MAX_DETECTIONS` 提供上界）。优化：仅在需要时渲染、ROI/降采样、CSV 批量 flush、灰度化预处理。

**部署建议**：`--no-display` 无头运行；线程模型上采集与跟踪分离、检测异步化；日志分级（`-v` 调试）；程序退出前 `finally` 正确释放 `VideoCapture`/`VideoWriter`/CSV。

---

## 6. 常见问题与改进

1. **配置 `iou_threshold: 0` 崩溃（G1）**：改为 config 层开区间校验或 `app.run` 捕获 `ValueError` → 退出码 2。
2. **`reset` 后 CSV 帧号回绕（G2/A1，P0）**：改用会话全局帧号，或 `r` 时轮转 CSV。
3. **`_refresh` 异常穿透（G3/A5，P1）**：捕获后 `misses+=1` 转 `LOST`。
4. **`matched` 语义误用（G7）**：补文档 + `track_id` 视图。
5. **死代码（G8/A9）**：`TrajectoryStore` / `write_points_atomic` / `pairwise_centroid_distance` 标注 experimental 或接入。
6. **`app.run` 约 200 行过程式编排（A10）**：抽取 `_process_frame` 等小函数，便于机器人侧复用。

---

## 附录：关键参数速查

| 参数 | 默认 | 范围/约束 | 语义 |
| --- | --- | --- | --- |
| `tracking.backend` | `csrt` | csrt/kcf/mil/mosse | 跟踪后端 |
| `tracking.iou_threshold` | `0.3` | config `[0,1]`，manager `(0,1]` | 关联 + 去重阈值（包含式） |
| `tracking.max_missing` | `15` | `[0,1000]` | 丢失容忍帧数（`>` 才销毁） |
| `tracking.trail_length` | `120` | `[1,10000]` | 每目标轨迹点上限 |
| `tracking.max_tracks` | `256` | `[1,10000]` | 同时存活 ID 硬上限 |
| `tracking.match_method` | `greedy` | greedy/hungarian | 关联算法 |
| `source.max_frames` | `null` | `[1,1e7]` | 处理帧数上限 |
| `output.csv_path` | `outputs/trajectories.csv` | null 关闭 | 轨迹输出 |
| `MIN_BOX_SIDE` / `MIN_DRAG_SIDE` | `1.0` / `8.0` | px | 最小框/最小拖拽 |
| `MAX_DETECTIONS` / `MAX_SEED_BOXES` | `256` / `100` | — | 单帧检测/种子框上限 |
| 退出码 | — | 0/1/2/130 | 正常/无帧/配置错误/中断 |
