# 架构说明

本文记录 `opencv-multi-object-tracking` 的分层、数据流、ID 生命周期与关键设计决策。实现细节以 `src/mot/` 为准，本文与代码同步维护。

## 1. 分层

```text
┌──────────────────────────────────────────────────────────────┐
│ 表现层   src/mot/app.py             CLI、窗口、主循环编排      │
│          src/mot/visualization.py   绘制（纯函数）             │
│          src/mot/interactive.py     交互状态机（不依赖 cv2）   │
├──────────────────────────────────────────────────────────────┤
│ 应用层   src/mot/pipeline.py        单帧管道：检测→去重→跟踪   │
├──────────────────────────────────────────────────────────────┤
│ 领域层   src/mot/track_manager.py   ID 生命周期与关联          │
│          src/mot/matching.py        贪心 / 匈牙利匹配          │
│          src/mot/iou.py             重叠与距离度量             │
│          src/mot/models.py          数据模型与校验             │
├──────────────────────────────────────────────────────────────┤
│ 基础设施 src/mot/trackers.py        OpenCV 跟踪器兼容封装      │
│          src/mot/video_source.py    视频输入 / 录像输出        │
│          src/mot/trajectory.py      CSV 轨迹导出               │
│          src/mot/config.py          YAML 配置校验              │
│          src/mot/utils.py           FPS、日志、文件名          │
└──────────────────────────────────────────────────────────────┘
```

约束：**领域层与基础设施层之间只通过接口交互**。`TrackManager` 不 `import cv2`，跟踪器以零参工厂注入；`MOTPipeline` 的检测器是 `Callable[[Frame], Sequence[Detection]]`。因此领域层可以在无 OpenCV 的环境里用假跟踪器完整测试。

## 2. 模块职责

| 模块 | 职责 | 关键符号 |
| --- | --- | --- |
| `models.py` | 数据模型、输入校验、ID 分配、轨迹缓冲 | `BBox`、`Detection`、`TrackedObject`、`TrackState`、`IdAllocator`、`StepResult`、`normalize_bbox`、`clip_bbox` |
| `iou.py` | 重叠度量 | `box_iou`、`iou_matrix`、`pairwise_centroid_distance` |
| `matching.py` | 关联算法 | `greedy_match`、`hungarian_match`、`match`、`match_boxes` |
| `track_manager.py` | ID 生命周期 | `TrackManager.step`、`add_or_refresh`、`remove`、`reset` |
| `pipeline.py` | 单帧管道 | `MOTPipeline.step`、`add_manual_box`、`delete_at`、`stats` |
| `trackers.py` | OpenCV 后端 | `create_tracker`、`make_tracker_factory`、`init_tracker`、`update_tracker` |
| `video_source.py` | 输入输出 | `VideoSource`、`VideoWriterSink`、`resolve_source` |
| `trajectory.py` | 轨迹导出 | `CsvTrajectoryWriter`、`TrackPoint`、`write_points_atomic` |
| `config.py` | 配置 | `AppConfig.from_file`、`validate`、`describe` |
| `interactive.py` | 交互状态 | `InteractiveSession.handle_key` / `handle_mouse` |
| `visualization.py` | 绘制 | `draw_tracks`、`draw_trail`、`draw_hud`、`draw_help`、`draw_banner` |
| `app.py` | 编排 | `main`、`run`、`build_config`、`save_snapshot` |

## 3. 单帧数据流

`MOTPipeline.step(frame, extra_detections=None)`：

1. **规范化检测**：接受 `Detection` 或 `(x, y, w, h)` 元组，转成 `Detection`（越界框裁剪，非法框拒绝）。
2. **可选检测器**：按 `detection_interval` 调用注入的 `detector`；异常被捕获并告警一次，不中断跟踪。
3. **同帧去重**：IoU 超过 `manager.iou_threshold` 的检测只保留置信度最高的一个（手工框与检测框共用同一阈值）。
4. **推进跟踪**：`TrackManager.step(frame, detections)`：
   - 逐目标 `update_tracker` 推进；成功则更新包围框、`hits`、`last_seen_frame`、`trail`；
   - 失败则 `misses += 1` 并转 `LOST`，`misses > max_missing` 时销毁 ID；
   - 有检测时用 `iou_matrix` + `match` 关联：命中已有目标则重新初始化其跟踪器（恢复 ID），未命中的检测生成新 ID；
   - 新 ID 数量受 `max_tracks` 限制，超限告警并跳过。
5. **返回 `FrameResult`**：`tracks`（当帧快照）、`created`、`removed`、`matched`、`detections`、`spawn_failures`。

`app.py` 只在渲染、写 CSV、写视频、处理按键处接触 OpenCV 与文件系统；替换 UI 时只需要消费 `FrameResult`。

## 4. ID 生命周期

```text
             新检测 / 手工框
                   │
                   ▼
   ┌──────────────────────────────┐
   │ ACTIVE  ──tracker 更新成功──► │  hits+1, misses=0, 轨迹追加
   └──────┬───────────────────────┘
          │ tracker 丢失
          ▼
   ┌──────────────────────────────┐
   │ LOST    ──IoU 关联命中──►     │  重新 init 跟踪器，ID 保留
   └──────┬───────────────────────┘
          │ misses > max_missing
          ▼
        ID 销毁（本轮 removed，同一位置再出现时分配新 ID）
```

- ID 由 `IdAllocator` 单调分配，会话内**不复用**；`reset()` 默认保留计数器（避免 CSV 里同 ID 指向不同目标），显式传 `start_id` 才重排。
- `max_missing=0` 时目标首次丢失即销毁；默认 `15` 帧。
- `trail` 是 `deque(maxlen=trail_length)`，连续相同点不重复追加。

## 5. 关联语义

- `iou_threshold` 是**包含式**阈值（`>=`），且必须落在 `(0, 1]`；`0` 会让任意检测劫持已有 ID，因此被拒绝。
- `greedy_match`：按 IoU 从高到低贪心配对，平局按 `(row, col)` 排序，结果确定。
- `hungarian_match`：`scipy.optimize.linear_sum_assignment` 全局最优；低于阈值用大代价禁止配对。**没有 scipy 时回退贪心并告警一次**。
- `FrameResult.matched` 是**当帧 `tracks` 列表的下标对** `(track_index, detection_index)`，不是 `track_id`——下游消费时需自行映射。

## 6. 轨迹与 CSV 契约

列固定为：

```text
track_id, frame, timestamp, x, y, w, h, cx, cy, label
```

- 只写 `ACTIVE` 目标；`timestamp = frame / fps`，假定恒定帧率。
- 默认覆盖写；`--append` 追加；父目录自动创建。
- 标签清洗：删除换行、零宽字符 / BOM，把 `= + - @` 开头加 `'` 前缀，限 64 字符（CSV 公式注入防护）。
- 写盘前拒绝符号链接目标（CWE-59）。

## 7. 关键设计决策

| 决策 | 理由 | 代价 |
| --- | --- | --- |
| 跟踪器工厂注入，而非 `TrackManager` 内直接 `create_tracker` | 领域层零 cv2 依赖，可用假跟踪器测试生命周期 | 调用方必须多传一个工厂 |
| ID 单调不复用，`reset` 默认不重排 | 同一 CSV 内 `track_id` 始终指向同一物理目标 | `r` 之后 `frame` 归零，需按会话切分 CSV |
| 手工框与现有 ID 的 IoU 达阈值时刷新该 ID | 防止同一物体被重复框选后拿到两个 ID | 快速移动时可能误合并 |
| 配置严格校验（未知键报错、类型/上界检查） | 配置错误尽早暴露，避免静默截断或类型异常穿透 | 新增配置项必须同步 schema |
| 有界资源（`max_tracks`、`MAX_DETECTIONS`、配置 1 MB、轨迹点上限） | 不可信输入下避免资源耗尽 | 极端场景需调参 |
| 写入前拒绝符号链接、CSV 注入转义 | 本地文件安全默认值 | 使用符号链接做输出目录的用户需改用真实路径 |

## 8. 已知问题（与 ROADMAP 对应）

| 编号 | 问题 | 影响 |
| --- | --- | --- |
| A1 | `r` 重置后 `frame` 列回绕，同一 CSV 出现两段 1..N | 按 `frame` 关联视频会串帧 |
| A5 | 关联命中后重新 `init` 跟踪器未包 try/except | 非法框可能让异常穿出 `step`（自定义检测器场景） |
| A8 | `detection_interval` 从 1 计数，首个检测发生在第 `interval` 帧 | 短片段可能无检测 |
| A9 | `TrajectoryStore`、`write_points_atomic`、`pairwise_centroid_distance` 等导出未被内部使用 | 公开 API 面偏大，文档负担 |
| — | GUI 窗口缩放后的鼠标坐标映射 | 已用 `WINDOW_AUTOSIZE`（1:1 像素）规避 |
| — | 时间戳基于 FPS 估算 | 摄像头会漂移，`--loop` 后失准 |

## 9. 性能画像（静态推断，未基准测试）

热点按成本排序：

1. 每帧每个目标一次跟踪器 `update`（线性于目标数；CSRT 在 640×480 上通常十几到几十 FPS）。
2. CSV 每帧 `flush`（可调 `flush_every`）。
3. 每帧整帧复制与绘制（仅在显示 / 录像时发生）。
4. 关联矩阵 `O(N·M)`（`max_tracks` 与 `MAX_DETECTIONS` 提供上界）。

## 10. 扩展点

- **新增跟踪后端**：在 `trackers.py` 的 `SUPPORTED_BACKENDS` 与候选工厂链里登记即可；启动期会做一次 eager 校验。
- **接入检测器**：实现 `Callable[[Frame], Sequence[Detection]]`，传给 `MOTPipeline(detector=...)`；若要暴露到 CLI，需要新增配置段并接线（当前未做）。
- **替换 UI / 下游**：消费 `FrameResult` 事件（`created` / `removed` / `matched`）即可，不依赖窗口。
- **机器人 / ROS**：`TrackManager` 可直接复用；建议把时间戳改为随帧传入，并异步化解耦检测与跟踪。

## 11. 测试策略

- 领域层：假跟踪器（`tests/conftest.py` 的 `FakeTracker`）驱动，覆盖 ID 生命周期、匹配、去重、检测间隔、重置语义。
- 基础设施层：临时目录 + 合成视频（mp4v），覆盖 CSV 写盘、追加、注入防护、符号链接拒绝、录像输出、错误路径退出码。
- OpenCV 相关用例模块级 `pytest.importorskip("cv2")`；无 GUI 环境下 `--no-display`。
