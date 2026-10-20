"""应用层会话编排：把单帧用例串成可无 GUI 运行的主循环。

本层**不 import cv2**：数据源 / 时钟 / 轨迹接收器 / 事件接收器 / 渲染器全部以
``domain.ports`` 的端口实例注入。接口层（组合根）负责装配并调用
:meth:`TrackingSession.run`。

相对既有 ``app.run`` 的重构（修 A10）：把约 200 行的过程式编排拆成可无头测试的
用例；检测→关联逻辑留在 :mod:`mot.application.pipeline`，渲染与写入交给端口。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, List, Optional, Sequence

from mot.domain.models import BBox, Point, clip_bbox
from mot.domain.ports import (
    Clock,
    EventSink,
    Frame,
    FrameSource,
    Renderer,
    TrajectorySink,
)

from .pipeline import MOTPipeline

LOGGER = logging.getLogger(__name__)


@dataclass
class SessionStats:
    """一次会话的汇总统计与退出码。"""

    frames: int = 0
    created_ids: int = 0
    removed_ids: int = 0
    spawn_failures: int = 0
    trail_points: int = 0
    exit_code: int = 0


class TrackingSession:
    """编排“读帧 → 种子 → 交互 → step → 落盘 → 事件 → 渲染”的可无头用例。

    Args:
        pipeline: 单帧管道。
        source: 帧输入源（``FrameSource`` 端口）。
        clock: 时间来源（``Clock`` 端口），用于时间戳（G6）。
        sinks: 轨迹接收器（``TrajectorySink``）；只写 ``ACTIVE`` 目标。
        event_sinks: 帧事件接收器（``EventSink``）。
        renderer: 可选渲染/交互端口（``Renderer``）；``None`` 表示无头运行。
        max_frames: 处理帧数上限。
        seed_boxes: 首帧启动种子框。

    说明：``save_frame``（截图）涉及画面与文件 I/O，属接口/适配层职责，故不在本层
    事件里；渲染器可通过 :attr:`last_frame` / :attr:`last_result` 自行取画面。
    """

    def __init__(
        self,
        pipeline: MOTPipeline,
        source: FrameSource,
        clock: Clock,
        *,
        sinks: Sequence[TrajectorySink] = (),
        event_sinks: Sequence[EventSink] = (),
        renderer: Optional[Renderer] = None,
        max_frames: Optional[int] = None,
        seed_boxes: Sequence[BBox] = (),
    ) -> None:
        self._pipeline = pipeline
        self._source = source
        self._clock = clock
        self._sinks = tuple(sinks)
        self._event_sinks = tuple(event_sinks)
        self._renderer = renderer
        self._max_frames = None if max_frames is None else max(1, int(max_frames))
        self._seed_boxes = tuple(seed_boxes)

        self._pending_boxes: List[BBox] = []
        self._pending_deletes: List[Point] = []
        self._quit = False
        self._paused = False
        self._seeded = False
        self._last_frame: Optional[Frame] = None
        self._last_result: Any = None

    # -- 只读视图 --------------------------------------------------------
    @property
    def last_frame(self) -> Optional[Frame]:
        """最近一次呈现的原始帧（供截图等外层消费）。"""
        return self._last_frame

    @property
    def last_result(self) -> Any:
        """最近一次的 ``FrameResult``。"""
        return self._last_result

    def attach_renderer(self, renderer: Optional[Renderer]) -> None:
        """由组合根在装配完成后接上渲染器。

        用于解决装配顺序上的循环依赖：渲染器需要会话以回灌交互动作，而会话需要
        渲染器以呈现画面，故先建会话、再建渲染器、最后回接。
        """
        self._renderer = renderer

    # -- 交互 ------------------------------------------------------------
    def queue_box(self, bbox: Sequence[float]) -> None:
        """入队一个待新增的手工框（下一帧生效）。"""
        self._pending_boxes.append(tuple(float(v) for v in bbox))

    def queue_delete(self, point: Point) -> None:
        """入队一个待删除的点击位置（下一帧生效）。"""
        self._pending_deletes.append((float(point[0]), float(point[1])))

    def handle_event(self, event: Optional[str]) -> None:
        """处理一个交互事件；``None`` 与未知事件被忽略。"""
        if not event:
            return
        if event == "quit":
            self._quit = True
        elif event == "pause":
            self._paused = not self._paused
        elif event == "reset":
            self._pipeline.reset()
        elif event == "clear_trails":
            self._pipeline.clear_trails()
        else:
            LOGGER.debug("ignoring unknown session event %r", event)

    # -- 主循环 ----------------------------------------------------------
    def run(self) -> SessionStats:
        """运行一次会话并返回统计（含 ``exit_code``：无帧则为 ``1``）。"""
        started = self._clock.now()
        processed = 0
        try:
            while not self._quit:
                if self._paused and self._renderer is not None:
                    # 暂停时仍呈现最后一帧以接收输入（quit / pause）。
                    self.handle_event(self._renderer.present(self._last_frame, self._last_result))
                    continue

                ok, frame = self._source.read()
                if not ok or frame is None:
                    break

                if not self._seeded:
                    self._seeded = True
                    self._apply_seed(frame)
                self._apply_pending(frame)

                timestamp = self._clock.now() - started
                result = self._pipeline.step(frame, timestamp)
                self._persist(result, timestamp)
                for sink in self._event_sinks:
                    sink.on_frame(result)

                self._last_frame = frame
                self._last_result = result
                if self._renderer is not None:
                    self.handle_event(self._renderer.present(frame, result))

                processed += 1
                if self._max_frames is not None and processed >= self._max_frames:
                    break
        finally:
            for sink in self._sinks:
                sink.close()
            self._source.release()

        return self._collect_stats(processed)

    # -- 内部实现 --------------------------------------------------------
    @staticmethod
    def _clip(box: BBox, frame: Frame) -> Optional[BBox]:
        shape = getattr(frame, "shape", None)
        if shape is None or len(shape) < 2:
            return tuple(float(v) for v in box)  # 假帧：无尺寸信息，原样使用
        height, width = int(shape[0]), int(shape[1])
        if width <= 0 or height <= 0:
            return tuple(float(v) for v in box)
        return clip_bbox(box, width, height)

    def _apply_seed(self, frame: Frame) -> None:
        for index, box in enumerate(self._seed_boxes):
            clipped = self._clip(box, frame)
            if clipped is None:
                LOGGER.warning("seed box %d lies outside the frame; skipped", index)
                continue
            if self._pipeline.add_manual_box(frame, clipped) is None:
                LOGGER.warning("seed box %d could not initialise a tracker", index)

    def _apply_pending(self, frame: Frame) -> None:
        for box in self._pending_boxes:
            clipped = self._clip(box, frame)
            if clipped is None:
                continue
            self._pipeline.add_manual_box(frame, clipped)
        self._pending_boxes.clear()
        for point in self._pending_deletes:
            self._pipeline.delete_at(point)
        self._pending_deletes.clear()

    def _persist(self, result: Any, timestamp: float) -> None:
        if not self._sinks:
            return
        active = tuple(track for track in result.tracks if track.is_active)
        for sink in self._sinks:
            sink.write_frame(result.frame_index, timestamp, active)

    def _collect_stats(self, processed: int) -> SessionStats:
        raw = self._pipeline.stats()
        return SessionStats(
            frames=processed,
            created_ids=int(raw.get("created_ids", 0)),
            removed_ids=int(raw.get("removed_ids", 0)),
            spawn_failures=int(raw.get("spawn_failures", 0)),
            trail_points=int(raw.get("trail_points", 0)),
            exit_code=1 if processed == 0 else 0,
        )


__all__ = ["SessionStats", "TrackingSession"]
