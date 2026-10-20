"""应用层单帧用例：检测 → 去重 → 跟踪。

本层编排单帧流程，**不 import cv2**（帧对象对应用层不透明）。检测器为
``Callable[[Frame, float], Sequence[Detection]]``，与 ``domain.ports.Detector`` 对齐；
时间戳随帧传入（G6，替代按 FPS 估算）。

相对既有实现的修复：
* ``FrameResult`` 增加 ``timestamp`` 与 ``matched_ids``（G6 / G7）。
* 检测器首帧即触发（G9，取代“第 interval 帧才首次检测”）。
* 去重与关联共用 ``TrackManager.iou_threshold``。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from mot.domain.errors import ValidationError
from mot.domain.geometry import box_iou
from mot.domain.lifecycle import TrackManager
from mot.domain.limits import MAX_DETECTIONS
from mot.domain.models import (
    Detection,
    Point,
    TrackedObject,
    bbox_area,
    bbox_contains,
    normalize_bbox,
)

LOGGER = logging.getLogger(__name__)

#: 检测器契约：``(frame, timestamp) -> Sequence[Detection]``。
DetectorFn = Callable[[Any, float], Sequence[Detection]]


@dataclass
class FrameResult:
    """渲染器与导出器消费的“帧消息”。

    Attributes:
        frame_index: 帧序号（来自 ``TrackManager``）。
        timestamp: 该帧时间戳（秒），由会话按注入的 ``Clock`` 计算。
        tracks: 当帧存活（含 LOST）目标。
        created: 当帧新建的 ``track_id``。
        removed: 当帧销毁的 ``track_id``。
        matched: 关联命中的**下标对** ``(track_index, detection_index)``。
        matched_ids: 可直接消费的 ``(track_id, label)`` 视图（G7）。
        detections: 当帧去重后的检测。
        spawn_failures: 当帧建 ID 失败次数。
    """

    frame_index: int
    timestamp: float
    tracks: Tuple[TrackedObject, ...] = ()
    created: Tuple[int, ...] = ()
    removed: Tuple[int, ...] = ()
    matched: Tuple[Tuple[int, int], ...] = ()
    matched_ids: Tuple[Tuple[int, str], ...] = ()
    detections: Tuple[Detection, ...] = ()
    spawn_failures: int = 0


class MOTPipeline:
    """持有一个 :class:`TrackManager` 与可选周期检测器。

    Args:
        manager: 持有 id 与跟踪器的管理器。
        detector: 可选 ``(frame, timestamp) -> Sequence[Detection]``；``None`` 时
            进入手工模式（框来自鼠标或 ``--seed-box``）。
        detection_interval: 每 N 帧跑一次检测器；``1`` 表示每帧。首帧总是立即检测。
    """

    def __init__(
        self,
        manager: TrackManager,
        detector: Optional[DetectorFn] = None,
        detection_interval: int = 1,
    ) -> None:
        if detector is not None and not callable(detector):
            raise TypeError("detector must be callable: (frame, timestamp) -> Sequence[Detection]")
        if int(detection_interval) < 1:
            raise ValidationError(
                f"detection_interval must be >= 1, got {detection_interval}",
                field="detection_interval",
                value=detection_interval,
            )
        self._manager = manager
        self._detector = detector
        self._detection_interval = int(detection_interval)
        self._since_detection = 0
        self.frame_count = 0
        self._warned_detector_failure = False
        self._warned_truncation = False

    # -- 只读视图 --------------------------------------------------------
    @property
    def manager(self) -> TrackManager:
        return self._manager

    @property
    def tracks(self) -> Tuple[TrackedObject, ...]:
        return self._manager.tracks

    @property
    def detector(self) -> Optional[DetectorFn]:
        return self._detector

    def stats(self) -> Dict[str, int]:
        return {
            "frames": self.frame_count,
            "tracks": len(self._manager.tracks),
            "created_ids": self._manager.total_created,
            "removed_ids": self._manager.total_removed,
            "spawn_failures": self._manager.spawn_failures,
            "trail_points": sum(len(track.trail) for track in self._manager.tracks),
        }

    # -- 单帧 ------------------------------------------------------------
    def step(
        self,
        frame: Any,
        timestamp: float = 0.0,
        extra_detections: Sequence[Detection] = (),
    ) -> FrameResult:
        """推进管道一帧并返回其账目结果。"""
        detections = self._normalize(extra_detections)
        self.frame_count += 1
        if self._detector is not None and self._should_run_detector():
            try:
                produced = self._detector(frame, float(timestamp))
            except Exception as exc:  # noqa: BLE001 - 检测器抖动不得中断主循环
                if not self._warned_detector_failure:
                    LOGGER.warning(
                        "detector failed on frame %d (%s); tracking continues without it",
                        self.frame_count,
                        exc,
                    )
                    self._warned_detector_failure = True
                produced = None
            if produced:
                detections.extend(self._normalize(produced))
        detections = self._dedupe(detections)
        result = self._manager.step(frame, detections)
        return FrameResult(
            frame_index=self._manager.frame_index,
            timestamp=float(timestamp),
            tracks=result.tracks,
            created=result.created,
            removed=result.removed,
            matched=result.matched,
            matched_ids=result.matched_ids,
            detections=tuple(detections),
            spawn_failures=result.spawn_failures,
        )

    # -- 交互式编辑 ------------------------------------------------------
    def add_manual_box(self, frame: Any, bbox: Sequence[float]) -> Optional[TrackedObject]:
        """用用户绘制的框初始化目标，复用其重叠到的已有 id。"""
        detection = Detection(bbox=normalize_bbox(bbox))
        return self._manager.add_or_refresh(frame, detection)

    def delete_at(self, point: Point) -> Optional[int]:
        """删除包含 ``point`` 的最小存活目标。"""
        best: Optional[TrackedObject] = None
        for track in self._manager.tracks:
            if not track.is_active or not bbox_contains(track.bbox, point):
                continue
            if best is None or bbox_area(track.bbox) < bbox_area(best.bbox):
                best = track
        if best is None:
            return None
        self._manager.remove(best.track_id)
        return best.track_id

    def remove(self, track_id: int) -> bool:
        return self._manager.remove(track_id)

    def reset(self) -> None:
        self._manager.reset()
        self.frame_count = 0
        self._since_detection = 0

    def clear_trails(self) -> None:
        self._manager.clear_trails()

    # -- 内部实现 --------------------------------------------------------
    def _should_run_detector(self) -> bool:
        """是否应在当帧运行检测器（首帧立即检测，修 G9）。"""
        self._since_detection += 1
        if self.frame_count == 1:
            self._since_detection = 0
            return True
        if self._since_detection >= self._detection_interval:
            self._since_detection = 0
            return True
        return False

    @staticmethod
    def _normalize(detections: Sequence[Detection]) -> List[Detection]:
        out: List[Detection] = []
        for item in detections:
            if isinstance(item, Detection):
                out.append(item)
            elif isinstance(item, (list, tuple)) and len(item) == 4:
                out.append(Detection(bbox=normalize_bbox(item)))
            else:
                raise TypeError(
                    "detections must be Detection instances or (x, y, w, h) sequences, "
                    f"got {type(item).__name__}"
                )
        return out

    def _dedupe(self, detections: Sequence[Detection]) -> List[Detection]:
        """在相互重叠的框中只保留置信度最高的一个。

        阈值即关联阈值：同帧内两个“会被当作同一目标”的框不得都去建 id。
        """
        threshold = self._manager.iou_threshold
        ordered = sorted(detections, key=lambda det: det.confidence, reverse=True)
        kept: List[Detection] = []
        for detection in ordered:
            if any(box_iou(detection.bbox, other.bbox) >= threshold for other in kept):
                continue
            kept.append(detection)
        if len(kept) > MAX_DETECTIONS:
            if not self._warned_truncation:
                LOGGER.warning(
                    "%d detections after dedupe; keeping the %d most confident",
                    len(kept),
                    MAX_DETECTIONS,
                )
                self._warned_truncation = True
            kept = kept[:MAX_DETECTIONS]
        return kept


__all__ = ["DetectorFn", "FrameResult", "MOTPipeline"]
