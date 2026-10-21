"""应用层单帧用例：检测 → 去重 → 跟踪。

本层编排单帧流程，**不 import cv2**（帧对象对应用层不透明）。检测器为
``Callable[[Frame, float], Sequence[Detection]]``，与 ``domain.ports.Detector`` 对齐；
时间戳随帧传入（G6，替代按 FPS 估算）。

相对既有实现的修复：
* ``FrameResult`` 增加 ``timestamp`` 与 ``matched_ids``（G6 / G7）。
* 检测器首帧即触发（G9，取代“第 interval 帧才首次检测”）。
* 去重与关联共用 ``TrackManager.iou_threshold``。
* **轨迹二次校验**：与活跃目标高度重叠、但尺寸相对已有轨迹突变过大的检测会被丢弃，
  避免同一目标的局部框顶替整体框或派生出新 id。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from mot.domain.errors import ValidationError
from mot.domain.geometry import box_iou, box_scale_change
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
        max_box_change: 轨迹二次校验的尺寸突变上限（默认 ``1.6``）。与活跃目标 IoU
            达到 ``manager.iou_threshold`` 的检测，若宽或高的相对变化超过该倍率，则
            视为同一目标的局部框/碎片被丢弃；``1.0`` 关闭该校验。
        static_iou: 静止判定的 IoU 阈值（默认 ``0.95``）。活跃目标在当前框与判定
            锚框的 IoU 连续 ``static_frames`` 帧不低于该值时，判定为“已静止”并销毁该 id。
        static_frames: 静止判定所需的连续帧数（默认 ``0`` 关闭；运动检测模式由配置开启）。
    """

    def __init__(
        self,
        manager: TrackManager,
        detector: Optional[DetectorFn] = None,
        detection_interval: int = 1,
        max_box_change: float = 1.6,
        static_iou: float = 0.95,
        static_frames: int = 0,
    ) -> None:
        if detector is not None and not callable(detector):
            raise TypeError("detector must be callable: (frame, timestamp) -> Sequence[Detection]")
        if int(detection_interval) < 1:
            raise ValidationError(
                f"detection_interval must be >= 1, got {detection_interval}",
                field="detection_interval",
                value=detection_interval,
            )
        if not 1.0 <= float(max_box_change) <= 100.0:
            raise ValidationError(
                f"max_box_change must be within [1, 100], got {max_box_change}",
                field="max_box_change",
                value=max_box_change,
            )
        if not 0.0 < float(static_iou) <= 1.0:
            raise ValidationError(
                f"static_iou must be within (0, 1], got {static_iou}",
                field="static_iou",
                value=static_iou,
            )
        if int(static_frames) < 0:
            raise ValidationError(
                f"static_frames must be >= 0, got {static_frames}",
                field="static_frames",
                value=static_frames,
            )
        self._manager = manager
        self._detector = detector
        self._detection_interval = int(detection_interval)
        self.max_box_change = float(max_box_change)
        self.static_iou = float(static_iou)
        self.static_frames = int(static_frames)
        self.size_rejects = 0
        self.static_removed = 0
        self._static_anchor: Dict[int, Tuple[float, float, float, float]] = {}
        self._static_hits: Dict[int, int] = {}
        self._warned_size_jump = False
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
            "size_rejects": self.size_rejects,
            "static_removed": self.static_removed,
            "tentative_dropped": self._manager.tentative_dropped,
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
        detections = self._reject_size_outliers(detections)
        detections = self._dedupe(detections)
        result = self._manager.step(frame, detections)
        removed_static = self._drop_static_tracks()
        # 轨迹生命周期过滤：未确认的暂定轨迹不进入渲染/导出/CSV。
        tracks = tuple(track for track in result.tracks if self._manager.is_confirmed(track))
        if removed_static:
            dropped = set(removed_static)
            tracks = tuple(track for track in tracks if track.track_id not in dropped)
            removed = result.removed + removed_static
        else:
            removed = result.removed
        return FrameResult(
            frame_index=self._manager.frame_index,
            timestamp=float(timestamp),
            tracks=tracks,
            created=result.created,
            removed=removed,
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
        self._static_anchor.clear()
        self._static_hits.clear()
        self.size_rejects = 0
        self.static_removed = 0

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

    def _drop_static_tracks(self) -> Tuple[int, ...]:
        """IoU 时间连续性判定：销毁长时间几乎不动的目标。

        以“判定锚框”为基准（而不是上一帧）：目标每帧的微小抖动不会重置计数，
        只有相对锚框累计位移足够大（IoU 掉到 ``static_iou`` 以下）才重新计时。
        ``LOST`` 目标不参与（遮挡期间没有可靠框）；重新激活后从头计时。
        """
        limit = self.static_frames
        if limit <= 0:
            return ()
        live = {track.track_id for track in self._manager.tracks}
        for track_id in list(self._static_anchor):
            if track_id not in live:
                self._static_anchor.pop(track_id, None)
                self._static_hits.pop(track_id, None)

        removed: List[int] = []
        for track in self._manager.tracks:
            if not track.is_active or not self._manager.is_confirmed(track):
                self._static_anchor.pop(track.track_id, None)
                self._static_hits.pop(track.track_id, None)
                continue
            anchor = self._static_anchor.get(track.track_id)
            if anchor is None:
                self._static_anchor[track.track_id] = track.bbox
                self._static_hits[track.track_id] = 1
                continue
            if box_iou(track.bbox, anchor) >= self.static_iou:
                hits = self._static_hits.get(track.track_id, 1) + 1
            else:
                self._static_anchor[track.track_id] = track.bbox
                hits = 1
            self._static_hits[track.track_id] = hits
            if hits >= limit and self._manager.remove(track.track_id):
                removed.append(track.track_id)
                self._static_anchor.pop(track.track_id, None)
                self._static_hits.pop(track.track_id, None)
        self.static_removed += len(removed)
        return tuple(removed)

    def _reject_size_outliers(self, detections: Sequence[Detection]) -> List[Detection]:
        """轨迹二次校验：丢弃与活跃目标重叠但尺寸突变的检测。

        典型误检是同一物体的局部框（面积只有整体框的一小部分）或碎片框；它们与
        已有轨迹 IoU 很高，会被关联成同一 id，随后把整体框替换成局部框。这里在
        去重/关联之前丢弃这类检测，目标继续由自身跟踪器推进，框尺寸保持稳定。

        只与 ``ACTIVE`` 目标比较：``LOST`` 目标重新出现时尺寸可能已变化，应允许
        重新捕获。``max_box_change <= 1.0`` 时关闭。
        """
        limit = self.max_box_change
        if limit <= 1.0 or not detections:
            return list(detections)
        tracks = self._manager.active_tracks
        if not tracks:
            return list(detections)

        kept: List[Detection] = []
        for detection in detections:
            best_iou = 0.0
            best_box: Optional[Sequence[float]] = None
            for track in tracks:
                iou = box_iou(detection.bbox, track.bbox)
                if iou > best_iou:
                    best_iou = iou
                    best_box = track.bbox
            if (
                best_box is not None
                and best_iou >= self._manager.iou_threshold
                and box_scale_change(detection.bbox, best_box) > limit
            ):
                self.size_rejects += 1
                if not self._warned_size_jump:
                    LOGGER.info(
                        "dropping detection with %.2fx size jump at frame %d",
                        box_scale_change(detection.bbox, best_box),
                        self.frame_count,
                    )
                    self._warned_size_jump = True
                continue
            kept.append(detection)
        return kept

    def _dedupe(self, detections: Sequence[Detection]) -> List[Detection]:
        """在相互重叠的框中只保留置信度最高的一个（同分时保留更大的整体框）。

        阈值即关联阈值：同帧内两个“会被当作同一目标”的框不得都去建 id。
        置信度相同的整体框与局部框必须让整体框胜出，否则局部框会先建 id、
        整体框被当重复丢弃。
        """
        threshold = self._manager.iou_threshold
        ordered = sorted(
            detections,
            key=lambda det: (det.confidence, bbox_area(det.bbox)),
            reverse=True,
        )
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
