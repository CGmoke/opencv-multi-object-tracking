"""领域层 ID 生命周期：新建、推进、关联与过期。

这是项目的核心：每个目标一个跟踪器；每帧推进所有跟踪器，再用 IoU 把新检测
（鼠标框或检测器输出）关联到已有 id。丢失的目标保留 ``max_missing`` 帧以便恢复，
超时后销毁 id，之后同位置的检测分配新 id。

设计约定
--------
* **领域层不 import cv2**：每个目标的跟踪器由 ``tracker_factory``（零参工厂）创建，
  驱动跟踪器的 ``init`` / ``update`` 通过 ``init_fn`` / ``update_fn`` 注入
  （默认由适配层 ``adapters/trackers.py`` 提供）。因此整条生命周期可用假跟踪器在
  无 OpenCV 环境下测试（见 ``docs/ARCHITECTURE_DESIGN.md`` §3.1）。
* **G3 修复**：关联命中后重新 ``init`` 失败**不再让异常穿出** ``step``，而是降级为
  一次 ``miss``（``misses += 1`` → ``LOST``）并记一次告警。
* 数值边界统一走 :mod:`mot.domain.limits` 的校验器（G1：单一真源，配置阶段即报错）。
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from .association import MATCH_METHODS, match
from .errors import ValidationError
from .geometry import iou_matrix
from .limits import (
    validate_iou_threshold,
    validate_max_missing,
    validate_max_tracks,
    validate_trail_length,
)
from .models import (
    DEFAULT_LABEL,
    BBox,
    Detection,
    IdAllocator,
    StepResult,
    TrackedObject,
    TrackState,
    bbox_center,
    clip_bbox,
    normalize_bbox,
)
from .ports import Frame, TrackerFactory

LOGGER = logging.getLogger(__name__)

#: 注入的跟踪器初始化回调：``(tracker, frame, bbox) -> None``；失败可抛异常。
TrackerInitFn = Callable[[Any, Frame, Sequence[float]], None]

#: 注入的跟踪器推进回调：``(tracker, frame) -> Optional[BBox]``；丢失时返回 ``None``。
TrackerUpdateFn = Callable[[Any, Frame], Optional[BBox]]


class TrackManager:
    """持有全部存活目标及其唯一 id。

    Args:
        tracker_factory: 零参可调用，返回一个全新的跟踪器。
        init_fn: 注入的跟踪器初始化回调（``tracker, frame, bbox``）。
        update_fn: 注入的跟踪器推进回调（``tracker, frame -> Optional[BBox]``）。
        iou_threshold: 检测与已有 id 关联所需的最小 IoU，也用于同帧去重。
        max_missing: 丢失目标保留 id 的帧数；``0`` 表示首次丢失即销毁。
        trail_length: 每个目标保留的轨迹点上限。
        match_method: ``"greedy"``（默认）或 ``"hungarian"``。
        id_allocator: 可选共享分配器；id 单调递增。
        max_tracks: 同时存活 id 的硬上限；超限的检测被拒绝（计入 ``spawn_failures``）
            而非无限增长内存。
        box_smoothing: 检测框 EMA 平滑系数（``0 < alpha <= 1``）。``1`` 表示直接使用
            检测框；调小可抑制检测框抖动，但会带来轻微滞后。仅在用检测重锚时生效。
    """

    def __init__(
        self,
        tracker_factory: TrackerFactory,
        *,
        init_fn: TrackerInitFn,
        update_fn: TrackerUpdateFn,
        iou_threshold: float = 0.3,
        max_missing: int = 15,
        trail_length: int = 120,
        match_method: str = "greedy",
        id_allocator: Optional[IdAllocator] = None,
        max_tracks: int = 256,
        box_smoothing: float = 1.0,
    ) -> None:
        if match_method not in MATCH_METHODS:
            raise ValidationError(
                f"unknown match_method {match_method!r}; expected {MATCH_METHODS}",
                field="match_method",
                value=match_method,
            )
        if not 0.0 < float(box_smoothing) <= 1.0:
            raise ValidationError(
                f"box_smoothing must be within (0, 1], got {box_smoothing}",
                field="box_smoothing",
                value=box_smoothing,
            )

        self._tracker_factory = tracker_factory
        self._init_fn = init_fn
        self._update_fn = update_fn
        self.iou_threshold = validate_iou_threshold(iou_threshold)
        self.max_missing = validate_max_missing(max_missing)
        self.trail_length = validate_trail_length(trail_length)
        self.match_method = match_method
        self.max_tracks = validate_max_tracks(max_tracks)
        self.box_smoothing = float(box_smoothing)
        self._capacity_warned = False
        self._allocator = id_allocator or IdAllocator()
        self._tracks: Dict[int, TrackedObject] = {}
        self.frame_index = 0
        self.total_created = 0
        self.total_removed = 0
        self.spawn_failures = 0

    # -- 只读视图 --------------------------------------------------------
    @property
    def tracks(self) -> Tuple[TrackedObject, ...]:
        """存活（含 LOST）目标，按 id 升序（渲染顺序稳定）。"""
        return tuple(self._tracks[key] for key in sorted(self._tracks))

    @property
    def active_tracks(self) -> Tuple[TrackedObject, ...]:
        """仅 ``ACTIVE`` 状态的目标。"""
        return tuple(t for t in self.tracks if t.is_active)

    @property
    def allocator(self) -> IdAllocator:
        return self._allocator

    def get(self, track_id: int) -> Optional[TrackedObject]:
        return self._tracks.get(int(track_id))

    def __len__(self) -> int:
        return len(self._tracks)

    def __contains__(self, track_id: object) -> bool:
        return isinstance(track_id, int) and track_id in self._tracks

    # -- 生命周期 --------------------------------------------------------
    def step(self, frame: Frame, detections: Sequence[Detection] = ()) -> StepResult:
        """推进所有跟踪器一帧，再吸收新检测。

        ``detections`` 是当前帧的观测（检测器输出或交互框），按 IoU 关联到存活 id：
        命中则刷新并重锚该目标的跟踪器，未命中则新建 id。

        Returns:
            :class:`~mot.domain.models.StepResult`，含当帧新建与销毁的 id。
        """
        self.frame_index += 1
        frame_index = self.frame_index
        created: List[int] = []
        removed: List[int] = []
        spawn_failures = 0

        pairs: List[Tuple[int, int]] = []
        updated_ids: set = set()
        # 1. 推进已有跟踪器。
        for track_id in list(self._tracks):
            track = self._tracks[track_id]
            track.age += 1
            box = self._update_fn(track.tracker, frame)
            if box is not None:
                box = self._clip_to_frame(box, frame)
            if box is not None:
                self._accept_box(track, box, frame_index)
                updated_ids.add(track_id)
            else:
                track.misses += 1
                track.state = TrackState.LOST
                if track.misses > self.max_missing:
                    del self._tracks[track_id]
                    self.total_removed += 1
                    removed.append(track_id)

        # 2. 把当前帧检测关联到存活 id。
        detections = tuple(detections)
        if detections:
            pairs, _unmatched_tracks, unmatched_detections = self._associate(detections)
            snapshot = self.tracks
            for track_idx, det_idx in pairs:
                track = snapshot[track_idx]
                detection = detections[det_idx]
                # G3：重锚失败已降级为一次 miss（LOST），此处不再覆盖状态。
                if not self._refresh(track, detection, frame):
                    continue
                track.state = TrackState.ACTIVE
                # 跟踪器本帧已计过一次成功更新；检测只在该目标被“救回”时补一次 hit。
                if track.track_id not in updated_ids:
                    track.hits += 1
                track.misses = 0
                track.last_seen_frame = frame_index
            for det_idx in unmatched_detections:
                detection = detections[det_idx]
                track = self._spawn(detection, frame, frame_index)
                if track is None:
                    spawn_failures += 1  # _spawn 已更新总数
                else:
                    created.append(track.track_id)

        return StepResult(
            tracks=self.tracks,
            created=tuple(created),
            removed=tuple(removed),
            matched=tuple(pairs),
            spawn_failures=spawn_failures,
        )

    def add_or_refresh(self, frame: Frame, detection: Detection) -> Optional[TrackedObject]:
        """新增手工框，或刷新其重叠到的 id。

        同一物体被重复框选不得产生第二个身份：当新框与某存活目标的 IoU 达到
        ``iou_threshold`` 时刷新该目标而非新建。
        """
        if len(self._tracks) == 0:
            return self._spawn(detection, frame, self.frame_index)

        pairs, _unmatched_tracks, _unmatched = self._associate([detection])
        if pairs:
            track = self.tracks[pairs[0][0]]
            # G3：重锚失败时保持 LOST 并返回该目标，不重复建 id。
            if not self._refresh(track, detection, frame):
                return track
            track.hits += 1
            track.misses = 0
            track.state = TrackState.ACTIVE
            track.last_seen_frame = self.frame_index
            return track
        return self._spawn(detection, frame, self.frame_index)

    def remove(self, track_id: int) -> bool:
        """立即销毁一个 id（交互式删除）。"""
        track_id = int(track_id)
        if track_id not in self._tracks:
            return False
        del self._tracks[track_id]
        self.total_removed += 1
        return True

    def clear_trails(self) -> None:
        """清空轨迹历史，保留身份与存活跟踪器。"""
        for track in self._tracks.values():
            track.trail.clear()

    def reset(self, start_id: Optional[int] = None) -> None:
        """销毁所有目标。

        默认继续计数，使 id 在同一 CSV / 会话内保持唯一。显式传 ``start_id``
        才重排编号（会让两个不同目标共用同一 ``track_id``）。
        """
        self._tracks.clear()
        self._capacity_warned = False
        self.frame_index = 0
        if start_id is not None:
            self._allocator.reset(start_id)
        self.total_created = 0
        self.total_removed = 0
        self.spawn_failures = 0

    # -- 内部实现 --------------------------------------------------------
    @staticmethod
    def _clip_to_frame(box: BBox, frame: Frame) -> Optional[BBox]:
        """丢弃（部分）离开画面的跟踪器框。

        跟踪器会报出画面外的坐标；保留它们会向 CSV 写入无意义行、并在屏幕外
        绘制轨迹。
        """
        shape = getattr(frame, "shape", None)
        if shape is None or len(shape) < 2:
            return box  # 单测里的假帧：没有尺寸信息
        height, width = int(shape[0]), int(shape[1])
        if width <= 0 or height <= 0:
            return box
        return clip_bbox(box, width, height)

    def _accept_box(self, track: TrackedObject, box: BBox, frame_index: int) -> None:
        track.bbox = box
        track.state = TrackState.ACTIVE
        track.hits += 1
        track.misses = 0
        track.last_seen_frame = frame_index
        track.push_trail(bbox_center(box), self.trail_length)

    def _refresh(self, track: TrackedObject, detection: Detection, frame: Frame) -> bool:
        """用新观测框重锚一个目标的跟踪器。

        Returns:
            ``True`` 表示刷新成功；``False`` 表示初始化失败（已降级为一次 miss
            并转为 ``LOST``，异常不外泄）。
        """
        box = self._smooth_box(track, detection.bbox)
        try:
            self._init_fn(track.tracker, frame, box)
        except Exception as exc:  # noqa: BLE001 - 非致命：单次重锚失败不得中断主循环
            LOGGER.warning("re-init failed for id %s: %s", track.track_id, exc)
            track.misses += 1
            track.state = TrackState.LOST
            return False
        track.bbox = box
        track.label = detection.label
        track.confidence = detection.confidence
        track.push_trail(bbox_center(box), self.trail_length)
        return True

    def _smooth_box(self, track: TrackedObject, box: BBox) -> BBox:
        """检测框 EMA 平滑；LOST 目标重新出现时直接吸附，避免“缓慢追框”。"""
        alpha = self.box_smoothing
        if alpha >= 1.0 or track.state is TrackState.LOST:
            return box
        previous = track.bbox
        return (
            alpha * box[0] + (1.0 - alpha) * previous[0],
            alpha * box[1] + (1.0 - alpha) * previous[1],
            alpha * box[2] + (1.0 - alpha) * previous[2],
            alpha * box[3] + (1.0 - alpha) * previous[3],
        )

    def _spawn(
        self, detection: Detection, frame: Frame, frame_index: int
    ) -> Optional[TrackedObject]:
        if len(self._tracks) >= self.max_tracks:
            if not self._capacity_warned:
                LOGGER.warning(
                    "max_tracks=%d reached; refusing to create more ids", self.max_tracks
                )
                self._capacity_warned = True
            return None
        try:
            tracker = self._tracker_factory()
            self._init_fn(tracker, frame, detection.bbox)
        except Exception:  # noqa: BLE001 - 坏检测不得中断主循环
            self.spawn_failures += 1
            return None

        track = TrackedObject(
            track_id=self._allocator(),
            tracker=tracker,
            bbox=normalize_bbox(detection.bbox),
            label=detection.label or DEFAULT_LABEL,
            confidence=detection.confidence,
            hits=1,
            created_frame=frame_index,
            last_seen_frame=frame_index,
        )
        track.push_trail(bbox_center(track.bbox), self.trail_length)
        self._tracks[track.track_id] = track
        self.total_created += 1
        return track

    def _associate(
        self, detections: Sequence[Detection]
    ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        track_boxes = [track.bbox for track in self.tracks]
        det_boxes = [detection.bbox for detection in detections]
        overlap = iou_matrix(track_boxes, det_boxes)
        pairs, unmatched_tracks, unmatched_dets = match(
            overlap, threshold=self.iou_threshold, method=self.match_method
        )
        return pairs, unmatched_tracks, unmatched_dets


def detections_from_boxes(boxes: Iterable[Sequence[float]]) -> List[Detection]:
    """把裸框转成检测（用于交互式点击与测试）。"""
    return [Detection(bbox=normalize_bbox(box)) for box in boxes]


__all__ = ["TrackerInitFn", "TrackerUpdateFn", "TrackManager", "detections_from_boxes"]
