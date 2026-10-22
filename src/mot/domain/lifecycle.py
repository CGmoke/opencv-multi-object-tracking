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
from collections import deque
from concurrent.futures import ThreadPoolExecutor
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
    bbox_area,
    bbox_center,
    clip_bbox,
    normalize_bbox,
)
from .ports import Frame, TrackerFactory
from .postprocess import intersection_over_min_area

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
        max_coast: 轨迹处于 ``LOST`` 且距最后一次检测支持达到该帧数时销毁；
            与 ``max_missing`` 同时生效，取二者更严者。默认 ``8``。
        trail_length: 每个目标保留的轨迹点上限。
        match_method: ``"greedy"``（默认）或 ``"hungarian"``。
        id_allocator: 可选共享分配器；id 单调递增。
        max_tracks: 同时存活 id 的硬上限；超限的检测被拒绝（计入 ``spawn_failures``）
            而非无限增长内存。
        max_parallel: 并行推进跟踪器的线程数；``0``（默认）表示完全串行。仅将
            ``update_fn`` 的纯计算并行化，关联与状态机仍在主线程按轨迹顺序执行。
            线程池惰性创建、跨帧复用，``close()`` / ``reset()`` 时释放。
        box_smoothing: 检测框 EMA 平滑系数（``0 < alpha <= 1``）。``1`` 表示直接使用
            检测框；调小可抑制检测框抖动，但会带来轻微滞后。仅在用检测重锚时生效。
        min_hits: 轨迹确认所需的命中数（默认 ``1`` 即不确认）。未确认的“暂定轨迹”
            不向渲染/导出输出，且一次丢失就立即销毁，用于过滤 1~2 帧的误检。
        smooth_window: 包围盒滑动平均窗口（``<=1`` 关闭）。开启后对跟踪器输出与检测框
            都做窗口平均，轨迹点因此更平滑；优先级高于 ``box_smoothing``。
        nest_containment: “大框套小框”判定阈值：小框落在更大目标框内的面积比例达到该值
            即视为重复框；``0`` 关闭。手动框选时用于保证大框内不出现重复的小框。
    """

    def __init__(
        self,
        tracker_factory: TrackerFactory,
        *,
        init_fn: TrackerInitFn,
        update_fn: TrackerUpdateFn,
        iou_threshold: float = 0.3,
        max_missing: int = 15,
        max_coast: int = 8,
        trail_length: int = 120,
        match_method: str = "greedy",
        id_allocator: Optional[IdAllocator] = None,
        max_tracks: int = 256,
        max_parallel: int = 0,
        box_smoothing: float = 1.0,
        min_hits: int = 1,
        smooth_window: int = 0,
        nest_containment: float = 0.85,
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
        if int(min_hits) < 1:
            raise ValidationError(
                f"min_hits must be >= 1, got {min_hits}", field="min_hits", value=min_hits
            )
        if int(smooth_window) < 0:
            raise ValidationError(
                f"smooth_window must be >= 0, got {smooth_window}",
                field="smooth_window",
                value=smooth_window,
            )
        if int(max_coast) < 0:
            raise ValidationError(
                f"max_coast must be >= 0, got {max_coast}",
                field="max_coast",
                value=max_coast,
            )
        if not 0 <= int(max_parallel) <= 256:
            raise ValidationError(
                f"max_parallel must be within [0, 256], got {max_parallel}",
                field="max_parallel",
                value=max_parallel,
            )
        if not 0.0 <= float(nest_containment) <= 1.0:
            raise ValidationError(
                f"nest_containment must be within [0, 1], got {nest_containment}",
                field="nest_containment",
                value=nest_containment,
            )

        self._tracker_factory = tracker_factory
        self._init_fn = init_fn
        self._update_fn = update_fn
        self.iou_threshold = validate_iou_threshold(iou_threshold)
        self.max_missing = validate_max_missing(max_missing)
        self.max_coast = int(max_coast)
        self.trail_length = validate_trail_length(trail_length)
        self.match_method = match_method
        self.max_tracks = validate_max_tracks(max_tracks)
        self.max_parallel = int(max_parallel)
        self._executor: Optional[ThreadPoolExecutor] = None
        self.box_smoothing = float(box_smoothing)
        self.min_hits = int(min_hits)
        self.smooth_window = int(smooth_window)
        self.nest_containment = float(nest_containment)
        self.tentative_dropped = 0
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
    def confirmed_tracks(self) -> Tuple[TrackedObject, ...]:
        """命中数达到 ``min_hits`` 的轨迹（渲染与导出只应使用这一组）。"""
        return tuple(t for t in self.tracks if self.is_confirmed(t))

    def is_confirmed(self, track: TrackedObject) -> bool:
        """该轨迹是否已经过确认（``min_hits=1`` 时恒为 ``True``）。"""
        return track.hits >= self.min_hits

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
        assoc_tracks: Tuple[TrackedObject, ...] = ()
        # 1. 推进已有跟踪器。跟踪器自报成功只刷新几何，不计命中、不改状态：
        #    ``hits`` 只表示“被检测支持的命中次数”。
        track_ids = list(self._tracks)
        if self.max_parallel > 0 and len(track_ids) > 1:
            # 并行路径：只把纯计算（update）放进线程池，状态机仍按原顺序在主线程执行。
            for track_id in track_ids:
                self._tracks[track_id].age += 1
            boxes = self._update_boxes_parallel(track_ids, frame)
            for track_id, box in zip(track_ids, boxes):
                removed_id = self._apply_update_result(track_id, box, frame, frame_index)
                if removed_id is not None:
                    removed.append(removed_id)
        else:
            # 串行路径（默认）：与既有实现完全一致。
            for track_id in track_ids:
                track = self._tracks[track_id]
                track.age += 1
                box = self._update_fn(track.tracker, frame)
                removed_id = self._apply_update_result(track_id, box, frame, frame_index)
                if removed_id is not None:
                    removed.append(removed_id)

        # 2. 把当前帧检测关联到存活 id。
        detections = tuple(detections)
        if detections:
            pairs, _unmatched_tracks, unmatched_detections = self._associate(detections)
            snapshot = self.tracks
            assoc_tracks = snapshot
            for track_idx, det_idx in pairs:
                track = snapshot[track_idx]
                detection = detections[det_idx]
                # G3：重锚失败已降级为一次 miss（LOST），此处不再覆盖状态。
                if not self._refresh(track, detection, frame):
                    continue
                track.state = TrackState.ACTIVE
                # 检测匹配是唯一的命中来源：同帧即使跟踪器也自报成功，
                # 也只记一次（且不计跟踪器的“成功”）。
                track.hits += 1
                track.misses = 0
                track.last_seen_frame = frame_index
                track.last_detection_frame = frame_index
            for det_idx in unmatched_detections:
                detection = detections[det_idx]
                track = self._spawn(detection, frame, frame_index)
                if track is None:
                    spawn_failures += 1  # _spawn 已更新总数
                else:
                    created.append(track.track_id)

        # 3. 滑行上限：LOST 且距最后一次检测支持达到 ``max_coast`` 的轨迹销毁。
        #    与 max_missing 并行生效，谁先触发谁负责回收。
        for track_id in list(self._tracks):
            track = self._tracks[track_id]
            if track.state is not TrackState.LOST:
                continue
            if frame_index - track.last_detection_frame < self.max_coast:
                continue
            del self._tracks[track_id]
            self.total_removed += 1
            removed.append(track_id)

        # 第 3 步可能销毁已滑行到期的 LOST 轨迹，导致存活列表缩短、下标整体前移；
        # 因此把关联下标按 ``track_id`` 重映射到最终列表，避免 matched_ids 越界。
        tracks = self.tracks
        index_by_id = {track.track_id: index for index, track in enumerate(tracks)}
        matched = tuple(
            (index_by_id[track_id], det_index)
            for track_id, det_index in (
                (assoc_tracks[track_index].track_id, det_index)
                for track_index, det_index in pairs
            )
            if track_id in index_by_id
        )
        return StepResult(
            tracks=tracks,
            created=tuple(created),
            removed=tuple(removed),
            matched=matched,
            spawn_failures=spawn_failures,
        )

    def add_or_refresh(self, frame: Frame, detection: Detection) -> Optional[TrackedObject]:
        """新增手工框，或刷新其重叠到的 id。

        同一物体被重复框选不得产生第二个身份：

        * 新框与某存活目标的 IoU 达到 ``iou_threshold`` 时刷新该目标而非新建；
        * 新框大部分落在更大的已有目标内时，视为重复的小框，直接返回该大框
          （不新建、不缩小大框）；
        * 新框包含已有的小框时，先销毁这些小框，再关联或新建——保证“大框中
          不会出现重复的小框”。
        """
        if len(self._tracks) == 0:
            return self._spawn(detection, frame, self.frame_index)

        box = normalize_bbox(detection.bbox)
        enclosing = self._enclosing_track(box)
        if enclosing is not None:
            return enclosing

        for track in list(self._tracks.values()):
            if self._contains(box, track.bbox):
                self.remove(track.track_id)

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
            track.last_detection_frame = self.frame_index
            return track
        return self._spawn(detection, frame, self.frame_index)

    def _enclosing_track(self, box: BBox) -> Optional[TrackedObject]:
        """返回把 ``box`` 大部分包住的更大存活目标；没有则返回 ``None``。"""
        if self.nest_containment <= 0.0:
            return None
        area = bbox_area(box)
        if area <= 0.0:
            return None
        best: Optional[TrackedObject] = None
        for track in self._tracks.values():
            if bbox_area(track.bbox) <= area:
                continue
            if intersection_over_min_area(track.bbox, box) < self.nest_containment:
                continue
            if best is None or bbox_area(track.bbox) < bbox_area(best.bbox):
                best = track
        return best

    def _contains(self, outer: BBox, inner: BBox) -> bool:
        """``inner`` 是否为 ``outer`` 内的重复小框（面积更小且被包含比例达标）。"""
        if self.nest_containment <= 0.0:
            return False
        inner_area, outer_area = bbox_area(inner), bbox_area(outer)
        if inner_area <= 0.0 or outer_area <= inner_area:
            return False
        return intersection_over_min_area(outer, inner) >= self.nest_containment

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
        self.close()
        self._tracks.clear()
        self._capacity_warned = False
        self.frame_index = 0
        if start_id is not None:
            self._allocator.reset(start_id)
        self.total_created = 0
        self.total_removed = 0
        self.spawn_failures = 0

    # -- 内部实现 --------------------------------------------------------
    def close(self) -> None:
        """释放并行 update 线程池；可重复调用。"""
        executor = self._executor
        self._executor = None
        if executor is not None:
            executor.shutdown(wait=True)

    def _update_boxes_parallel(
        self, track_ids: Sequence[int], frame: Frame
    ) -> List[Optional[BBox]]:
        """在复用线程池中并行执行每条轨迹的 update，并按输入顺序返回结果。

        一轨迹一任务，同一 tracker 不会被两个线程同时 update；结果顺序与
        ``track_ids`` 一致，后续状态机仍在主线程按原顺序执行，保证可复现。
        """
        if self._executor is None:
            self._executor = ThreadPoolExecutor(
                max_workers=self.max_parallel, thread_name_prefix="mot-update"
            )
        futures = [
            self._executor.submit(self._update_fn, self._tracks[track_id].tracker, frame)
            for track_id in track_ids
        ]
        return [future.result() for future in futures]

    def _apply_update_result(
        self, track_id: int, box: Optional[BBox], frame: Frame, frame_index: int
    ) -> Optional[int]:
        """串行处理单条轨迹的 update 结果；返回需要写入 removed 事件的 id。"""
        track = self._tracks[track_id]
        if box is not None:
            box = self._clip_to_frame(box, frame)
        if box is not None:
            self._accept_box(track, box, frame_index)
            return None
        track.misses += 1
        track.state = TrackState.LOST
        if not self.is_confirmed(track):
            # 暂定轨迹：一次丢失立即销毁（不进 removed 事件，避免噪声事件）
            del self._tracks[track_id]
            self.total_removed += 1
            self.tentative_dropped += 1
            return None
        if track.misses > self.max_missing:
            del self._tracks[track_id]
            self.total_removed += 1
            return track_id
        return None

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
        """用跟踪器框刷新几何与轨迹。

        跟踪器自报成功**不是**检测支持：这里不计 ``hits``，也不把 ``LOST``
        改回 ``ACTIVE``；状态只能由检测匹配恢复。
        """
        box = self._smooth_box(track, box)
        track.bbox = box
        track.misses = 0
        track.last_seen_frame = frame_index
        track.push_trail(bbox_center(box), self.trail_length)

    def _refresh(self, track: TrackedObject, detection: Detection, frame: Frame) -> bool:
        """用新观测框重锚一个目标的跟踪器。

        Returns:
            ``True`` 表示刷新成功；``False`` 表示初始化失败（已降级为一次 miss
            并转为 ``LOST``，异常不外泄）。
        """
        raw = normalize_bbox(detection.bbox)
        try:
            self._init_fn(track.tracker, frame, raw)  # 跟踪器用原始检测框，避免滞后
        except Exception as exc:  # noqa: BLE001 - 非致命：单次重锚失败不得中断主循环
            LOGGER.warning("re-init failed for id %s: %s", track.track_id, exc)
            track.misses += 1
            track.state = TrackState.LOST
            return False
        box = self._smooth_box(track, raw)
        track.bbox = box
        track.label = detection.label
        track.confidence = detection.confidence
        track.push_trail(bbox_center(box), self.trail_length)
        return True

    def _smooth_box(self, track: TrackedObject, box: BBox) -> BBox:
        """包围盒平滑：优先滑动平均（``smooth_window``），否则 EMA（``box_smoothing``）。

        滑动平均窗口对跟踪器输出与检测框一律生效，轨迹点（中心）随之平滑。
        ``LOST`` 目标重新出现时清空历史并直接吸附，避免旧框拖拽新位置。
        """
        if self.smooth_window > 1:
            if track.state is TrackState.LOST:
                track.box_history.clear()
            if track.box_history.maxlen != self.smooth_window:
                track.box_history = deque(track.box_history, maxlen=self.smooth_window)
            track.box_history.append(box)
            count = float(len(track.box_history))
            sums = [0.0, 0.0, 0.0, 0.0]
            for item in track.box_history:
                for index in range(4):
                    sums[index] += item[index]
            return (
                sums[0] / count,
                sums[1] / count,
                sums[2] / count,
                sums[3] / count,
            )

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
            last_detection_frame=frame_index,
            box_history=deque(maxlen=self.smooth_window if self.smooth_window > 1 else 0),
        )
        if self.smooth_window > 1:
            track.box_history.append(track.bbox)
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
