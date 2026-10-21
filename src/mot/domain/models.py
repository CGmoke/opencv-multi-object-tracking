"""领域层数据模型与包围盒几何原语。

本模块是领域层的“名词层”：定义包围盒、检测、存活目标、ID 分配等数据模型，
以及与之绑定的输入校验。纯逻辑、零框架依赖（不得 import ``cv2`` / ``yaml`` /
``csv`` / ``os``），可在无 OpenCV 环境下单元测试。

职责边界
--------
* 数据模型与基础校验：``BBox`` / ``Point`` / ``Detection`` / ``TrackedObject``
  / ``TrackState`` / ``StepResult`` / ``IdAllocator``。
* 包围盒几何原语：``normalize_bbox`` / ``clip_bbox`` / ``bbox_center`` /
  ``bbox_area`` / ``bbox_contains``（IoU 等关联度量见 ``domain/geometry.py``）。
* 校验失败统一抛 :class:`~mot.domain.errors.ValidationError`（它同时是
  ``ValueError``，保持既有 "非法入参用 ValueError 表达" 的调用方兼容）。

设计约定
--------
* 所有数值约束引用 :mod:`mot.domain.limits` 的单一真源（如 ``MIN_BOX_SIDE``）。
* ``StepResult.matched`` 保持“下标对”语义（``(track_index, detection_index)``），
  避免下游误用为 ``track_id``；同时提供 ``matched_ids`` 只读视图（G7 修复）。
* 帧对象对领域层不透明：``TrackedObject.tracker`` 与 ``frame`` 均为任意对象，
  由适配层的端口实现，领域层不感知其类型。
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Deque, Optional, Sequence, Tuple

from .errors import ValidationError
from .limits import MIN_BOX_SIDE

#: 包围盒：``(x, y, w, h)``，左上角坐标 + 宽高，单位像素。
BBox = Tuple[float, float, float, float]

#: 点：``(x, y)``。
Point = Tuple[float, float]

#: 未显式提供标签时使用的默认标签。
DEFAULT_LABEL = "object"


# ---------------------------------------------------------------------------
# 包围盒几何原语
# ---------------------------------------------------------------------------


def normalize_bbox(bbox: Sequence[float]) -> BBox:
    """校验 ``(x, y, w, h)`` 并返回宽高为正的包围盒。

    负的宽/高被解释为“从对角反向绘制的框”（鼠标拖拽可能产生），会被翻转为正。

    Args:
        bbox: 形如 ``(x, y, w, h)`` 的四元素序列。

    Returns:
        归一后的 ``(x, y, w, h)`` 浮点元组。

    Raises:
        ValidationError: 元素个数不为 4、含非数值、含非有限值，或任一边长
            小于 :data:`~mot.domain.limits.MIN_BOX_SIDE`。
    """
    try:
        values = tuple(float(v) for v in bbox)
    except (TypeError, ValueError):
        raise ValidationError(
            f"bbox must be four numbers, got {bbox!r}", field="bbox", value=bbox
        ) from None
    if len(values) != 4:
        raise ValidationError(
            f"bbox must be (x, y, w, h), got {len(values)} values", field="bbox", value=bbox
        )
    if not all(math.isfinite(v) for v in values):
        raise ValidationError(
            f"bbox contains non-finite values: {values}", field="bbox", value=bbox
        )

    x, y, w, h = values
    if w < 0:
        x, w = x + w, -w
    if h < 0:
        y, h = y + h, -h
    if w < MIN_BOX_SIDE or h < MIN_BOX_SIDE:
        raise ValidationError(
            f"bbox must be at least {MIN_BOX_SIDE:g}x{MIN_BOX_SIDE:g} pixels, got {values}",
            field="bbox",
            value=bbox,
        )
    return (x, y, w, h)


def bbox_center(bbox: Sequence[float]) -> Point:
    """返回包围盒的中心点。"""
    x, y, w, h = bbox
    return (x + w / 2.0, y + h / 2.0)


def bbox_area(bbox: Sequence[float]) -> float:
    """返回包围盒面积，恒不为负。"""
    _, _, w, h = bbox
    return max(0.0, float(w)) * max(0.0, float(h))


def bbox_diagonal(bbox: Sequence[float]) -> float:
    """返回包围盒对角线长度（像素），用于尺度无关的距离阈值。"""
    _, _, w, h = bbox
    return math.hypot(max(0.0, float(w)), max(0.0, float(h)))


def bbox_contains(bbox: Sequence[float], point: Point) -> bool:
    """判断 ``point`` 是否落在盒内（含边界）。"""
    x, y, w, h = bbox
    px, py = point
    return x <= px <= x + w and y <= py <= y + h


def clip_bbox(bbox: Sequence[float], width: int, height: int) -> Optional[BBox]:
    """把包围盒裁剪到 ``width x height`` 的画面内。

    Args:
        bbox: 待裁剪的包围盒。
        width: 画面宽（正整数）。
        height: 画面高（正整数）。

    Returns:
        裁剪后的盒；若盒与画面已无重叠则返回 ``None``（目标离开画面即由此判定）。

    Raises:
        ValidationError: 画面尺寸非正。
    """
    if width <= 0 or height <= 0:
        raise ValidationError(
            f"frame size must be positive, got {width}x{height}",
            field="frame_size",
            value=(width, height),
        )
    x, y, w, h = bbox
    x1 = max(0.0, float(x))
    y1 = max(0.0, float(y))
    x2 = min(float(width), float(x) + float(w))
    y2 = min(float(height), float(y) + float(h))
    if x2 - x1 < MIN_BOX_SIDE or y2 - y1 < MIN_BOX_SIDE:
        return None
    return (x1, y1, x2 - x1, y2 - y1)


# ---------------------------------------------------------------------------
# 数据模型
# ---------------------------------------------------------------------------


class TrackState(str, Enum):
    """目标 ID 的生命周期状态（由 ``domain/lifecycle.py`` 维护）。"""

    ACTIVE = "active"
    LOST = "lost"


@dataclass(frozen=True)
class Detection:
    """一次观测：由检测器输出或交互式框选产生。

    Args:
        bbox: 观测框，构造时经 :func:`normalize_bbox` 校验归一并冻结。
        confidence: 置信度，构造时校验有限并夹取到 ``[0, 1]``。
        label: 标签，写入 CSV 前会被清洗（清洗属适配层职责）。
    """

    bbox: BBox
    confidence: float = 1.0
    label: str = DEFAULT_LABEL

    def __post_init__(self) -> None:
        object.__setattr__(self, "bbox", normalize_bbox(self.bbox))
        if not math.isfinite(self.confidence):
            raise ValidationError(
                f"confidence must be finite, got {self.confidence}",
                field="confidence",
                value=self.confidence,
            )
        object.__setattr__(self, "confidence", float(min(1.0, max(0.0, self.confidence))))


@dataclass
class TrackedObject:
    """一个存活目标：唯一 ID、一个跟踪器、当前包围盒与其轨迹。

    ``tracker`` 是任意对象（由适配层端口实现），领域层只通过注入的
    ``init_fn`` / ``update_fn`` 驱动它，不直接依赖 OpenCV。
    """

    track_id: int
    tracker: object
    bbox: BBox
    label: str = DEFAULT_LABEL
    confidence: float = 1.0
    state: TrackState = TrackState.ACTIVE
    hits: int = 0
    misses: int = 0
    age: int = 0
    created_frame: int = 0
    last_seen_frame: int = 0
    #: 最近一次被检测（而非跟踪器自报）支持的帧号；``max_coast`` 以此为基准。
    last_detection_frame: int = 0
    trail: Deque[Point] = field(default_factory=deque)
    #: 最近的原始包围盒（滑动平均平滑用；由 ``TrackManager`` 维护）。
    box_history: Deque[BBox] = field(default_factory=deque)

    @property
    def center(self) -> Point:
        """当前包围盒中心。"""
        return bbox_center(self.bbox)

    @property
    def is_active(self) -> bool:
        """是否处于 ``ACTIVE`` 状态。"""
        return self.state is TrackState.ACTIVE

    def push_trail(self, point: Point, max_points: int) -> None:
        """追加一个轨迹点，最多保留 ``max_points`` 个采样。

        相邻重复点会被去重（快速移动前目标静止时避免轨迹堆积同一点）；
        超出上限时从最早的点开始丢弃。
        """
        if self.trail and self.trail[-1] == point:
            return
        self.trail.append(point)
        while max_points > 0 and len(self.trail) > max_points:
            self.trail.popleft()


@dataclass
class StepResult:
    """一次 ``TrackManager.step`` 的账目结果。

    Attributes:
        tracks: 当帧所有存活（含 LOST）目标。
        created: 当帧新建的 ``track_id``。
        removed: 当帧销毁的 ``track_id``。
        matched: 关联命中的**下标对** ``(track_index, detection_index)``，
            索引对应 ``tracks`` 与当帧检测列表，**不是** ``track_id``。
        spawn_failures: 当帧建 ID 失败次数（容量上限或跟踪器初始化失败）。
    """

    tracks: Tuple[TrackedObject, ...] = ()
    created: Tuple[int, ...] = ()
    removed: Tuple[int, ...] = ()
    matched: Tuple[Tuple[int, int], ...] = ()
    spawn_failures: int = 0

    @property
    def matched_ids(self) -> Tuple[Tuple[int, str], ...]:
        """G7 便捷视图：把下标对映射为可直接消费的 ``(track_id, label)``。

        关联命中后 ``TrackManager`` 会用该检测重新锚定目标并把 ``track.label``
        更新为检测标签，因此这里取 ``tracks[i].label`` 即等价于命中的检测标签，
        无需再持有检测列表（保持 ``StepResult`` 只依赖 ``tracks``）。

        Returns:
            与 ``matched`` 等长、顺序一致的 ``(track_id, label)`` 元组。
        """
        return tuple(
            (self.tracks[track_index].track_id, self.tracks[track_index].label)
            for track_index, _det_index in self.matched
        )


class IdAllocator:
    """单调 ID 分配器。

    ID 在会话内**不复用**，因此即使目标消失后同一位置再出现新目标，CSV 中的
    ``track_id`` 也始终唯一且稳定。

    Args:
        start: 起始 ID，必须 ``>= 1``。

    Raises:
        ValidationError: ``start < 1``。
    """

    def __init__(self, start: int = 1) -> None:
        if int(start) < 1:
            raise ValidationError(
                f"start id must be >= 1, got {start}", field="start_id", value=start
            )
        self._next = int(start)

    def __call__(self) -> int:
        track_id = self._next
        self._next += 1
        return track_id

    @property
    def next_id(self) -> int:
        """下一个将被分配的 ID。"""
        return self._next

    def reset(self, start: int = 1) -> None:
        """重置计数器；``start`` 必须 ``>= 1``。"""
        if int(start) < 1:
            raise ValidationError(
                f"start id must be >= 1, got {start}", field="start_id", value=start
            )
        self._next = int(start)

    def __len__(self) -> int:
        """已分配的 ID 数量。"""
        return self._next - 1


__all__ = [
    "BBox",
    "DEFAULT_LABEL",
    "Detection",
    "IdAllocator",
    "Point",
    "StepResult",
    "TrackState",
    "TrackedObject",
    "bbox_area",
    "bbox_center",
    "bbox_contains",
    "bbox_diagonal",
    "clip_bbox",
    "normalize_bbox",
]
