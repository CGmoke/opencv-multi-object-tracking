"""领域层纯数据记录：轨迹点与内存轨迹历史。

本模块只包含**数据与纯逻辑**，不含任何 I/O（``csv`` / ``os`` 归
``adapters/sinks/csv.py``）。CSV 的表头、行格式与标签清洗属于序列化/安全
关注点，由适配层实现；领域层只持有原始数据记录。

职责边界
--------
* :class:`TrackPoint`：一个目标在某一帧的采样位置（不可变记录）。
* :class:`TrajectoryStore`：按 ``track_id`` 分组、按目标有界的内存历史。

不在此处的部分（迁移目标见 ``docs/ARCHITECTURE_DESIGN.md`` §2.6）：
``CsvTrajectoryWriter`` / ``write_points_atomic`` / ``sanitize_label`` /
``CSV_HEADER`` → ``adapters/sinks/csv.py``（唯一允许 import ``csv`` / ``os`` 处）。
"""

from __future__ import annotations

import math
from collections import OrderedDict
from dataclasses import dataclass
from typing import Iterable, List, Tuple

from .errors import ValidationError
from .limits import MAX_TRAIL_LENGTH
from .models import DEFAULT_LABEL, BBox, Point, bbox_center, normalize_bbox


@dataclass(frozen=True)
class TrackPoint:
    """一个目标在某一帧的采样位置（纯数据，不含格式化）。

    Args:
        track_id: 目标 ID。
        frame: 帧序号。
        timestamp: 该帧的时间戳（秒）。
        bbox: 该帧的包围盒。
        label: 标签。
    """

    track_id: int
    frame: int
    timestamp: float
    bbox: BBox
    label: str = DEFAULT_LABEL

    @property
    def center(self) -> Point:
        """包围盒中心点。"""
        return bbox_center(self.bbox)


class TrajectoryStore:
    """按 ``track_id`` 分组、按目标有界的内存轨迹历史。

    .. note::
       当前主流程**未接线**到本类（见 ``docs/ROADMAP.md`` 的 A9）。它作为公共
       API 保留，供未来“回退/重放流”或外部消费使用；若确定无用可整体移除。

    Args:
        max_points_per_track: 每个目标保留的最大点数，默认取
            :data:`~mot.domain.limits.MAX_TRAIL_LENGTH`。

    Raises:
        ValidationError: ``max_points_per_track < 1``。
    """

    def __init__(self, max_points_per_track: int = MAX_TRAIL_LENGTH) -> None:
        if int(max_points_per_track) < 1:
            raise ValidationError(
                "max_points_per_track must be >= 1",
                field="max_points_per_track",
                value=max_points_per_track,
            )
        self._max_points = int(max_points_per_track)
        self._points: OrderedDict[int, List[TrackPoint]] = OrderedDict()

    def add(
        self,
        track_id: int,
        frame: int,
        timestamp: float,
        bbox: BBox,
        label: str = DEFAULT_LABEL,
    ) -> TrackPoint:
        """记录一个轨迹点并返回它。

        Args:
            track_id: 目标 ID。
            frame: 帧序号。
            timestamp: 时间戳（秒），必须有限且 ``>= 0``。
            bbox: 包围盒，经 :func:`~mot.domain.models.normalize_bbox` 校验。
            label: 标签。

        Returns:
            构造出的 :class:`TrackPoint`。

        Raises:
            ValidationError: ``timestamp`` 非有限或为负，或 ``bbox`` 非法。
        """
        if not math.isfinite(timestamp) or timestamp < 0:
            raise ValidationError(
                f"timestamp must be finite and >= 0, got {timestamp}",
                field="timestamp",
                value=timestamp,
            )
        point = TrackPoint(
            track_id=int(track_id),
            frame=int(frame),
            timestamp=float(timestamp),
            bbox=normalize_bbox(bbox),
            label=label,
        )
        bucket = self._points.setdefault(point.track_id, [])
        bucket.append(point)
        if len(bucket) > self._max_points:
            del bucket[: len(bucket) - self._max_points]
        return point

    def points(self, track_id: int) -> Tuple[TrackPoint, ...]:
        """返回某目标的全部已存轨迹点（按写入顺序）。"""
        return tuple(self._points.get(int(track_id), ()))

    def items(self) -> Iterable[Tuple[int, Tuple[TrackPoint, ...]]]:
        """按插入顺序迭代 ``(track_id, points)``。"""
        for track_id, points in self._points.items():
            yield track_id, tuple(points)

    def clear(self) -> None:
        """清空全部历史。"""
        self._points.clear()

    def __len__(self) -> int:
        """已存轨迹点总数。"""
        return sum(len(points) for points in self._points.values())

    def __contains__(self, track_id: object) -> bool:
        return isinstance(track_id, int) and track_id in self._points


__all__ = ["TrackPoint", "TrajectoryStore"]
