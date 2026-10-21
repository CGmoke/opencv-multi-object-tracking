"""领域层：检测后处理（碎片合并 + 非极大值抑制）。

运动检测、YOLO 或任何检测器都可能把**同一个物体**输出成多个局部框（碎片），
或者同时输出“整体框 + 局部框”。本模块提供两个纯几何步骤解决它：

* :func:`merge_fragments` —— 把相互重叠（或间隙很小）的框合并成外接框，修复碎片；
* :func:`nms` —— 抑制与更大框高度重叠/被包含的局部框，优先保留整体框。

两者都只做几何运算、无 I/O，可独立复用与单测。

**相邻物体不会被误合并**：合并要求明显的重叠（``IoU >= iou_threshold``）或一个
**正的小间隙**（``0 < gap <= gap_px``）。默认 ``gap_px=0`` 完全关闭间隙合并；
仅轻微重叠（IoU 低于阈值）与紧贴边缘（间隙 0）都视为两个独立物体。
"""

from __future__ import annotations

from typing import Iterable, List, Optional, Sequence

from .geometry import box_iou
from .models import BBox


def _intersection_area(a: Sequence[float], b: Sequence[float]) -> float:
    ax, ay, aw, ah = (float(value) for value in a)
    bx, by, bw, bh = (float(value) for value in b)
    if aw <= 0 or ah <= 0 or bw <= 0 or bh <= 0:
        return 0.0
    left = max(ax, bx)
    top = max(ay, by)
    right = min(ax + aw, bx + bw)
    bottom = min(ay + ah, by + bh)
    if right <= left or bottom <= top:
        return 0.0
    return float((right - left) * (bottom - top))


def intersection_over_min_area(a: Sequence[float], b: Sequence[float]) -> float:
    """交集面积 / 较小框面积；用于判定“局部框被整体框包含”。

    返回 ``0.0``（无交集）到 ``1.0``（小框完全落在大框内）。
    """
    aw, ah = float(a[2]), float(a[3])
    bw, bh = float(b[2]), float(b[3])
    if aw <= 0 or ah <= 0 or bw <= 0 or bh <= 0:
        return 0.0
    return _intersection_area(a, b) / min(aw * ah, bw * bh)


def box_gap(a: Sequence[float], b: Sequence[float]) -> float:
    """两个框边缘之间的最短距离；重叠或相接时为 ``0``。"""
    ax, ay, aw, ah = (float(value) for value in a)
    bx, by, bw, bh = (float(value) for value in b)
    dx = max(0.0, max(ax, bx) - min(ax + aw, bx + bw))
    dy = max(0.0, max(ay, by) - min(ay + ah, by + bh))
    return (dx * dx + dy * dy) ** 0.5


def union_box(a: Sequence[float], b: Sequence[float]) -> BBox:
    """两个框的外接框。"""
    ax, ay, aw, ah = (float(value) for value in a)
    bx, by, bw, bh = (float(value) for value in b)
    left = min(ax, bx)
    top = min(ay, by)
    right = max(ax + aw, bx + bw)
    bottom = max(ay + ah, by + bh)
    return (left, top, right - left, bottom - top)


def _find(parent: List[int], index: int) -> int:
    while parent[index] != index:
        parent[index] = parent[parent[index]]
        index = parent[index]
    return index


def merge_fragments(
    boxes: Iterable[Sequence[float]],
    *,
    iou_threshold: float = 0.15,
    gap_px: float = 0.0,
    growth: float = 3.0,
) -> List[BBox]:
    """把属于同一物体的碎片框合并为外接框。

    两个框满足任一条件即归入同一组（并向上传递）：

    * ``IoU >= iou_threshold``（明显重叠，属于同一物体）；
    * 边缘存在正间隙且 ``gap <= gap_px``（用于修复被噪声切开的碎片）。

    轻微重叠（IoU 不足）或边缘紧贴（间隙恰为 0）**不会**合并，以免把并排的
    两个独立目标当成一个。

    **面积护栏**：每次尝试合并两个分组前，先计算两者的并集外接框面积；若它超过
    ``growth × 两个分组中最大的单个碎片面积``，本次合并被拒绝，两个分组保持独立。
    护栏作用在 union-find 的每一次合并尝试上（含传递合并的中间步骤），而不是最后
    对结果做过滤——否则链式膨胀已经发生，再过滤只会把整个目标丢掉。基准取
    “最大的单个碎片面积”而不是分组的累计面积：后者会让每一跳再放大 ``growth`` 倍，
    长链依然能吞掉整片背景。

    Args:
        boxes: 待合并的框。
        iou_threshold: 触发合并的 IoU 下限（``0`` 表示任意重叠都合并）。
        gap_px: 触发合并的最大边缘间隙（像素）；默认 ``0`` 关闭间隙合并，只按
            IoU 合并；调到 ``2~3`` 可修复被 1~2px 噪声线切开的碎片。
        growth: 合并后外接框面积相对最大单框面积的上限倍数（默认 ``3.0``）；
            ``<= 0`` 关闭护栏（退化为无上限的传递合并）。

    Returns:
        合并后的框，按 ``(y, x)`` 排序，保证输出顺序稳定。
    """
    items = [tuple(float(value) for value in box) for box in boxes]
    if len(items) <= 1:
        return [item for item in items]  # type: ignore[misc]

    limit = float(growth)
    parent = list(range(len(items)))
    group_box: dict[int, BBox] = {index: item for index, item in enumerate(items)}
    group_max_area: dict[int, float] = {
        index: _box_area(item) for index, item in enumerate(items)
    }

    def try_union(i: int, j: int) -> None:
        root_i, root_j = _find(parent, i), _find(parent, j)
        if root_i == root_j:
            return
        merged = union_box(group_box[root_i], group_box[root_j])
        if limit > 0.0:
            cap = limit * max(group_max_area[root_i], group_max_area[root_j])
            if _box_area(merged) > cap:
                return  # 拒绝链式膨胀：保留两个独立框，不做截断/裁剪
        parent[root_j] = root_i
        group_box[root_i] = merged
        group_max_area[root_i] = max(group_max_area[root_i], group_max_area[root_j])

    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            if _should_merge(items[i], items[j], iou_threshold, gap_px):
                try_union(i, j)

    roots: List[int] = []
    for index in range(len(items)):
        root = _find(parent, index)
        if root not in roots:
            roots.append(root)
    return sorted((group_box[root] for root in roots), key=lambda box: (box[1], box[0]))


def _should_merge(
    a: Sequence[float], b: Sequence[float], iou_threshold: float, gap_px: float
) -> bool:
    """重叠足够多才合并；边缘存在**正间隙**且不超过 ``gap_px`` 时才按间隙合并。

    只轻微重叠（IoU 低于阈值）或边缘紧贴（间隙恰为 0）的框视为两个独立物体，
    避免把并排/紧挨的目标误合成一个。
    """
    if box_iou(a, b) >= iou_threshold:
        return True
    gap = box_gap(a, b)
    return 0.0 < gap <= gap_px


def _box_area(box: Sequence[float]) -> float:
    return max(0.0, float(box[2])) * max(0.0, float(box[3]))


def nms_indices(
    boxes: Sequence[Sequence[float]],
    *,
    scores: Optional[Sequence[float]] = None,
    iou_threshold: float = 0.6,
    containment_threshold: float = 0.8,
) -> List[int]:
    """非极大值抑制，返回保留下来的下标（按分数从高到低）。

    除了标准 IoU 抑制，还会抑制**大部分落在更大框内部的局部框**（
    ``交集 / 本框面积 >= containment_threshold``），这正是“整体框与局部框同时
    存在”时保留整体框的关键。判定是单向的：大框不会被小框以包含关系反向抑制。

    Args:
        boxes: 候选框。
        scores: 与 ``boxes`` 等长的分数；默认用面积（大框优先）。
        iou_threshold: IoU 超过该值的框被抑制。
        containment_threshold: 局部框被大框覆盖的面积比例阈值。
    """
    if not boxes:
        return []
    values = [float(score) for score in (scores if scores is not None else map(_box_area, boxes))]
    order = sorted(range(len(boxes)), key=lambda index: values[index], reverse=True)
    keep: List[int] = []
    for index in order:
        candidate = boxes[index]
        if all(
            box_iou(candidate, boxes[kept]) <= iou_threshold
            and _contained_ratio(candidate, boxes[kept]) < containment_threshold
            for kept in keep
        ):
            keep.append(index)
    return keep


def _contained_ratio(candidate: Sequence[float], other: Sequence[float]) -> float:
    """``candidate`` 被 ``other`` 覆盖的面积比例（单向）。"""
    area = _box_area(candidate)
    if area <= 0.0:
        return 0.0
    return _intersection_area(candidate, other) / area


def nms(
    boxes: Sequence[Sequence[float]],
    *,
    scores: Optional[Sequence[float]] = None,
    iou_threshold: float = 0.6,
    containment_threshold: float = 0.8,
) -> List[BBox]:
    """同 :func:`nms_indices`，直接返回保留下来的框。"""
    return [
        tuple(float(value) for value in boxes[index])  # type: ignore[misc]
        for index in nms_indices(
            boxes,
            scores=scores,
            iou_threshold=iou_threshold,
            containment_threshold=containment_threshold,
        )
    ]


__all__ = [
    "box_gap",
    "intersection_over_min_area",
    "merge_fragments",
    "nms",
    "nms_indices",
    "union_box",
]
