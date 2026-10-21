"""领域层几何度量：IoU 与质心距离。

IoU 是本项目的关联度量：廉价、无额外依赖、语言中立，便于后续移植到 C++ 或
ROS 节点。本模块只依赖 ``numpy``（领域层允许的向量化依赖），不含框架与 I/O。

职责边界
--------
* 标量几何：``intersection_area`` / ``union_area`` / ``box_iou``。
* 矩阵几何（关联算法的输入）：``iou_matrix`` / ``pairwise_centroid_distance``。
* 便利函数：``boxes_to_list`` / ``describe``。

包围盒校验统一走 :func:`mot.domain.models.normalize_bbox`，因此零面积盒会抛
:class:`~mot.domain.errors.ValidationError`（而非静默返回 0）——零面积盒一定是
上游的 bug。
"""

from __future__ import annotations

from typing import Iterable, List, Sequence

import numpy as np

from .models import BBox, bbox_area, normalize_bbox

EPS = 1e-9


def intersection_area(a: Sequence[float], b: Sequence[float]) -> float:
    """返回两个 ``(x, y, w, h)`` 盒的相交面积。"""
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix = min(ax + aw, bx + bw) - max(ax, bx)
    iy = min(ay + ah, by + bh) - max(ay, by)
    if ix <= 0.0 or iy <= 0.0:
        return 0.0
    return float(ix * iy)


def union_area(a: Sequence[float], b: Sequence[float]) -> float:
    """返回两个盒覆盖的并集面积。"""
    return bbox_area(a) + bbox_area(b) - intersection_area(a, b)


def box_iou(a: Sequence[float], b: Sequence[float]) -> float:
    """返回 ``[0, 1]`` 的 IoU；退化/相离盒为 ``0.0``。

    输入经 :func:`~mot.domain.models.normalize_bbox` 校验，因此零面积盒会抛
    :class:`~mot.domain.errors.ValidationError`，而不是静默返回 ``0``。
    """
    a = normalize_bbox(a)
    b = normalize_bbox(b)
    inter = intersection_area(a, b)
    if inter <= 0.0:
        return 0.0
    union = union_area(a, b)
    if union <= EPS:
        return 0.0
    return float(inter / union)


def iou_matrix(
    boxes_a: Iterable[Sequence[float]], boxes_b: Iterable[Sequence[float]]
) -> np.ndarray:
    """返回 ``(len(a), len(b))`` 的两两 IoU 矩阵。

    矩阵以向量化 numpy 运算构建，是 :mod:`mot.domain.association` 中贪心 /
    匈牙利关联的输入。
    """
    a = np.asarray([normalize_bbox(box) for box in boxes_a], dtype=np.float64)
    b = np.asarray([normalize_bbox(box) for box in boxes_b], dtype=np.float64)
    if a.size == 0 or b.size == 0:
        return np.zeros((len(a), len(b)), dtype=np.float64)

    ax1, ay1 = a[:, 0], a[:, 1]
    ax2, ay2 = a[:, 0] + a[:, 2], a[:, 1] + a[:, 3]
    bx1, by1 = b[:, 0], b[:, 1]
    bx2, by2 = b[:, 0] + b[:, 2], b[:, 1] + b[:, 3]

    ix = np.clip(
        np.minimum(ax2[:, None], bx2[None, :]) - np.maximum(ax1[:, None], bx1[None, :]), 0, None
    )
    iy = np.clip(
        np.minimum(ay2[:, None], by2[None, :]) - np.maximum(ay1[:, None], by1[None, :]), 0, None
    )
    inter = ix * iy
    area_a = (a[:, 2] * a[:, 3])[:, None]
    area_b = (b[:, 2] * b[:, 3])[None, :]
    union = area_a + area_b - inter
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(union > EPS, inter / union, 0.0)
    return np.clip(out, 0.0, 1.0)


def pairwise_centroid_distance(
    boxes_a: Iterable[Sequence[float]], boxes_b: Iterable[Sequence[float]]
) -> np.ndarray:
    """返回 ``(len(a), len(b))`` 的质心距离矩阵。

    作为未来“检测器驱动模式”的次级关联线索保留（仅凭 IoU 对小而快的目标不稳）。
    当前主流程未使用（见 ``docs/ROADMAP.md`` 的 A9）。
    """
    a = np.asarray([normalize_bbox(box) for box in boxes_a], dtype=np.float64)
    b = np.asarray([normalize_bbox(box) for box in boxes_b], dtype=np.float64)
    if a.size == 0 or b.size == 0:
        return np.zeros((len(a), len(b)), dtype=np.float64)
    ca = a[:, :2] + a[:, 2:] / 2.0
    cb = b[:, :2] + b[:, 2:] / 2.0
    delta = ca[:, None, :] - cb[None, :, :]
    return np.sqrt(np.sum(delta * delta, axis=2))


def boxes_to_list(boxes: Iterable[Sequence[float]]) -> List[BBox]:
    """把盒的可迭代对象归一为已校验的盒列表。"""
    return [normalize_bbox(box) for box in boxes]


def describe(box: Sequence[float]) -> str:
    """人类可读的盒表示，便于日志与错误信息。"""
    x, y, w, h = normalize_bbox(box)
    return f"<x={x:.1f} y={y:.1f} w={w:.1f} h={h:.1f}>"


def box_scale_change(a: Sequence[float], b: Sequence[float]) -> float:
    """两个盒的尺寸相对变化倍率，返回 ``>= 1.0``（``1.0`` 表示宽高完全一致）。

    取宽、高两个方向“放大/缩小”倍率的较大值。用于轨迹二次校验：同一 id 的框
    在一帧内尺寸突变（例如整体框被局部框顶替、或被拆成碎片）时倍率明显大于 1，
    据此丢弃该检测。
    """
    aw, ah = float(a[2]), float(a[3])
    bw, bh = float(b[2]), float(b[3])
    return max(aw / bw, bw / aw, ah / bh, bh / ah)


__all__ = [
    "EPS",
    "box_iou",
    "box_scale_change",
    "boxes_to_list",
    "describe",
    "intersection_area",
    "iou_matrix",
    "pairwise_centroid_distance",
    "union_area",
]
