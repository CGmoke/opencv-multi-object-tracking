"""IoU geometry.

Intersection-over-union is the association metric of this project: it is
cheap, dependency free and language neutral, which makes the matching layer
easy to port to C++ or to a ROS node later.
"""

from __future__ import annotations

from typing import Iterable, List, Sequence

import numpy as np

from .models import BBox, bbox_area, normalize_bbox

EPS = 1e-9


def intersection_area(a: Sequence[float], b: Sequence[float]) -> float:
    """Return the area shared by two ``(x, y, w, h)`` boxes."""
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix = min(ax + aw, bx + bw) - max(ax, bx)
    iy = min(ay + ah, by + bh) - max(ay, by)
    if ix <= 0.0 or iy <= 0.0:
        return 0.0
    return float(ix * iy)


def union_area(a: Sequence[float], b: Sequence[float]) -> float:
    """Return the area covered by either box."""
    return bbox_area(a) + bbox_area(b) - intersection_area(a, b)


def box_iou(a: Sequence[float], b: Sequence[float]) -> float:
    """Return IoU in ``[0, 1]``; ``0.0`` for degenerate/disjoint boxes.

    The inputs are validated with :func:`mot.models.normalize_bbox`, so a
    zero-area box raises ``ValueError`` instead of silently yielding ``0``:
    a zero-area box is always a bug upstream.
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


def iou_matrix(boxes_a: Iterable[Sequence[float]], boxes_b: Iterable[Sequence[float]]) -> np.ndarray:
    """Return a ``(len(a), len(b))`` float array of pairwise IoU values.

    The matrix is built with vectorised numpy operations; it is the input of
    the greedy / Hungarian association in :mod:`mot.matching`.
    """
    a = np.asarray([normalize_bbox(box) for box in boxes_a], dtype=np.float64)
    b = np.asarray([normalize_bbox(box) for box in boxes_b], dtype=np.float64)
    if a.size == 0 or b.size == 0:
        return np.zeros((len(a), len(b)), dtype=np.float64)

    ax1, ay1 = a[:, 0], a[:, 1]
    ax2, ay2 = a[:, 0] + a[:, 2], a[:, 1] + a[:, 3]
    bx1, by1 = b[:, 0], b[:, 1]
    bx2, by2 = b[:, 0] + b[:, 2], b[:, 1] + b[:, 3]

    ix = np.clip(np.minimum(ax2[:, None], bx2[None, :]) - np.maximum(ax1[:, None], bx1[None, :]), 0, None)
    iy = np.clip(np.minimum(ay2[:, None], by2[None, :]) - np.maximum(ay1[:, None], by1[None, :]), 0, None)
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
    """Return a ``(len(a), len(b))`` matrix of centroid distances.

    Provided as a secondary association cue for future detector-backed modes
    (IoU alone is unstable for small, fast targets).
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
    """Normalise an iterable of boxes into a list of validated tuples."""
    return [normalize_bbox(box) for box in boxes]


def describe(box: Sequence[float]) -> str:
    """Human readable box, handy in log lines and error messages."""
    x, y, w, h = normalize_bbox(box)
    return f"<x={x:.1f} y={y:.1f} w={w:.1f} h={h:.1f}>"


__all__ = [
    "box_iou",
    "boxes_to_list",
    "describe",
    "intersection_area",
    "iou_matrix",
    "pairwise_centroid_distance",
    "union_area",
]
