"""领域层关联：把已有目标与当前帧检测配对。

两种算法：

* ``greedy`` —— 按重叠度从高到低贪心配对，``O(n*m log(n*m))``，确定性强、无额外
  依赖，是默认算法。
* ``hungarian`` —— 经 ``scipy.optimize.linear_sum_assignment`` 的全局最优一对一
  指派；装了 scipy 且调用方显式要求 ``method="hungarian"`` 时使用，否则回退贪心
  并告警一次。

两者都返回“匹配下标对 + 未匹配行/列”，供调用方据此新建 id 或让丢失目标过期。
"""

from __future__ import annotations

import logging
from typing import List, Sequence, Tuple

import numpy as np

from .errors import ValidationError
from .models import BBox

Pair = Tuple[int, int]
MatchResult = Tuple[List[Pair], List[int], List[int]]

#: 支持的关联算法名（单一真源，config 与领域层共用）。
MATCH_METHODS = ("greedy", "hungarian")

LOGGER = logging.getLogger(__name__)
_warned_scipy_fallback = False

try:  # pragma: no cover - 仅在装了 scipy 时走到
    from scipy.optimize import linear_sum_assignment

    _HAVE_SCIPY = True
except Exception:  # pragma: no cover - scipy 为可选依赖
    linear_sum_assignment = None  # type: ignore[assignment]
    _HAVE_SCIPY = False

#: 匈牙利求解中用于“禁止低于阈值配对”的代价，大于任何真实代价（1.0），
#: 因此仅在没有其它可能时才可能被选中，随后被阈值过滤掉。
_FORBIDDEN_COST = 1e6


def scipy_available() -> bool:
    """返回可选的匈牙利求解器是否可用。"""
    return _HAVE_SCIPY


def _validate_overlap(overlap: np.ndarray, threshold: float) -> np.ndarray:
    matrix = np.asarray(overlap, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValidationError(
            f"overlap matrix must be 2-D, got shape {matrix.shape}", field="overlap"
        )
    if not 0.0 <= float(threshold) <= 1.0:
        raise ValidationError(
            f"threshold must be within [0, 1], got {threshold}", field="threshold", value=threshold
        )
    if not np.all(np.isfinite(matrix)):
        raise ValidationError("overlap matrix contains non-finite values", field="overlap")
    return matrix


def greedy_match(overlap: np.ndarray, threshold: float = 0.3) -> MatchResult:
    """按重叠度降序贪心配对行与列。

    平局按 ``(row, column)`` 排序，保证结果跨运行/跨平台可复现。

    Args:
        overlap: ``(n_rows, n_cols)`` 矩阵，通常为 IoU。
        threshold: 视为匹配的最小重叠（包含式）。

    Returns:
        ``(pairs, unmatched_rows, unmatched_cols)``，``pairs`` 按重叠度降序。
    """
    matrix = _validate_overlap(overlap, threshold)
    n_rows, n_cols = matrix.shape

    candidates = [
        (float(matrix[i, j]), i, j)
        for i in range(n_rows)
        for j in range(n_cols)
        if matrix[i, j] >= threshold
    ]
    candidates.sort(key=lambda item: (-item[0], item[1], item[2]))

    used_rows: set = set()
    used_cols: set = set()
    pairs: List[Pair] = []
    for _score, row, col in candidates:
        if row in used_rows or col in used_cols:
            continue
        used_rows.add(row)
        used_cols.add(col)
        pairs.append((row, col))

    unmatched_rows = [i for i in range(n_rows) if i not in used_rows]
    unmatched_cols = [j for j in range(n_cols) if j not in used_cols]
    return pairs, unmatched_rows, unmatched_cols


def hungarian_match(overlap: np.ndarray, threshold: float = 0.3) -> MatchResult:
    """最大化总重叠的最优一对一指派。

    低于 ``threshold`` 的配对永不返回。未装 scipy 时回退到 :func:`greedy_match`。
    """
    matrix = _validate_overlap(overlap, threshold)
    if not _HAVE_SCIPY or linear_sum_assignment is None:
        global _warned_scipy_fallback
        if not _warned_scipy_fallback:
            LOGGER.warning(
                "hungarian matching needs scipy (pip install scipy); "
                "falling back to greedy matching"
            )
            _warned_scipy_fallback = True
        return greedy_match(matrix, threshold)

    n_rows, n_cols = matrix.shape
    if n_rows == 0 or n_cols == 0:
        return [], list(range(n_rows)), list(range(n_cols))

    cost = np.where(matrix >= threshold, 1.0 - matrix, _FORBIDDEN_COST)
    row_ind, col_ind = linear_sum_assignment(cost)
    pairs: List[Pair] = [
        (int(r), int(c))
        for r, c in zip(row_ind.tolist(), col_ind.tolist())
        if matrix[r, c] >= threshold
    ]
    pairs.sort(key=lambda item: (-float(matrix[item[0], item[1]]), item[0], item[1]))
    matched_rows = {r for r, _ in pairs}
    matched_cols = {c for _, c in pairs}
    unmatched_rows = [i for i in range(n_rows) if i not in matched_rows]
    unmatched_cols = [j for j in range(n_cols) if j not in matched_cols]
    return pairs, unmatched_rows, unmatched_cols


def match(overlap: np.ndarray, threshold: float = 0.3, method: str = "greedy") -> MatchResult:
    """分派到指定的关联算法。

    Raises:
        ValidationError: 方法名未知或矩阵非法。
    """
    if method not in MATCH_METHODS:
        raise ValidationError(
            f"unknown match method {method!r}; expected one of {MATCH_METHODS}",
            field="match_method",
            value=method,
        )
    if method == "hungarian":
        return hungarian_match(overlap, threshold)
    return greedy_match(overlap, threshold)


def match_boxes(
    track_boxes: Sequence[BBox],
    detections: Sequence[BBox],
    threshold: float = 0.3,
    method: str = "greedy",
) -> MatchResult:
    """便利封装：先算 IoU 再关联两组盒。"""
    from .geometry import iou_matrix

    return match(iou_matrix(track_boxes, detections), threshold=threshold, method=method)


__all__ = [
    "MATCH_METHODS",
    "MatchResult",
    "Pair",
    "greedy_match",
    "hungarian_match",
    "match",
    "match_boxes",
    "scipy_available",
]
