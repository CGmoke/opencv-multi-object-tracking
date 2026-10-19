"""Association between existing tracks and new detections.

Two algorithms are provided:

* ``greedy`` - sort all candidate pairs by overlap, assign while free.
  ``O(n * m log(n * m))``, deterministic and dependency free. This is the
  default and matches what the project description asks for.
* ``hungarian`` - globally optimal one-to-one assignment through
  ``scipy.optimize.linear_sum_assignment``. Used when scipy is installed and
  the caller explicitly asks for ``method="hungarian"``; otherwise the code
  falls back to greedy with a warning logged by the caller.

Both return matched index pairs plus the unmatched rows/columns so the
caller can spawn new ids or expire lost tracks.
"""

from __future__ import annotations

import logging
from typing import List, Sequence, Tuple

import numpy as np

from .models import BBox

Pair = Tuple[int, int]
MatchResult = Tuple[List[Pair], List[int], List[int]]

MATCH_METHODS = ("greedy", "hungarian")

LOGGER = logging.getLogger(__name__)
_warned_scipy_fallback = False

try:  # pragma: no cover - exercised only when scipy is installed
    from scipy.optimize import linear_sum_assignment

    _HAVE_SCIPY = True
except Exception:  # pragma: no cover - scipy is an optional dependency
    linear_sum_assignment = None  # type: ignore[assignment]
    _HAVE_SCIPY = False

#: Cost used to forbid assignments below the IoU threshold in the Hungarian
#: solver. Larger than any real cost (1.0), so forbidden pairs are only chosen
#: when nothing else is possible, and are then filtered out.
_FORBIDDEN_COST = 1e6


def scipy_available() -> bool:
    """Return whether the optional Hungarian solver can be used."""
    return _HAVE_SCIPY


def _validate_overlap(overlap: np.ndarray, threshold: float) -> np.ndarray:
    matrix = np.asarray(overlap, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError(f"overlap matrix must be 2-D, got shape {matrix.shape}")
    if not 0.0 <= float(threshold) <= 1.0:
        raise ValueError(f"threshold must be within [0, 1], got {threshold}")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("overlap matrix contains non-finite values")
    return matrix


def greedy_match(overlap: np.ndarray, threshold: float = 0.3) -> MatchResult:
    """Greedily pair rows and columns by descending overlap.

    Ties are broken by ``(row, column)`` so results are reproducible across
    runs and platforms.

    Args:
        overlap: ``(n_rows, n_cols)`` matrix, usually IoU.
        threshold: minimum overlap for a pair to be considered a match.

    Returns:
        ``(pairs, unmatched_rows, unmatched_cols)`` with pairs ordered by
        descending overlap.
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
    """Optimal one-to-one assignment maximising total overlap.

    Pairs with overlap below ``threshold`` are never returned. Falls back to
    :func:`greedy_match` if scipy is not installed.
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
        (int(r), int(c)) for r, c in zip(row_ind.tolist(), col_ind.tolist()) if matrix[r, c] >= threshold
    ]
    pairs.sort(key=lambda item: (-float(matrix[item[0], item[1]]), item[0], item[1]))
    matched_rows = {r for r, _ in pairs}
    matched_cols = {c for _, c in pairs}
    unmatched_rows = [i for i in range(n_rows) if i not in matched_rows]
    unmatched_cols = [j for j in range(n_cols) if j not in matched_cols]
    return pairs, unmatched_rows, unmatched_cols


def match(overlap: np.ndarray, threshold: float = 0.3, method: str = "greedy") -> MatchResult:
    """Dispatch to the requested association algorithm.

    Raises:
        ValueError: for an unknown method name or malformed matrix.
    """
    if method not in MATCH_METHODS:
        raise ValueError(f"unknown match method {method!r}; expected one of {MATCH_METHODS}")
    if method == "hungarian":
        return hungarian_match(overlap, threshold)
    return greedy_match(overlap, threshold)


def match_boxes(
    track_boxes: Sequence[BBox],
    detections: Sequence[BBox],
    threshold: float = 0.3,
    method: str = "greedy",
) -> MatchResult:
    """Convenience wrapper: compute IoU then associate two box lists."""
    from .iou import iou_matrix

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
