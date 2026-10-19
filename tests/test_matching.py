"""Greedy and Hungarian association."""

from __future__ import annotations

import numpy as np
import pytest

import mot.matching as matching
from mot.matching import (
    MATCH_METHODS,
    greedy_match,
    hungarian_match,
    match,
    match_boxes,
    scipy_available,
)


class TestGreedyMatch:
    def test_picks_highest_overlap_first(self):
        pairs, rows, cols = greedy_match(np.array([[0.9, 0.4], [0.5, 0.2]]), 0.3)
        assert pairs == [(0, 0)]
        assert rows == [1]
        assert cols == [1]

    def test_each_row_and_column_is_used_once(self):
        pairs, rows, cols = greedy_match(np.array([[0.5, 0.5], [0.5, 0.5]]), 0.3)
        assert pairs == [(0, 0), (1, 1)]
        assert rows == []
        assert cols == []

    def test_threshold_is_inclusive(self):
        pairs, _, _ = greedy_match(np.array([[0.3]]), 0.3)
        assert pairs == [(0, 0)]

    def test_below_threshold_is_unmatched(self):
        pairs, rows, cols = greedy_match(np.array([[0.29]]), 0.3)
        assert pairs == []
        assert rows == [0]
        assert cols == [0]

    def test_empty_matrix(self):
        pairs, rows, cols = greedy_match(np.zeros((0, 0)), 0.3)
        assert pairs == [] and rows == [] and cols == []

    def test_single_row_multiple_columns(self):
        pairs, rows, cols = greedy_match(np.array([[0.2, 0.8, 0.6]]), 0.3)
        assert pairs == [(0, 1)]
        assert rows == []
        assert cols == [0, 2]

    def test_pairs_sorted_by_descending_overlap(self):
        pairs, _, _ = greedy_match(np.array([[0.4, 0.9], [0.95, 0.1]]), 0.3)
        assert pairs == [(1, 0), (0, 1)]


class TestHungarianMatch:
    def test_respects_threshold(self):
        pairs, rows, cols = hungarian_match(np.array([[0.9, 0.1]]), 0.3)
        assert pairs == [(0, 0)]
        assert cols == [1]
        assert rows == []

    def test_beats_greedy_on_a_conflict(self):
        # Greedy grabs (0, 0)=0.60 and strands row 1; the optimal assignment
        # is (0, 1)+(1, 0) with a larger total overlap.
        matrix = np.array([[0.60, 0.55], [0.59, 0.0]])
        greedy_pairs, _, _ = greedy_match(matrix, 0.3)
        optimal_pairs, _, _ = hungarian_match(matrix, 0.3)
        assert greedy_pairs == [(0, 0)]
        assert optimal_pairs == [(1, 0), (0, 1)]

    def test_returns_every_row_when_no_column_is_available(self):
        pairs, rows, cols = hungarian_match(np.zeros((2, 1)), 0.3)
        assert pairs == []
        assert rows == [0, 1]
        assert cols == [0]

    def test_empty_matrix(self):
        pairs, rows, cols = hungarian_match(np.zeros((0, 3)), 0.3)
        assert pairs == [] and rows == [] and cols == [0, 1, 2]

    def test_falls_back_to_greedy_without_scipy(self, monkeypatch):
        matrix = np.array([[0.60, 0.55], [0.59, 0.0]])
        monkeypatch.setattr(matching, "_HAVE_SCIPY", False)
        assert hungarian_match(matrix, 0.3) == greedy_match(matrix, 0.3)


class TestMatchDispatch:
    def test_greedy_is_the_default(self):
        matrix = np.array([[0.7]])
        assert match(matrix) == greedy_match(matrix)

    def test_unknown_method_raises(self):
        with pytest.raises(ValueError):
            match(np.zeros((1, 1)), 0.3, method="magic")

    def test_methods_are_the_documented_pair(self):
        assert MATCH_METHODS == ("greedy", "hungarian")

    def test_match_boxes_uses_iou(self):
        pairs, _, _ = match_boxes([(0, 0, 10, 10)], [(0, 0, 10, 10)], 0.3)
        assert pairs == [(0, 0)]


class TestValidation:
    @pytest.mark.parametrize("threshold", [-0.1, 1.1])
    def test_threshold_must_be_in_unit_range(self, threshold):
        with pytest.raises(ValueError):
            greedy_match(np.zeros((1, 1)), threshold)

    def test_matrix_must_be_two_dimensional(self):
        with pytest.raises(ValueError):
            greedy_match(np.zeros(3), 0.3)

    def test_matrix_must_be_finite(self):
        with pytest.raises(ValueError):
            greedy_match(np.array([[np.nan]]), 0.3)


def test_scipy_available_reports_a_bool():
    assert isinstance(scipy_available(), bool)
