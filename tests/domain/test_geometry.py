"""领域层几何度量（步骤 3 验收测试）。

移植自基线 v0.2.0 的 ``tests/test_iou.py``，导入路径改到 ``mot.domain.geometry``；
零面积盒断言改为 ``ValidationError``（其父类仍是 ``ValueError``）。
"""

from __future__ import annotations

import numpy as np
import pytest

from mot.domain.errors import ValidationError
from mot.domain.geometry import (
    box_iou,
    boxes_to_list,
    describe,
    intersection_area,
    iou_matrix,
    pairwise_centroid_distance,
    union_area,
)


class TestIntersection:
    def test_identical_boxes(self):
        assert intersection_area((0, 0, 10, 10), (0, 0, 10, 10)) == 100.0

    def test_disjoint_boxes(self):
        assert intersection_area((0, 0, 10, 10), (20, 20, 10, 10)) == 0.0

    def test_partial_overlap(self):
        assert intersection_area((0, 0, 10, 10), (5, 0, 10, 10)) == 50.0

    def test_touching_edges_do_not_intersect(self):
        assert intersection_area((0, 0, 10, 10), (10, 0, 10, 10)) == 0.0

    def test_union(self):
        assert union_area((0, 0, 10, 10), (5, 0, 10, 10)) == 150.0


class TestBoxIou:
    def test_identical_boxes_have_iou_one(self):
        assert box_iou((0, 0, 10, 10), (0, 0, 10, 10)) == pytest.approx(1.0)

    def test_half_overlap(self):
        assert box_iou((0, 0, 10, 10), (5, 0, 10, 10)) == pytest.approx(1 / 3)

    def test_disjoint_boxes_have_iou_zero(self):
        assert box_iou((0, 0, 10, 10), (50, 50, 10, 10)) == 0.0

    def test_is_symmetric(self):
        a, b = (1, 2, 30, 40), (10, 15, 30, 40)
        assert box_iou(a, b) == pytest.approx(box_iou(b, a))

    def test_negative_size_is_flipped(self):
        assert box_iou((10, 0, -10, 10), (0, 0, 10, 10)) == pytest.approx(1.0)

    def test_invalid_box_raises(self):
        with pytest.raises(ValidationError):
            box_iou((0, 0, 0, 10), (0, 0, 10, 10))


class TestIouMatrix:
    def test_shape_and_values(self):
        matrix = iou_matrix([(0, 0, 10, 10), (5, 0, 10, 10)], [(5, 0, 10, 10)])
        assert matrix.shape == (2, 1)
        assert matrix[0, 0] == pytest.approx(1 / 3)
        assert matrix[1, 0] == pytest.approx(1.0)

    def test_matrix_matches_pairwise_function(self):
        a = [(0, 0, 10, 10), (20, 20, 5, 5)]
        b = [(5, 5, 10, 10), (0, 25, 10, 10)]
        matrix = iou_matrix(a, b)
        for i in range(len(a)):
            for j in range(len(b)):
                assert matrix[i, j] == pytest.approx(box_iou(a[i], b[j]))

    def test_values_are_clipped_to_unit_range(self):
        matrix = iou_matrix([(0, 0, 10, 10)], [(0, 0, 10, 10), (100, 100, 5, 5)])
        assert float(matrix.min()) >= 0.0
        assert float(matrix.max()) <= 1.0

    def test_empty_inputs(self):
        assert iou_matrix([], []).shape == (0, 0)
        assert iou_matrix([(0, 0, 10, 10)], []).shape == (1, 0)
        assert iou_matrix([], [(0, 0, 10, 10)]).shape == (0, 1)

    def test_invalid_box_raises(self):
        with pytest.raises(ValidationError):
            iou_matrix([(0, 0, 0, 10)], [(0, 0, 10, 10)])


class TestCentroidDistance:
    def test_identical_boxes_have_zero_distance(self):
        matrix = pairwise_centroid_distance([(0, 0, 10, 10)], [(0, 0, 10, 10)])
        assert matrix[0, 0] == pytest.approx(0.0)

    def test_distance_between_centres(self):
        matrix = pairwise_centroid_distance([(0, 0, 10, 10)], [(30, 40, 10, 10)])
        assert matrix[0, 0] == pytest.approx(50.0)

    def test_empty_inputs(self):
        assert pairwise_centroid_distance([], []).shape == (0, 0)


def test_boxes_to_list_normalises():
    assert boxes_to_list([(10, 10, -5, -5)]) == [(5.0, 5.0, 5.0, 5.0)]


def test_describe_formats_box():
    assert describe((1.234, 2.0, 3.0, 4.0)) == "<x=1.2 y=2.0 w=3.0 h=4.0>"


def test_matrix_is_float64():
    assert iou_matrix([(0, 0, 10, 10)], [(0, 0, 10, 10)]).dtype == np.float64
