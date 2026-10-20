"""领域层检测后处理：碎片合并与 NMS（纯几何，无 cv2 依赖）。"""

from __future__ import annotations

import pytest

from mot.domain.postprocess import (
    box_gap,
    intersection_over_min_area,
    merge_fragments,
    nms,
    nms_indices,
    union_box,
)

WHOLE = (0.0, 0.0, 40.0, 40.0)
HALF_LEFT = (0.0, 0.0, 22.0, 40.0)
HALF_TOP = (0.0, 0.0, 40.0, 22.0)
INSIDE = (10.0, 10.0, 10.0, 10.0)
FAR = (200.0, 200.0, 20.0, 20.0)


class TestGeometryHelpers:
    def test_contained_box_has_full_overlap_over_min_area(self):
        assert intersection_over_min_area(WHOLE, INSIDE) == pytest.approx(1.0)

    def test_disjoint_boxes_have_no_overlap(self):
        assert intersection_over_min_area(WHOLE, FAR) == 0.0

    def test_partial_overlap_is_intersection_over_smaller(self):
        assert intersection_over_min_area((0, 0, 10, 10), (5, 0, 10, 10)) == pytest.approx(0.5)

    def test_gap_is_zero_for_overlap_and_touching_boxes(self):
        assert box_gap(WHOLE, INSIDE) == 0.0
        assert box_gap((0, 0, 10, 10), (10, 0, 10, 10)) == 0.0

    def test_gap_is_the_shortest_edge_distance(self):
        assert box_gap((0, 0, 10, 10), (14, 15, 10, 10)) == pytest.approx(41**0.5)

    def test_union_covers_both_boxes(self):
        assert union_box((0, 0, 10, 10), (20, 30, 5, 5)) == (0.0, 0.0, 25.0, 35.0)


class TestMergeFragments:
    def test_single_and_empty_input_passes_through(self):
        assert merge_fragments([]) == []
        assert merge_fragments([WHOLE]) == [WHOLE]

    def test_overlapping_fragments_become_one_whole_box(self):
        merged = merge_fragments([HALF_LEFT, HALF_TOP])
        assert merged == [(0.0, 0.0, 40.0, 40.0)]

    def test_chain_of_fragments_is_merged_transitively(self):
        boxes = [(0, 0, 20, 20), (10, 0, 20, 20), (20, 0, 20, 20)]
        merged = merge_fragments(boxes, iou_threshold=0.3)
        assert merged == [(0.0, 0.0, 40.0, 20.0)]

    def test_separate_neighbours_are_never_merged(self):
        left = (0.0, 0.0, 20.0, 20.0)
        right = (25.0, 0.0, 20.0, 20.0)  # 5px gap
        assert merge_fragments([left, right]) == [left, right]

    def test_touching_boxes_are_kept_separate(self):
        left = (0.0, 0.0, 20.0, 20.0)
        right = (20.0, 0.0, 20.0, 20.0)  # 边缘紧贴，但 IoU=0
        assert merge_fragments([left, right]) == [left, right]

    def test_slightly_overlapping_boxes_below_iou_are_not_merged(self):
        left = (0.0, 0.0, 20.0, 20.0)
        right = (19.0, 0.0, 20.0, 20.0)  # IoU≈0.026 < 0.15
        assert merge_fragments([left, right]) == [left, right]

    def test_positive_gap_merges_small_splits(self):
        left = (0.0, 0.0, 20.0, 20.0)
        right = (22.0, 0.0, 20.0, 20.0)  # 2px gap
        assert merge_fragments([left, right], gap_px=2) == [(0.0, 0.0, 42.0, 20.0)]

    def test_output_is_sorted_by_position(self):
        boxes = [(100, 100, 10, 10), (0, 50, 10, 10), (0, 0, 10, 10)]
        assert merge_fragments(boxes) == [
            (0.0, 0.0, 10.0, 10.0),
            (0.0, 50.0, 10.0, 10.0),
            (100.0, 100.0, 10.0, 10.0),
        ]


class TestNms:
    def test_empty_input(self):
        assert nms([]) == []
        assert nms_indices([]) == []

    def test_contained_local_box_is_suppressed_in_favour_of_the_whole_box(self):
        assert nms([INSIDE, WHOLE]) == [WHOLE]
        assert nms([WHOLE, INSIDE]) == [WHOLE]

    def test_heavy_overlap_keeps_the_higher_scoring_box(self):
        boxes = [(0, 0, 20, 20), (2, 2, 20, 20)]
        assert nms(boxes, scores=[0.4, 0.9], iou_threshold=0.5) == [(2.0, 2.0, 20.0, 20.0)]

    def test_disjoint_boxes_are_both_kept(self):
        assert len(nms([WHOLE, FAR])) == 2

    def test_custom_scores_decide_the_winner(self):
        # 小框分数更高时不会被包含关系抑制（分数优先于面积）。
        kept = nms_indices(
            [WHOLE, INSIDE], scores=[0.1, 0.9], containment_threshold=1.0
        )
        assert kept == [1, 0]

    def test_slightly_jittered_neighbour_boxes_are_both_kept(self):
        left = (0.0, 0.0, 20.0, 20.0)
        right = (24.0, 0.0, 20.0, 20.0)
        assert len(nms([left, right], iou_threshold=0.3)) == 2
