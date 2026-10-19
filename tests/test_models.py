"""Validation helpers and data structures."""

from __future__ import annotations

import dataclasses
import math

import pytest

from mot.models import (
    Detection,
    IdAllocator,
    StepResult,
    TrackedObject,
    TrackState,
    bbox_area,
    bbox_center,
    bbox_contains,
    clip_bbox,
    normalize_bbox,
)


class TestNormalizeBbox:
    def test_keeps_positive_box(self):
        assert normalize_bbox((1, 2, 3, 4)) == (1.0, 2.0, 3.0, 4.0)

    def test_flips_negative_width_and_height(self):
        assert normalize_bbox((100, 200, -20, -30)) == (80.0, 170.0, 20.0, 30.0)

    def test_flips_numpy_scalars(self):
        assert normalize_bbox([0, 0, 10.5, 20.5]) == (0.0, 0.0, 10.5, 20.5)

    @pytest.mark.parametrize("bbox", [(1, 2, 3), (1, 2, 3, 4, 5), ()])
    def test_rejects_wrong_arity(self, bbox):
        with pytest.raises(ValueError):
            normalize_bbox(bbox)

    @pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
    def test_rejects_non_finite(self, bad):
        with pytest.raises(ValueError):
            normalize_bbox((0, 0, bad, 10))

    def test_rejects_zero_sized_box(self):
        with pytest.raises(ValueError):
            normalize_bbox((0, 0, 0, 10))

    def test_accepts_minimum_side(self):
        assert normalize_bbox((0, 0, 1, 1)) == (0.0, 0.0, 1.0, 1.0)

    def test_rejects_sub_pixel_side(self):
        with pytest.raises(ValueError):
            normalize_bbox((0, 0, 0.99, 10))


class TestGeometryHelpers:
    def test_center(self):
        assert bbox_center((10, 20, 40, 60)) == (30.0, 50.0)

    def test_area_is_never_negative(self):
        assert bbox_area((0, 0, -5, -5)) == 0.0
        assert bbox_area((0, 0, 5, 6)) == 30.0

    @pytest.mark.parametrize(
        "point,expected",
        [((0, 0), True), ((10, 10), True), ((11, 5), False), ((5, -1), False)],
    )
    def test_contains(self, point, expected):
        assert bbox_contains((0, 0, 10, 10), point) is expected


class TestClipBbox:
    def test_inside_is_unchanged(self):
        assert clip_bbox((10, 10, 20, 20), 100, 100) == (10.0, 10.0, 20.0, 20.0)

    def test_partial_overlap_is_clipped(self):
        assert clip_bbox((-10, -10, 30, 30), 100, 100) == (0.0, 0.0, 20.0, 20.0)

    def test_fully_outside_returns_none(self):
        assert clip_bbox((200, 200, 10, 10), 100, 100) is None

    def test_box_beyond_bottom_right_is_clipped(self):
        assert clip_bbox((90, 90, 50, 50), 100, 100) == (90.0, 90.0, 10.0, 10.0)

    @pytest.mark.parametrize("size", [(0, 10), (10, 0), (-1, 10)])
    def test_rejects_invalid_frame_size(self, size):
        with pytest.raises(ValueError):
            clip_bbox((0, 0, 10, 10), *size)


class TestDetection:
    def test_defaults(self):
        det = Detection(bbox=(0, 0, 10, 10))
        assert det.confidence == 1.0
        assert det.label == "object"

    def test_confidence_is_clamped(self):
        assert Detection(bbox=(0, 0, 10, 10), confidence=7).confidence == 1.0
        assert Detection(bbox=(0, 0, 10, 10), confidence=-3).confidence == 0.0

    def test_confidence_must_be_finite(self):
        with pytest.raises(ValueError):
            Detection(bbox=(0, 0, 10, 10), confidence=math.nan)

    def test_invalid_box_is_rejected(self):
        with pytest.raises(ValueError):
            Detection(bbox=(0, 0, 0, 10))

    def test_is_frozen(self):
        det = Detection(bbox=(0, 0, 10, 10))
        with pytest.raises(dataclasses.FrozenInstanceError):
            det.bbox = (0, 0, 5, 5)  # type: ignore[misc]


class TestTrackedObject:
    def test_state_defaults_to_active(self):
        track = TrackedObject(track_id=1, tracker=None, bbox=(0, 0, 10, 10))
        assert track.state is TrackState.ACTIVE
        assert track.is_active is True

    def test_lost_state_is_not_active(self):
        track = TrackedObject(
            track_id=1, tracker=None, bbox=(0, 0, 10, 10), state=TrackState.LOST
        )
        assert track.is_active is False

    def test_push_trail_deduplicates_adjacent_points(self):
        track = TrackedObject(track_id=1, tracker=None, bbox=(0, 0, 10, 10))
        track.push_trail((5.0, 5.0), 10)
        track.push_trail((5.0, 5.0), 10)
        assert list(track.trail) == [(5.0, 5.0)]

    def test_push_trail_keeps_newest_points(self):
        track = TrackedObject(track_id=1, tracker=None, bbox=(0, 0, 10, 10))
        for x in range(5):
            track.push_trail((float(x), 0.0), 3)
        assert list(track.trail) == [(2.0, 0.0), (3.0, 0.0), (4.0, 0.0)]

    def test_center_property(self):
        track = TrackedObject(track_id=1, tracker=None, bbox=(10, 10, 20, 20))
        assert track.center == (20.0, 20.0)


class TestIdAllocator:
    def test_ids_are_monotonic(self):
        allocator = IdAllocator(start=5)
        assert allocator() == 5
        assert allocator() == 6
        assert allocator.next_id == 7

    def test_ids_are_never_reused(self):
        allocator = IdAllocator()
        assert allocator() == 1
        assert allocator() == 2
        assert allocator() == 3

    def test_len_counts_issued_ids(self):
        allocator = IdAllocator()
        assert len(allocator) == 0
        allocator()
        allocator()
        assert len(allocator) == 2

    def test_reset(self):
        allocator = IdAllocator()
        allocator()
        allocator.reset(10)
        assert allocator.next_id == 10

    @pytest.mark.parametrize("start", [0, -1])
    def test_rejects_invalid_start(self, start):
        with pytest.raises(ValueError):
            IdAllocator(start=start)


def test_step_result_defaults_are_empty():
    result = StepResult()
    assert result.tracks == ()
    assert result.created == ()
    assert result.removed == ()
    assert result.matched == ()
    assert result.spawn_failures == 0
