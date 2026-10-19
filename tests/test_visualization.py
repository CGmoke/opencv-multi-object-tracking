"""Rendering helpers: colours, boxes, trails, HUD."""

from __future__ import annotations

import pytest

cv2 = pytest.importorskip("cv2")

import numpy as np

from mot.models import TrackedObject, TrackState
from mot.visualization import (
    color_for_id,
    draw_banner,
    draw_box,
    draw_help,
    draw_hud,
    draw_selection,
    draw_tracks,
    draw_trail,
    font_scale_for,
    scale_color,
    thickness_for,
)


def blank(height: int = 120, width: int = 160) -> np.ndarray:
    return np.zeros((height, width, 3), dtype=np.uint8)


def track(track_id: int = 1, state: TrackState = TrackState.ACTIVE) -> TrackedObject:
    obj = TrackedObject(
        track_id=track_id,
        tracker=None,
        bbox=(20.0, 20.0, 40.0, 30.0),
        state=state,
        label="person",
        confidence=0.87,
    )
    for x in range(10):
        obj.push_trail((20.0 + x * 2, 30.0), 100)
    return obj


class TestColors:
    def test_color_is_stable_per_id(self):
        assert color_for_id(7) == color_for_id(7)

    def test_colors_differ_between_ids(self):
        colors = {color_for_id(track_id) for track_id in range(1, 9)}
        assert len(colors) == 8

    def test_components_are_valid_bgr(self):
        for track_id in range(1, 20):
            assert all(0 <= channel <= 255 for channel in color_for_id(track_id))

    def test_scale_color_blends(self):
        assert scale_color((255, 100, 0), 1.0) == (255, 100, 0)
        assert scale_color((255, 100, 0), 0.0) == (0, 0, 0)
        assert scale_color((255, 100, 0), 0.5) == (128, 50, 0)

    def test_scale_color_clamps_factor(self):
        assert scale_color((10, 20, 30), 5.0) == (10, 20, 30)
        assert scale_color((10, 20, 30), -1.0) == (0, 0, 0)


class TestScaleHelpers:
    def test_font_scale_is_bounded(self):
        assert font_scale_for(blank(10, 10)) == pytest.approx(0.38)
        assert font_scale_for(blank(2000, 10)) == pytest.approx(0.9)

    def test_thickness_grows_with_height(self):
        assert thickness_for(blank(100, 100)) >= 1
        assert thickness_for(blank(1000, 100)) > thickness_for(blank(100, 100))


class TestDrawing:
    def test_draw_box_changes_pixels_along_the_border(self):
        frame = blank()
        draw_box(frame, (10, 10, 30, 20), (0, 255, 0), 2, label="1", score=0.9)
        assert frame[10, 15].any()
        assert frame[30, 15].any()

    def test_draw_box_without_label(self):
        frame = blank()
        assert draw_box(frame, (10, 10, 30, 20), (0, 255, 0), show_label=False) is frame

    def test_draw_box_label_near_top_edge_does_not_crash(self):
        frame = blank()
        draw_box(frame, (10, 0, 30, 20), (0, 255, 0), label="1")
        assert frame.shape == (120, 160, 3)

    def test_draw_trail_draws_a_line(self):
        frame = blank()
        draw_trail(frame, [(10, 10), (30, 30), (50, 20)], (0, 0, 255), 2)
        assert frame.any()

    def test_draw_trail_single_point_draws_a_dot(self):
        frame = blank()
        draw_trail(frame, [(10, 10)], (0, 0, 255), 2)
        assert frame.any()

    def test_draw_trail_empty_is_a_noop(self):
        frame = blank()
        assert draw_trail(frame, [], (0, 0, 255)) is frame
        assert not frame.any()

    def test_draw_tracks_renders_active_and_lost(self):
        frame = blank()
        draw_tracks(frame, [track(1), track(2, TrackState.LOST)], show_lost=True)
        assert frame.any()

    def test_draw_tracks_can_hide_lost_targets(self):
        visible = blank()
        hidden = blank()
        draw_tracks(visible, [track(1)], show_lost=True)
        draw_tracks(hidden, [track(1, TrackState.LOST)], show_lost=False)
        assert visible.any()
        assert not hidden.any()

    def test_draw_tracks_without_trails_or_labels(self):
        frame = blank()
        draw_tracks(frame, [track(1)], show_trails=False, show_labels=False, fade=False)
        assert frame.any()

    def test_hud_renders_text(self):
        frame = blank(240, 320)
        draw_hud(
            frame,
            fps=25.0,
            n_active=2,
            total_ids=5,
            frame_index=42,
            paused=False,
            backend="csrt",
            source_name="demo.mp4",
        )
        assert frame.any()

    def test_selection_overlay(self):
        frame = blank()
        draw_selection(frame, (10, 10), (80, 60))
        assert frame.any()

    def test_help_overlay(self):
        frame = blank(240, 320)
        draw_help(frame)
        assert frame.any()

    def test_banner_overlay(self):
        frame = blank(240, 320)
        draw_banner(frame, "paused")
        assert frame.any()

    def test_banner_does_not_overflow_small_frames(self):
        frame = blank(40, 60)
        draw_banner(frame, "a rather long notice that should be truncated")
        assert frame.shape == (40, 60, 3)
