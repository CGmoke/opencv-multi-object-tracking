"""Pipeline orchestration: detectors, dedupe, manual edits."""

from __future__ import annotations

import pytest

from conftest import TrackerPool
from mot import Detection, MOTPipeline, TrackManager

BOX = (10.0, 10.0, 20.0, 20.0)


def make_pipeline(pool: TrackerPool, **kwargs) -> MOTPipeline:
    return MOTPipeline(TrackManager(pool, **kwargs))


class TestManualMode:
    def test_add_manual_box_then_track(self, pool):
        pipeline = make_pipeline(pool)
        track = pipeline.add_manual_box(None, BOX)
        assert track is not None
        result = pipeline.step(None)
        assert result.frame_index == 1
        assert len(result.tracks) == 1
        assert result.tracks[0].track_id == track.track_id

    def test_manual_box_with_invalid_geometry_raises(self, pool):
        pipeline = make_pipeline(pool)
        with pytest.raises(ValueError):
            pipeline.add_manual_box(None, (0, 0, 0, 0))

    def test_delete_at_picks_the_smallest_containing_target(self, pool):
        pipeline = make_pipeline(pool)
        large = pipeline.add_manual_box(None, (0, 0, 100, 100))
        small = pipeline.add_manual_box(None, (10, 10, 20, 20))
        assert large is not None and small is not None
        removed = pipeline.delete_at((15.0, 15.0))
        assert removed == small.track_id
        assert pipeline.remove(large.track_id) is True

    def test_delete_at_returns_none_for_empty_space(self, pool):
        pipeline = make_pipeline(pool)
        pipeline.add_manual_box(None, BOX)
        assert pipeline.delete_at((500.0, 500.0)) is None

    def test_delete_at_ignores_lost_targets(self, pool):
        pipeline = make_pipeline(pool, max_missing=5)
        pipeline.add_manual_box(None, BOX)
        pool.trackers[0].alive = False
        pipeline.step(None)
        assert pipeline.delete_at((20.0, 20.0)) is None


class TestDetectorMode:
    def test_detector_runs_on_the_configured_interval(self, pool):
        calls = []

        def detector(frame):
            calls.append(frame)
            return []

        pipeline = MOTPipeline(TrackManager(pool), detector=detector, detection_interval=2)
        for _ in range(5):
            pipeline.step("frame")
        assert len(calls) == 2

    def test_detector_output_spawns_ids(self, pool):
        pipeline = MOTPipeline(TrackManager(pool), detector=lambda frame: [Detection(BOX)])
        result = pipeline.step(None)
        assert result.created == (1,)
        assert len(result.detections) == 1

    def test_detector_returning_none_is_tolerated(self, pool):
        pipeline = MOTPipeline(TrackManager(pool), detector=lambda frame: None)
        result = pipeline.step(None)
        assert result.created == ()

    def test_duplicate_detections_are_collapsed(self, pool):
        pipeline = MOTPipeline(TrackManager(pool))
        result = pipeline.step(
            None, [Detection(BOX, confidence=0.9), Detection((11, 11, 20, 20), confidence=0.5)]
        )
        assert result.created == (1,)
        assert len(result.detections) == 1

    def test_highest_confidence_detection_wins_dedupe(self, pool):
        pipeline = MOTPipeline(TrackManager(pool))
        result = pipeline.step(
            None,
            [Detection((12, 12, 20, 20), confidence=0.4), Detection(BOX, confidence=0.95)],
        )
        assert result.detections[0].bbox == BOX

    def test_non_overlapping_detections_both_survive(self, pool):
        pipeline = MOTPipeline(TrackManager(pool))
        result = pipeline.step(None, [Detection(BOX), Detection((200, 200, 20, 20))])
        assert len(result.created) == 2

    def test_invalid_detector_is_rejected(self, pool):
        with pytest.raises(TypeError):
            MOTPipeline(TrackManager(pool), detector="not-callable")

    def test_invalid_interval_is_rejected(self, pool):
        with pytest.raises(ValueError):
            MOTPipeline(TrackManager(pool), detection_interval=0)


class TestDetectionNormalisation:
    def test_tuples_are_accepted(self, pool):
        pipeline = make_pipeline(pool)
        result = pipeline.step(None, [(0, 0, 10, 10)])
        assert result.created == (1,)
        assert result.detections[0].bbox == (0.0, 0.0, 10.0, 10.0)

    def test_wrong_type_raises(self, pool):
        pipeline = make_pipeline(pool)
        with pytest.raises(TypeError):
            pipeline.step(None, ["nope"])


class TestBookkeeping:
    def test_stats_report_frames_and_trails(self, pool):
        pipeline = make_pipeline(pool)
        pipeline.add_manual_box(None, BOX)
        pipeline.step(None)
        stats = pipeline.stats()
        assert stats["frames"] == 1
        assert stats["tracks"] == 1
        assert stats["created_ids"] == 1
        assert stats["trail_points"] == 1  # a static target does not duplicate its trail point

    def test_reset_clears_tracks_and_frame_count(self, pool):
        pipeline = make_pipeline(pool)
        pipeline.add_manual_box(None, BOX)
        pipeline.step(None)
        pipeline.reset()
        assert pipeline.frame_count == 0
        assert pipeline.tracks == ()

    def test_clear_trails_through_pipeline(self, pool):
        pipeline = make_pipeline(pool)
        track = pipeline.add_manual_box(None, BOX)
        pipeline.clear_trails()
        assert track is not None
        assert list(track.trail) == []
