"""应用层单帧管道（步骤 5 验收测试）。

移植自基线 v0.2.0 的 ``tests/test_pipeline.py``：``TrackManager`` 由夹具注入
``init_fn`` / ``update_fn``；检测器签名升级为 ``(frame, timestamp)``；并因 G9
（首帧即检测）调整了检测间隔的期望值。
"""

from __future__ import annotations

import pytest

from mot.application.pipeline import MOTPipeline
from mot.domain.errors import ValidationError
from mot.domain.models import Detection

BOX = (10.0, 10.0, 20.0, 20.0)


def make_pipeline(make_manager, pool, **kwargs) -> MOTPipeline:
    return MOTPipeline(make_manager(pool, **kwargs))


class TestManualMode:
    def test_add_manual_box_then_track(self, make_manager, pool):
        pipeline = make_pipeline(make_manager, pool)
        track = pipeline.add_manual_box(None, BOX)
        assert track is not None
        result = pipeline.step(None)
        assert result.frame_index == 1
        assert len(result.tracks) == 1
        assert result.tracks[0].track_id == track.track_id

    def test_manual_box_with_invalid_geometry_raises(self, make_manager, pool):
        pipeline = make_pipeline(make_manager, pool)
        with pytest.raises(ValidationError):
            pipeline.add_manual_box(None, (0, 0, 0, 0))

    def test_delete_at_picks_the_smallest_containing_target(self, make_manager, pool):
        pipeline = make_pipeline(make_manager, pool)
        large = pipeline.add_manual_box(None, (0, 0, 100, 100))
        small = pipeline.add_manual_box(None, (10, 10, 20, 20))
        assert large is not None and small is not None
        removed = pipeline.delete_at((15.0, 15.0))
        assert removed == small.track_id
        assert pipeline.remove(large.track_id) is True

    def test_delete_at_returns_none_for_empty_space(self, make_manager, pool):
        pipeline = make_pipeline(make_manager, pool)
        pipeline.add_manual_box(None, BOX)
        assert pipeline.delete_at((500.0, 500.0)) is None

    def test_delete_at_ignores_lost_targets(self, make_manager, pool):
        pipeline = make_pipeline(make_manager, pool, max_missing=5)
        pipeline.add_manual_box(None, BOX)
        pool.trackers[0].alive = False
        pipeline.step(None)
        assert pipeline.delete_at((20.0, 20.0)) is None


class TestDetectorMode:
    def test_first_frame_always_runs_the_detector(self, make_manager, pool):
        calls = []

        def detector(frame, timestamp):
            calls.append((frame, timestamp))
            return []

        pipeline = MOTPipeline(make_manager(pool), detector=detector, detection_interval=3)
        pipeline.step("f", 0.0)
        assert len(calls) == 1  # G9：首帧立即检测

    def test_detector_runs_on_the_configured_interval(self, make_manager, pool):
        calls = []

        def detector(frame, timestamp):
            calls.append(frame)
            return []

        pipeline = MOTPipeline(make_manager(pool), detector=detector, detection_interval=2)
        for _ in range(5):
            pipeline.step("frame", 0.0)
        # 首帧立即检测，之后每 2 帧一次 → 第 1/3/5 帧，共 3 次。
        assert len(calls) == 3

    def test_detector_receives_the_timestamp(self, make_manager, pool):
        seen = []

        def detector(frame, timestamp):
            seen.append(timestamp)
            return []

        pipeline = MOTPipeline(make_manager(pool), detector=detector)
        pipeline.step("f", 1.25)
        assert seen == [1.25]

    def test_detector_output_spawns_ids(self, make_manager, pool):
        pipeline = MOTPipeline(
            make_manager(pool), detector=lambda frame, timestamp: [Detection(BOX)]
        )
        result = pipeline.step(None)
        assert result.created == (1,)
        assert len(result.detections) == 1

    def test_detector_returning_none_is_tolerated(self, make_manager, pool):
        pipeline = MOTPipeline(make_manager(pool), detector=lambda frame, timestamp: None)
        result = pipeline.step(None)
        assert result.created == ()

    def test_duplicate_detections_are_collapsed(self, make_manager, pool):
        pipeline = make_pipeline(make_manager, pool)
        result = pipeline.step(
            None,
            extra_detections=[
                Detection(BOX, confidence=0.9),
                Detection((11, 11, 20, 20), confidence=0.5),
            ],
        )
        assert result.created == (1,)
        assert len(result.detections) == 1

    def test_highest_confidence_detection_wins_dedupe(self, make_manager, pool):
        pipeline = make_pipeline(make_manager, pool)
        result = pipeline.step(
            None,
            extra_detections=[
                Detection((12, 12, 20, 20), confidence=0.4),
                Detection(BOX, confidence=0.95),
            ],
        )
        assert result.detections[0].bbox == BOX

    def test_non_overlapping_detections_both_survive(self, make_manager, pool):
        pipeline = make_pipeline(make_manager, pool)
        result = pipeline.step(
            None, extra_detections=[Detection(BOX), Detection((200, 200, 20, 20))]
        )
        assert len(result.created) == 2

    def test_invalid_detector_is_rejected(self, make_manager, pool):
        with pytest.raises(TypeError):
            MOTPipeline(make_manager(pool), detector="not-callable")

    def test_invalid_interval_is_rejected(self, make_manager, pool):
        with pytest.raises(ValidationError):
            MOTPipeline(make_manager(pool), detection_interval=0)


class TestDetectionNormalisation:
    def test_tuples_are_accepted(self, make_manager, pool):
        pipeline = make_pipeline(make_manager, pool)
        result = pipeline.step(None, extra_detections=[(0, 0, 10, 10)])
        assert result.created == (1,)
        assert result.detections[0].bbox == (0.0, 0.0, 10.0, 10.0)

    def test_wrong_type_raises(self, make_manager, pool):
        pipeline = make_pipeline(make_manager, pool)
        with pytest.raises(TypeError):
            pipeline.step(None, extra_detections=["nope"])


class TestFrameResult:
    def test_carries_timestamp_and_matched_ids(self, make_manager, pool):
        pipeline = make_pipeline(make_manager, pool)
        pipeline.step(None, 0.0, extra_detections=[Detection(BOX)])
        result = pipeline.step(None, 0.5, extra_detections=[Detection(BOX)])
        assert result.timestamp == 0.5
        assert result.matched == ((0, 0),)
        assert result.matched_ids == ((1, "object"),)


class TestBookkeeping:
    def test_stats_report_frames_and_trails(self, make_manager, pool):
        pipeline = make_pipeline(make_manager, pool)
        pipeline.add_manual_box(None, BOX)
        pipeline.step(None)
        stats = pipeline.stats()
        assert stats["frames"] == 1
        assert stats["tracks"] == 1
        assert stats["created_ids"] == 1
        assert stats["trail_points"] == 1  # 静止目标不会重复追加同一轨迹点

    def test_reset_clears_tracks_and_frame_count(self, make_manager, pool):
        pipeline = make_pipeline(make_manager, pool)
        pipeline.add_manual_box(None, BOX)
        pipeline.step(None)
        pipeline.reset()
        assert pipeline.frame_count == 0
        assert pipeline.tracks == ()

    def test_clear_trails_through_pipeline(self, make_manager, pool):
        pipeline = make_pipeline(make_manager, pool)
        track = pipeline.add_manual_box(None, BOX)
        pipeline.clear_trails()
        assert track is not None
        assert list(track.trail) == []
