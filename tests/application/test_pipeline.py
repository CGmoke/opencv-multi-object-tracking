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

    def test_equal_confidence_prefers_the_whole_box(self, make_manager, pool):
        pipeline = make_pipeline(make_manager, pool)
        result = pipeline.step(
            None,
            extra_detections=[
                Detection((10, 10, 20, 10), confidence=1.0),  # 局部框（先入列）
                Detection((10, 10, 20, 20), confidence=1.0),  # 整体框
            ],
        )
        assert [detection.bbox for detection in result.detections] == [(10.0, 10.0, 20.0, 20.0)]

    def test_invalid_detector_is_rejected(self, make_manager, pool):
        with pytest.raises(TypeError):
            MOTPipeline(make_manager(pool), detector="not-callable")

    def test_invalid_interval_is_rejected(self, make_manager, pool):
        with pytest.raises(ValidationError):
            MOTPipeline(make_manager(pool), detection_interval=0)

    def test_invalid_track_dedupe_parameters_are_rejected(self, make_manager, pool):
        with pytest.raises(ValidationError):
            MOTPipeline(make_manager(pool), dedupe_tracks="yes")
        with pytest.raises(ValidationError):
            MOTPipeline(make_manager(pool), merge_track_iou=0.0)
        with pytest.raises(ValidationError):
            MOTPipeline(make_manager(pool), merge_track_frames=0)


class TestTrackDedupe:
    """R3：轨迹端去重——检测端去重之后的兜底。"""

    @staticmethod
    def _two_overlapping_tracks(make_manager, pool, **kwargs) -> MOTPipeline:
        """id1 在 frame1 建立，id2 在 frame2 建立；随后让 id2 的框贴近 id1。"""
        manager = make_manager(pool, min_hits=1)
        pipeline = MOTPipeline(manager, **kwargs)
        first = pipeline.step(None, extra_detections=[Detection((10, 10, 20, 20))])
        assert first.created == (1,)
        second = pipeline.step(None, extra_detections=[Detection((100, 100, 20, 20))])
        assert second.created == (2,)
        # FakeTracker 每帧原样返回内部 box；把 id2 的框移到与 id1 几乎重合。
        pool.trackers[1].box = (12.0, 10.0, 20.0, 20.0)
        return pipeline

    def test_five_overlapping_frames_merge_and_keep_the_older_id(
        self, make_manager, pool
    ):
        pipeline = self._two_overlapping_tracks(
            make_manager, pool, merge_track_frames=5
        )
        for _ in range(4):
            assert pipeline.step(None).removed == ()
        result = pipeline.step(None)  # 第 5 帧达标 → 销毁较年轻的 id2
        assert result.removed == (2,)
        assert pipeline.manager.get(1) is not None
        assert pipeline.manager.get(2) is None
        assert pipeline.duplicate_removed == 1

    def test_three_overlapping_frames_do_not_merge(self, make_manager, pool):
        pipeline = self._two_overlapping_tracks(
            make_manager, pool, merge_track_frames=5
        )
        result = None
        for _ in range(3):
            result = pipeline.step(None)
        assert result is not None and result.removed == ()
        assert pipeline.manager.get(1) is not None
        assert pipeline.manager.get(2) is not None
        assert pipeline.duplicate_removed == 0

    def test_contained_small_box_is_not_merged(self, make_manager, pool):
        manager = make_manager(pool, min_hits=1)
        pipeline = MOTPipeline(manager, merge_track_iou=0.01, merge_track_frames=1)
        created = pipeline.step(
            None, extra_detections=[Detection((0, 0, 100, 100)), Detection((40, 40, 20, 20))]
        )
        assert created.created == (1, 2)
        result = pipeline.step(None)  # 面积比 0.04 < 0.15 → 不得合并
        assert result.removed == ()
        assert pipeline.manager.get(1) is not None
        assert pipeline.manager.get(2) is not None

    def test_disabled_dedupe_keeps_both_tracks(self, make_manager, pool):
        pipeline = self._two_overlapping_tracks(
            make_manager, pool, dedupe_tracks=False
        )
        result = None
        for _ in range(6):
            result = pipeline.step(None)
        assert result is not None and result.removed == ()
        assert pipeline.manager.get(1) is not None
        assert pipeline.manager.get(2) is not None
        assert pipeline.duplicate_removed == 0


class TestTrackSanityCheck:
    """轨迹二次校验：与轨迹重叠但尺寸突变的检测应被丢弃。"""

    def test_oversized_detection_does_not_replace_the_box(self, make_manager, pool):
        manager = make_manager(pool)
        pipeline = MOTPipeline(manager, max_box_change=1.6)
        pipeline.step(None, extra_detections=[Detection(BOX)])  # 20x20 轨迹
        result = pipeline.step(None, extra_detections=[Detection((10, 10, 40, 10))])
        assert result.created == ()
        assert result.detections == ()
        assert result.matched == ()
        assert pipeline.size_rejects == 1
        assert manager.get(1).bbox == BOX  # 继续使用跟踪器自身的框

    def test_similar_size_detection_still_refreshes_the_track(self, make_manager, pool):
        manager = make_manager(pool)
        pipeline = MOTPipeline(manager, max_box_change=1.6)
        pipeline.step(None, extra_detections=[Detection(BOX)])
        result = pipeline.step(None, extra_detections=[Detection((11, 10, 21, 20))])
        assert result.matched == ((0, 0),)
        assert pipeline.size_rejects == 0

    def test_non_overlapping_new_object_is_not_filtered(self, make_manager, pool):
        pipeline = MOTPipeline(make_manager(pool), max_box_change=1.6)
        pipeline.step(None, extra_detections=[Detection(BOX)])
        result = pipeline.step(None, extra_detections=[Detection((200, 200, 80, 80))])
        assert result.created == (2,)
        assert pipeline.size_rejects == 0

    def test_guard_can_be_disabled(self, make_manager, pool):
        pipeline = MOTPipeline(make_manager(pool), max_box_change=1.0)
        pipeline.step(None, extra_detections=[Detection(BOX)])
        result = pipeline.step(None, extra_detections=[Detection((10, 10, 40, 10))])
        assert pipeline.size_rejects == 0
        assert len(result.detections) == 1

    def test_lost_tracks_are_exempt_from_the_guard(self, make_manager, pool):
        manager = make_manager(pool, max_missing=5)
        pipeline = MOTPipeline(manager, max_box_change=1.6)
        pipeline.step(None, extra_detections=[Detection(BOX)])
        pool.trackers[0].alive = False
        pipeline.step(None)  # 跟踪器丢失 → LOST
        result = pipeline.step(None, extra_detections=[Detection((10, 10, 40, 10))])
        assert pipeline.size_rejects == 0
        assert len(result.detections) == 1
        assert result.created == ()

    def test_invalid_limit_is_rejected(self, make_manager, pool):
        with pytest.raises(ValidationError):
            MOTPipeline(make_manager(pool), max_box_change=0.5)


class TestStaticTrackRemoval:
    """IoU 时间连续性：长时间不动的目标会被销毁。"""

    def test_static_track_is_removed_after_the_window(self, make_manager, pool):
        manager = make_manager(pool)
        pipeline = MOTPipeline(manager, static_frames=3)
        pipeline.step(None, extra_detections=[Detection(BOX)])
        results = [pipeline.step(None, extra_detections=[Detection(BOX)]) for _ in range(4)]
        removed_ids = [track_id for result in results for track_id in result.removed]
        assert 1 in removed_ids
        assert pipeline.static_removed >= 1
        removal = next(result for result in results if 1 in result.removed)
        assert all(track.track_id != 1 for track in removal.tracks)

    def test_moving_track_is_kept(self, make_manager, tracker_pool):
        pool = tracker_pool(dx=8.0)
        manager = make_manager(pool)
        pipeline = MOTPipeline(manager, static_frames=3)
        pipeline.step(None, extra_detections=[Detection(BOX)])
        for _ in range(6):
            pipeline.step(None)
        assert manager.get(1) is not None
        assert pipeline.static_removed == 0

    def test_zero_disables_removal(self, make_manager, pool):
        manager = make_manager(pool)
        pipeline = MOTPipeline(manager, static_frames=0)
        pipeline.step(None, extra_detections=[Detection(BOX)])
        for _ in range(6):
            pipeline.step(None, extra_detections=[Detection(BOX)])
        assert manager.get(1) is not None
        assert pipeline.static_removed == 0

    def test_lost_tracks_do_not_accumulate(self, make_manager, pool):
        manager = make_manager(pool, max_missing=10)
        pipeline = MOTPipeline(manager, static_frames=3)
        pipeline.step(None, extra_detections=[Detection(BOX)])
        pool.trackers[0].alive = False
        for _ in range(5):
            pipeline.step(None)  # LOST：不参与静止计数
        assert manager.get(1) is not None
        assert pipeline.static_removed == 0

    def test_reset_clears_static_state(self, make_manager, pool):
        pipeline = MOTPipeline(make_manager(pool), static_frames=5)
        pipeline.step(None, extra_detections=[Detection(BOX)])
        pipeline.step(None, extra_detections=[Detection(BOX)])
        assert pipeline._static_hits
        pipeline.reset()
        assert pipeline._static_hits == {}
        assert pipeline.static_removed == 0

    def test_invalid_parameters_are_rejected(self, make_manager, pool):
        with pytest.raises(ValidationError):
            MOTPipeline(make_manager(pool), static_frames=-1)
        with pytest.raises(ValidationError):
            MOTPipeline(make_manager(pool), static_iou=0.0)


class TestLifecycleFiltering:
    def test_tentative_tracks_are_hidden_from_frame_result(self, make_manager, pool):
        manager = make_manager(pool, min_hits=3)
        pipeline = MOTPipeline(manager)
        first = pipeline.step(None, extra_detections=[Detection(BOX)])
        assert first.created == (1,)
        assert first.tracks == ()  # 未确认 → 不输出
        pipeline.step(None, extra_detections=[Detection(BOX)])
        assert pipeline.step(None, extra_detections=[Detection(BOX)]).tracks

    def test_tentative_track_disappears_after_a_miss(self, make_manager, pool):
        manager = make_manager(pool, min_hits=3)
        pipeline = MOTPipeline(manager)
        pipeline.step(None, extra_detections=[Detection(BOX)])
        pool.trackers[0].alive = False
        result = pipeline.step(None)
        assert result.tracks == ()
        assert result.removed == ()
        assert pipeline.stats()["tentative_dropped"] == 1


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
