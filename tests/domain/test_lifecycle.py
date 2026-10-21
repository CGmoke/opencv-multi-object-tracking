"""领域层 ID 生命周期（步骤 4 验收测试）。

移植自基线 v0.2.0 的 ``tests/test_track_manager.py``：跟踪器改由 ``init_fn`` /
``update_fn`` 注入（夹具见 ``conftest.py``），并补充 G3（重锚失败降级）用例。
"""

from __future__ import annotations

import pytest

from mot.domain.errors import ValidationError
from mot.domain.lifecycle import TrackManager, detections_from_boxes
from mot.domain.models import Detection, IdAllocator, TrackState

BOX = (10.0, 10.0, 20.0, 20.0)


class TestSpawning:
    def test_new_detection_creates_an_id(self, manager):
        result = manager.step(None, [Detection(BOX)])
        assert result.created == (1,)
        assert len(manager) == 1
        track = manager.get(1)
        assert track is not None
        assert track.bbox == BOX
        assert track.hits == 1
        assert track.created_frame == 1
        assert list(track.trail) == [(20.0, 20.0)]

    def test_tracker_is_advanced_before_association(self, manager):
        manager.step(None, [Detection(BOX)])
        result = manager.step(None)  # dx=0，框不动
        assert result.tracks[0].bbox == BOX
        assert result.tracks[0].hits == 2

    def test_moving_tracker_updates_bbox_and_trail(self, make_manager, tracker_pool):
        pool = tracker_pool(dx=5.0)
        manager = make_manager(pool)
        manager.step(None, [Detection(BOX)])
        manager.step(None)
        track = manager.get(1)
        assert track.bbox == (15.0, 10.0, 20.0, 20.0)
        assert list(track.trail) == [(20.0, 20.0), (25.0, 20.0)]

    def test_ids_are_unique_per_target(self, manager):
        manager.step(None, [Detection(BOX), Detection((100, 100, 20, 20))])
        assert [track.track_id for track in manager.tracks] == [1, 2]
        assert manager.total_created == 2


class TestAssociation:
    def test_overlapping_detection_refreshes_the_id(self, manager):
        manager.step(None, [Detection(BOX)])
        result = manager.step(None, [Detection(BOX)])
        assert result.created == ()
        assert result.matched == ((0, 0),)
        assert len(manager) == 1
        assert manager.get(1).hits == 2

    def test_non_overlapping_detection_creates_a_new_id(self, manager):
        manager.step(None, [Detection(BOX)])
        result = manager.step(None, [Detection((200, 200, 20, 20))])
        assert result.created == (2,)
        assert len(manager) == 2

    def test_two_detections_match_two_tracks(self, manager):
        manager.step(None, [Detection(BOX), Detection((100, 100, 20, 20))])
        result = manager.step(None, [Detection((12, 10, 20, 20)), Detection((101, 100, 20, 20))])
        assert len(result.matched) == 2
        assert result.created == ()
        assert manager.get(1).bbox == (12.0, 10.0, 20.0, 20.0)
        assert manager.get(2).bbox == (101.0, 100.0, 20.0, 20.0)


class TestBoxSmoothing:
    def test_default_uses_the_detection_box_verbatim(self, make_manager, pool):
        manager = make_manager(pool)
        manager.step(None, [Detection(BOX)])
        manager.step(None, [Detection((12, 10, 20, 20))])
        assert manager.get(1).bbox == (12.0, 10.0, 20.0, 20.0)

    def test_ema_blends_detection_into_the_existing_box(self, make_manager, pool):
        manager = make_manager(pool, box_smoothing=0.5)
        manager.step(None, [Detection(BOX)])
        manager.step(None, [Detection((20, 10, 20, 20))])
        assert manager.get(1).bbox == (15.0, 10.0, 20.0, 20.0)

    def test_lost_track_snaps_to_the_detection_instead_of_drifting(self, make_manager, pool):
        manager = make_manager(pool, box_smoothing=0.5, max_missing=5)
        manager.step(None, [Detection(BOX)])
        pool.trackers[0].alive = False
        manager.step(None)  # 跟踪器丢失 → LOST
        manager.step(None, [Detection((20, 10, 20, 20))])
        assert manager.get(1).bbox == (20.0, 10.0, 20.0, 20.0)


class TestLifecycleFiltering:
    def test_tentative_track_dies_on_its_first_miss(self, make_manager, pool):
        manager = make_manager(pool, min_hits=3)
        manager.step(None, [Detection(BOX)])
        pool.trackers[0].alive = False
        result = manager.step(None)
        assert manager.get(1) is None  # 未确认 → 不保留 max_missing 宽限
        assert result.removed == ()  # 不产生移除事件
        assert manager.tentative_dropped == 1

    def test_confirmed_track_keeps_its_grace_period(self, make_manager, pool):
        manager = make_manager(pool, min_hits=2, max_missing=3)
        manager.step(None, [Detection(BOX)])
        manager.step(None)  # hits=2 → 已确认
        pool.trackers[0].alive = False
        manager.step(None)
        assert manager.get(1) is not None
        assert manager.tentative_dropped == 0

    def test_confirmed_view_excludes_tentative_tracks(self, make_manager, pool):
        manager = make_manager(pool, min_hits=3)
        manager.step(None, [Detection(BOX)])
        assert manager.confirmed_tracks == ()
        manager.step(None)
        manager.step(None)
        assert [track.track_id for track in manager.confirmed_tracks] == [1]

    def test_min_hits_one_confirms_immediately(self, manager):
        manager.step(None, [Detection(BOX)])
        assert [track.track_id for track in manager.confirmed_tracks] == [1]


class TestBoxSmoothingWindow:
    def test_moving_average_blends_the_last_frames(self, make_manager, tracker_pool):
        pool = tracker_pool(dx=6.0)
        manager = make_manager(pool, smooth_window=3)
        manager.step(None, [Detection(BOX)])  # 原始 (10,10,20,20)
        manager.step(None)  # 原始 16 → 均值 (10+16)/2 = 13
        assert manager.get(1).bbox == (13.0, 10.0, 20.0, 20.0)
        manager.step(None)  # 原始 22 → 均值 (10+16+22)/3 = 16
        assert manager.get(1).bbox == (16.0, 10.0, 20.0, 20.0)
        manager.step(None)  # 原始 28 → 均值 (16+22+28)/3 = 22
        assert manager.get(1).bbox == (22.0, 10.0, 20.0, 20.0)

    def test_trail_centres_follow_the_smoothed_box(self, make_manager, tracker_pool):
        pool = tracker_pool(dx=6.0)
        manager = make_manager(pool, smooth_window=3)
        manager.step(None, [Detection(BOX)])
        manager.step(None)
        manager.step(None)
        assert list(manager.get(1).trail) == [(20.0, 20.0), (23.0, 20.0), (26.0, 20.0)]

    def test_detection_refresh_is_also_smoothed(self, make_manager, pool):
        manager = make_manager(pool, smooth_window=3)
        manager.step(None, [Detection(BOX)])
        manager.step(None, [Detection((20, 10, 20, 20))])
        # 窗口内是 10(初始)、10(跟踪器)、20(检测) → 均值 13.33
        assert manager.get(1).bbox[0] == pytest.approx(40 / 3)
        assert manager.get(1).bbox[1:] == (10.0, 20.0, 20.0)

    def test_lost_track_snaps_instead_of_blending_stale_boxes(self, make_manager, pool):
        manager = make_manager(pool, smooth_window=5, max_missing=5)
        manager.step(None, [Detection(BOX)])
        manager.step(None)
        pool.trackers[0].alive = False
        manager.step(None)  # LOST
        manager.step(None, [Detection((20, 10, 20, 20))])
        # 若与历史混合会得到 12；吸附应直接采用新框
        assert manager.get(1).bbox == (20.0, 10.0, 20.0, 20.0)


class TestLossAndExpiry:
    def test_track_keeps_its_id_while_missing(self, make_manager, pool):
        manager = make_manager(pool, max_missing=3)
        manager.step(None, [Detection(BOX)])
        pool.trackers[0].alive = False
        result = manager.step(None)
        assert result.removed == ()
        assert len(manager) == 1
        track = manager.get(1)
        assert track.state is TrackState.LOST
        assert track.misses == 1

    def test_id_is_destroyed_after_max_missing_frames(self, make_manager, pool):
        manager = make_manager(pool, max_missing=2)
        manager.step(None, [Detection(BOX)])
        pool.trackers[0].alive = False
        assert manager.step(None).removed == ()
        assert manager.step(None).removed == ()
        result = manager.step(None)  # 第三次 miss > max_missing
        assert result.removed == (1,)
        assert len(manager) == 0
        assert manager.total_removed == 1

    def test_max_missing_zero_removes_on_first_failure(self, make_manager, pool):
        manager = make_manager(pool, max_missing=0)
        manager.step(None, [Detection(BOX)])
        pool.trackers[0].alive = False
        assert manager.step(None).removed == (1,)

    def test_track_recovers_when_the_tracker_returns(self, make_manager, pool):
        manager = make_manager(pool, max_missing=5)
        manager.step(None, [Detection(BOX)])
        pool.trackers[0].alive = False
        manager.step(None)
        pool.trackers[0].alive = True
        result = manager.step(None)
        track = manager.get(1)
        assert track.state is TrackState.ACTIVE
        assert track.misses == 0
        assert result.removed == ()

    def test_new_target_after_expiry_gets_a_new_id(self, make_manager, pool):
        manager = make_manager(pool, max_missing=0)
        manager.step(None, [Detection(BOX)])
        pool.trackers[0].alive = False
        manager.step(None)
        result = manager.step(None, [Detection(BOX)])
        assert result.created == (2,)
        assert manager.get(1) is None


class TestManualBoxes:
    def test_overlapping_manual_box_reuses_the_id(self, manager):
        first = manager.add_or_refresh(None, Detection(BOX))
        second = manager.add_or_refresh(None, Detection((12, 10, 20, 20)))
        assert first is not None and second is not None
        assert second.track_id == first.track_id
        assert len(manager) == 1
        assert second.hits == 2

    def test_disjoint_manual_box_creates_a_new_id(self, manager):
        manager.add_or_refresh(None, Detection(BOX))
        added = manager.add_or_refresh(None, Detection((200, 200, 20, 20)))
        assert added is not None
        assert added.track_id == 2

    def test_empty_frame_index_is_used_for_manual_boxes(self, manager):
        track = manager.add_or_refresh(None, Detection(BOX))
        assert track is not None
        assert track.created_frame == 0


class TestRefreshFailureIsNonFatal:
    """G3：关联命中后重锚失败必须降级为一次 miss，异常不得穿出 ``step``。"""

    def _flaky_manager(self, pool, fail_from: int):
        calls = {"n": 0}

        def flaky_init(tracker, frame, bbox) -> None:
            calls["n"] += 1
            if calls["n"] >= fail_from:
                raise RuntimeError("re-init boom")
            tracker.init(frame, bbox)

        def steady_update(tracker, frame):
            # 始终报回一个与 BOX 重叠 0.6 的框，确保下一帧关联命中。
            return (15.0, 10.0, 20.0, 20.0)

        return TrackManager(pool, init_fn=flaky_init, update_fn=steady_update)

    def test_refresh_failure_degrades_to_miss_and_lost(self, pool):
        manager = self._flaky_manager(pool, fail_from=2)  # 第一次 spawn 成功，之后失败
        manager.step(None, [Detection(BOX)])
        result = manager.step(None, [Detection(BOX)])  # 关联命中 → 重锚抛错
        track = manager.get(1)
        assert track is not None  # id 未因一次重锚失败而被销毁
        assert track.state is TrackState.LOST
        assert track.misses >= 1
        assert result.removed == ()  # 异常未穿出 step
        assert result.matched == ((0, 0),)  # 关联本身仍成立（下标对语义不变）

    def test_refresh_failure_in_add_or_refresh_does_not_raise(self, pool):
        manager = self._flaky_manager(pool, fail_from=2)
        manager.add_or_refresh(None, Detection(BOX))
        track = manager.add_or_refresh(None, Detection((12, 10, 20, 20)))
        assert track is not None
        assert track.track_id == 1
        assert track.state is TrackState.LOST


class TestFailureHandling:
    def test_spawn_failure_is_counted_and_does_not_raise(self, make_manager):
        def broken_factory():
            raise RuntimeError("tracker init exploded")

        manager = make_manager(broken_factory)
        track = manager.add_or_refresh(None, Detection(BOX))
        assert track is None
        assert manager.spawn_failures == 1
        assert len(manager) == 0

    def test_spawn_failure_in_step(self, make_manager):
        def broken_factory():
            raise RuntimeError("nope")

        manager = make_manager(broken_factory)
        result = manager.step(None, [Detection(BOX)])
        assert result.spawn_failures == 1
        assert result.created == ()
        assert manager.spawn_failures == 1


class TestBookkeeping:
    def test_remove_destroys_one_id(self, manager):
        manager.step(None, [Detection(BOX), Detection((100, 100, 20, 20))])
        assert manager.remove(1) is True
        assert manager.remove(1) is False
        assert len(manager) == 1

    def test_clear_trails_keeps_identities(self, manager):
        manager.step(None, [Detection(BOX)])
        manager.clear_trails()
        assert list(manager.get(1).trail) == []
        assert len(manager) == 1

    def test_tracks_are_sorted_by_id(self, manager):
        manager.step(None, [Detection(BOX), Detection((100, 100, 20, 20))])
        assert [track.track_id for track in manager.tracks] == [1, 2]

    def test_active_tracks_excludes_lost(self, make_manager, pool):
        manager = make_manager(pool, max_missing=5)
        manager.step(None, [Detection(BOX)])
        pool.trackers[0].alive = False
        manager.step(None)
        assert manager.active_tracks == ()
        assert len(manager.tracks) == 1

    def test_reset_clears_state_and_restarts_numbering(self, manager):
        manager.step(None, [Detection(BOX), Detection((100, 100, 20, 20))])
        manager.reset(start_id=7)
        assert len(manager) == 0
        assert manager.frame_index == 0
        assert manager.total_created == 0
        result = manager.step(None, [Detection(BOX)])
        assert result.created == (7,)

    def test_default_reset_keeps_numbering(self, manager):
        manager.step(None, [Detection(BOX)])
        manager.reset()
        result = manager.step(None, [Detection(BOX)])
        assert result.created == (2,)

    def test_shared_allocator_is_respected(self, tracker_pool, make_manager):
        pool = tracker_pool()
        manager = make_manager(pool, id_allocator=IdAllocator(start=42))
        manager.step(None, [Detection(BOX)])
        assert manager.get(42) is not None


class TestValidation:
    @pytest.mark.parametrize(
        "kwargs",
        [
            {"iou_threshold": -0.1},
            {"iou_threshold": 1.1},
            {"max_missing": -1},
            {"trail_length": 0},
            {"match_method": "nope"},
            {"box_smoothing": 0.0},
            {"box_smoothing": 1.5},
        ],
    )
    def test_invalid_arguments_raise(self, make_manager, pool, kwargs):
        with pytest.raises(ValidationError):
            make_manager(pool, **kwargs)

    def test_hungarian_method_is_accepted(self, make_manager, pool):
        manager = make_manager(pool, match_method="hungarian")
        result = manager.step(None, [Detection(BOX)])
        assert result.created == (1,)

    def test_missing_injected_fns_are_required(self, pool):
        with pytest.raises(TypeError):
            TrackManager(pool)  # type: ignore[call-arg]


def test_detections_from_boxes_normalises():
    detections = detections_from_boxes([(10, 10, -5, -5)])
    assert detections[0].bbox == (5.0, 5.0, 5.0, 5.0)
