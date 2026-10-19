"""Identity lifecycle: spawning, association, losing and expiring ids."""

from __future__ import annotations

import pytest

from conftest import TrackerPool
from mot import Detection, IdAllocator, TrackManager, TrackState
from mot.track_manager import detections_from_boxes

BOX = (10.0, 10.0, 20.0, 20.0)


def make_manager(pool: TrackerPool, **kwargs) -> TrackManager:
    return TrackManager(pool, **kwargs)


class TestSpawning:
    def test_new_detection_creates_an_id(self, pool):
        manager = make_manager(pool)
        result = manager.step(None, [Detection(BOX)])
        assert result.created == (1,)
        assert len(manager) == 1
        track = manager.get(1)
        assert track is not None
        assert track.bbox == BOX
        assert track.hits == 1
        assert track.created_frame == 1
        assert list(track.trail) == [(20.0, 20.0)]

    def test_tracker_is_advanced_before_association(self, pool):
        manager = make_manager(pool)
        manager.step(None, [Detection(BOX)])
        result = manager.step(None)  # dx=0, so the box stays put
        assert result.tracks[0].bbox == BOX
        assert result.tracks[0].hits == 2

    def test_moving_tracker_updates_bbox_and_trail(self):
        pool = TrackerPool(dx=5.0)
        manager = make_manager(pool)
        manager.step(None, [Detection(BOX)])
        manager.step(None)
        track = manager.get(1)
        assert track.bbox == (15.0, 10.0, 20.0, 20.0)
        assert list(track.trail) == [(20.0, 20.0), (25.0, 20.0)]

    def test_ids_are_unique_per_target(self, pool):
        manager = make_manager(pool)
        manager.step(None, [Detection(BOX), Detection((100, 100, 20, 20))])
        assert [track.track_id for track in manager.tracks] == [1, 2]
        assert manager.total_created == 2


class TestAssociation:
    def test_overlapping_detection_refreshes_the_id(self, pool):
        manager = make_manager(pool)
        manager.step(None, [Detection(BOX)])
        result = manager.step(None, [Detection(BOX)])
        assert result.created == ()
        assert result.matched == ((0, 0),)
        assert len(manager) == 1
        assert manager.get(1).hits == 2

    def test_non_overlapping_detection_creates_a_new_id(self, pool):
        manager = make_manager(pool)
        manager.step(None, [Detection(BOX)])
        result = manager.step(None, [Detection((200, 200, 20, 20))])
        assert result.created == (2,)
        assert len(manager) == 2

    def test_two_detections_match_two_tracks(self, pool):
        manager = make_manager(pool)
        manager.step(None, [Detection(BOX), Detection((100, 100, 20, 20))])
        result = manager.step(
            None, [Detection((12, 10, 20, 20)), Detection((101, 100, 20, 20))]
        )
        assert len(result.matched) == 2
        assert result.created == ()
        assert manager.get(1).bbox == (12.0, 10.0, 20.0, 20.0)
        assert manager.get(2).bbox == (101.0, 100.0, 20.0, 20.0)


class TestLossAndExpiry:
    def test_track_keeps_its_id_while_missing(self, pool):
        manager = make_manager(pool, max_missing=3)
        manager.step(None, [Detection(BOX)])
        pool.trackers[0].alive = False
        result = manager.step(None)
        assert result.removed == ()
        assert len(manager) == 1
        track = manager.get(1)
        assert track.state is TrackState.LOST
        assert track.misses == 1

    def test_id_is_destroyed_after_max_missing_frames(self, pool):
        manager = make_manager(pool, max_missing=2)
        manager.step(None, [Detection(BOX)])
        pool.trackers[0].alive = False
        assert manager.step(None).removed == ()
        assert manager.step(None).removed == ()
        result = manager.step(None)  # third miss > max_missing
        assert result.removed == (1,)
        assert len(manager) == 0
        assert manager.total_removed == 1

    def test_max_missing_zero_removes_on_first_failure(self, pool):
        manager = make_manager(pool, max_missing=0)
        manager.step(None, [Detection(BOX)])
        pool.trackers[0].alive = False
        assert manager.step(None).removed == (1,)

    def test_track_recovers_when_the_tracker_returns(self, pool):
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

    def test_new_target_after_expiry_gets_a_new_id(self, pool):
        manager = make_manager(pool, max_missing=0)
        manager.step(None, [Detection(BOX)])
        pool.trackers[0].alive = False
        manager.step(None)
        result = manager.step(None, [Detection(BOX)])
        assert result.created == (2,)
        assert manager.get(1) is None


class TestManualBoxes:
    def test_overlapping_manual_box_reuses_the_id(self, pool):
        manager = make_manager(pool)
        first = manager.add_or_refresh(None, Detection(BOX))
        second = manager.add_or_refresh(None, Detection((12, 10, 20, 20)))
        assert first is not None and second is not None
        assert second.track_id == first.track_id
        assert len(manager) == 1
        assert second.hits == 2

    def test_disjoint_manual_box_creates_a_new_id(self, pool):
        manager = make_manager(pool)
        manager.add_or_refresh(None, Detection(BOX))
        added = manager.add_or_refresh(None, Detection((200, 200, 20, 20)))
        assert added is not None
        assert added.track_id == 2

    def test_empty_frame_index_is_used_for_manual_boxes(self, pool):
        manager = make_manager(pool)
        track = manager.add_or_refresh(None, Detection(BOX))
        assert track is not None
        assert track.created_frame == 0


class TestFailureHandling:
    def test_spawn_failure_is_counted_and_does_not_raise(self):
        def broken_factory():
            raise RuntimeError("tracker init exploded")

        manager = TrackManager(broken_factory)
        track = manager.add_or_refresh(None, Detection(BOX))
        assert track is None
        assert manager.spawn_failures == 1
        assert len(manager) == 0

    def test_spawn_failure_in_step(self):
        def broken_factory():
            raise RuntimeError("nope")

        manager = TrackManager(broken_factory)
        result = manager.step(None, [Detection(BOX)])
        assert result.spawn_failures == 1
        assert result.created == ()
        assert manager.spawn_failures == 1


class TestBookkeeping:
    def test_remove_destroys_one_id(self, pool):
        manager = make_manager(pool)
        manager.step(None, [Detection(BOX), Detection((100, 100, 20, 20))])
        assert manager.remove(1) is True
        assert manager.remove(1) is False
        assert len(manager) == 1

    def test_clear_trails_keeps_identities(self, pool):
        manager = make_manager(pool)
        manager.step(None, [Detection(BOX)])
        manager.clear_trails()
        assert list(manager.get(1).trail) == []
        assert len(manager) == 1

    def test_tracks_are_sorted_by_id(self, pool):
        manager = make_manager(pool)
        manager.step(None, [Detection(BOX), Detection((100, 100, 20, 20))])
        assert [track.track_id for track in manager.tracks] == [1, 2]

    def test_active_tracks_excludes_lost(self, pool):
        manager = make_manager(pool, max_missing=5)
        manager.step(None, [Detection(BOX)])
        pool.trackers[0].alive = False
        manager.step(None)
        assert manager.active_tracks == ()
        assert len(manager.tracks) == 1

    def test_reset_clears_state_and_restarts_numbering(self, pool):
        manager = make_manager(pool)
        manager.step(None, [Detection(BOX), Detection((100, 100, 20, 20))])
        manager.reset(start_id=7)
        assert len(manager) == 0
        assert manager.frame_index == 0
        assert manager.total_created == 0
        result = manager.step(None, [Detection(BOX)])
        assert result.created == (7,)

    def test_default_reset_keeps_numbering(self, pool):
        manager = make_manager(pool)
        manager.step(None, [Detection(BOX)])
        manager.reset()
        result = manager.step(None, [Detection(BOX)])
        assert result.created == (2,)

    def test_shared_allocator_is_respected(self):
        pool = TrackerPool()
        manager = TrackManager(pool, id_allocator=IdAllocator(start=42))
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
        ],
    )
    def test_invalid_arguments_raise(self, pool, kwargs):
        with pytest.raises(ValueError):
            TrackManager(pool, **kwargs)

    def test_hungarian_method_is_accepted(self, pool):
        manager = TrackManager(pool, match_method="hungarian")
        result = manager.step(None, [Detection(BOX)])
        assert result.created == (1,)


def test_detections_from_boxes_normalises():
    detections = detections_from_boxes([(10, 10, -5, -5)])
    assert detections[0].bbox == (5.0, 5.0, 5.0, 5.0)
