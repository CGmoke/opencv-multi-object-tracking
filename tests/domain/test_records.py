"""领域层纯数据记录（步骤 2 验收测试）。

移植自基线 v0.2.0 的 ``tests/test_trajectory.py`` 中属于领域层的部分
（``TrackPoint`` 与 ``TrajectoryStore``）。CSV 写入器 / 标签清洗 / 原子写等
依赖 ``csv`` / ``os`` 的部分将在 ``adapters/sinks/csv.py`` 落地时另行测试。
"""

from __future__ import annotations

import dataclasses

import pytest

from mot.domain.errors import ValidationError
from mot.domain.records import TrackPoint, TrajectoryStore


class TestTrackPoint:
    def test_defaults(self):
        point = TrackPoint(track_id=1, frame=1, timestamp=0.0, bbox=(0, 0, 10, 10))
        assert point.label == "object"

    def test_center_property(self):
        point = TrackPoint(track_id=1, frame=1, timestamp=0.0, bbox=(0, 0, 10, 10))
        assert point.center == (5.0, 5.0)

    def test_is_frozen(self):
        point = TrackPoint(track_id=1, frame=1, timestamp=0.0, bbox=(0, 0, 10, 10))
        with pytest.raises(dataclasses.FrozenInstanceError):
            point.frame = 2  # type: ignore[misc]


class TestTrajectoryStore:
    def test_add_and_read_back(self):
        store = TrajectoryStore()
        store.add(1, 1, 0.0, (0, 0, 10, 10))
        store.add(1, 2, 0.1, (5, 0, 10, 10))
        store.add(2, 2, 0.1, (50, 50, 10, 10))
        assert len(store.points(1)) == 2
        assert len(store.points(2)) == 1
        assert len(store) == 3
        assert 1 in store and 3 not in store

    def test_added_point_is_validated(self):
        store = TrajectoryStore()
        point = store.add(1, 1, 0.0, (0, 0, 10, 10))
        assert point.bbox == (0.0, 0.0, 10.0, 10.0)
        with pytest.raises(ValidationError):
            store.add(1, 2, 0.1, (0, 0, 0, 10))

    def test_points_are_bounded_per_track(self):
        store = TrajectoryStore(max_points_per_track=2)
        for frame in range(1, 5):
            store.add(1, frame, frame * 0.1, (0, 0, 10, 10))
        points = store.points(1)
        assert [point.frame for point in points] == [3, 4]

    def test_default_bound_matches_limits(self):
        from mot.domain.limits import MAX_TRAIL_LENGTH

        assert TrajectoryStore()._max_points == MAX_TRAIL_LENGTH

    def test_items_preserve_insertion_order(self):
        store = TrajectoryStore()
        store.add(2, 1, 0.0, (0, 0, 10, 10))
        store.add(1, 1, 0.0, (0, 0, 10, 10))
        assert [track_id for track_id, _ in store.items()] == [2, 1]

    def test_invalid_timestamp_is_rejected(self):
        store = TrajectoryStore()
        with pytest.raises(ValidationError):
            store.add(1, 1, -1.0, (0, 0, 10, 10))
        with pytest.raises(ValidationError):
            store.add(1, 1, float("nan"), (0, 0, 10, 10))

    def test_invalid_max_points_is_rejected(self):
        with pytest.raises(ValidationError):
            TrajectoryStore(max_points_per_track=0)

    def test_clear(self):
        store = TrajectoryStore()
        store.add(1, 1, 0.0, (0, 0, 10, 10))
        store.clear()
        assert len(store) == 0
