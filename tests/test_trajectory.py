"""Trail storage and csv export."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from mot.models import TrackedObject
from mot.trajectory import (
    CSV_HEADER,
    CsvTrajectoryWriter,
    TrackPoint,
    TrajectoryStore,
    sanitize_label,
    write_points_atomic,
)


class TestSanitizeLabel:
    @pytest.mark.parametrize("label", ["=SUM(A1)", "+1", "-1", "@cmd", "\t=cmd"])
    def test_formula_prefixes_are_escaped(self, label):
        assert sanitize_label(label).startswith("'")

    def test_newlines_are_replaced(self):
        assert sanitize_label("a\nb\rc") == "a b c"

    def test_empty_label_falls_back(self):
        assert sanitize_label("   ") == "object"

    def test_label_is_truncated(self):
        assert len(sanitize_label("x" * 200)) == 64

    def test_plain_label_is_kept(self):
        assert sanitize_label("person") == "person"


class TestTrackPoint:
    def test_as_row_formats_floats(self):
        point = TrackPoint(
            track_id=3, frame=7, timestamp=1.5, bbox=(10.0, 20.0, 30.0, 40.0), label="car"
        )
        row = point.as_row()
        assert row[0] == 3
        assert row[1] == 7
        assert row[2] == "1.500000"
        assert row[3:7] == ("10.000", "20.000", "30.000", "40.000")
        assert row[7:9] == ("25.000", "40.000")
        assert row[9] == "car"

    def test_center_property(self):
        point = TrackPoint(track_id=1, frame=1, timestamp=0.0, bbox=(0, 0, 10, 10))
        assert point.center == (5.0, 5.0)

    def test_label_is_escaped_in_the_row(self):
        point = TrackPoint(track_id=1, frame=1, timestamp=0.0, bbox=(0, 0, 10, 10), label="=cmd")
        assert point.as_row()[-1] == "'=cmd"


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

    def test_points_are_bounded_per_track(self):
        store = TrajectoryStore(max_points_per_track=2)
        for frame in range(1, 5):
            store.add(1, frame, frame * 0.1, (0, 0, 10, 10))
        points = store.points(1)
        assert [point.frame for point in points] == [3, 4]

    def test_items_preserve_insertion_order(self):
        store = TrajectoryStore()
        store.add(2, 1, 0.0, (0, 0, 10, 10))
        store.add(1, 1, 0.0, (0, 0, 10, 10))
        assert [track_id for track_id, _ in store.items()] == [2, 1]

    def test_invalid_timestamp_is_rejected(self):
        store = TrajectoryStore()
        with pytest.raises(ValueError):
            store.add(1, 1, -1.0, (0, 0, 10, 10))
        with pytest.raises(ValueError):
            store.add(1, 1, float("nan"), (0, 0, 10, 10))

    def test_invalid_max_points_is_rejected(self):
        with pytest.raises(ValueError):
            TrajectoryStore(max_points_per_track=0)

    def test_clear(self):
        store = TrajectoryStore()
        store.add(1, 1, 0.0, (0, 0, 10, 10))
        store.clear()
        assert len(store) == 0


def make_track(track_id: int = 1, label: str = "object") -> TrackedObject:
    return TrackedObject(
        track_id=track_id, tracker=None, bbox=(10.0, 20.0, 30.0, 40.0), label=label
    )


class TestCsvTrajectoryWriter:
    def test_writes_header_and_rows(self, tmp_path):
        path = tmp_path / "nested" / "trajectories.csv"
        with CsvTrajectoryWriter(path) as writer:
            assert writer.write_frame(1, 0.0, [make_track(1)]) == 1
            assert writer.write_frame(2, 0.5, [make_track(1), make_track(2)]) == 2
            rows_written = writer.rows_written
        assert rows_written == 3

        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.reader(handle))
        assert tuple(rows[0]) == CSV_HEADER
        assert len(rows) == 4
        assert rows[1][0] == "1"
        assert rows[1][1] == "1"
        assert rows[1][3] == "10.000"
        assert rows[1][9] == "object"

    def test_rows_are_flushed_before_close(self, tmp_path):
        path = tmp_path / "traj.csv"
        writer = CsvTrajectoryWriter(path)
        try:
            writer.write_frame(1, 0.0, [make_track()])
            assert path.read_text(encoding="utf-8").count("\n") == 2
        finally:
            writer.close()

    def test_append_does_not_duplicate_the_header(self, tmp_path):
        path = tmp_path / "traj.csv"
        with CsvTrajectoryWriter(path) as writer:
            writer.write_frame(1, 0.0, [make_track(1)])
        with CsvTrajectoryWriter(path, append=True) as writer:
            writer.write_frame(2, 0.1, [make_track(2)])

        lines = path.read_text(encoding="utf-8").splitlines()
        assert sum(1 for line in lines if line.startswith("track_id")) == 1
        assert len(lines) == 3

    def test_overwrite_by_default(self, tmp_path):
        path = tmp_path / "traj.csv"
        with CsvTrajectoryWriter(path) as writer:
            writer.write_frame(1, 0.0, [make_track(1)])
        with CsvTrajectoryWriter(path) as writer:
            writer.write_frame(5, 0.0, [make_track(9)])
        lines = path.read_text(encoding="utf-8").splitlines()
        assert len(lines) == 2
        assert lines[1].startswith("9,")

    def test_write_points(self, tmp_path):
        path = tmp_path / "traj.csv"
        point = TrackPoint(track_id=4, frame=2, timestamp=0.2, bbox=(1, 2, 3, 4))
        with CsvTrajectoryWriter(path) as writer:
            assert writer.write_points([point, point]) == 2
        assert len(path.read_text(encoding="utf-8").splitlines()) == 3

    def test_writing_after_close_raises(self, tmp_path):
        writer = CsvTrajectoryWriter(tmp_path / "traj.csv")
        writer.close()
        with pytest.raises(RuntimeError):
            writer.write_frame(1, 0.0, [make_track()])

    def test_unsafe_label_is_escaped_on_disk(self, tmp_path):
        path = tmp_path / "traj.csv"
        with CsvTrajectoryWriter(path) as writer:
            writer.write_frame(1, 0.0, [make_track(label="=HYPERLINK(...)")])
        assert "'=HYPERLINK" in path.read_text(encoding="utf-8")


class TestAtomicWrite:
    def test_writes_complete_file(self, tmp_path):
        path = write_points_atomic(
            tmp_path / "out" / "traj.csv",
            [TrackPoint(track_id=1, frame=1, timestamp=0.0, bbox=(0, 0, 10, 10))],
        )
        lines = path.read_text(encoding="utf-8").splitlines()
        assert tuple(lines[0].split(",")) == CSV_HEADER
        assert len(lines) == 2
        assert isinstance(path, Path)

    def test_no_temporary_files_are_left_behind(self, tmp_path):
        write_points_atomic(
            tmp_path / "traj.csv",
            [TrackPoint(track_id=1, frame=1, timestamp=0.0, bbox=(0, 0, 10, 10))],
        )
        assert [p.name for p in tmp_path.iterdir()] == ["traj.csv"]
