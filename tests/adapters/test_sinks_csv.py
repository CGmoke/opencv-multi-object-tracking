"""适配层 CSV 轨迹落盘（步骤 6a 验收测试）。

移植自基线 v0.2.0 的 ``tests/test_trajectory.py``：``TrackPoint`` 保留在领域层
（``domain/records``），行格式化、标签清洗与写入器迁到 ``adapters/sinks/csv``。
"""

from __future__ import annotations

import csv

import pytest

from mot.adapters.sinks.csv import (
    CSV_HEADER,
    CsvTrajectoryWriter,
    sanitize_label,
    write_points_atomic,
)
from mot.domain.errors import SinkError
from mot.domain.models import TrackedObject
from mot.domain.records import TrackPoint


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

    def test_row_format_matches_the_contract(self, tmp_path):
        path = tmp_path / "t.csv"
        with CsvTrajectoryWriter(path) as writer:
            writer.write_frame(7, 1.5, [make_track(3, "car")])
        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.reader(handle))
        row = rows[1]
        assert row[0] == "3"
        assert row[1] == "7"
        assert row[2] == "1.500000"
        assert row[3:7] == ["10.000", "20.000", "30.000", "40.000"]
        assert row[7:9] == ["25.000", "40.000"]
        assert row[9] == "car"

    def test_label_is_escaped_in_the_row(self, tmp_path):
        path = tmp_path / "t.csv"
        with CsvTrajectoryWriter(path) as writer:
            writer.write_frame(1, 0.0, [make_track(1, "=cmd")])
        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.reader(handle))
        assert rows[1][-1] == "'=cmd"

    def test_append_keeps_existing_rows(self, tmp_path):
        path = tmp_path / "t.csv"
        with CsvTrajectoryWriter(path) as writer:
            writer.write_frame(1, 0.0, [make_track(1)])
        with CsvTrajectoryWriter(path, append=True) as writer:
            writer.write_frame(2, 0.1, [make_track(1)])
        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.reader(handle))
        assert len(rows) == 3  # header + 2 data rows

    def test_symlink_target_is_refused(self, tmp_path):
        real = tmp_path / "real.csv"
        real.write_text("", encoding="utf-8")
        link = tmp_path / "link.csv"
        try:
            link.symlink_to(real)
        except OSError:
            pytest.skip("symlinks are not supported here")
        with pytest.raises(SinkError, match="symlink"):
            CsvTrajectoryWriter(link)

    def test_write_points_atomic(self, tmp_path):
        path = tmp_path / "atomic.csv"
        points = [TrackPoint(track_id=1, frame=1, timestamp=0.0, bbox=(0, 0, 10, 10), label="a")]
        result = write_points_atomic(path, points)
        assert result == path
        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.reader(handle))
        assert tuple(rows[0]) == CSV_HEADER
        assert len(rows) == 2
