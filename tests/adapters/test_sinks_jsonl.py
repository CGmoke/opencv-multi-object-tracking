"""适配层：事件 JSONL 落盘（P0-B 验收测试）。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mot.adapters.sinks.jsonl import JsonlEventSink
from mot.application.pipeline import FrameResult
from mot.domain.errors import SinkError
from mot.domain.models import TrackedObject, TrackState


def track(track_id: int = 1, state: TrackState = TrackState.ACTIVE, label: str = "object"):
    return TrackedObject(
        track_id=track_id,
        tracker=None,
        bbox=(10.0, 20.0, 30.0, 40.0),
        label=label,
        state=state,
    )


def result(frame_index=1, timestamp=0.5, tracks=(), created=(), removed=()):
    return FrameResult(
        frame_index=frame_index,
        timestamp=timestamp,
        tracks=tracks,
        created=created,
        removed=removed,
    )


def read_lines(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_writes_one_record_per_frame(tmp_path):
    path = tmp_path / "events.jsonl"
    with JsonlEventSink(path) as sink:
        sink.on_frame(result(frame_index=1, tracks=(track(1),), created=(1,)))
        sink.on_frame(result(frame_index=2, tracks=(track(1),)))
        assert sink.rows_written == 2
    records = read_lines(path)
    assert records[0]["frame"] == 1
    assert records[0]["created"] == [1]
    assert records[0]["tracks"][0]["id"] == 1
    assert records[0]["tracks"][0]["bbox"] == [10.0, 20.0, 30.0, 40.0]


def test_skips_empty_frames(tmp_path):
    path = tmp_path / "events.jsonl"
    with JsonlEventSink(path) as sink:
        sink.on_frame(result())  # 无目标、无事件 → 跳过
        sink.on_frame(result(created=(2,)))
    assert len(read_lines(path)) == 1


def test_only_active_is_respected(tmp_path):
    path = tmp_path / "events.jsonl"
    with JsonlEventSink(path) as sink:
        sink.on_frame(result(tracks=(track(1), track(2, TrackState.LOST))))
    ids = [item["id"] for item in read_lines(path)[0]["tracks"]]
    assert ids == [1]


def test_append_mode(tmp_path):
    path = tmp_path / "events.jsonl"
    with JsonlEventSink(path) as sink:
        sink.on_frame(result(created=(1,)))
    with JsonlEventSink(path, append=True) as sink:
        sink.on_frame(result(created=(2,)))
    assert len(read_lines(path)) == 2


def test_symlink_is_refused(tmp_path):
    real = tmp_path / "real.jsonl"
    real.write_text("", encoding="utf-8")
    link = tmp_path / "link.jsonl"
    try:
        link.symlink_to(real)
    except OSError:
        pytest.skip("symlinks are not supported here")
    with pytest.raises(SinkError, match="symlink"):
        JsonlEventSink(link)


def test_closed_sink_rejects_writes(tmp_path):
    sink = JsonlEventSink(tmp_path / "e.jsonl")
    sink.close()
    with pytest.raises(SinkError, match="closed"):
        sink.on_frame(result(created=(1,)))
