"""适配层预录视频（先录后决定保存/删除）。"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

cv2 = pytest.importorskip("cv2")

import numpy as np  # noqa: E402

from mot.adapters.recording import PendingVideoRecorder  # noqa: E402
from mot.domain.errors import SinkError  # noqa: E402


def frame(width: int = 64, height: int = 48) -> np.ndarray:
    return np.zeros((height, width, 3), dtype=np.uint8)


RESULT = SimpleNamespace(tracks=())


def make_recorder(tmp_path, **kwargs) -> PendingVideoRecorder:
    return PendingVideoRecorder(
        fps=20.0, pending_dir=tmp_path / "pending", name="test", **kwargs
    )


def feed(recorder: PendingVideoRecorder, count: int = 3) -> None:
    for _ in range(count):
        recorder.write(frame(), RESULT)
    if recorder.failed:
        pytest.skip("video writer unavailable in this OpenCV build")


class TestPendingVideoRecorder:
    def test_records_frames_and_saves_to_the_target(self, tmp_path):
        recorder = make_recorder(tmp_path)
        feed(recorder)
        assert recorder.pending is True
        assert recorder.frames == 3

        recorder.close()
        target = tmp_path / "saved" / "clip.mp4"
        saved = recorder.commit(target)
        assert saved == target
        assert target.exists() and target.stat().st_size > 0
        assert recorder.pending is False

        capture = cv2.VideoCapture(str(target))
        assert capture.isOpened()
        ok, frame_read = capture.read()
        assert ok and frame_read.shape[:2] == (48, 64)
        capture.release()

    def test_default_target_lives_next_to_the_pending_dir(self, tmp_path):
        recorder = make_recorder(tmp_path)
        feed(recorder)
        recorder.close()
        assert recorder.commit() == tmp_path / "test_recording.mp4"

    def test_discard_removes_the_pending_file(self, tmp_path):
        recorder = make_recorder(tmp_path)
        feed(recorder)
        recorder.close()
        assert recorder.discard() is True
        assert recorder.pending is False
        assert recorder.discard() is False
        assert not (tmp_path / "pending").exists() or not any(
            (tmp_path / "pending").iterdir()
        )

    def test_saving_while_recording_is_rejected(self, tmp_path):
        recorder = make_recorder(tmp_path)
        feed(recorder)
        with pytest.raises(SinkError, match="stop the session"):
            recorder.commit(tmp_path / "x.mp4")

    def test_discarding_while_recording_is_rejected(self, tmp_path):
        recorder = make_recorder(tmp_path)
        feed(recorder)
        with pytest.raises(SinkError, match="stop the session"):
            recorder.discard()

    def test_existing_target_is_not_overwritten(self, tmp_path):
        recorder = make_recorder(tmp_path)
        feed(recorder)
        recorder.close()
        target = tmp_path / "taken.mp4"
        target.write_bytes(b"existing")
        with pytest.raises(SinkError, match="already exists"):
            recorder.commit(target)
        assert target.read_bytes() == b"existing"
        assert recorder.pending is True  # 失败后仍可另存

    def test_empty_recording_cannot_be_saved(self, tmp_path):
        recorder = make_recorder(tmp_path)
        recorder.write(None, RESULT)  # 假帧被忽略
        recorder.close()
        assert recorder.frames == 0
        assert recorder.pending is False
        with pytest.raises(SinkError, match="no recorded video"):
            recorder.commit(tmp_path / "empty.mp4")

    def test_commit_twice_returns_the_same_path(self, tmp_path):
        recorder = make_recorder(tmp_path)
        feed(recorder)
        recorder.close()
        first = recorder.commit(tmp_path / "once.mp4")
        assert recorder.commit(tmp_path / "twice.mp4") == first
