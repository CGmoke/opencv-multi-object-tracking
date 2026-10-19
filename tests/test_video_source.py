"""Video input/output wrappers."""

from __future__ import annotations

import pytest

cv2 = pytest.importorskip("cv2")

import numpy as np

from mot.video_source import (
    VideoSource,
    VideoSourceError,
    VideoWriterSink,
    resolve_source,
)


class TestResolveSource:
    def test_numeric_string_is_a_camera_index(self):
        assert resolve_source("0") == 0
        assert resolve_source("3") == 3

    def test_int_camera_index(self):
        assert resolve_source(2) == 2

    def test_negative_index_is_rejected(self):
        with pytest.raises(VideoSourceError, match=">= 0"):
            resolve_source(-1)

    def test_empty_spec_is_rejected(self):
        with pytest.raises(VideoSourceError, match="empty"):
            resolve_source("   ")

    def test_missing_file_is_rejected(self, tmp_path):
        with pytest.raises(VideoSourceError, match="does not exist"):
            resolve_source(str(tmp_path / "nope.mp4"))

    def test_directory_is_rejected(self, tmp_path):
        with pytest.raises(VideoSourceError, match="not a regular file"):
            resolve_source(str(tmp_path))

    def test_existing_file_is_resolved(self, sample_video):
        resolved = resolve_source(str(sample_video))
        assert resolved.endswith("input.mp4")


class TestVideoSource:
    def test_reads_a_synthetic_video(self, sample_video):
        with VideoSource(str(sample_video)) as source:
            assert source.is_camera is False
            assert source.name == "input.mp4"
            assert source.width == 160
            assert source.height == 120
            assert source.fps == pytest.approx(20.0, rel=0.1)
            assert source.frame_count == 20

            frames = 0
            while True:
                ok, frame = source.read()
                if not ok:
                    break
                assert frame is not None and frame.shape == (120, 160, 3)
                frames += 1
            assert frames == 20
            assert source.frames_read == 20

    def test_rewind(self, sample_video):
        with VideoSource(str(sample_video)) as source:
            source.read()
            source.read()
            assert source.rewind() is True
            ok, frame = source.read()
            assert ok and frame is not None

    def test_loop_rewinds_once(self, sample_video):
        with VideoSource(str(sample_video), loop=True) as source:
            for _ in range(20):
                assert source.read()[0] is True
            ok, _frame = source.read()
            assert ok is True
            assert source.frames_read == 21

    def test_loop_is_ignored_for_cameras(self):
        source = VideoSource.__new__(VideoSource)
        source._resolved = 0
        source._is_camera = True
        assert source.is_camera is True

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(VideoSourceError):
            VideoSource(str(tmp_path / "nope.mp4"))


class TestVideoWriterSink:
    def test_writes_a_playable_file(self, tmp_path):
        path = tmp_path / "out" / "annotated.mp4"
        with VideoWriterSink(str(path), 20.0, (160, 120)) as sink:
            for _ in range(5):
                sink.write(np.zeros((120, 160, 3), dtype=np.uint8))
            assert sink.codec
        assert path.exists()
        assert path.stat().st_size > 0

    def test_invalid_frame_size_is_rejected(self, tmp_path):
        with pytest.raises(VideoSourceError, match="invalid frame size"):
            VideoWriterSink(str(tmp_path / "out.mp4"), 20.0, (0, 120))

    def test_directory_target_is_rejected(self, tmp_path):
        with pytest.raises(VideoSourceError, match="directory"):
            VideoWriterSink(str(tmp_path), 20.0, (160, 120))
