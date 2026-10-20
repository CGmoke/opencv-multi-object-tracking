"""适配层视频输入/输出（步骤 6a 验收测试）。

移植自基线 v0.2.0 的 ``tests/test_video_source.py``；在目标架构中
``VideoSource`` 迁到 ``mot.adapters.sources``，``VideoWriterSink`` 迁到
``mot.adapters.sinks.video``，错误类型统一为 ``SourceError`` / ``SinkError``。
"""

from __future__ import annotations

import pytest

cv2 = pytest.importorskip("cv2")

import numpy as np

from mot.adapters.sinks.video import VideoWriterSink
from mot.adapters.sources import (
    UrlSource,
    VideoSource,
    is_url,
    open_source,
    resolve_source,
)
from mot.domain.errors import SinkError, SourceError


class TestResolveSource:
    def test_numeric_string_is_a_camera_index(self):
        assert resolve_source("0") == 0
        assert resolve_source("3") == 3

    def test_int_camera_index(self):
        assert resolve_source(2) == 2

    def test_negative_index_is_rejected(self):
        with pytest.raises(SourceError, match=">= 0"):
            resolve_source(-1)

    def test_empty_spec_is_rejected(self):
        with pytest.raises(SourceError, match="empty"):
            resolve_source("   ")

    def test_missing_file_is_rejected(self, tmp_path):
        with pytest.raises(SourceError, match="does not exist"):
            resolve_source(str(tmp_path / "nope.mp4"))

    def test_directory_is_rejected(self, tmp_path):
        with pytest.raises(SourceError, match="not a regular file"):
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
        with pytest.raises(SourceError):
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
        with pytest.raises(SinkError, match="invalid frame size"):
            VideoWriterSink(str(tmp_path / "out.mp4"), 20.0, (0, 120))

    def test_directory_target_is_rejected(self, tmp_path):
        with pytest.raises(SinkError, match="directory"):
            VideoWriterSink(str(tmp_path), 20.0, (160, 120))


class _FakeCapture:
    """最小 ``cv2.VideoCapture`` 替身：按脚本返回帧，帧用尽即“断流”。"""

    def __init__(self, frames=()) -> None:
        self._frames = list(frames)
        self._index = 0
        self.released = False

    def isOpened(self) -> bool:
        return True

    def read(self):
        if self._index >= len(self._frames):
            return False, None
        frame = self._frames[self._index]
        self._index += 1
        return True, frame

    def release(self) -> None:
        self.released = True


class _ClosedCapture:
    """打不开的 capture 替身。"""

    def isOpened(self) -> bool:
        return False

    def release(self) -> None:
        pass


class _CaptureFactory:
    """逐个返回 capture；第 ``open_fail_at`` 次起返回“打不开”。"""

    def __init__(self, frames=(), open_fail_at=None) -> None:
        self._frames = frames
        self._open_fail_at = open_fail_at
        self.calls = 0
        self.captures = []

    def __call__(self, spec):
        self.calls += 1
        if self._open_fail_at is not None and self.calls >= self._open_fail_at:
            return _ClosedCapture()
        capture = _FakeCapture(self._frames)
        self.captures.append(capture)
        return capture


class TestUrlRouting:
    def test_url_is_passed_through_untouched(self):
        assert resolve_source("rtsp://cam/stream") == "rtsp://cam/stream"
        assert resolve_source("http://host/v.mp4") == "http://host/v.mp4"

    @pytest.mark.parametrize(
        "spec", ["rtsp://a", "RTSP://A", "rtmp://a", "https://a", "udp://a", "tcp://a"]
    )
    def test_is_url_true(self, spec):
        assert is_url(spec) is True

    @pytest.mark.parametrize("spec", ["clip.mp4", "0", 0, None, "  "])
    def test_is_url_false(self, spec):
        assert is_url(spec) is False

    def test_open_source_dispatches_by_scheme(self, monkeypatch, sample_video):
        factory = _CaptureFactory([])
        monkeypatch.setattr(cv2, "VideoCapture", factory)
        assert isinstance(open_source("rtsp://cam/stream"), UrlSource)
        assert isinstance(open_source(str(sample_video)), VideoSource)


class TestUrlSource:
    def test_reads_frames_and_counts(self, monkeypatch):
        frame = np.zeros((4, 4, 3), dtype=np.uint8)
        monkeypatch.setattr(cv2, "VideoCapture", _CaptureFactory([frame, frame]))
        source = UrlSource("rtsp://cam/stream", backoff=0.0)
        ok, got = source.read()
        assert ok is True and got is frame
        assert source.read()[0] is True
        assert source.frames_read == 2
        assert source.read()[0] is False  # 帧用尽
        assert source.name == "rtsp://cam/stream"
        source.release()

    def test_reconnects_after_a_drop(self, monkeypatch):
        frame = np.zeros((4, 4, 3), dtype=np.uint8)
        factory = _CaptureFactory([frame])  # 每个新 capture 只有 1 帧
        monkeypatch.setattr(cv2, "VideoCapture", factory)
        source = UrlSource("rtsp://cam/stream", max_retries=2, backoff=0.0)
        assert source.read()[0] is True  # 首帧
        assert source.read()[0] is False  # 断流
        assert source.read()[0] is True  # 重连成功后再读一帧
        assert factory.calls >= 2
        source.release()

    def test_gives_up_after_max_retries(self, monkeypatch):
        factory = _CaptureFactory([], open_fail_at=2)
        monkeypatch.setattr(cv2, "VideoCapture", factory)
        source = UrlSource("rtsp://cam/stream", max_retries=2, backoff=0.0)
        assert source.read()[0] is False  # 首帧即失败
        assert source.read()[0] is False  # 第 1 次重连失败
        assert source.read()[0] is False  # 第 2 次重连失败 = 达上限
        assert factory.calls == 3  # 初次打开 + 2 次重连
        assert source.read()[0] is False  # 已放弃，不再重连
        assert factory.calls == 3
        source.release()

    def test_rejects_non_url(self):
        with pytest.raises(SourceError, match="not a network stream"):
            UrlSource("clip.mp4")
