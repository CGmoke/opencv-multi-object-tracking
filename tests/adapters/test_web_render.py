"""适配层：Web JPEG 渲染器与 MJPEG 分块（P1 验收测试）。"""

from __future__ import annotations

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from mot.adapters.web_render import JpegRenderer, LatestFrameHolder, mjpeg_chunks  # noqa: E402
from mot.application.pipeline import FrameResult  # noqa: E402
from mot.domain.models import TrackedObject  # noqa: E402


def frame(height: int = 120, width: int = 160) -> np.ndarray:
    return np.zeros((height, width, 3), dtype=np.uint8)


def result(frame_index: int = 1, tracks=()) -> FrameResult:
    return FrameResult(frame_index=frame_index, timestamp=0.0, tracks=tracks)


def renderer(**kwargs) -> JpegRenderer:
    return JpegRenderer(stats_fn=lambda: {"created_ids": 0}, **kwargs)


class TestLatestFrameHolder:
    def test_set_get_and_version(self):
        holder = LatestFrameHolder()
        assert holder.get() is None and holder.version == 0
        holder.set(b"abc")
        assert holder.get() == b"abc" and holder.version == 1
        holder.set(b"def")
        assert holder.get() == b"def" and holder.version == 2


class TestJpegRenderer:
    def test_present_encodes_a_decodable_frame(self):
        renderer_ = renderer(max_fps=0)
        renderer_.present(frame(), result())
        jpeg = renderer_.holder.get()
        assert jpeg is not None
        decoded = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
        assert decoded is not None and decoded.shape == (120, 160, 3)

    def test_present_returns_none(self):
        assert renderer(max_fps=0).present(frame(), result()) is None

    def test_max_fps_throttles_encoding(self):
        renderer_ = renderer(max_fps=1.0)
        renderer_.present(frame(), result())  # 第一帧编码
        first = renderer_.holder.version
        renderer_.present(frame(), result())  # 立即第二次 → 被节流
        assert renderer_.holder.version == first

    def test_tracks_are_drawn_into_the_canvas(self):
        track = TrackedObject(track_id=1, tracker=None, bbox=(10.0, 10.0, 40.0, 30.0))
        renderer_ = renderer(max_fps=0, show_hud=False)
        renderer_.present(frame(), result(tracks=(track,)))
        decoded = cv2.imdecode(
            np.frombuffer(renderer_.holder.get(), dtype=np.uint8), cv2.IMREAD_COLOR
        )
        assert decoded.any()  # 画了框 → 非全黑


class TestMjpegChunks:
    def test_yields_a_chunk_with_boundary_and_content_type(self):
        holder = LatestFrameHolder()
        holder.set(b"jpegbytes")
        chunks = mjpeg_chunks(holder, boundary="bnd", fps=60.0)
        first = next(chunks)
        assert first.startswith(b"--bnd\r\n")
        assert b"Content-Type: image/jpeg" in first
        assert first.endswith(b"\r\n")
        chunks.close()

    def test_stops_when_should_stop_is_true(self):
        holder = LatestFrameHolder()
        chunks = mjpeg_chunks(holder, fps=60.0, should_stop=lambda: True)
        with pytest.raises(StopIteration):
            next(chunks)
