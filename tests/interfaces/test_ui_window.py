"""接口层窗口渲染器（步骤 6b 验收测试）。

无头环境无法开真窗口，故主要覆盖：端口一致性、``show_display=False`` 的纯录制路径，
以及把鼠标动作回灌会话（通过 monkeypatch 屏蔽真实窗口调用）。
"""

from __future__ import annotations

import pytest

cv2 = pytest.importorskip("cv2")

import numpy as np  # noqa: E402 - importorskip 必须先行

from mot.adapters.sinks.video import VideoWriterSink  # noqa: E402
from mot.adapters.ui import (  # noqa: E402
    MOUSE_LBUTTON_DOWN,
    MOUSE_LBUTTON_UP,
    MOUSE_MOVE,
    InteractiveSession,
)
from mot.application.pipeline import FrameResult  # noqa: E402
from mot.domain.ports import Renderer  # noqa: E402
from mot.interfaces.ui_window import WindowRenderer  # noqa: E402


class _StubSession:
    def __init__(self) -> None:
        self.boxes: list = []
        self.deletes: list = []

    def queue_box(self, bbox) -> None:
        self.boxes.append(tuple(bbox))

    def queue_delete(self, point) -> None:
        self.deletes.append(tuple(point))


def _frame() -> np.ndarray:
    return np.zeros((120, 160, 3), dtype=np.uint8)


def _result(frame_index: int = 1) -> FrameResult:
    return FrameResult(frame_index=frame_index, timestamp=0.0)


def _renderer(ui: InteractiveSession, session: _StubSession, **kwargs) -> WindowRenderer:
    return WindowRenderer("w", ui=ui, session=session, stats=lambda: {"created_ids": 0}, **kwargs)


def test_implements_the_renderer_port():
    assert isinstance(_renderer(InteractiveSession(), _StubSession(), show_display=False), Renderer)


def test_headless_open_is_a_noop():
    renderer = _renderer(InteractiveSession(), _StubSession(), show_display=False)
    renderer.open()  # 不应创建窗口 / 不应抛异常
    renderer.close()


def test_headless_present_returns_none_and_records_video(tmp_path):
    video = VideoWriterSink(str(tmp_path / "out.mp4"), 10.0, (160, 120))
    renderer = _renderer(InteractiveSession(), _StubSession(), show_display=False, video_sink=video)
    assert renderer.present(_frame(), _result()) is None
    renderer.close()
    assert (tmp_path / "out.mp4").exists()
    assert (tmp_path / "out.mp4").stat().st_size > 0


def test_mouse_actions_are_drained_into_the_session(monkeypatch):
    # 屏蔽真实窗口调用，使交互路径可在无头环境跑通。
    monkeypatch.setattr(cv2, "imshow", lambda *a, **k: None)
    monkeypatch.setattr(cv2, "waitKey", lambda *a, **k: -1)
    monkeypatch.setattr(cv2, "getWindowProperty", lambda *a, **k: 1.0)

    ui = InteractiveSession()
    session = _StubSession()
    renderer = _renderer(ui, session)  # show_display=True（默认）

    ui.handle_key(ord("n"))
    ui.handle_mouse(MOUSE_LBUTTON_DOWN, 10, 10)
    ui.handle_mouse(MOUSE_MOVE, 60, 50)
    ui.handle_mouse(MOUSE_LBUTTON_UP, 60, 50)

    renderer.present(_frame(), _result())
    assert session.boxes == [(10.0, 10.0, 50.0, 40.0)]


def test_window_closed_becomes_a_quit_event(monkeypatch):
    monkeypatch.setattr(cv2, "imshow", lambda *a, **k: None)
    monkeypatch.setattr(cv2, "waitKey", lambda *a, **k: -1)
    monkeypatch.setattr(cv2, "getWindowProperty", lambda *a, **k: 0.0)  # 不可见

    renderer = _renderer(InteractiveSession(), _StubSession())
    assert renderer.present(_frame(), _result()) == "quit"
