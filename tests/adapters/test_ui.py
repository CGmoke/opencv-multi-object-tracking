"""适配层交互状态机（步骤 6a 验收测试）。

移植自基线 v0.2.0 的 ``tests/test_interactive.py``；``ui`` 不 import cv2，鼠标事件码
用本模块自带的 ``MOUSE_*`` 常量（数值与 ``cv2.EVENT_*`` 一致），故本测试无需 cv2。
"""

from __future__ import annotations

import pytest

from mot.adapters.ui import (
    KEY_EVENTS,
    MIN_DRAG_SIDE,
    MOUSE_LBUTTON_DOWN,
    MOUSE_LBUTTON_UP,
    MOUSE_MOVE,
    InteractiveSession,
)


def drag(session: InteractiveSession, start, end) -> None:
    session.handle_mouse(MOUSE_LBUTTON_DOWN, start[0], start[1])
    session.handle_mouse(MOUSE_MOVE, end[0], end[1])
    session.handle_mouse(MOUSE_LBUTTON_UP, end[0], end[1])


class TestKeys:
    def test_quit_key_sets_the_flag(self):
        session = InteractiveSession()
        assert session.handle_key(ord("q")) == "quit"
        assert session.quit_requested is True

    def test_escape_quits(self):
        session = InteractiveSession()
        assert session.handle_key(27) == "quit"

    def test_space_toggles_pause(self):
        session = InteractiveSession()
        session.handle_key(ord(" "))
        assert session.paused is True
        session.handle_key(ord(" "))
        assert session.paused is False

    def test_unknown_key_is_ignored(self):
        session = InteractiveSession()
        assert session.handle_key(ord("z")) is None
        assert session.handle_key(-1) is None

    def test_add_mode_toggles(self):
        session = InteractiveSession()
        session.handle_key(ord("n"))
        assert session.mode == "add"
        session.handle_key(ord("n"))
        assert session.mode == "browse"

    def test_delete_mode_and_switch(self):
        session = InteractiveSession()
        session.handle_key(ord("x"))
        assert session.mode == "delete"
        session.handle_key(ord("n"))
        assert session.mode == "add"

    def test_help_toggles(self):
        session = InteractiveSession()
        assert session.help_visible is True
        session.handle_key(ord("h"))
        assert session.help_visible is False

    def test_key_map_covers_documented_actions(self):
        actions = set(KEY_EVENTS.values())
        assert {"quit", "pause", "add_mode", "delete_mode", "clear_trails", "reset"} <= actions


class TestMouse:
    def test_drag_queues_a_box_and_returns_to_browse(self):
        session = InteractiveSession()
        session.handle_key(ord("n"))
        drag(session, (10, 10), (60, 50))
        boxes = session.consume_new_boxes()
        assert boxes == [(10.0, 10.0, 50.0, 40.0)]
        assert session.mode == "browse"
        assert session.consume_new_boxes() == []

    def test_drag_from_any_corner(self):
        session = InteractiveSession()
        session.handle_key(ord("n"))
        drag(session, (60, 50), (10, 10))
        assert session.consume_new_boxes() == [(10.0, 10.0, 50.0, 40.0)]

    def test_tiny_drag_is_rejected(self):
        session = InteractiveSession()
        session.handle_key(ord("n"))
        drag(session, (10, 10), (10 + MIN_DRAG_SIDE - 1, 40))
        assert session.consume_new_boxes() == []
        assert session.mode == "add"

    def test_delete_click_is_queued(self):
        session = InteractiveSession()
        session.handle_key(ord("x"))
        drag(session, (30, 30), (30, 30))
        assert session.consume_delete_points() == [(30.0, 30.0)]
        assert session.mode == "browse"

    def test_browse_mode_drag_does_nothing(self):
        session = InteractiveSession()
        drag(session, (10, 10), (60, 50))
        assert session.consume_new_boxes() == []
        assert session.consume_delete_points() == []

    def test_drag_box_preview(self):
        session = InteractiveSession()
        assert session.drag_box is None
        session.handle_key(ord("n"))
        session.handle_mouse(MOUSE_LBUTTON_DOWN, 10, 10)
        session.handle_mouse(MOUSE_MOVE, 40, 40)
        assert session.drag_box == ((10.0, 10.0), (40.0, 40.0))
        session.handle_mouse(MOUSE_LBUTTON_UP, 40, 40)
        assert session.drag_box is None


class TestNotices:
    def test_notice_ages_out(self):
        session = InteractiveSession()
        session.set_notice("hello", ttl=2)
        assert session.notice == "hello"
        session.tick()
        assert session.notice == "hello"
        session.tick()
        assert session.notice == ""

    def test_notice_is_set_by_actions(self):
        session = InteractiveSession()
        session.handle_key(ord(" "))
        assert session.notice == "paused"
        session.handle_key(ord(" "))
        assert session.notice == "resumed"


def test_ui_does_not_import_cv2():
    import mot.adapters.ui as ui_module

    source = ui_module.__file__
    assert source is not None
    with open(source, encoding="utf-8") as handle:
        text = handle.read()
    statements = [line.strip() for line in text.splitlines()]
    assert not any(line.startswith(("import cv2", "from cv2")) for line in statements), (
        "adapters/ui.py must not import cv2 (cv2 constants live in interfaces/ui_window.py)"
    )


def test_mouse_constants_match_opencv_values():
    cv2 = pytest.importorskip("cv2")
    assert MOUSE_MOVE == cv2.EVENT_MOUSEMOVE
    assert MOUSE_LBUTTON_DOWN == cv2.EVENT_LBUTTONDOWN
    assert MOUSE_LBUTTON_UP == cv2.EVENT_LBUTTONUP
