"""Interactive state machine (keys, drag selection, notices)."""

from __future__ import annotations

import pytest

cv2 = pytest.importorskip("cv2")

from mot.interactive import KEY_EVENTS, MIN_DRAG_SIDE, InteractiveSession


def drag(session: InteractiveSession, start, end) -> None:
    session.handle_mouse(cv2.EVENT_LBUTTONDOWN, start[0], start[1])
    session.handle_mouse(cv2.EVENT_MOUSEMOVE, end[0], end[1])
    session.handle_mouse(cv2.EVENT_LBUTTONUP, end[0], end[1])


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
        session.handle_mouse(cv2.EVENT_LBUTTONDOWN, 10, 10)
        session.handle_mouse(cv2.EVENT_MOUSEMOVE, 40, 40)
        assert session.drag_box == ((10.0, 10.0), (40.0, 40.0))
        session.handle_mouse(cv2.EVENT_LBUTTONUP, 40, 40)
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
