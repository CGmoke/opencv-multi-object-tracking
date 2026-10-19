"""Interactive state machine for the demo window.

Kept free of blocking calls so it can be driven from a main loop:

* ``handle_key`` returns a short event name for the application to act on.
* ``handle_mouse`` performs the drag-to-select state machine and queues boxes
  or delete clicks for the application to consume.

Only the *state* lives here; every cv2 window call stays in :mod:`mot.app`.
"""

from __future__ import annotations

from collections import deque
from typing import Deque, List, Optional, Tuple

import cv2

from .models import BBox, Point, normalize_bbox

#: Minimum drag size, in pixels, for a box to be accepted as a new target.
MIN_DRAG_SIDE = 8.0

KEY_EVENTS = {
    ord("q"): "quit",
    27: "quit",  # esc
    ord(" "): "pause",
    ord("n"): "add_mode",
    ord("x"): "delete_mode",
    ord("c"): "clear_trails",
    ord("r"): "reset",
    ord("s"): "save_frame",
    ord("h"): "toggle_help",
}


class InteractiveSession:
    """Pause flag, selection modes and pending user actions."""

    MODES = ("browse", "add", "delete")

    def __init__(self) -> None:
        self.paused = False
        self.quit_requested = False
        self.help_visible = True
        self.mode = "browse"
        self._drag_start: Optional[Point] = None
        self._drag_current: Optional[Point] = None
        self._new_boxes: Deque[BBox] = deque()
        self._delete_points: Deque[Point] = deque()
        self.notice = ""
        self.notice_ttl = 0

    # -- key handling ----------------------------------------------------
    def handle_key(self, key: int) -> Optional[str]:
        """Translate a ``cv2.waitKey`` code into an action name."""
        if key is None or key < 0:
            return None
        event = KEY_EVENTS.get(key)
        if event is None:
            return None
        if event == "quit":
            self.quit_requested = True
        elif event == "pause":
            self.paused = not self.paused
            self.notice = "resumed" if not self.paused else "paused"
            self.notice_ttl = 90
        elif event in ("add_mode", "delete_mode"):
            target = "add" if event == "add_mode" else "delete"
            self.mode = "browse" if self.mode == target else target
            self.notice = (
                "drag a box around the target" if self.mode == "add" else "click a target to delete it"
            )
            self.notice_ttl = 120
        elif event == "toggle_help":
            self.help_visible = not self.help_visible
        return event

    # -- mouse handling --------------------------------------------------
    def handle_mouse(self, event: int, x: int, y: int, flags: int = 0) -> None:
        """Process a mouse event; ``(x, y)`` are window coordinates."""
        point = (float(x), float(y))
        if event == cv2.EVENT_LBUTTONDOWN:
            self._drag_start = point
            self._drag_current = point
        elif event == cv2.EVENT_MOUSEMOVE and self._drag_start is not None:
            self._drag_current = point
        elif event == cv2.EVENT_LBUTTONUP:
            start = self._drag_start
            self._drag_start = None
            self._drag_current = None
            if start is None:
                return
            if self.mode == "delete":
                self._delete_points.append(point)
                self.mode = "browse"
                self.notice = "delete click queued"
                self.notice_ttl = 60
                return
            box = self._box_from_drag(start, point)
            if box is not None and self.mode == "add":
                self._new_boxes.append(box)
                self.mode = "browse"
                self.notice = "target added"
                self.notice_ttl = 60

    def _box_from_drag(self, start: Point, end: Point) -> Optional[BBox]:
        x = min(start[0], end[0])
        y = min(start[1], end[1])
        w = abs(end[0] - start[0])
        h = abs(end[1] - start[1])
        if w < MIN_DRAG_SIDE or h < MIN_DRAG_SIDE:
            return None
        try:
            return normalize_bbox((x, y, w, h))
        except ValueError:
            return None

    # -- state consumed by the main loop ---------------------------------
    @property
    def drag_box(self) -> Optional[Tuple[Point, Point]]:
        """The rectangle currently being dragged, for the preview overlay."""
        if self._drag_start is None or self._drag_current is None:
            return None
        return self._drag_start, self._drag_current

    def consume_new_boxes(self) -> List[BBox]:
        boxes = list(self._new_boxes)
        self._new_boxes.clear()
        return boxes

    def consume_delete_points(self) -> List[Point]:
        points = list(self._delete_points)
        self._delete_points.clear()
        return points

    def set_notice(self, text: str, ttl: int = 90) -> None:
        self.notice = text
        self.notice_ttl = int(ttl)

    def tick(self) -> None:
        """Age the transient notice; call once per processed frame."""
        if self.notice_ttl > 0:
            self.notice_ttl -= 1
            if self.notice_ttl == 0:
                self.notice = ""


__all__ = ["KEY_EVENTS", "MIN_DRAG_SIDE", "InteractiveSession"]
