"""适配层：交互状态机。

保持无阻塞调用，可由主循环驱动：

* ``handle_key`` 把 ``waitKey`` 的键码翻译成事件名供应用层处理；
* ``handle_mouse`` 执行拖拽选择状态机，把框或删除点入队待应用层消费。

只放**状态**，所有 cv2 窗口调用留在接口层。鼠标事件码与 OpenCV ``cv2.EVENT_*``
数值一致，但本模块**不 import cv2**（解耦由 ``interfaces/ui_window.py`` 负责）。
"""

from __future__ import annotations

from collections import deque
from typing import Deque, List, Optional, Tuple

from mot.domain.models import BBox, Point, normalize_bbox

#: 接受为新建目标的最小拖拽边长（像素）。
MIN_DRAG_SIDE = 8.0

#: 鼠标事件码（与 OpenCV ``cv2.EVENT_*`` 数值一致，但不 import cv2）。
MOUSE_MOVE = 0
MOUSE_LBUTTON_DOWN = 1
MOUSE_LBUTTON_UP = 4

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
    """暂停标志、选择模式与待处理用户动作。"""

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

    # -- 按键 ------------------------------------------------------------
    def handle_key(self, key: int) -> Optional[str]:
        """把 ``cv2.waitKey`` 键码翻译成动作名。"""
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
                "drag a box around the target"
                if self.mode == "add"
                else "click a target to delete it"
            )
            self.notice_ttl = 120
        elif event == "toggle_help":
            self.help_visible = not self.help_visible
        return event

    # -- 鼠标 ------------------------------------------------------------
    def handle_mouse(self, event: int, x: int, y: int, flags: int = 0) -> None:
        """处理一个鼠标事件；``(x, y)`` 为窗口坐标。"""
        point = (float(x), float(y))
        if event == MOUSE_LBUTTON_DOWN:
            self._drag_start = point
            self._drag_current = point
        elif event == MOUSE_MOVE and self._drag_start is not None:
            self._drag_current = point
        elif event == MOUSE_LBUTTON_UP:
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

    # -- 主循环消费的状态 ------------------------------------------------
    @property
    def drag_box(self) -> Optional[Tuple[Point, Point]]:
        """正在拖拽的矩形，用于预览叠加。"""
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
        """老化瞬时提示；每处理一帧调用一次。"""
        if self.notice_ttl > 0:
            self.notice_ttl -= 1
            if self.notice_ttl == 0:
                self.notice = ""


__all__ = [
    "KEY_EVENTS",
    "MIN_DRAG_SIDE",
    "MOUSE_LBUTTON_DOWN",
    "MOUSE_LBUTTON_UP",
    "MOUSE_MOVE",
    "InteractiveSession",
]
