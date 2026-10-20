"""接口层：窗口呈现与输入轮询（唯一直接调用 cv2 窗口 API 的地方）。

把 :class:`mot.adapters.ui.InteractiveSession`（零 cv2 的状态机）与
:class:`mot.adapters.rendering` 的绘制函数接到 OpenCV 窗口上，并实现
:class:`mot.domain.ports.Renderer` 端口：``present(frame, result)`` 负责“绘制 +
（可选）录像 + 呈现 + 轮询输入”，返回一个交互事件名给会话。
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import cv2
import numpy as np

from mot.adapters.rendering import (
    draw_banner,
    draw_help,
    draw_hud,
    draw_selection,
    draw_tracks,
)
from mot.adapters.sinks.video import VideoWriterSink
from mot.adapters.ui import (
    MOUSE_LBUTTON_DOWN,
    MOUSE_LBUTTON_UP,
    MOUSE_MOVE,
    InteractiveSession,
)
from mot.application.pipeline import FrameResult
from mot.application.session import TrackingSession
from mot.domain.errors import SinkError
from mot.utils.fps import FpsMeter
from mot.utils.paths import sanitize_filename

LOGGER = logging.getLogger(__name__)

#: cv2 事件常量 -> ``adapters.ui`` 事件码（数值一致，这里显式桥接以保持 ui 零 cv2）。
_CV2_TO_UI_MOUSE = {
    cv2.EVENT_MOUSEMOVE: MOUSE_MOVE,
    cv2.EVENT_LBUTTONDOWN: MOUSE_LBUTTON_DOWN,
    cv2.EVENT_LBUTTONUP: MOUSE_LBUTTON_UP,
}


def display_available() -> bool:
    """探测是否有可用 GUI（创建窗口前的最佳努力判断）。"""
    if sys.platform.startswith("linux"):
        if not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY"):
            return False
    return True


def save_snapshot(
    frame: np.ndarray, directory: str | Path, source_name: str, frame_index: int
) -> Path:
    """把当前帧写成 PNG 并返回路径。

    同一帧反复按 ``s`` 不得覆盖上一张截图；预先放置的符号链接不得被跟随写入。
    """
    base = Path(directory) / (
        f"{sanitize_filename(source_name, 'source')}_frame_{int(frame_index):06d}"
    )
    path = base.with_name(base.name + ".png")
    if path.is_symlink():
        raise SinkError(f"refusing to overwrite a symlink snapshot: {path}")
    counter = 1
    while path.exists():
        path = base.with_name(f"{base.name}_{counter}.png")
        counter += 1
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), frame):
        raise SinkError(f"could not write snapshot {path}")
    return path


class WindowRenderer:
    """组合“绘制 + 录像 + 窗口呈现 + 输入轮询”的 ``Renderer`` 实现。

    Args:
        window_name: 预览窗口名。
        ui: 交互状态机（拖拽/按键状态）。
        session: 会话，用于回灌交互动作（新增框 / 删除点）。
        stats: 返回管道统计的可调用对象（供 HUD 显示累计 id）。
        source_name: 源名（HUD / 截图文件名）。
        show_display: 是否创建并呈现窗口；``False`` 时仅绘制（可只录像）。
        video_sink: 可选录像输出。
        snapshot_dir: 截图目录。
    """

    def __init__(
        self,
        window_name: str,
        *,
        ui: InteractiveSession,
        session: TrackingSession,
        stats: Callable[[], Dict[str, int]],
        backend: str = "csrt",
        source_name: str = "",
        show_trails: bool = True,
        show_labels: bool = True,
        show_hud: bool = True,
        trail_fade: bool = True,
        show_display: bool = True,
        video_sink: Optional[VideoWriterSink] = None,
        snapshot_dir: str = "outputs/snapshots",
    ) -> None:
        self.window_name = window_name
        self._ui = ui
        self._session = session
        self._stats = stats
        self._backend = backend
        self._source_name = source_name
        self._show_trails = show_trails
        self._show_labels = show_labels
        self._show_hud = show_hud
        self._trail_fade = trail_fade
        self._show_display = show_display
        self._video_sink = video_sink
        self._snapshot_dir = snapshot_dir
        self._fps = FpsMeter()
        self._last_canvas: Optional[np.ndarray] = None

    # -- 生命周期 --------------------------------------------------------
    def open(self) -> None:
        """创建窗口并绑定鼠标回调；失败则降级为不显示。"""
        if not self._show_display:
            return
        try:
            cv2.namedWindow(self.window_name, cv2.WINDOW_AUTOSIZE)
            cv2.setMouseCallback(self.window_name, self._on_mouse)
        except cv2.error as exc:
            LOGGER.warning("cannot create preview window (%s); continuing without it", exc)
            self._show_display = False

    def close(self) -> None:
        if self._video_sink is not None:
            self._video_sink.release()
        if self._show_display:
            cv2.destroyAllWindows()

    # -- Renderer 端口 ---------------------------------------------------
    def present(self, frame: Any, result: FrameResult) -> Optional[str]:
        """绘制并呈现一帧，返回交互事件名（或 ``None``）。"""
        canvas = frame.copy()
        canvas = draw_tracks(
            canvas,
            result.tracks,
            show_trails=self._show_trails,
            show_labels=self._show_labels,
            fade=self._trail_fade,
        )
        drag = self._ui.drag_box
        if drag is not None:
            canvas = draw_selection(canvas, drag[0], drag[1])
        if self._show_hud:
            active = sum(1 for track in result.tracks if track.is_active)
            created = int(self._stats().get("created_ids", 0))
            canvas = draw_hud(
                canvas,
                fps=self._fps.fps,
                n_active=active,
                total_ids=created,
                frame_index=result.frame_index,
                paused=self._ui.paused,
                backend=self._backend,
                source_name=self._source_name,
            )
        if self._ui.help_visible:
            canvas = draw_help(canvas)
        if self._ui.notice:
            canvas = draw_banner(canvas, self._ui.notice)
        self._last_canvas = canvas

        if self._video_sink is not None:
            self._video_sink.write(canvas)

        event: Optional[str] = None
        if self._show_display:
            cv2.imshow(self.window_name, canvas)
            if self._window_closed():
                return "quit"
            event = self._ui.handle_key(cv2.waitKey(1) & 0xFF)
            self._drain_mouse()

        self._fps.tick()
        self._ui.tick()
        if event == "save_frame":
            self._handle_save_snapshot(result.frame_index)
        return event

    # -- 内部 ------------------------------------------------------------
    def _on_mouse(self, event: int, x: int, y: int, flags: int, _param: object) -> None:
        ui_event = _CV2_TO_UI_MOUSE.get(event)
        if ui_event is not None:
            self._ui.handle_mouse(ui_event, x, y, flags)

    def _drain_mouse(self) -> None:
        for box in self._ui.consume_new_boxes():
            self._session.queue_box(box)
        for point in self._ui.consume_delete_points():
            self._session.queue_delete(point)

    def _window_closed(self) -> bool:
        try:
            return cv2.getWindowProperty(self.window_name, cv2.WND_PROP_VISIBLE) < 1
        except cv2.error:
            return True

    def _handle_save_snapshot(self, frame_index: int) -> None:
        if self._last_canvas is None:
            self._ui.set_notice("nothing to save yet", 45)
            return
        try:
            path = save_snapshot(
                self._last_canvas, self._snapshot_dir, self._source_name, frame_index
            )
        except (SinkError, OSError) as exc:
            LOGGER.error("%s", exc)
            self._ui.set_notice("snapshot failed", 60)
        else:
            self._ui.set_notice(f"saved {path.name}")


__all__ = ["WindowRenderer", "display_available", "save_snapshot"]
