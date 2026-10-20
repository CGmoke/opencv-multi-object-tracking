"""适配层：把帧渲染成 JPEG 并线程安全地保存“最新一帧”（Web 实时画面）。

实现 ``domain.ports.Renderer``：``present(frame, result)`` 只做「绘制 → 编码 JPEG →
覆盖写最新帧」，不打开任何窗口，也不处理键鼠事件（Web 场景下交互由 HTTP 路由负责）。

``result`` 是应用层 ``FrameResult``；本模块不 import application，只按属性访问，
保持 ``adapters → application`` 的依赖禁令。
"""

from __future__ import annotations

import threading
import time
from typing import Any, Callable, Dict, Iterator, Optional

import cv2
import numpy as np

from mot.adapters.rendering import draw_hud, draw_tracks
from mot.utils.fps import FpsMeter


class LatestFrameHolder:
    """线程安全地保存“最新一帧 JPEG”（覆盖写，无积压）。"""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._jpeg: Optional[bytes] = None
        self._version = 0

    def set(self, jpeg: bytes) -> None:
        with self._lock:
            self._jpeg = jpeg
            self._version += 1

    def get(self) -> Optional[bytes]:
        with self._lock:
            return self._jpeg

    @property
    def version(self) -> int:
        """写入计数（每写一帧 +1），供 MJPEG 生成器判断是否有新帧。"""
        with self._lock:
            return self._version


class JpegRenderer:
    """Web 渲染器：绘制 → 编码 JPEG → 存最新帧。

    Args:
        stats_fn: 返回管道统计的可调用对象（供 HUD 显示累计 id）。
        source_name / backend: HUD 文本。
        show_hud / show_trails / show_labels / trail_fade: 绘制开关。
        quality: JPEG 质量（1–100）。
        max_fps: 编码上限；超过则本帧不重编码（复用上一帧），避免慢客户端拖垮会话。
    """

    def __init__(
        self,
        *,
        stats_fn: Callable[[], Dict[str, int]],
        source_name: str = "",
        backend: str = "csrt",
        show_hud: bool = True,
        show_trails: bool = True,
        show_labels: bool = True,
        trail_fade: bool = True,
        quality: int = 80,
        max_fps: float = 15.0,
        pixels_per_meter: Optional[float] = None,
        small_max_m: float = 0.5,
        large_min_m: float = 2.0,
    ) -> None:
        self._stats_fn = stats_fn
        self._source_name = source_name
        self._backend = backend
        self._show_hud = show_hud
        self._show_trails = show_trails
        self._show_labels = show_labels
        self._trail_fade = trail_fade
        self._quality = int(min(100, max(1, quality)))
        self._max_fps = max(0.0, float(max_fps))
        self._pixels_per_meter = pixels_per_meter
        self._small_max_m = float(small_max_m)
        self._large_min_m = float(large_min_m)
        self._fps = FpsMeter()
        self.holder = LatestFrameHolder()
        self._last_encode = 0.0
        self.frames = 0

    def present(self, frame: Any, result: Any) -> Optional[str]:
        """绘制并编码一帧；始终返回 ``None``（无交互事件）。"""
        self.frames += 1
        self._fps.tick()

        now = time.monotonic()
        if self._max_fps > 0 and now - self._last_encode < 1.0 / self._max_fps:
            return None
        self._last_encode = now

        canvas = np.asarray(frame).copy()
        canvas = draw_tracks(
            canvas,
            result.tracks,
            show_trails=self._show_trails,
            show_labels=self._show_labels,
            fade=self._trail_fade,
            pixels_per_meter=self._pixels_per_meter,
            small_max_m=self._small_max_m,
            large_min_m=self._large_min_m,
        )
        if self._show_hud:
            active = sum(1 for track in result.tracks if track.is_active)
            created = int(self._stats_fn().get("created_ids", 0))
            canvas = draw_hud(
                canvas,
                fps=self._fps.fps,
                n_active=active,
                total_ids=created,
                frame_index=int(result.frame_index),
                backend=self._backend,
                source_name=self._source_name,
            )
        ok, buffer = cv2.imencode(".jpg", canvas, [int(cv2.IMWRITE_JPEG_QUALITY), self._quality])
        if ok:
            self.holder.set(buffer.tobytes())
        return None


def decode_jpeg(data: bytes) -> Optional[np.ndarray]:
    """把浏览器 POST 上来的 JPEG 字节解码成 BGR 帧；失败返回 ``None``。"""
    if not data:
        return None
    buffer = np.frombuffer(data, dtype=np.uint8)
    frame = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    return frame


def mjpeg_chunks(
    holder: LatestFrameHolder,
    *,
    boundary: str = "motframe",
    fps: float = 15.0,
    should_stop: Optional[Callable[[], bool]] = None,
) -> Iterator[bytes]:
    """把最新帧生成 MJPEG（``multipart/x-mixed-replace``）分块。

    仅在出现新帧（``holder.version`` 变化）时产出一块，否则按 ``fps`` 节流等待；
    ``should_stop()`` 为真时结束。客户端断开时由 WSGI 服务器关闭生成器。
    """
    interval = 1.0 / max(1.0, float(fps))
    marker = boundary.encode("ascii")
    last_version = -1
    while True:
        if should_stop is not None and should_stop():
            return
        version = holder.version
        jpeg = holder.get()
        if jpeg is not None and version != last_version:
            last_version = version
            yield (
                b"--"
                + marker
                + b"\r\nContent-Type: image/jpeg\r\nContent-Length: "
                + str(len(jpeg)).encode("ascii")
                + b"\r\n\r\n"
                + jpeg
                + b"\r\n"
            )
        time.sleep(interval)


__all__ = ["JpegRenderer", "LatestFrameHolder", "decode_jpeg", "mjpeg_chunks"]
