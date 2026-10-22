"""适配层：视频输入与源解析（实现 ``domain.ports.FrameSource``）。

``VideoSource`` 隐藏“摄像头索引 vs 视频文件”的差异，``UrlSource`` 处理网络流
（RTSP/RTMP/HTTP）并在断流时自动重连；``open_source`` 按 spec 自动选择。打开失败会
给出清晰错误而不是一片黑窗。本模块是唯一允许 import ``cv2`` 的层之一。
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Optional, Tuple, Union

import cv2
import numpy as np

from mot.domain.errors import SourceError
from mot.domain.ports import FrameSource

SourceSpec = Union[str, int]

#: 视为网络流（而非本地文件）的 URL 前缀。
URL_SCHEMES: Tuple[str, ...] = (
    "rtsp://",
    "rtmp://",
    "rtmps://",
    "http://",
    "https://",
    "udp://",
    "tcp://",
)


def is_url(spec: object) -> bool:
    """返回 ``spec`` 是否为网络流地址（rtsp / rtmp / http(s) / udp / tcp）。"""
    return isinstance(spec, str) and spec.strip().lower().startswith(URL_SCHEMES)


def camera_apis() -> Tuple[int, ...]:
    """返回本平台优先尝试的摄像头后端。

    OpenCV 5 的 ``CAP_ANY`` 可能先尝试 FFMPEG / obsensor，对普通 UVC 摄像头会报
    ``Camera index out of range``，因此摄像头索引优先显式指定系统后端。
    """
    if sys.platform.startswith("linux"):
        return (cv2.CAP_V4L2,)
    if sys.platform == "win32":
        return (cv2.CAP_DSHOW, cv2.CAP_MSMF)
    if sys.platform == "darwin":
        return (cv2.CAP_AVFOUNDATION,)
    return ()


def resolve_source(spec: SourceSpec) -> Union[str, int]:
    """把 CLI 字符串映射为摄像头索引、网络流地址或绝对文件路径。

    Raises:
        SourceError: 输入为空、索引为负，或本地路径不存在 / 非普通文件。
    """
    if isinstance(spec, int):
        if spec < 0:
            raise SourceError(f"camera index must be >= 0, got {spec}", field="source", value=spec)
        return spec

    text = str(spec).strip()
    if not text:
        raise SourceError(
            "video source is empty; pass a camera index or a video path", field="source"
        )
    if text.isdigit():
        return int(text)
    if is_url(text):
        return text

    path = Path(text).expanduser()
    if not path.exists():
        raise SourceError(f"video file does not exist: {path}", field="source", value=text)
    if not path.is_file():
        raise SourceError(f"video source is not a regular file: {path}", field="source", value=text)
    return str(path.resolve())


class VideoSource:
    """``cv2.VideoCapture`` 的上下文管理包装。

    Args:
        spec: 摄像头索引（``0`` / ``"0"``）或视频文件路径。
        loop: 文件结束时重播（对摄像头忽略）。
    """

    def __init__(self, spec: SourceSpec, loop: bool = False) -> None:
        self._resolved = resolve_source(spec)
        self._is_camera = isinstance(self._resolved, int)
        self._is_url = is_url(self._resolved)
        self.loop = bool(loop) and not self._is_camera and not self._is_url
        self._capture: Optional[cv2.VideoCapture] = self._open_capture()
        if self._capture is None or not self._capture.isOpened():
            if self._is_camera:
                raise SourceError(
                    f"could not open camera index {self._resolved}. Check that the device "
                    "exists (`ls /dev/video*`), that your user is in the `video` group "
                    "(`sudo usermod -aG video $USER`, then re-login), and that no other app "
                    "is using it. On a headless/server host without a camera, use the "
                    "browser camera source instead.",
                    field="source",
                    value=self._resolved,
                )
            raise SourceError(
                f"could not open video source {self._resolved!r}. "
                "Check the path and that OpenCV has the matching codec (FFMPEG/GStreamer).",
                field="source",
                value=self._resolved,
            )
        self._frame_count = 0

    def _open_capture(self) -> cv2.VideoCapture:
        """打开采集器；摄像头索引优先使用平台原生后端，失败再回退 ``CAP_ANY``。"""
        if self._is_camera:
            for api in camera_apis():
                capture = cv2.VideoCapture(self._resolved, api)
                if capture.isOpened():
                    return capture
                capture.release()
        return cv2.VideoCapture(self._resolved)

    # -- 只读视图 --------------------------------------------------------
    @property
    def is_camera(self) -> bool:
        return self._is_camera

    @property
    def resolved(self) -> Union[str, int]:
        return self._resolved

    @property
    def name(self) -> str:
        if self._is_camera:
            return "camera"
        if self._is_url:
            return str(self._resolved)
        return Path(str(self._resolved)).name

    @property
    def width(self) -> int:
        return int(round(self._capture.get(cv2.CAP_PROP_FRAME_WIDTH)))

    @property
    def height(self) -> int:
        return int(round(self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    @property
    def fps(self) -> float:
        value = float(self._capture.get(cv2.CAP_PROP_FPS))
        if not np.isfinite(value) or value <= 1e-3:
            return 30.0
        return min(value, 240.0)

    @property
    def frame_count(self) -> int:
        value = int(self._capture.get(cv2.CAP_PROP_FRAME_COUNT))
        return max(0, value)

    @property
    def frames_read(self) -> int:
        return self._frame_count

    # -- io --------------------------------------------------------------
    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """读下一帧；启用 ``loop`` 时在结束时回绕一次。"""
        assert self._capture is not None
        ok, frame = self._capture.read()
        if not ok and self.loop:
            self._capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = self._capture.read()
        if ok:
            self._frame_count += 1
        return bool(ok), frame

    def rewind(self) -> bool:
        """回到第一帧（仅文件源）。"""
        if self._is_camera or self._is_url or self._capture is None:
            return False
        return bool(self._capture.set(cv2.CAP_PROP_POS_FRAMES, 0))

    def release(self) -> None:
        if self._capture is not None:
            self._capture.release()
            self._capture = None

    def __enter__(self) -> "VideoSource":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.release()


class UrlSource:
    """网络流（RTSP / RTMP / HTTP）源：断流自动重连（实现 ``FrameSource``）。

    Args:
        spec: 网络流地址，如 ``rtsp://user:pass@host:554/stream``。
        max_retries: 连续重连失败的最大次数；超过后 ``read()`` 恒返回 ``(False, None)``。
        backoff: 重连退避基数（秒），第 n 次退避 ``min(backoff * n, 5.0)``。

    Raises:
        SourceError: 地址不是网络流，或首次打开失败。
    """

    def __init__(self, spec: str, *, max_retries: int = 5, backoff: float = 0.5) -> None:
        if not is_url(spec):
            raise SourceError(f"not a network stream URL: {spec!r}", field="source", value=spec)
        self._spec = str(spec).strip()
        self._max_retries = max(0, int(max_retries))
        self._backoff = max(0.0, float(backoff))
        self._retries = 0
        self._frames_read = 0
        self._capture: Optional[cv2.VideoCapture] = None
        self._open()

    def _open(self) -> None:
        capture = cv2.VideoCapture(self._spec)
        if not capture.isOpened():
            capture.release()
            raise SourceError(
                f"could not open network stream {self._spec!r}. Check the URL and credentials, "
                "and that OpenCV has the matching backend (FFMPEG/GStreamer).",
                field="source",
                value=self._spec,
            )
        self._capture = capture

    # -- 只读视图 --------------------------------------------------------
    @property
    def name(self) -> str:
        return self._spec

    @property
    def fps(self) -> float:
        if self._capture is None:
            return 30.0
        value = float(self._capture.get(cv2.CAP_PROP_FPS))
        if not np.isfinite(value) or value <= 1e-3:
            return 30.0
        return min(value, 240.0)

    @property
    def frames_read(self) -> int:
        return self._frames_read

    # -- io --------------------------------------------------------------
    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """读一帧；断流时按退避重连，连续失败 ``max_retries`` 次后放弃。"""
        if self._capture is None and not self._reconnect():
            return False, None
        assert self._capture is not None
        ok, frame = self._capture.read()
        if ok:
            self._retries = 0
            self._frames_read += 1
            return True, frame
        self._capture.release()
        self._capture = None
        return False, None

    def _reconnect(self) -> bool:
        if self._retries >= self._max_retries:
            return False
        self._retries += 1
        time.sleep(min(self._backoff * self._retries, 5.0))
        capture = cv2.VideoCapture(self._spec)
        if not capture.isOpened():
            capture.release()
            return False
        self._capture = capture
        return True

    def release(self) -> None:
        if self._capture is not None:
            self._capture.release()
            self._capture = None

    def __enter__(self) -> "UrlSource":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.release()


def open_source(spec: SourceSpec, loop: bool = False) -> FrameSource:
    """按 ``spec`` 自动选择本地/摄像头源（``VideoSource``）或网络流源（``UrlSource``）。"""
    if is_url(spec):
        return UrlSource(str(spec))
    return VideoSource(spec, loop=loop)


__all__ = [
    "URL_SCHEMES",
    "SourceSpec",
    "UrlSource",
    "VideoSource",
    "camera_apis",
    "is_url",
    "open_source",
    "resolve_source",
]
