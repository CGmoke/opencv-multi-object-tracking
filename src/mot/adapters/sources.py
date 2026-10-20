"""适配层：视频输入与源解析（实现 ``domain.ports.FrameSource``）。

``VideoSource`` 隐藏“摄像头索引 vs 视频文件”的差异，并在打开前校验源，让用户得到
清晰错误而不是一片黑窗。本模块是唯一允许 import ``cv2`` 的层之一。
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple, Union

import cv2
import numpy as np

from mot.domain.errors import SourceError

SourceSpec = Union[str, int]


def resolve_source(spec: SourceSpec) -> Union[str, int]:
    """把 CLI 字符串映射为摄像头索引或绝对文件路径。

    Raises:
        SourceError: 输入为空、索引为负，或路径不存在/非普通文件。
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
        self.loop = bool(loop) and not self._is_camera
        self._capture: Optional[cv2.VideoCapture] = cv2.VideoCapture(self._resolved)
        if not self._capture.isOpened():
            raise SourceError(
                f"could not open video source {self._resolved!r}. "
                "For a camera, check the index and permissions; for a file, check the path "
                "and that OpenCV has the matching codec (FFMPEG/GStreamer).",
                field="source",
                value=self._resolved,
            )
        self._frame_count = 0

    # -- 只读视图 --------------------------------------------------------
    @property
    def is_camera(self) -> bool:
        return self._is_camera

    @property
    def resolved(self) -> Union[str, int]:
        return self._resolved

    @property
    def name(self) -> str:
        return "camera" if self._is_camera else Path(str(self._resolved)).name

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
        if self._is_camera or self._capture is None:
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


__all__ = ["SourceSpec", "VideoSource", "resolve_source"]
