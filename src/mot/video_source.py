"""Video input/output.

``VideoSource`` hides the difference between a camera index and a video file,
and validates the source up front so the user gets a clear error instead of a
silent black window.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Tuple, Union

import cv2
import numpy as np

SourceSpec = Union[str, int]


class VideoSourceError(RuntimeError):
    """Raised when a camera or video file cannot be opened."""


def resolve_source(spec: SourceSpec) -> Union[str, int]:
    """Map a CLI string to a camera index or an absolute file path.

    Raises:
        VideoSourceError: for empty input, a negative index, or a path that
            does not exist.
    """
    if isinstance(spec, int):
        if spec < 0:
            raise VideoSourceError(f"camera index must be >= 0, got {spec}")
        return spec

    text = str(spec).strip()
    if not text:
        raise VideoSourceError("video source is empty; pass a camera index or a video path")
    if text.isdigit():
        return int(text)

    path = Path(text).expanduser()
    if not path.exists():
        raise VideoSourceError(f"video file does not exist: {path}")
    if not path.is_file():
        raise VideoSourceError(f"video source is not a regular file: {path}")
    return str(path.resolve())


class VideoSource:
    """Context-managed ``cv2.VideoCapture`` wrapper.

    Args:
        spec: camera index (``0`` / ``"0"``) or path to a video file.
        loop: restart a file when it ends (ignored for cameras).

    Example:
        >>> with VideoSource("assets/demo.mp4") as source:  # doctest: +SKIP
        ...     ok, frame = source.read()
    """

    def __init__(self, spec: SourceSpec, loop: bool = False) -> None:
        self._resolved = resolve_source(spec)
        self._is_camera = isinstance(self._resolved, int)
        self.loop = bool(loop) and not self._is_camera
        self._capture = cv2.VideoCapture(self._resolved)
        if not self._capture.isOpened():
            raise VideoSourceError(
                f"could not open video source {self._resolved!r}. "
                "For a camera, check the index and permissions; for a file, check the path "
                "and that OpenCV has the matching codec (FFMPEG/GStreamer)."
            )
        self._frame_count = 0

    # -- properties ------------------------------------------------------
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
        """Read the next frame; rewinds once when ``loop`` is enabled."""
        ok, frame = self._capture.read()
        if not ok and self.loop:
            self._capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = self._capture.read()
        if ok:
            self._frame_count += 1
        return bool(ok), frame

    def rewind(self) -> bool:
        """Go back to the first frame (file sources only)."""
        if self._is_camera:
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


class VideoWriterSink:
    """Best-effort mp4/avi writer for the annotated stream.

    Several fourcc codecs are probed because headless OpenCV builds differ in
    what they support; the first one that opens wins.
    """

    CODECS = ("mp4v", "avc1", "XVID")

    def __init__(self, path: str, fps: float, frame_size: Tuple[int, int]) -> None:
        target = Path(os.fspath(path)).expanduser()
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.is_symlink():
            raise VideoSourceError(f"refusing to record through a symlink: {target}")
        if target.exists() and target.is_dir():
            raise VideoSourceError(f"output video path is a directory: {target}")

        fourcc_candidates = list(self.CODECS)
        suffix = target.suffix.lower()
        if suffix == ".avi":
            fourcc_candidates = ["XVID", "MJPG", *fourcc_candidates]
        width, height = int(frame_size[0]), int(frame_size[1])
        if width <= 0 or height <= 0:
            raise VideoSourceError(f"invalid frame size for writer: {frame_size}")

        last_error: Optional[Exception] = None
        self._writer: Optional[cv2.VideoWriter] = None
        for fourcc in fourcc_candidates:
            try:
                writer = cv2.VideoWriter(
                    str(target), cv2.VideoWriter_fourcc(*fourcc), float(fps), (width, height)
                )
            except Exception as exc:  # noqa: BLE001 - try the next codec
                last_error = exc
                continue
            if writer.isOpened():
                self._writer = writer
                self.path = target
                self.codec = fourcc
                break
            writer.release()
        if self._writer is None:
            raise VideoSourceError(
                f"could not open video writer for {target} (tried {fourcc_candidates})"
                + (f": {last_error}" if last_error else "")
            )

    def write(self, frame: np.ndarray) -> None:
        if self._writer is not None:
            self._writer.write(frame)

    def release(self) -> None:
        if self._writer is not None:
            self._writer.release()
            self._writer = None

    def __enter__(self) -> "VideoWriterSink":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.release()


__all__ = [
    "SourceSpec",
    "VideoSourceError",
    "VideoSource",
    "VideoWriterSink",
    "resolve_source",
]
