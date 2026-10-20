"""适配层：标注视频输出（录像）。

尽力而为的 mp4/avi 写入器：不同无头 OpenCV 构建支持的 fourcc 不同，故依次探测，
第一个能打开者胜出。
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

from mot.domain.errors import SinkError


class VideoWriterSink:
    """把标注后的帧写入视频文件。"""

    CODECS = ("mp4v", "avc1", "XVID")

    def __init__(self, path: str, fps: float, frame_size: Tuple[int, int]) -> None:
        target = Path(os.fspath(path)).expanduser()
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.is_symlink():
            raise SinkError(f"refusing to record through a symlink: {target}")
        if target.exists() and target.is_dir():
            raise SinkError(f"output video path is a directory: {target}")

        fourcc_candidates = list(self.CODECS)
        suffix = target.suffix.lower()
        if suffix == ".avi":
            fourcc_candidates = ["XVID", "MJPG", *fourcc_candidates]
        width, height = int(frame_size[0]), int(frame_size[1])
        if width <= 0 or height <= 0:
            raise SinkError(f"invalid frame size for writer: {frame_size}")

        last_error: Optional[Exception] = None
        self._writer: Optional[cv2.VideoWriter] = None
        for fourcc in fourcc_candidates:
            try:
                writer = cv2.VideoWriter(
                    str(target), cv2.VideoWriter_fourcc(*fourcc), float(fps), (width, height)
                )
            except Exception as exc:  # noqa: BLE001 - 尝试下一个 codec
                last_error = exc
                continue
            if writer.isOpened():
                self._writer = writer
                self.path = target
                self.codec = fourcc
                break
            writer.release()
        if self._writer is None:
            raise SinkError(
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


__all__ = ["VideoWriterSink"]
