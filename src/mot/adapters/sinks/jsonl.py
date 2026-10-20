"""适配层：把每帧事件写成 JSON Lines（实现 ``domain.ports.EventSink``）。

每行一个 JSON 对象，便于 ``jq`` / pandas / 日志管线消费：

    {"frame": 42, "timestamp": 2.1, "created": [3], "removed": [], "tracks": [...]}

与 CSV sink 一样拒绝符号链接目标（CWE-59），并支持 ``append`` 与按行 flush。
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Optional

from mot.domain.errors import SinkError


class JsonlEventSink:
    """把 ``FrameResult`` 逐帧写成 JSON Lines 文件。

    Args:
        path: 输出文件路径。
        append: 追加写（默认覆盖写）。
        flush_every: 每写多少行 flush 一次（默认 1）。
        only_active: 只记录 ``ACTIVE`` 状态的目标（默认 ``True``）。
        skip_empty: 无目标且无新建 / 销毁事件时不写该帧（默认 ``True``，避免空转噪声）。
    """

    def __init__(
        self,
        path: str | Path,
        append: bool = False,
        flush_every: int = 1,
        only_active: bool = True,
        skip_empty: bool = True,
    ) -> None:
        self.path = Path(os.fspath(path)).expanduser()
        self.append = bool(append)
        self._flush_every = max(1, int(flush_every))
        self._only_active = bool(only_active)
        self._skip_empty = bool(skip_empty)
        self._rows_written = 0
        self._handle: Optional[Any] = None
        self._ensure_parent()
        self._open()

    # -- 生命周期 --------------------------------------------------------
    def _ensure_parent(self) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise SinkError(f"cannot create jsonl directory {self.path.parent}: {exc}") from exc

    def _open(self) -> None:
        if self.path.is_symlink():
            raise SinkError(f"refusing to write events through a symlink: {self.path}")
        mode = "a" if self.append else "w"
        self._handle = self.path.open(mode, encoding="utf-8")

    def on_frame(self, result: Any) -> None:
        """记录一帧的事件（``result`` 为应用层 ``FrameResult``）。"""
        if self._handle is None:
            raise SinkError("jsonl sink is closed")
        tracks = result.tracks
        if self._only_active:
            tracks = tuple(track for track in tracks if track.is_active)
        created = tuple(result.created)
        removed = tuple(result.removed)
        if self._skip_empty and not tracks and not created and not removed:
            return
        record = {
            "frame": int(result.frame_index),
            "timestamp": float(result.timestamp),
            "created": list(created),
            "removed": list(removed),
            "tracks": [
                {
                    "id": int(track.track_id),
                    "label": track.label,
                    "bbox": [float(value) for value in track.bbox],
                }
                for track in tracks
            ],
        }
        self._handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._rows_written += 1
        if self._rows_written % self._flush_every == 0:
            self.flush()

    def flush(self) -> None:
        assert self._handle is not None
        self._handle.flush()

    def close(self) -> None:
        if self._handle is not None:
            try:
                self._handle.flush()
            finally:
                self._handle.close()
                self._handle = None

    @property
    def rows_written(self) -> int:
        return self._rows_written

    def __enter__(self) -> "JsonlEventSink":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


__all__ = ["JsonlEventSink"]
