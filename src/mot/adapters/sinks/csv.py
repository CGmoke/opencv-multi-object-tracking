"""适配层：轨迹 CSV 落盘（实现 ``domain.ports.TrajectorySink``）。

列固定为 ``track_id,frame,timestamp,x,y,w,h,cx,cy,label``，可直接被 pandas /
matplotlib / 机器人日志管线读取。标签在写入前清洗，使 CSV 在电子表格中打开也不会
执行公式；写盘前拒绝符号链接（CWE-59）。
"""

from __future__ import annotations

import csv
import os
import tempfile
from pathlib import Path
from typing import Iterable, Optional, Sequence, Tuple

from mot.domain.errors import SinkError
from mot.domain.models import TrackedObject, normalize_bbox
from mot.domain.records import TrackPoint

CSV_HEADER: Tuple[str, ...] = (
    "track_id",
    "frame",
    "timestamp",
    "x",
    "y",
    "w",
    "h",
    "cx",
    "cy",
    "label",
)

#: 电子表格会把以这些字符开头的单元格当公式执行；来自检测器或用户输入的标签
#: 绝不能未经转义进入 CSV。
_FORMULA_PREFIXES = ("=", "+", "-", "@", "\t", "\r")

#: 电子表格会忽略却仍把后随字符当前缀的不可见码点（BOM / 零宽）。
_INVISIBLE_CHARS = ("\ufeff", "\u200b", "\u200c", "\u200d", "\u2060", "\u180e")


def sanitize_label(label: str, max_length: int = 64) -> str:
    """让标签对 CSV / 电子表格安全并有界。"""
    text = str(label)
    for invisible in _INVISIBLE_CHARS:
        text = text.replace(invisible, "")
    text = text.replace("\r", " ").replace("\n", " ").strip()
    if not text:
        return "object"
    if text.startswith(_FORMULA_PREFIXES):
        text = "'" + text
    return text[:max_length]


def _row_from_point(point: TrackPoint) -> Tuple[object, ...]:
    """把轨迹点格式化为一行 CSV 值。"""
    cx, cy = point.center
    x, y, w, h = point.bbox
    return (
        point.track_id,
        point.frame,
        f"{point.timestamp:.6f}",
        f"{x:.3f}",
        f"{y:.3f}",
        f"{w:.3f}",
        f"{h:.3f}",
        f"{cx:.3f}",
        f"{cy:.3f}",
        sanitize_label(point.label),
    )


class CsvTrajectoryWriter:
    """把轨迹行流式写入 CSV，默认每帧 flush。

    写入由 csv 模块缓冲但每帧 flush，因此 Ctrl+C 也会留下可用文件。写盘前校验目标
    路径，拒绝符号链接。
    """

    def __init__(self, path: str | Path, append: bool = False, flush_every: int = 1) -> None:
        self.path = Path(os.fspath(path)).expanduser()
        self.append = bool(append)
        self._rows_written = 0
        self._flush_every = max(1, int(flush_every))
        self._handle: Optional[object] = None
        self._writer: Optional[object] = None
        self._ensure_parent()
        self._open()

    # -- 生命周期 --------------------------------------------------------
    def _ensure_parent(self) -> None:
        parent = self.path.parent
        try:
            parent.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise SinkError(f"cannot create csv directory {parent}: {exc}") from exc

    def _open(self) -> None:
        if self.path.is_symlink():
            raise SinkError(f"refusing to write trajectories through a symlink: {self.path}")
        write_header = not self.append or not self.path.exists() or self.path.stat().st_size == 0
        mode = "a" if self.append else "w"
        self._handle = self.path.open(mode, newline="", encoding="utf-8")
        self._writer = csv.writer(self._handle, lineterminator="\n")
        if write_header:
            self._writer.writerow(CSV_HEADER)

    def write_frame(
        self, frame_index: int, timestamp: float, tracks: Sequence[TrackedObject]
    ) -> int:
        """为每个存活目标追加一行并返回写入的行数。"""
        if self._writer is None:
            raise SinkError("csv writer is closed")
        rows = 0
        for track in tracks:
            point = TrackPoint(
                track_id=track.track_id,
                frame=int(frame_index),
                timestamp=float(timestamp),
                bbox=normalize_bbox(track.bbox),
                label=track.label,
            )
            self._writer.writerow(_row_from_point(point))
            rows += 1
        self._rows_written += rows
        if rows and self._rows_written % self._flush_every == 0:
            self.flush()
        return rows

    def write_points(self, points: Iterable[TrackPoint]) -> int:
        """追加已物化的轨迹点（回退/重放流时使用）。"""
        if self._writer is None:
            raise SinkError("csv writer is closed")
        rows = 0
        for point in points:
            self._writer.writerow(_row_from_point(point))
            rows += 1
        self._rows_written += rows
        self.flush()
        return rows

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
                self._writer = None

    @property
    def rows_written(self) -> int:
        return self._rows_written

    def __enter__(self) -> "CsvTrajectoryWriter":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


def write_points_atomic(path: str | Path, points: Iterable[TrackPoint]) -> Path:
    """原子地把完整轨迹写入文件（临时文件 + rename）。"""
    target = Path(os.fspath(path)).expanduser()
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=target.name + ".", dir=str(target.parent))
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle, lineterminator="\n")
            writer.writerow(CSV_HEADER)
            for point in points:
                writer.writerow(_row_from_point(point))
        os.replace(tmp_path, target)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise
    return target


__all__ = [
    "CSV_HEADER",
    "CsvTrajectoryWriter",
    "sanitize_label",
    "write_points_atomic",
]
