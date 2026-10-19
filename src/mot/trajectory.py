"""Trail storage and csv export.

Every confirmed tracker update contributes one row. The schema is stable so
the file can be loaded directly by pandas, matplotlib or a robotics log
pipeline::

    track_id,frame,timestamp,x,y,w,h,cx,cy,label

``timestamp`` is seconds from the start of the source. Labels are sanitised
before writing so a csv opened in a spreadsheet cannot execute a formula.
"""

from __future__ import annotations

import csv
import math
import os
import tempfile
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence, Tuple

from .models import BBox, TrackedObject, bbox_center, normalize_bbox

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

#: Spreadsheet applications execute cells starting with these characters. A
#: label coming from a detector or from user input must never reach a csv
#: unescaped.
_FORMULA_PREFIXES = ("=", "+", "-", "@", "\t", "\r")

#: Invisible code points that a spreadsheet ignores while still treating the
#: following character as the start of a cell (BOM / zero-width joiners). They
#: are removed before the prefix check, otherwise "\ufeff=1+1" reaches Excel
#: as a formula.
_INVISIBLE_CHARS = ("\ufeff", "\u200b", "\u200c", "\u200d", "\u2060", "\u180e")


def sanitize_label(label: str, max_length: int = 64) -> str:
    """Make a label safe for csv/spreadsheet consumers and bounded in size."""
    text = str(label)
    for invisible in _INVISIBLE_CHARS:
        text = text.replace(invisible, "")
    text = text.replace("\r", " ").replace("\n", " ").strip()
    if not text:
        return "object"
    if text.startswith(_FORMULA_PREFIXES):
        text = "'" + text
    return text[:max_length]


@dataclass(frozen=True)
class TrackPoint:
    """One sampled position of a target."""

    track_id: int
    frame: int
    timestamp: float
    bbox: BBox
    label: str = "object"

    @property
    def center(self) -> Tuple[float, float]:
        return bbox_center(self.bbox)

    def as_row(self) -> Tuple[object, ...]:
        cx, cy = self.center
        x, y, w, h = self.bbox
        return (
            self.track_id,
            self.frame,
            f"{self.timestamp:.6f}",
            f"{x:.3f}",
            f"{y:.3f}",
            f"{w:.3f}",
            f"{h:.3f}",
            f"{cx:.3f}",
            f"{cy:.3f}",
            sanitize_label(self.label),
        )


class TrajectoryStore:
    """In-memory trail history, keyed by id, bounded per target."""

    def __init__(self, max_points_per_track: int = 10000) -> None:
        if int(max_points_per_track) < 1:
            raise ValueError("max_points_per_track must be >= 1")
        self._max_points = int(max_points_per_track)
        self._points: "OrderedDict[int, list[TrackPoint]]" = OrderedDict()

    def add(self, track_id: int, frame: int, timestamp: float, bbox: Sequence[float], label: str = "object") -> TrackPoint:
        if not math.isfinite(timestamp) or timestamp < 0:
            raise ValueError(f"timestamp must be finite and >= 0, got {timestamp}")
        point = TrackPoint(
            track_id=int(track_id),
            frame=int(frame),
            timestamp=float(timestamp),
            bbox=normalize_bbox(bbox),
            label=label,
        )
        bucket = self._points.setdefault(point.track_id, [])
        bucket.append(point)
        if len(bucket) > self._max_points:
            del bucket[: len(bucket) - self._max_points]
        return point

    def points(self, track_id: int) -> Tuple[TrackPoint, ...]:
        return tuple(self._points.get(int(track_id), ()))

    def items(self) -> Iterable[Tuple[int, Tuple[TrackPoint, ...]]]:
        for track_id, points in self._points.items():
            yield track_id, tuple(points)

    def clear(self) -> None:
        self._points.clear()

    def __len__(self) -> int:
        return sum(len(points) for points in self._points.values())

    def __contains__(self, track_id: object) -> bool:
        return isinstance(track_id, int) and track_id in self._points


class CsvTrajectoryWriter:
    """Stream trail rows to a csv file, one flush per frame.

    Writes are buffered by the csv module but flushed each frame so a Ctrl+C
    still leaves a usable file. The writer validates the destination path and
    refuses to write into a directory that cannot be created.
    """

    def __init__(self, path: str | Path, append: bool = False, flush_every: int = 1) -> None:
        self.path = Path(os.fspath(path)).expanduser()
        self.append = bool(append)
        self._rows_written = 0
        self._flush_every = max(1, int(flush_every))
        self._handle: Optional[object] = None
        self._writer: Optional[csv.writer] = None
        self._ensure_parent()
        self._open()

    # -- lifecycle -------------------------------------------------------
    def _ensure_parent(self) -> None:
        parent = self.path.parent
        try:
            parent.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise OSError(f"cannot create csv directory {parent}: {exc}") from exc

    def _open(self) -> None:
        if self.path.is_symlink():
            raise OSError(f"refusing to write trajectories through a symlink: {self.path}")
        write_header = not self.append or not self.path.exists() or self.path.stat().st_size == 0
        mode = "a" if self.append else "w"
        self._handle = self.path.open(mode, newline="", encoding="utf-8")
        self._writer = csv.writer(self._handle, lineterminator="\n")
        if write_header:
            self._writer.writerow(CSV_HEADER)

    def write_frame(
        self, frame_index: int, timestamp: float, tracks: Iterable[TrackedObject]
    ) -> int:
        """Append one row per live track and return the number of rows written."""
        if self._writer is None:
            raise RuntimeError("csv writer is closed")
        rows = 0
        for track in tracks:
            point = TrackPoint(
                track_id=track.track_id,
                frame=int(frame_index),
                timestamp=float(timestamp),
                bbox=normalize_bbox(track.bbox),
                label=track.label,
            )
            self._writer.writerow(point.as_row())
            rows += 1
        self._rows_written += rows
        if rows and self._rows_written % self._flush_every == 0:
            self.flush()
        return rows

    def write_points(self, points: Iterable[TrackPoint]) -> int:
        """Append already materialised points (used when rewinding a stream)."""
        if self._writer is None:
            raise RuntimeError("csv writer is closed")
        rows = 0
        for point in points:
            self._writer.writerow(point.as_row())
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
    """Write a complete trajectory file atomically (temp file + rename)."""
    target = Path(os.fspath(path)).expanduser()
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=target.name + ".", dir=str(target.parent))
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle, lineterminator="\n")
            writer.writerow(CSV_HEADER)
            for point in points:
                writer.writerow(point.as_row())
        os.replace(tmp_path, target)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise
    return target


__all__ = [
    "CSV_HEADER",
    "CsvTrajectoryWriter",
    "TrackPoint",
    "TrajectoryStore",
    "sanitize_label",
    "write_points_atomic",
]
