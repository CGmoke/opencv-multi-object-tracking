"""Core data structures and validation helpers.

The pipeline is deliberately tracker-agnostic: any object that follows the
OpenCV tracker contract (``init(frame, bbox)`` / ``update(frame)``) can be
injected, which keeps the association and identity-lifecycle logic unit
testable without OpenCV.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Deque, Dict, Optional, Sequence, Tuple

BBox = Tuple[float, float, float, float]
Point = Tuple[float, float]
Color = Tuple[int, int, int]

DEFAULT_LABEL = "object"

#: Minimum side length, in pixels, accepted for any bounding box.
MIN_BOX_SIDE = 1.0


def normalize_bbox(bbox: Sequence[float]) -> BBox:
    """Validate ``(x, y, w, h)`` and return it with positive width/height.

    Negative width/height are interpreted as a box drawn from the opposite
    corner (which is what a mouse drag can produce) and are flipped.

    Raises:
        ValueError: if the values are not four finite numbers, or the box is
            smaller than :data:`MIN_BOX_SIDE` in either dimension.
    """
    values = tuple(float(v) for v in bbox)
    if len(values) != 4:
        raise ValueError(f"bbox must be (x, y, w, h), got {len(values)} values")
    if not all(math.isfinite(v) for v in values):
        raise ValueError(f"bbox contains non-finite values: {values}")

    x, y, w, h = values
    if w < 0:
        x, w = x + w, -w
    if h < 0:
        y, h = y + h, -h
    if w < MIN_BOX_SIDE or h < MIN_BOX_SIDE:
        raise ValueError(
            f"bbox must be at least {MIN_BOX_SIDE:g}x{MIN_BOX_SIDE:g} pixels, got {values}"
        )
    return (x, y, w, h)


def bbox_center(bbox: Sequence[float]) -> Point:
    """Return the center point of a bounding box."""
    x, y, w, h = bbox
    return (x + w / 2.0, y + h / 2.0)


def bbox_area(bbox: Sequence[float]) -> float:
    """Return the area of a bounding box, never negative."""
    _, _, w, h = bbox
    return max(0.0, float(w)) * max(0.0, float(h))


def bbox_contains(bbox: Sequence[float], point: Point) -> bool:
    """Return ``True`` when ``point`` lies inside the box (edges included)."""
    x, y, w, h = bbox
    px, py = point
    return x <= px <= x + w and y <= py <= y + h


def clip_bbox(bbox: Sequence[float], width: int, height: int) -> Optional[BBox]:
    """Clip a bounding box to a ``width x height`` frame.

    Returns:
        The clipped box, or ``None`` when nothing of it remains inside the
        frame (which is how a target leaving the picture is detected).
    """
    if width <= 0 or height <= 0:
        raise ValueError(f"frame size must be positive, got {width}x{height}")
    x, y, w, h = bbox
    x1 = max(0.0, float(x))
    y1 = max(0.0, float(y))
    x2 = min(float(width), float(x) + float(w))
    y2 = min(float(height), float(y) + float(h))
    if x2 - x1 < MIN_BOX_SIDE or y2 - y1 < MIN_BOX_SIDE:
        return None
    return (x1, y1, x2 - x1, y2 - y1)


class TrackState(str, Enum):
    """Identity lifecycle states kept by :class:`~mot.track_manager.TrackManager`."""

    ACTIVE = "active"
    LOST = "lost"


@dataclass(frozen=True)
class Detection:
    """One observed target, produced by a detector or by an interactive box."""

    bbox: BBox
    confidence: float = 1.0
    label: str = DEFAULT_LABEL

    def __post_init__(self) -> None:
        object.__setattr__(self, "bbox", normalize_bbox(self.bbox))
        if not math.isfinite(self.confidence):
            raise ValueError(f"confidence must be finite, got {self.confidence}")
        object.__setattr__(self, "confidence", float(min(1.0, max(0.0, self.confidence))))


@dataclass
class TrackedObject:
    """A live target: one unique id, one tracker, one bbox and its trail."""

    track_id: int
    tracker: object
    bbox: BBox
    label: str = DEFAULT_LABEL
    confidence: float = 1.0
    state: TrackState = TrackState.ACTIVE
    hits: int = 0
    misses: int = 0
    age: int = 0
    created_frame: int = 0
    last_seen_frame: int = 0
    trail: Deque[Point] = field(default_factory=deque)

    @property
    def center(self) -> Point:
        return bbox_center(self.bbox)

    @property
    def is_active(self) -> bool:
        return self.state is TrackState.ACTIVE

    def push_trail(self, point: Point, max_points: int) -> None:
        """Append a trail point, keeping at most ``max_points`` samples."""
        if self.trail and self.trail[-1] == point:
            return
        self.trail.append(point)
        while max_points > 0 and len(self.trail) > max_points:
            self.trail.popleft()


@dataclass
class StepResult:
    """Bookkeeping returned by one :meth:`TrackManager.step` call."""

    tracks: Tuple[TrackedObject, ...] = ()
    created: Tuple[int, ...] = ()
    removed: Tuple[int, ...] = ()
    matched: Tuple[Tuple[int, int], ...] = ()
    spawn_failures: int = 0


class IdAllocator:
    """Monotonic id source.

    Ids are never recycled during a session, so a csv file stays unambiguous
    even when a target disappears and a new one appears in the same place.
    """

    def __init__(self, start: int = 1) -> None:
        if int(start) < 1:
            raise ValueError(f"start id must be >= 1, got {start}")
        self._next = int(start)

    def __call__(self) -> int:
        track_id = self._next
        self._next += 1
        return track_id

    @property
    def next_id(self) -> int:
        return self._next

    def reset(self, start: int = 1) -> None:
        if int(start) < 1:
            raise ValueError(f"start id must be >= 1, got {start}")
        self._next = int(start)

    def __len__(self) -> int:
        return self._next - 1


#: Frame type alias; frames are opaque to this package's pure-logic layer.
Frame = object


@dataclass
class AppStats:
    """Cumulative statistics, printed when the application shuts down."""

    frames: int = 0
    created_ids: int = 0
    removed_ids: int = 0
    spawn_failures: int = 0
    trail_points: int = 0
    started_at_frame: int = 0

    def as_dict(self) -> Dict[str, int]:
        return {
            "frames": self.frames,
            "created_ids": self.created_ids,
            "removed_ids": self.removed_ids,
            "spawn_failures": self.spawn_failures,
            "trail_points": self.trail_points,
        }
