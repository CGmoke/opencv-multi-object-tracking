"""Pure-OpenCV multi-object tracking framework.

Public API::

    from mot import MOTPipeline, TrackManager, make_tracker_factory, Detection

    manager = TrackManager(make_tracker_factory("csrt"), iou_threshold=0.3)
    pipeline = MOTPipeline(manager)

    pipeline.add_manual_box(frame, (x, y, w, h))
    result = pipeline.step(frame)          # one tracker update per target
    for track in result.tracks:
        print(track.track_id, track.bbox, track.trail)

The command line entry point is ``mot-track`` (see :mod:`mot.app`).
"""

from __future__ import annotations

from .models import (
    Detection,
    IdAllocator,
    StepResult,
    TrackedObject,
    TrackState,
    bbox_area,
    bbox_center,
    bbox_contains,
    clip_bbox,
    normalize_bbox,
)
from .pipeline import MOTPipeline
from .track_manager import TrackManager
from .trackers import (
    SUPPORTED_BACKENDS,
    TrackerBackendError,
    create_tracker,
    make_tracker_factory,
    tracker_available,
)
from .trajectory import CsvTrajectoryWriter, TrackPoint, TrajectoryStore

__version__ = "0.1.0"

__all__ = [
    "CsvTrajectoryWriter",
    "Detection",
    "IdAllocator",
    "MOTPipeline",
    "SUPPORTED_BACKENDS",
    "StepResult",
    "TrackManager",
    "TrackPoint",
    "TrackState",
    "TrackedObject",
    "TrackerBackendError",
    "TrajectoryStore",
    "bbox_area",
    "bbox_center",
    "bbox_contains",
    "clip_bbox",
    "create_tracker",
    "make_tracker_factory",
    "normalize_bbox",
    "tracker_available",
]
