"""Identity lifecycle: spawning, updating, associating and expiring tracks.

This is the heart of the project. One tracker instance per target; every frame
each tracker is advanced, and new detections (a mouse-drawn box or a detector
output) are associated to existing ids by IoU. A track that the tracker loses
keeps its id for ``max_missing`` frames so it can recover, then its id is
destroyed and a later detection gets a new one.

The manager has no OpenCV import: trackers are injected as factories, which
lets the whole lifecycle be tested with fake trackers.
"""

from __future__ import annotations

import logging
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from .iou import iou_matrix
from .matching import MATCH_METHODS, match
from .models import (
    DEFAULT_LABEL,
    BBox,
    Detection,
    IdAllocator,
    StepResult,
    TrackedObject,
    TrackState,
    bbox_center,
    clip_bbox,
    normalize_bbox,
)
from .trackers import TrackerFactory, init_tracker, update_tracker

LOGGER = logging.getLogger(__name__)


class TrackManager:
    """Owns all live targets and their unique ids.

    Args:
        tracker_factory: zero-argument callable returning a fresh tracker that
            implements ``init(frame, bbox)`` / ``update(frame)``.
        iou_threshold: minimum IoU for a detection to be associated with an
            existing id. Also used to suppress duplicate boxes on one target.
        max_missing: frames a lost track may stay alive before its id is
            destroyed. ``0`` means "destroy immediately when the tracker fails".
        trail_length: maximum trail points kept per target.
        match_method: ``"greedy"`` (default) or ``"hungarian"``.
        id_allocator: optional shared allocator; ids are monotonic.
        max_tracks: hard cap on simultaneously live ids; detections beyond the
            cap are refused (counted in ``spawn_failures``) instead of growing
            memory without bound.
    """

    def __init__(
        self,
        tracker_factory: TrackerFactory,
        iou_threshold: float = 0.3,
        max_missing: int = 15,
        trail_length: int = 120,
        match_method: str = "greedy",
        id_allocator: Optional[IdAllocator] = None,
        max_tracks: int = 256,
    ) -> None:
        if not 0.0 < float(iou_threshold) <= 1.0:
            raise ValueError(
                f"iou_threshold must be within (0, 1], got {iou_threshold} "
                "(0 would let any detection hijack an existing id)"
            )
        if int(max_missing) < 0:
            raise ValueError(f"max_missing must be >= 0, got {max_missing}")
        if int(trail_length) < 1:
            raise ValueError(f"trail_length must be >= 1, got {trail_length}")
        if int(max_tracks) < 1:
            raise ValueError(f"max_tracks must be >= 1, got {max_tracks}")
        if match_method not in MATCH_METHODS:
            raise ValueError(f"unknown match_method {match_method!r}; expected {MATCH_METHODS}")

        self._tracker_factory = tracker_factory
        self.iou_threshold = float(iou_threshold)
        self.max_missing = int(max_missing)
        self.trail_length = int(trail_length)
        self.match_method = match_method
        self.max_tracks = int(max_tracks)
        self._capacity_warned = False
        self._allocator = id_allocator or IdAllocator()
        self._tracks: Dict[int, TrackedObject] = {}
        self.frame_index = 0
        self.total_created = 0
        self.total_removed = 0
        self.spawn_failures = 0

    # -- read-only views -------------------------------------------------
    @property
    def tracks(self) -> Tuple[TrackedObject, ...]:
        """Active and lost tracks, ordered by id (stable for rendering)."""
        return tuple(self._tracks[key] for key in sorted(self._tracks))

    @property
    def active_tracks(self) -> Tuple[TrackedObject, ...]:
        return tuple(t for t in self.tracks if t.is_active)

    @property
    def allocator(self) -> IdAllocator:
        return self._allocator

    def get(self, track_id: int) -> Optional[TrackedObject]:
        return self._tracks.get(int(track_id))

    def __len__(self) -> int:
        return len(self._tracks)

    def __contains__(self, track_id: object) -> bool:
        return isinstance(track_id, int) and track_id in self._tracks

    # -- lifecycle -------------------------------------------------------
    def step(
        self, frame: object, detections: Sequence[Detection] = ()
    ) -> StepResult:
        """Advance every tracker by one frame, then absorb new detections.

        ``detections`` are observations for the current frame (detector output
        or interactive boxes). They are associated to live ids by IoU; a match
        refreshes the box and re-initialises that target's tracker, an
        unmatched detection becomes a new id.

        Returns:
            A :class:`StepResult` describing the frame, including the ids that
            were created and destroyed.
        """
        self.frame_index += 1
        frame_index = self.frame_index
        created: List[int] = []
        removed: List[int] = []
        spawn_failures = 0

        pairs: List[Tuple[int, int]] = []
        updated_ids: set = set()
        # 1. Advance existing trackers.
        for track_id in list(self._tracks):
            track = self._tracks[track_id]
            track.age += 1
            box = update_tracker(track.tracker, frame)
            if box is not None:
                box = self._clip_to_frame(box, frame)
            if box is not None:
                self._accept_box(track, box, frame_index)
                updated_ids.add(track_id)
            else:
                track.misses += 1
                track.state = TrackState.LOST
                if track.misses > self.max_missing:
                    del self._tracks[track_id]
                    self.total_removed += 1
                    removed.append(track_id)

        # 2. Associate the current detections with the surviving ids.
        detections = tuple(detections)
        if detections:
            pairs, _unmatched_tracks, unmatched_detections = self._associate(detections)
            snapshot = self.tracks
            for track_idx, det_idx in pairs:
                track = snapshot[track_idx]
                detection = detections[det_idx]
                self._refresh(track, detection, frame)
                track.state = TrackState.ACTIVE
                # The tracker already counted a successful update this frame;
                # a detection only adds a hit when it rescues a lost target.
                if track.track_id not in updated_ids:
                    track.hits += 1
                track.misses = 0
                track.last_seen_frame = frame_index
            for det_idx in unmatched_detections:
                detection = detections[det_idx]
                track = self._spawn(detection, frame, frame_index)
                if track is None:
                    spawn_failures += 1  # _spawn already updated the total
                else:
                    created.append(track.track_id)

        return StepResult(
            tracks=self.tracks,
            created=tuple(created),
            removed=tuple(removed),
            matched=tuple(pairs),
            spawn_failures=spawn_failures,
        )

    def add_or_refresh(
        self, frame: object, detection: Detection
    ) -> Optional[TrackedObject]:
        """Add a manual box, or refresh the id it overlaps.

        Clicking twice on the same object must not create a second identity:
        when the new box overlaps a live track by at least ``iou_threshold``,
        that track is refreshed instead.
        """
        if len(self._tracks) == 0:
            return self._spawn(detection, frame, self.frame_index)

        pairs, _unmatched_tracks, unmatched = self._associate([detection])
        if pairs:
            track = self.tracks[pairs[0][0]]
            self._refresh(track, detection, frame)
            track.hits += 1
            track.misses = 0
            track.state = TrackState.ACTIVE
            track.last_seen_frame = self.frame_index
            return track
        return self._spawn(detection, frame, self.frame_index)

    def remove(self, track_id: int) -> bool:
        """Destroy one id immediately (interactive delete)."""
        track_id = int(track_id)
        if track_id not in self._tracks:
            return False
        del self._tracks[track_id]
        self.total_removed += 1
        return True

    def clear_trails(self) -> None:
        """Drop trail history but keep identities and live trackers."""
        for track in self._tracks.values():
            track.trail.clear()

    def reset(self, start_id: Optional[int] = None) -> None:
        """Destroy every track.

        Numbering continues by default, so ids stay unique within one csv /
        session. Pass ``start_id`` to restart numbering explicitly (the demo
        never does; that would make two different targets share a track id).
        """
        self._tracks.clear()
        self._capacity_warned = False
        self.frame_index = 0
        if start_id is not None:
            self._allocator.reset(start_id)
        self.total_created = 0
        self.total_removed = 0
        self.spawn_failures = 0

    # -- internals -------------------------------------------------------
    @staticmethod
    def _clip_to_frame(box: BBox, frame: object) -> Optional[BBox]:
        """Drop (part of) a tracker box that left the frame.

        A tracker happily reports coordinates outside the picture; keeping
        them would write nonsense rows to the csv and draw trails off-screen.
        """
        shape = getattr(frame, "shape", None)
        if shape is None or len(shape) < 2:
            return box  # fake frame in unit tests: no size information
        height, width = int(shape[0]), int(shape[1])
        if width <= 0 or height <= 0:
            return box
        return clip_bbox(box, width, height)

    def _accept_box(self, track: TrackedObject, box: BBox, frame_index: int) -> None:
        track.bbox = box
        track.state = TrackState.ACTIVE
        track.hits += 1
        track.misses = 0
        track.last_seen_frame = frame_index
        track.push_trail(bbox_center(box), self.trail_length)

    def _refresh(self, track: TrackedObject, detection: Detection, frame: object) -> None:
        """Re-initialise a target's tracker with a fresh observation box."""
        track.bbox = detection.bbox
        track.label = detection.label
        track.confidence = detection.confidence
        init_tracker(track.tracker, frame, detection.bbox)
        track.push_trail(bbox_center(detection.bbox), self.trail_length)

    def _spawn(
        self, detection: Detection, frame: object, frame_index: int
    ) -> Optional[TrackedObject]:
        if len(self._tracks) >= self.max_tracks:
            if not self._capacity_warned:
                LOGGER.warning(
                    "max_tracks=%d reached; refusing to create more ids", self.max_tracks
                )
                self._capacity_warned = True
            return None
        try:
            tracker = self._tracker_factory()
            init_tracker(tracker, frame, detection.bbox)
        except Exception:  # noqa: BLE001 - a bad detection must not kill the loop
            self.spawn_failures += 1
            return None

        track = TrackedObject(
            track_id=self._allocator(),
            tracker=tracker,
            bbox=normalize_bbox(detection.bbox),
            label=detection.label or DEFAULT_LABEL,
            confidence=detection.confidence,
            hits=1,
            created_frame=frame_index,
            last_seen_frame=frame_index,
        )
        track.push_trail(bbox_center(track.bbox), self.trail_length)
        self._tracks[track.track_id] = track
        self.total_created += 1
        return track

    def _associate(
        self, detections: Sequence[Detection]
    ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        track_boxes = [track.bbox for track in self.tracks]
        det_boxes = [detection.bbox for detection in detections]
        overlap = iou_matrix(track_boxes, det_boxes)
        pairs, unmatched_tracks, unmatched_dets = match(
            overlap, threshold=self.iou_threshold, method=self.match_method
        )
        return pairs, unmatched_tracks, unmatched_dets


def detections_from_boxes(boxes: Iterable[Sequence[float]]) -> List[Detection]:
    """Turn raw boxes into detections (used for interactive clicks and tests)."""
    return [Detection(bbox=normalize_bbox(box)) for box in boxes]


__all__ = ["TrackManager", "detections_from_boxes"]
