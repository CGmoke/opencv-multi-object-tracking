"""Frame orchestration: optional detector hook + track manager + bookkeeping.

The pipeline is OpenCV-free; frames are opaque objects passed to the injected
trackers and detector. That keeps the complete association logic testable with
synthetic frames and fake trackers.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Sequence, Tuple

from .models import Detection, Point, TrackedObject, bbox_area, bbox_contains, normalize_bbox
from .track_manager import TrackManager

DetectorFn = Callable[[Any], Sequence[Detection]]

LOGGER = logging.getLogger(__name__)

#: Hard cap on detections absorbed per frame. A detector that returns thousands
#: of boxes must not turn one frame into an O(n^2) dedupe + spawn storm.
MAX_DETECTIONS = 256

#: Detections whose boxes overlap by at least this IoU are treated as
#: duplicates of one target. Kept for callers that used the module constant;
#: the pipeline itself matches at ``TrackManager.iou_threshold`` so a manual
#: box and a detector box for the same object cannot mint two ids.
DEDUPE_IOU = 0.6


@dataclass
class FrameResult:
    """Everything the renderer and exporter need for one frame."""

    frame_index: int
    tracks: Tuple[TrackedObject, ...] = ()
    created: Tuple[int, ...] = ()
    removed: Tuple[int, ...] = ()
    matched: Tuple[Tuple[int, int], ...] = ()
    detections: Tuple[Detection, ...] = ()
    spawn_failures: int = 0


class MOTPipeline:
    """Owns a :class:`TrackManager` and optional periodic detector.

    Args:
        manager: track manager holding ids and trackers.
        detector: optional ``frame -> Sequence[Detection]`` callable. When
            ``None`` the application runs in manual mode (boxes come from the
            mouse or ``--seed-box``).
        detection_interval: run the detector every N frames; ``1`` means every
            frame. The tracker still covers the in-between frames.
    """

    def __init__(
        self,
        manager: TrackManager,
        detector: Optional[DetectorFn] = None,
        detection_interval: int = 1,
    ) -> None:
        if detector is not None and not callable(detector):
            raise TypeError("detector must be callable: frame -> Sequence[Detection]")
        if int(detection_interval) < 1:
            raise ValueError(f"detection_interval must be >= 1, got {detection_interval}")
        self._manager = manager
        self._detector = detector
        self._detection_interval = int(detection_interval)
        self.frame_count = 0
        self._warned_detector_failure = False
        self._warned_truncation = False

    # -- views -----------------------------------------------------------
    @property
    def manager(self) -> TrackManager:
        return self._manager

    @property
    def tracks(self) -> Tuple[TrackedObject, ...]:
        return self._manager.tracks

    @property
    def detector(self) -> Optional[DetectorFn]:
        return self._detector

    def stats(self) -> Dict[str, int]:
        return {
            "frames": self.frame_count,
            "tracks": len(self._manager.tracks),
            "created_ids": self._manager.total_created,
            "removed_ids": self._manager.total_removed,
            "spawn_failures": self._manager.spawn_failures,
            "trail_points": sum(len(track.trail) for track in self._manager.tracks),
        }

    # -- per frame -------------------------------------------------------
    def step(
        self, frame: Any, extra_detections: Sequence[Detection] = ()
    ) -> FrameResult:
        """Advance the pipeline by one frame and return its bookkeeping."""
        detections = self._normalize(extra_detections)
        self.frame_count += 1
        if self._detector is not None and self.frame_count % self._detection_interval == 0:
            try:
                produced = self._detector(frame)
            except Exception as exc:  # noqa: BLE001 - a detector hiccup must not kill the loop
                if not self._warned_detector_failure:
                    LOGGER.warning(
                        "detector failed on frame %d (%s); tracking continues without it",
                        self.frame_count,
                        exc,
                    )
                    self._warned_detector_failure = True
                produced = None
            if produced:
                detections.extend(self._normalize(produced))
        detections = self._dedupe(detections)
        result = self._manager.step(frame, detections)
        return FrameResult(
            frame_index=self._manager.frame_index,
            tracks=result.tracks,
            created=result.created,
            removed=result.removed,
            matched=result.matched,
            detections=tuple(detections),
            spawn_failures=result.spawn_failures,
        )

    # -- interactive edits ----------------------------------------------
    def add_manual_box(self, frame: Any, bbox: Sequence[float]) -> Optional[TrackedObject]:
        """Initialise a target from a user-drawn box, reusing an overlapping id."""
        detection = Detection(bbox=normalize_bbox(bbox))
        return self._manager.add_or_refresh(frame, detection)

    def delete_at(self, point: Point) -> Optional[int]:
        """Delete the smallest active target containing ``point``."""
        best: Optional[TrackedObject] = None
        for track in self._manager.tracks:
            if not track.is_active or not bbox_contains(track.bbox, point):
                continue
            if best is None or bbox_area(track.bbox) < bbox_area(best.bbox):
                best = track
        if best is None:
            return None
        self._manager.remove(best.track_id)
        return best.track_id

    def remove(self, track_id: int) -> bool:
        return self._manager.remove(track_id)

    def reset(self) -> None:
        self._manager.reset()
        self.frame_count = 0

    def clear_trails(self) -> None:
        self._manager.clear_trails()

    # -- internals -------------------------------------------------------
    @staticmethod
    def _normalize(detections: Sequence[Detection]) -> list:
        out = []
        for item in detections:
            if isinstance(item, Detection):
                out.append(item)
            elif isinstance(item, (list, tuple)) and len(item) == 4:
                out.append(Detection(bbox=normalize_bbox(item)))
            else:
                raise TypeError(
                    "detections must be Detection instances or (x, y, w, h) sequences, "
                    f"got {type(item).__name__}"
                )
        return out

    def _dedupe(self, detections: Sequence[Detection]) -> list:
        """Keep the most confident detection among mutually overlapping boxes.

        The threshold is the association threshold: two boxes that would be
        considered "the same target" must not both spawn an id in one frame.
        """
        from .iou import box_iou

        threshold = self._manager.iou_threshold
        ordered = sorted(detections, key=lambda det: det.confidence, reverse=True)
        kept: list = []
        for detection in ordered:
            if any(box_iou(detection.bbox, other.bbox) >= threshold for other in kept):
                continue
            kept.append(detection)
        if len(kept) > MAX_DETECTIONS:
            if not self._warned_truncation:
                LOGGER.warning(
                    "%d detections after dedupe; keeping the %d most confident",
                    len(kept),
                    MAX_DETECTIONS,
                )
                self._warned_truncation = True
            kept = kept[:MAX_DETECTIONS]
        return kept


__all__ = ["DEDUPE_IOU", "MAX_DETECTIONS", "DetectorFn", "FrameResult", "MOTPipeline"]
