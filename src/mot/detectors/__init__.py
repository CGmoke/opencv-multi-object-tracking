"""Optional detector adapters.

The core project is a *tracking* framework: targets are initialised by the
user (click or ``--seed-box``) and followed by per-object OpenCV trackers.
A detector is the documented extension point for "detect periodically, track
in between": implement ``frame -> Sequence[Detection]`` and pass it to
:class:`mot.pipeline.MOTPipeline`.
"""

from __future__ import annotations

from .yolo import YoloDetector, yolo_available

__all__ = ["YoloDetector", "yolo_available"]
