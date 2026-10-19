"""Ultralytics YOLO detector adapter.

This module is an *extension point*, not part of the core feature set:

* ``ultralytics`` is an optional dependency and is imported lazily, so the
  tracking package keeps working without it;
* the adapter has no automated test in this repository (the model weights are
  not vendored), it is smoke-tested manually against a local model;
* boxes are clipped to the frame and converted to :class:`mot.models.Detection`
  so the rest of the pipeline stays detector-agnostic.

Example::

    from mot.detectors import YoloDetector
    from mot.pipeline import MOTPipeline

    detector = YoloDetector("yolov8n.pt", confidence=0.35)
    pipeline = MOTPipeline(manager, detector=detector, detection_interval=5)
"""

from __future__ import annotations

from typing import Any, List, Optional, Sequence

from ..models import Detection, clip_bbox

DEFAULT_MODEL = "yolov8n.pt"


def yolo_available() -> bool:
    """Return ``True`` when the optional ``ultralytics`` package is installed."""
    try:
        import ultralytics  # noqa: F401
    except Exception:  # noqa: BLE001 - any import problem means "not available"
        return False
    return True


class YoloDetector:
    """Callable detector backed by an Ultralytics YOLO model.

    Args:
        model: model name or path (``yolov8n.pt``, ``yolo11s.pt``, ...).
        confidence: minimum confidence for a detection to be used.
        classes: optional list of class ids to keep (e.g. ``[0]`` for people).
        device: optional inference device string (``"cpu"``, ``"cuda:0"``).

    Raises:
        ImportError: when ``ultralytics`` is not installed.
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        confidence: float = 0.35,
        classes: Optional[Sequence[int]] = None,
        device: Optional[str] = None,
    ) -> None:
        try:
            from ultralytics import YOLO
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "YoloDetector requires the optional 'ultralytics' package: "
                "pip install ultralytics"
            ) from exc

        if not 0.0 < float(confidence) <= 1.0:
            raise ValueError(f"confidence must be within (0, 1], got {confidence}")
        self.model = model
        self.confidence = float(confidence)
        self.classes = list(classes) if classes else None
        self.device = device
        self._net = YOLO(model)

    def __call__(self, frame: Any) -> List[Detection]:
        """Run inference on one BGR frame and return validated detections."""
        height, width = frame.shape[:2]
        results = self._net.predict(
            frame,
            conf=self.confidence,
            classes=self.classes,
            device=self.device,
            verbose=False,
        )
        detections: List[Detection] = []
        for result in results:
            boxes = getattr(result, "boxes", None)
            if boxes is None:
                continue
            names = getattr(result, "names", {}) or {}
            for box in boxes:
                x, y, w, h = (float(value) for value in box.xywh[0].tolist())
                clipped = clip_bbox((x - w / 2.0, y - h / 2.0, w, h), width, height)
                if clipped is None:
                    continue
                class_id = int(box.cls[0]) if box.cls is not None else -1
                label = str(names.get(class_id, f"class_{class_id}"))
                score = float(box.conf[0]) if box.conf is not None else self.confidence
                detections.append(Detection(bbox=clipped, confidence=score, label=label))
        return detections


__all__ = ["DEFAULT_MODEL", "YoloDetector", "yolo_available"]
