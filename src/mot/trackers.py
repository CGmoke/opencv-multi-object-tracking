"""Tracker construction with OpenCV namespace compatibility.

OpenCV moved the contributed trackers to the ``cv2.legacy`` namespace in
4.5.1 and brought them back in later releases, so the factory probes several
entry points instead of hard-coding one. Supported backends are CSRT (accurate)
and KCF (fast); MIL/MOSSE remain available for experimentation.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple

TrackerFactory = Callable[[], Any]

#: Friendly name -> OpenCV tracker short name.
SUPPORTED_BACKENDS: Tuple[str, ...] = ("csrt", "kcf", "mil", "mosse")

#: Default kwargs per backend. CSRT is accurate but slow (~15-30 FPS on CPU);
#: KCF trades accuracy for speed (~60+ FPS on CPU at 640x480).
BACKEND_NOTES: Dict[str, str] = {
    "csrt": "Discriminative correlation filter with channel and spatial reliability; "
    "most accurate of the built-ins, best for offline analysis.",
    "kcf": "Kernelised correlation filter; fastest usable backend, drops boxes under "
    "fast motion and heavy occlusion.",
    "mil": "Multiple instance learning; robust to some occlusion, loose boxes, CPU heavy.",
    "mosse": "Minimum output sum of squared error; extremely fast, least accurate.",
}


class TrackerBackendError(RuntimeError):
    """Raised when the requested tracker backend cannot be created."""


def _candidate_factories(name: str) -> List[Tuple[str, Callable[[], Any]]]:
    """Return ``(description, factory)`` pairs to try, most specific first."""
    short = name.upper()
    candidates: List[Tuple[str, Callable[[], Any]]] = []
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - exercised in opencv-missing envs
        raise TrackerBackendError(
            "OpenCV is not installed. Install opencv-contrib-python (the tracking module "
            "is not part of plain opencv-python)."
        ) from exc

    modern = getattr(cv2, f"Tracker{short}_create", None)
    if callable(modern):
        candidates.append((f"cv2.Tracker{short}_create", modern))

    legacy = getattr(getattr(cv2, "legacy", None), f"Tracker{short}_create", None)
    if callable(legacy):
        candidates.append((f"cv2.legacy.Tracker{short}_create", legacy))

    # Some builds expose the class only, e.g. cv2.TrackerCSRT.create().
    cls = getattr(cv2, f"Tracker{short}", None)
    if cls is not None:
        cls_create = getattr(cls, "create", None)
        if callable(cls_create):
            candidates.append((f"cv2.Tracker{short}.create", cls_create))
    return candidates


def tracker_available(name: str) -> bool:
    """Return ``True`` when the backend can be constructed in this build."""
    try:
        create_tracker(name)
    except TrackerBackendError:
        return False
    return True


def create_tracker(name: str = "csrt") -> Any:
    """Create a fresh OpenCV tracker for the requested backend.

    Args:
        name: one of :data:`SUPPORTED_BACKENDS`, case insensitive.

    Raises:
        TrackerBackendError: when OpenCV has no factory for the backend, or the
            backend name is unknown. The message lists attempted entry points.
    """
    backend = str(name).strip().lower()
    if backend not in SUPPORTED_BACKENDS:
        raise TrackerBackendError(
            f"unknown tracker backend {name!r}; supported: {', '.join(SUPPORTED_BACKENDS)}"
        )

    candidates = _candidate_factories(backend)
    if not candidates:
        raise TrackerBackendError(
            f"OpenCV build has no Tracker{backend.upper()} factory. "
            "Install opencv-contrib-python>=4.5.1 (plain opencv-python has no trackers)."
        )

    errors: List[str] = []
    for description, factory in candidates:
        try:
            tracker = factory()
        except Exception as exc:  # noqa: BLE001 - report every entry point, then fail
            errors.append(f"{description}: {exc}")
            continue
        if tracker is not None:
            return tracker
        errors.append(f"{description}: returned None")
    raise TrackerBackendError(
        f"could not create tracker {backend!r}; tried: " + "; ".join(errors)
    )


def make_tracker_factory(name: str = "csrt") -> TrackerFactory:
    """Return a zero-argument factory suitable for :class:`TrackManager`.

    The backend is validated eagerly so the application fails at startup
    rather than on the first click.
    """
    create_tracker(name)  # validate now; the result is intentionally discarded
    backend = str(name).strip().lower()

    def factory() -> Any:
        return create_tracker(backend)

    return factory


def describe_backend(name: str) -> str:
    """Return a one-line description of a backend, for ``--help`` and docs."""
    return BACKEND_NOTES.get(str(name).strip().lower(), f"unknown backend {name!r}")


def init_tracker(tracker: Any, frame: Any, bbox: Tuple[float, float, float, float]) -> None:
    """Initialise a tracker with an integer pixel box.

    OpenCV's tracker implementations reject float tuples on some platforms, so
    the box is rounded to integer pixels first.
    """
    x, y, w, h = bbox
    tracker.init(frame, (int(round(x)), int(round(y)), int(round(max(1.0, w))), int(round(max(1.0, h)))))


def update_tracker(tracker: Any, frame: Any) -> Optional[Tuple[float, float, float, float]]:
    """Run one tracker update; return a validated float box or ``None`` on loss."""
    try:
        ok, raw = tracker.update(frame)
    except Exception:  # noqa: BLE001 - OpenCV raises cv2.error on some backends
        return None
    if not ok or raw is None:
        return None
    try:
        x, y, w, h = (float(v) for v in raw)
    except (TypeError, ValueError):
        return None
    if not (w >= 1.0 and h >= 1.0):
        return None
    return (x, y, w, h)


__all__ = [
    "BACKEND_NOTES",
    "SUPPORTED_BACKENDS",
    "TrackerBackendError",
    "TrackerFactory",
    "create_tracker",
    "describe_backend",
    "init_tracker",
    "make_tracker_factory",
    "tracker_available",
    "update_tracker",
]
