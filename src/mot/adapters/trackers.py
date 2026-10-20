"""适配层：OpenCV 跟踪器构造与命名空间兼容链。

OpenCV 在 4.5.1 把 contrib 跟踪器移入 ``cv2.legacy``，后续版本又移回，因此工厂
逐点探测多个入口，而不硬编码其一。支持的后端名来自
``domain.limits.SUPPORTED_BACKENDS``（单一真源）。

``init_tracker`` / ``update_tracker`` 是注入给 ``domain.lifecycle.TrackManager``
的 ``init_fn`` / ``update_fn``，使领域层保持零 cv2 依赖。
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple

from mot.domain.errors import TrackerSetupError
from mot.domain.limits import SUPPORTED_BACKENDS
from mot.domain.models import BBox
from mot.domain.ports import TrackerFactory

#: 后端 -> 一句话说明（用于 ``--help``、``--list-trackers`` 与文档）。
BACKEND_NOTES: Dict[str, str] = {
    "csrt": "Discriminative correlation filter with channel and spatial reliability; "
    "most accurate of the built-ins, best for offline analysis.",
    "kcf": "Kernelised correlation filter; fastest usable backend, drops boxes under "
    "fast motion and heavy occlusion.",
    "mil": "Multiple instance learning; robust to some occlusion, loose boxes, CPU heavy.",
    "mosse": "Minimum output sum of squared error; extremely fast, least accurate.",
}


def _candidate_factories(name: str) -> List[Tuple[str, Callable[[], Any]]]:
    """返回 ``(描述, 工厂)`` 列表，越具体的入口越靠前。"""
    short = name.upper()
    candidates: List[Tuple[str, Callable[[], Any]]] = []
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - 无 OpenCV 环境下走到
        raise TrackerSetupError(
            "OpenCV is not installed. Install opencv-contrib-python (the tracking module "
            "is not part of plain opencv-python)."
        ) from exc

    modern = getattr(cv2, f"Tracker{short}_create", None)
    if callable(modern):
        candidates.append((f"cv2.Tracker{short}_create", modern))

    legacy = getattr(getattr(cv2, "legacy", None), f"Tracker{short}_create", None)
    if callable(legacy):
        candidates.append((f"cv2.legacy.Tracker{short}_create", legacy))

    # 某些构建只暴露类，例如 cv2.TrackerCSRT.create()。
    cls = getattr(cv2, f"Tracker{short}", None)
    if cls is not None:
        cls_create = getattr(cls, "create", None)
        if callable(cls_create):
            candidates.append((f"cv2.Tracker{short}.create", cls_create))
    return candidates


def create_tracker(name: str = "csrt") -> Any:
    """为指定后端创建一个全新的 OpenCV 跟踪器。

    Args:
        name: :data:`~mot.domain.limits.SUPPORTED_BACKENDS` 中的名字，大小写不敏感。

    Raises:
        TrackerSetupError: 后端名未知，或本 OpenCV 构建没有对应工厂。
    """
    backend = str(name).strip().lower()
    if backend not in SUPPORTED_BACKENDS:
        raise TrackerSetupError(
            f"unknown tracker backend {name!r}; supported: {', '.join(SUPPORTED_BACKENDS)}",
            field="backend",
            value=name,
        )

    candidates = _candidate_factories(backend)
    if not candidates:
        raise TrackerSetupError(
            f"OpenCV build has no Tracker{backend.upper()} factory. "
            "Install opencv-contrib-python>=4.5.1 (plain opencv-python has no trackers).",
            field="backend",
            value=backend,
        )

    errors: List[str] = []
    for description, factory in candidates:
        try:
            tracker = factory()
        except Exception as exc:  # noqa: BLE001 - 逐入口报告后统一失败
            errors.append(f"{description}: {exc}")
            continue
        if tracker is not None:
            return tracker
        errors.append(f"{description}: returned None")
    raise TrackerSetupError(
        f"could not create tracker {backend!r}; tried: " + "; ".join(errors),
        field="backend",
        value=backend,
    )


def make_tracker_factory(name: str = "csrt") -> TrackerFactory:
    """返回适配 ``TrackManager`` 的零参工厂。

    后端在启动期即被 eager 校验，使应用在启动时失败而非首次点击时。
    """
    create_tracker(name)  # 立即校验；结果故意丢弃
    backend = str(name).strip().lower()

    def factory() -> Any:
        return create_tracker(backend)

    return factory


def tracker_available(name: str) -> bool:
    """返回该后端在当前构建中是否可构造。"""
    try:
        create_tracker(name)
    except TrackerSetupError:
        return False
    return True


def describe_backend(name: str) -> str:
    """返回后端的一句话说明（用于 ``--help`` 与文档）。"""
    return BACKEND_NOTES.get(str(name).strip().lower(), f"unknown backend {name!r}")


def init_tracker(tracker: Any, frame: Any, bbox: BBox) -> None:
    """用整数像素框初始化跟踪器。

    部分平台的 OpenCV 跟踪器拒绝浮点元组，故先把框四舍五入到整数像素。
    """
    x, y, w, h = bbox
    tracker.init(
        frame, (int(round(x)), int(round(y)), int(round(max(1.0, w))), int(round(max(1.0, h))))
    )


def update_tracker(tracker: Any, frame: Any) -> Optional[BBox]:
    """推进一次跟踪器；返回校验后的浮点框，丢失时返回 ``None``。"""
    try:
        ok, raw = tracker.update(frame)
    except Exception:  # noqa: BLE001 - 某些后端抛 cv2.error
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
    "create_tracker",
    "describe_backend",
    "init_tracker",
    "make_tracker_factory",
    "tracker_available",
    "update_tracker",
]
