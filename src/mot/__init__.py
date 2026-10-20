"""纯 OpenCV 多目标跟踪框架（目标架构 v0.2.0）。

分层与依赖方向（单向）::

    interfaces ──► application ──► domain
         │               │            ▲
         └───────────────┴──► adapters ┘

**无 OpenCV 也能导入核心**：本顶层包只**立即**再导出 ``domain`` 与 ``application``
的公开符号（二者不 import cv2）；适配层符号（``make_tracker_factory`` 等）通过
模块级 ``__getattr__`` **惰性**加载，故 ``import mot`` 不要求安装 OpenCV。

命令行入口：``mot-track`` 或 ``python -m mot``。
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Tuple

from .application.pipeline import FrameResult, MOTPipeline
from .domain.errors import DomainError, TrackerSetupError, ValidationError
from .domain.lifecycle import TrackManager
from .domain.limits import SUPPORTED_BACKENDS
from .domain.models import (
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
from .domain.records import TrackPoint, TrajectoryStore

if TYPE_CHECKING:  # pragma: no cover - 仅供类型检查，运行期惰性导入
    from .adapters.sinks.csv import CsvTrajectoryWriter

__version__ = "0.2.0"

#: 适配层符号的惰性加载表：``名称 -> (模块, 属性)``。
_LAZY_ATTRS: Dict[str, Tuple[str, str]] = {
    "CsvTrajectoryWriter": ("mot.adapters.sinks.csv", "CsvTrajectoryWriter"),
    "create_tracker": ("mot.adapters.trackers", "create_tracker"),
    "make_tracker_factory": ("mot.adapters.trackers", "make_tracker_factory"),
    "tracker_available": ("mot.adapters.trackers", "tracker_available"),
}


def __getattr__(name: str) -> Any:
    """按需加载适配层符号，使 ``import mot`` 无需 OpenCV。"""
    target = _LAZY_ATTRS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    value = getattr(importlib.import_module(target[0]), target[1])
    globals()[name] = value  # 缓存，后续访问直接命中
    return value


__all__ = [
    "SUPPORTED_BACKENDS",
    "CsvTrajectoryWriter",
    "Detection",
    "DomainError",
    "FrameResult",
    "IdAllocator",
    "MOTPipeline",
    "StepResult",
    "TrackManager",
    "TrackPoint",
    "TrackState",
    "TrackedObject",
    "TrackerSetupError",
    "TrajectoryStore",
    "ValidationError",
    "bbox_area",
    "bbox_center",
    "bbox_contains",
    "clip_bbox",
    "create_tracker",
    "make_tracker_factory",
    "normalize_bbox",
    "tracker_available",
]
