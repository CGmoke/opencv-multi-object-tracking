"""适配层：检测器端口的说明与测试用假检测器。

检测器端口签名为 ``(frame, timestamp) -> Sequence[Detection]``
（见 :class:`mot.domain.ports.Detector`）。真正基于模型的实现见
:mod:`mot.adapters.detectors.yolo`。
"""

from __future__ import annotations

from typing import Any, List, Sequence

from mot.domain.models import Detection


class NullDetector:
    """永远返回空检测的检测器（禁用检测或测试用）。"""

    def __call__(self, frame: Any, timestamp: float) -> List[Detection]:
        return []


class FixedDetector:
    """返回固定检测序列的检测器（测试 / 演示用）。"""

    def __init__(self, detections: Sequence[Detection] = ()) -> None:
        self._detections = tuple(detections)

    def __call__(self, frame: Any, timestamp: float) -> List[Detection]:
        return list(self._detections)


__all__ = ["FixedDetector", "NullDetector"]
