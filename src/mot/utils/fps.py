"""无状态工具：FPS 估计（滑动窗口）。"""

from __future__ import annotations

import time
from collections import deque
from typing import Deque, Optional


class FpsMeter:
    """滑动窗口 FPS 估计器。

    用窗口内首末 tick 的时间差而非指数平均，使帧率不规律时的读数更诚实。
    """

    def __init__(self, window: int = 30) -> None:
        if int(window) < 2:
            raise ValueError(f"window must be >= 2, got {window}")
        self._timestamps: Deque[float] = deque(maxlen=int(window))

    def tick(self, now: Optional[float] = None) -> float:
        self._timestamps.append(time.perf_counter() if now is None else float(now))
        return self.fps

    @property
    def fps(self) -> float:
        if len(self._timestamps) < 2:
            return 0.0
        elapsed = self._timestamps[-1] - self._timestamps[0]
        if elapsed <= 1e-9:
            return 0.0
        return (len(self._timestamps) - 1) / elapsed

    def reset(self) -> None:
        self._timestamps.clear()


__all__ = ["FpsMeter"]
