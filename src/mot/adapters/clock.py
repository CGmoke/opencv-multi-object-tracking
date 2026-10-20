"""适配层：时间来源（实现 ``domain.ports.Clock``）。

用单调时钟为每帧提供时间戳，替代“按 FPS 估算”（修 G6），避免摄像头漂移与
``--loop`` 后时间失准。
"""

from __future__ import annotations

import time


class MonotonicClock:
    """返回单调递增秒数的时钟。"""

    def now(self) -> float:
        return time.monotonic()


__all__ = ["MonotonicClock"]
