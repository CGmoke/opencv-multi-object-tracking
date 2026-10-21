"""无状态工具：FPS 估计（滑动窗口）与 Web 会话帧预算。"""

from __future__ import annotations

import threading
import time
from collections import deque
from typing import Any, Deque, Dict, Optional


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


class FrameBudget:
    """Web 会话三段帧率（滑动窗口，默认最近 30 帧）与推流跳帧策略。

    * ``capture_fps``：收到帧的速率（browser 为 POST 入队速率，其它源为 read 速率）；
    * ``process_fps``：``pipeline.step`` 的速率；
    * ``stream_fps``：MJPEG 分块成功产出的速率。

    当处理速率明显高于推流速率（默认 1.5 倍）时判定为积压，推流按 ``target_fps``
    限量跳帧：只发送最新帧，被跳过的帧计入 ``skipped``，绝不阻塞处理线程。
    """

    def __init__(
        self,
        *,
        window: int = 30,
        target_fps: float = 15.0,
        backlog_ratio: float = 1.5,
    ) -> None:
        self._lock = threading.Lock()
        self.capture = FpsMeter(window)
        self.process = FpsMeter(window)
        self.stream = FpsMeter(window)
        self.target_fps = max(0.0, float(target_fps))
        self.backlog_ratio = max(1.0, float(backlog_ratio))
        self.skipped = 0
        self._last_sent_at: Optional[float] = None

    def mark_capture(self, now: Optional[float] = None) -> None:
        with self._lock:
            self.capture.tick(now)

    def mark_process(self, now: Optional[float] = None) -> None:
        with self._lock:
            self.process.tick(now)

    def mark_stream(self, now: Optional[float] = None) -> None:
        stamp = time.monotonic() if now is None else float(now)
        with self._lock:
            self.stream.tick(stamp)
            self._last_sent_at = stamp

    def should_send(self, now: Optional[float] = None) -> bool:
        """是否放行当前帧的推送；积压时按目标帧率限量跳帧。"""
        stamp = time.monotonic() if now is None else float(now)
        with self._lock:
            process_fps = self.process.fps
            stream_fps = self.stream.fps
            backlog = (
                process_fps > 0.0
                and stream_fps > 0.0
                and process_fps > stream_fps * self.backlog_ratio
            )
            limited = (
                backlog
                and self.target_fps > 0.0
                and self._last_sent_at is not None
                and stamp - self._last_sent_at < 1.0 / self.target_fps
            )
            if limited:
                self.skipped += 1
                return False
            return True

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "capture_fps": round(self.capture.fps, 1),
                "process_fps": round(self.process.fps, 1),
                "stream_fps": round(self.stream.fps, 1),
                "stream_skipped": self.skipped,
            }


__all__ = ["FrameBudget", "FpsMeter"]
