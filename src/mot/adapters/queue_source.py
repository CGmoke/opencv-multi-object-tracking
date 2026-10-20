"""适配层：由外部推入帧的 ``FrameSource``（浏览器摄像头回传）。

浏览器前端用 ``getUserMedia`` 抓帧并定时 POST 给服务端，服务端解码后 ``push`` 到本
源的内部队列，跟踪会话再从队列取帧。队列默认 ``maxsize=1`` 且**覆盖写**：新帧挤掉
旧帧，慢客户端不会积压。

``read()`` 会**阻塞等待**（轮询 0.2s）而不是立刻返回“流结束”，因此浏览器摄像头会话
不会因为两帧之间没有数据而提前退出；会话结束由显式 ``release()`` 或 ``idle_timeout``
（可选，客户端长时间不再推帧）触发。
"""

from __future__ import annotations

import time
from queue import Empty, Full, Queue
from typing import Any, Optional, Tuple


class QueueFrameSource:
    """从内部队列取帧的 ``FrameSource``。

    Args:
        fps: 声明帧率（供 HUD / 参考）。
        maxsize: 队列容量（≥1）；满时覆盖最旧帧。
        name: 源名（HUD 显示）。
        idle_timeout: 客户端多久（秒）不再推帧即结束会话；``None`` 表示不设超时。
    """

    def __init__(
        self,
        fps: float = 15.0,
        maxsize: int = 1,
        name: str = "browser",
        idle_timeout: Optional[float] = None,
    ) -> None:
        self._queue: "Queue[Any]" = Queue(maxsize=max(1, int(maxsize)))
        self._fps = float(fps)
        self._name = name
        self._idle_timeout = None if idle_timeout is None else max(0.0, float(idle_timeout))
        self._released = False
        self._pushed = 0
        self._dropped = 0
        self._last_push = time.monotonic()

    # -- 只读视图 --------------------------------------------------------
    @property
    def fps(self) -> float:
        return self._fps

    @property
    def name(self) -> str:
        return self._name

    @property
    def pushed(self) -> int:
        """累计接受的帧数。"""
        return self._pushed

    @property
    def dropped(self) -> int:
        """累计因队列满而丢弃的帧数。"""
        return self._dropped

    @property
    def released(self) -> bool:
        return self._released

    def set_fps(self, fps: float) -> None:
        self._fps = max(0.0, float(fps))

    # -- 供帧 ------------------------------------------------------------
    def push(self, frame: Any) -> bool:
        """入队一帧；队列满则丢弃最旧一帧再放入。返回是否接受了新帧。"""
        if self._released:
            return False
        try:
            self._queue.put_nowait(frame)
        except Full:
            try:
                self._queue.get_nowait()
                self._dropped += 1
            except Empty:  # pragma: no cover - 与 Full 竞态时的兜底
                pass
            try:
                self._queue.put_nowait(frame)
            except Full:  # pragma: no cover - 见上
                self._dropped += 1
                return False
        self._pushed += 1
        self._last_push = time.monotonic()
        return True

    # -- FrameSource -----------------------------------------------------
    def read(self) -> Tuple[bool, Optional[Any]]:
        """取一帧；无帧时阻塞等待，直到有新帧、被释放或空闲超时。"""
        while not self._released:
            try:
                return True, self._queue.get(timeout=0.2)
            except Empty:
                if self._idle_timeout is not None:
                    if time.monotonic() - self._last_push > self._idle_timeout:
                        return False, None
        return False, None

    def release(self) -> None:
        """停止该源；阻塞中的 ``read()`` 会在 0.2 秒内返回流结束。"""
        self._released = True
        while True:
            try:
                self._queue.get_nowait()
            except Empty:
                break

    def __enter__(self) -> "QueueFrameSource":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.release()


__all__ = ["QueueFrameSource"]
