"""测试共享支持：纯 Python 的假件（无 cv2），供领域层与应用层测试复用。

跟踪器以零参工厂 + ``init_fn`` / ``update_fn`` 注入（领域层不 import cv2），
数据源/时钟/接收器/渲染器则以结构化实现满足 ``domain.ports`` 的端口契约。
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Optional, Sequence

import pytest

from mot.domain.lifecycle import TrackManager
from mot.domain.models import BBox

# 让子目录下的测试可用 ``from conftest import ...`` 取到本模块里的假件。
_TESTS_DIR = str(Path(__file__).resolve().parent)
if _TESTS_DIR not in sys.path:
    sys.path.insert(0, _TESTS_DIR)


class FakeTracker:
    """最小跟踪器：遵循 ``init`` / ``update`` 契约，每次 update 平移 ``(dx, dy)``。"""

    def __init__(self, dx: float = 0.0, dy: float = 0.0, alive: bool = True) -> None:
        self.dx = float(dx)
        self.dy = float(dy)
        self.alive = alive
        self.box = None
        self.init_calls = 0
        self.update_calls = 0

    def init(self, frame, bbox) -> None:
        self.init_calls += 1
        self.box = tuple(float(v) for v in bbox)

    def update(self, frame):
        self.update_calls += 1
        if not self.alive or self.box is None:
            return False, None
        x, y, w, h = self.box
        self.box = (x + self.dx, y + self.dy, w, h)
        return True, self.box


class TrackerPool:
    """零参工厂，记录它创建过的跟踪器（便于断言与开关 ``alive``）。"""

    def __init__(self, dx: float = 0.0, dy: float = 0.0, alive: bool = True) -> None:
        self.dx = dx
        self.dy = dy
        self.alive = alive
        self.trackers: list = []

    def __call__(self) -> FakeTracker:
        tracker = FakeTracker(dx=self.dx, dy=self.dy, alive=self.alive)
        self.trackers.append(tracker)
        return tracker


def init_fn(tracker, frame, bbox) -> None:
    """适配器风格的重锚回调。"""
    tracker.init(frame, bbox)


def update_fn(tracker, frame) -> Optional[BBox]:
    """适配器风格的推进回调：把跟踪器的 ``(ok, box)`` 折成 ``Optional[BBox]``。"""
    ok, box = tracker.update(frame)
    if not ok or box is None:
        return None
    return tuple(float(v) for v in box)


class FakeSource:
    """内存帧源，实现 ``FrameSource`` 契约（读完后返回流结束）。"""

    def __init__(self, frames: Sequence[Any], fps: float = 30.0, name: str = "fake") -> None:
        self._frames = list(frames)
        self.fps = float(fps)
        self.name = name
        self.released = False
        self._index = 0

    def read(self):
        if self._index >= len(self._frames):
            return False, None
        frame = self._frames[self._index]
        self._index += 1
        return True, frame

    def release(self) -> None:
        self.released = True


class FakeClock:
    """单调时钟：每次 ``now()`` 前进 ``step`` 秒。"""

    def __init__(self, start: float = 0.0, step: float = 0.0) -> None:
        self._now = float(start)
        self._step = float(step)

    def now(self) -> float:
        value = self._now
        self._now += self._step
        return value


class RecordingTrajectorySink:
    """内存轨迹接收器，实现 ``TrajectorySink`` 契约。"""

    def __init__(self) -> None:
        self.rows: list = []
        self.closed = False

    def write_frame(self, frame_index: int, timestamp: float, tracks) -> int:
        ids = tuple(track.track_id for track in tracks)
        self.rows.append((frame_index, timestamp, ids))
        return len(ids)

    def close(self) -> None:
        self.closed = True


class RecordingEventSink:
    """内存事件接收器，实现 ``EventSink`` 契约。"""

    def __init__(self) -> None:
        self.results: list = []

    def on_frame(self, result) -> None:
        self.results.append(result)


class ScriptedRenderer:
    """按脚本返回事件的假渲染器，实现 ``Renderer`` 契约；记录呈现过的帧。"""

    def __init__(self, events: Sequence[Optional[str]] = ()) -> None:
        self._events = list(events)
        self.presented: list = []

    def present(self, frame, result) -> Optional[str]:
        self.presented.append((frame, result))
        if self._events:
            return self._events.pop(0)
        return None


@pytest.fixture
def tracker_pool():
    """返回 ``TrackerPool(**kwargs)`` 的工厂。"""

    def _factory(**kwargs) -> TrackerPool:
        return TrackerPool(**kwargs)

    return _factory


@pytest.fixture
def make_manager():
    """返回已注入 ``init_fn`` / ``update_fn`` 的 ``TrackManager`` 工厂。"""

    def _factory(tracker_factory, **kwargs) -> TrackManager:
        return TrackManager(tracker_factory, init_fn=init_fn, update_fn=update_fn, **kwargs)

    return _factory


@pytest.fixture
def pool() -> TrackerPool:
    return TrackerPool()


@pytest.fixture
def manager(make_manager, pool) -> TrackManager:
    return make_manager(pool)


@pytest.fixture
def sample_video(tmp_path):
    """合成小视频：白色方块向右移动（需要 cv2，否则跳过）。"""
    cv2 = pytest.importorskip("cv2")
    import numpy as np

    path = tmp_path / "input.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 20.0, (160, 120))
    if not writer.isOpened():
        pytest.skip("no mp4v writer available in this OpenCV build")
    for index in range(20):
        frame = np.zeros((120, 160, 3), dtype=np.uint8)
        x = 20 + 4 * index
        cv2.rectangle(frame, (x, 40), (x + 39, 79), (255, 255, 255), -1)
        writer.write(frame)
    writer.release()
    return path
