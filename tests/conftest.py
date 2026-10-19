"""Shared test setup: make ``src/`` importable and provide fake trackers."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


class FakeTracker:
    """Minimal object that follows the OpenCV tracker contract.

    It moves its box by ``(dx, dy)`` on every update, so the identity
    lifecycle can be exercised without OpenCV or real video.
    """

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
    """Zero-argument tracker factory that remembers what it created."""

    def __init__(self, dx: float = 0.0, dy: float = 0.0, alive: bool = True) -> None:
        self.dx = dx
        self.dy = dy
        self.alive = alive
        self.trackers = []

    def __call__(self) -> FakeTracker:
        tracker = FakeTracker(dx=self.dx, dy=self.dy, alive=self.alive)
        self.trackers.append(tracker)
        return tracker


@pytest.fixture
def pool() -> TrackerPool:
    return TrackerPool()


@pytest.fixture
def sample_video(tmp_path):
    """A small synthetic video: a white square moving to the right."""
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


@pytest.fixture
def manager(pool):
    from mot import TrackManager

    return TrackManager(pool)
