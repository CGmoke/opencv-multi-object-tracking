"""适配层：浏览器帧队列源（P0-B 验收测试）。"""

from __future__ import annotations

import threading
import time

import numpy as np

from mot.adapters.queue_source import QueueFrameSource


def frame(value: int = 0) -> np.ndarray:
    return np.full((4, 4, 3), value, dtype=np.uint8)


def test_push_read_roundtrip():
    source = QueueFrameSource()
    assert source.push(frame(1)) is True
    ok, got = source.read()
    assert ok is True
    assert int(got[0, 0, 0]) == 1
    assert source.pushed == 1


def test_queue_overwrites_oldest_when_full():
    source = QueueFrameSource(maxsize=1)
    source.push(frame(1))
    source.push(frame(2))
    ok, got = source.read()
    assert ok is True
    assert int(got[0, 0, 0]) == 2
    assert source.dropped == 1


def test_read_blocks_until_a_frame_arrives():
    source = QueueFrameSource()
    box = {}

    def reader() -> None:
        box["read"] = source.read()

    thread = threading.Thread(target=reader)
    thread.start()
    time.sleep(0.1)
    assert "read" not in box  # 仍在等待帧
    source.push(frame(7))
    thread.join(timeout=2)
    assert box["read"][0] is True
    assert int(box["read"][1][0, 0, 0]) == 7


def test_release_ends_a_blocked_read():
    source = QueueFrameSource()
    box = {}

    def reader() -> None:
        box["read"] = source.read()

    thread = threading.Thread(target=reader)
    thread.start()
    time.sleep(0.1)
    source.release()
    thread.join(timeout=2)
    assert box["read"] == (False, None)
    assert source.push(frame(1)) is False  # 已释放，不再接受帧


def test_idle_timeout_ends_the_session():
    source = QueueFrameSource(idle_timeout=0.05)
    assert source.read() == (False, None)


def test_fps_and_name():
    source = QueueFrameSource(fps=8.0)
    assert source.fps == 8.0 and source.name == "browser"
    source.set_fps(12)
    assert source.fps == 12.0
