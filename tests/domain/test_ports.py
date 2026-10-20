"""领域层端口契约（步骤 3 验收测试）。

验证 ``typing.Protocol`` 端口能被“结构上匹配”的假实现满足（``isinstance``），
以及不匹配的对象会被拒绝。这保证适配层只需按结构实现即可被接入。
"""

from __future__ import annotations

from mot.domain.models import Detection, TrackedObject
from mot.domain.ports import (
    Clock,
    Detector,
    EventSink,
    FrameSource,
    SelfHostedTracker,
    TrajectorySink,
)


class _FakeTracker:
    def init(self, frame, bbox) -> bool:
        return True

    def update(self, frame):
        return False, ()


class _FakeSource:
    fps = 30.0
    name = "fake"

    def read(self):
        return False, None

    def release(self) -> None:
        pass


class _FakeDetector:
    def __call__(self, frame, timestamp) -> list:
        return [Detection(bbox=(0, 0, 10, 10))]


class _FakeSink:
    def write_frame(self, frame_index, timestamp, tracks) -> int:
        return len(tuple(tracks))

    def close(self) -> None:
        pass


class _FakeEventSink:
    def on_frame(self, result) -> None:
        pass


class _FakeClock:
    def now(self) -> float:
        return 0.0


def test_structural_conformance():
    assert isinstance(_FakeTracker(), SelfHostedTracker)
    assert isinstance(_FakeSource(), FrameSource)
    assert isinstance(_FakeDetector(), Detector)
    assert isinstance(_FakeSink(), TrajectorySink)
    assert isinstance(_FakeEventSink(), EventSink)
    assert isinstance(_FakeClock(), Clock)


def test_non_conforming_objects_are_rejected():
    assert not isinstance(object(), SelfHostedTracker)
    assert not isinstance(object(), FrameSource)
    assert not isinstance(object(), Detector)
    assert not isinstance(object(), TrajectorySink)
    assert not isinstance(object(), EventSink)
    assert not isinstance(object(), Clock)


def test_partial_implementation_is_rejected():
    class _HalfSource:
        fps = 30.0  # 缺 name / read / release

    assert not isinstance(_HalfSource(), FrameSource)


def test_fake_tracker_satisfies_used_contract_shape():
    tracker = _FakeTracker()
    assert tracker.init(object(), (0, 0, 10, 10)) is True
    ok, box = tracker.update(object())
    assert ok is False and box == ()


def test_fake_sink_counts_rows():
    sink = _FakeSink()
    tracks = [TrackedObject(track_id=1, tracker=None, bbox=(0, 0, 10, 10))]
    assert sink.write_frame(1, 0.0, tracks) == 1
