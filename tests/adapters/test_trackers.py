"""适配层跟踪器工厂与注入回调（步骤 6a 验收测试）。"""

from __future__ import annotations

import pytest

cv2 = pytest.importorskip("cv2")

from mot.adapters.trackers import (  # noqa: E402 - importorskip 必须先行
    BACKEND_NOTES,
    create_tracker,
    describe_backend,
    init_tracker,
    make_tracker_factory,
    tracker_available,
    update_tracker,
)
from mot.domain.errors import TrackerSetupError
from mot.domain.limits import SUPPORTED_BACKENDS


def test_supported_backends_are_documented():
    assert set(SUPPORTED_BACKENDS) <= set(BACKEND_NOTES)


def test_unknown_backend_is_rejected():
    with pytest.raises(TrackerSetupError, match="unknown"):
        create_tracker("nope")


def test_tracker_available_returns_bool():
    for backend in SUPPORTED_BACKENDS:
        assert isinstance(tracker_available(backend), bool)


def test_create_tracker_and_factory():
    if not tracker_available("csrt"):
        pytest.skip("csrt tracker not available in this OpenCV build")
    assert create_tracker("csrt") is not None
    factory = make_tracker_factory("csrt")
    assert callable(factory)
    assert factory() is not None


def test_make_tracker_factory_validates_eagerly():
    with pytest.raises(TrackerSetupError):
        make_tracker_factory("nope")


def test_describe_backend():
    assert "correlation" in describe_backend("csrt").lower()
    assert "unknown" in describe_backend("nope").lower()


class _FakeTracker:
    def __init__(self, ok: bool = True, raw=("1", "2", "3", "4"), explode: bool = False) -> None:
        self.ok = ok
        self.raw = raw
        self.explode = explode
        self.inited = None

    def init(self, frame, bbox) -> None:
        self.inited = bbox

    def update(self, frame):
        if self.explode:
            raise RuntimeError("boom")
        return self.ok, self.raw


def test_init_tracker_rounds_to_integer_pixels():
    tracker = _FakeTracker()
    init_tracker(tracker, None, (1.4, 2.6, 3.2, 4.0))
    assert tracker.inited == (1, 3, 3, 4)


def test_update_tracker_returns_validated_box():
    assert update_tracker(_FakeTracker(ok=True, raw=(1.0, 2.0, 3.0, 4.0)), None) == (
        1.0,
        2.0,
        3.0,
        4.0,
    )


def test_update_tracker_handles_failure_and_bad_boxes():
    assert update_tracker(_FakeTracker(ok=False), None) is None
    assert update_tracker(_FakeTracker(ok=True, raw=(1, 2, 0.5, 4)), None) is None  # 宽 < 1
    assert update_tracker(_FakeTracker(ok=True, raw=None), None) is None
    assert update_tracker(_FakeTracker(explode=True), None) is None
