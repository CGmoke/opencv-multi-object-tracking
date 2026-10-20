"""工具层 FpsMeter（步骤 6a 验收测试）。"""

from __future__ import annotations

import pytest

from mot.utils.fps import FpsMeter


class TestFpsMeter:
    def test_zero_before_two_samples(self):
        meter = FpsMeter()
        assert meter.tick(0.0) == 0.0

    def test_uniform_interval(self):
        meter = FpsMeter(window=4)
        assert meter.tick(0.0) == 0.0
        assert meter.tick(1.0) == pytest.approx(1.0)
        assert meter.tick(2.0) == pytest.approx(1.0)

    def test_irregular_interval_is_measured_honestly(self):
        meter = FpsMeter(window=2)
        meter.tick(0.0)
        assert meter.tick(2.0) == pytest.approx(0.5)

    def test_window_slides(self):
        meter = FpsMeter(window=3)
        for timestamp in (0.0, 1.0):
            meter.tick(timestamp)
        for timestamp in (2.0, 3.0):
            meter.tick(timestamp)
        assert meter.fps == pytest.approx(1.0)

    def test_reset(self):
        meter = FpsMeter()
        meter.tick(0.0)
        meter.tick(1.0)
        meter.reset()
        assert meter.fps == 0.0

    def test_window_must_be_at_least_two(self):
        with pytest.raises(ValueError):
            FpsMeter(window=1)
