"""领域层尺寸估算（验收测试）。"""

from __future__ import annotations

import pytest

from mot.domain.errors import ValidationError
from mot.domain.sizing import (
    SIZE_CLASSES,
    classify_size,
    estimate_size_meters,
    format_size,
    pixel_size,
)


class TestPixelSize:
    def test_dimensions_and_area(self):
        assert pixel_size((10.0, 20.0, 30.0, 40.0)) == (30.0, 40.0, 1200.0)

    def test_negative_sizes_are_clamped(self):
        assert pixel_size((0.0, 0.0, -5.0, -6.0)) == (0.0, 0.0, 0.0)


class TestEstimateSizeMeters:
    def test_converts_with_calibration(self):
        width_m, height_m, diagonal_m = estimate_size_meters((0, 0, 120.0, 160.0), 120.0)
        assert width_m == pytest.approx(1.0)
        assert height_m == pytest.approx(160.0 / 120.0)
        assert diagonal_m == pytest.approx((1.0 + (160.0 / 120.0) ** 2) ** 0.5)

    def test_missing_calibration_returns_none(self):
        assert estimate_size_meters((0, 0, 10, 10), None) is None
        assert estimate_size_meters((0, 0, 10, 10), 0) is None
        assert estimate_size_meters((0, 0, 10, 10), -3) is None

    @pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
    def test_non_finite_calibration_is_rejected(self, bad):
        with pytest.raises(ValidationError):
            estimate_size_meters((0, 0, 10, 10), bad)


class TestClassifySize:
    def test_levels(self):
        assert SIZE_CLASSES == ("small", "medium", "large")
        assert classify_size(0.1, 0.2) == "small"
        assert classify_size(0.8, 0.9) == "medium"
        assert classify_size(1.4, 3.0) == "large"

    def test_boundaries_are_inclusive(self):
        assert classify_size(0.5, 0.1) == "small"  # 恰好 small_max
        assert classify_size(2.0, 0.1) == "large"  # 恰好 large_min

    def test_invalid_thresholds_are_rejected(self):
        with pytest.raises(ValidationError):
            classify_size(1.0, 1.0, small_max=0.0)
        with pytest.raises(ValidationError):
            classify_size(1.0, 1.0, small_max=1.0, large_min=1.0)


class TestFormatSize:
    def test_metres_with_class(self):
        assert format_size((0, 0, 120, 160), 120.0) == "1.00x1.33m medium"

    def test_pixels_without_calibration(self):
        assert format_size((0, 0, 50, 60), None) == "50x60px"

    def test_custom_thresholds(self):
        assert format_size((0, 0, 120, 120), 120.0, small_max=1.2, large_min=2.0) == (
            "1.00x1.00m small"
        )
