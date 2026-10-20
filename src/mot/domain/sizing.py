"""领域层：目标尺寸估算与分级（纯逻辑，零框架依赖）。

跟踪器的包围盒单位是像素；要知道“实际尺寸”，需要一个标定比例
``pixels_per_meter``（每米多少像素，由现场标定得到，例如 1 米宽的参考物
在画面中占 120 px，则该值为 120）。

本模块只做纯数学换算与分级判定，不做任何 I/O，也不感知 OpenCV。
未提供标定时，仍可得到像素尺寸与面积，供灵敏度/阈值调参使用。
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

from .errors import ValidationError

#: 尺寸等级名。
SIZE_CLASSES: Tuple[str, ...] = ("small", "medium", "large")


def pixel_size(bbox: Sequence[float]) -> Tuple[float, float, float]:
    """返回包围盒的像素尺寸与面积：``(width_px, height_px, area_px)``。"""
    _, _, width, height = bbox
    w = max(0.0, float(width))
    h = max(0.0, float(height))
    return (w, h, w * h)


def estimate_size_meters(
    bbox: Sequence[float], pixels_per_meter: float
) -> Optional[Tuple[float, float, float]]:
    """按标定比例换算实际尺寸：``(width_m, height_m, diagonal_m)``。

    Args:
        bbox: ``(x, y, w, h)``，像素单位。
        pixels_per_meter: 每米像素数；``None`` 或 ``<= 0`` 表示未标定，返回 ``None``。

    Raises:
        ValidationError: 标定值非有限数。
    """
    if pixels_per_meter is None:
        return None
    scale = float(pixels_per_meter)
    if scale != scale or scale in (float("inf"), float("-inf")):  # NaN / inf
        raise ValidationError(
            f"pixels_per_meter must be finite, got {pixels_per_meter}",
            field="pixels_per_meter",
            value=pixels_per_meter,
        )
    if scale <= 0.0:
        return None
    width_px, height_px, _ = pixel_size(bbox)
    width_m = width_px / scale
    height_m = height_px / scale
    diagonal_m = (width_m * width_m + height_m * height_m) ** 0.5
    return (width_m, height_m, diagonal_m)


def classify_size(
    width_m: float,
    height_m: float,
    *,
    small_max: float = 0.5,
    large_min: float = 2.0,
) -> str:
    """按较长边把目标分为 ``small`` / ``medium`` / ``large``。

    Args:
        width_m: 实际宽度（米）。
        height_m: 实际高度（米）。
        small_max: 小于等于该值（米）判为 ``small``（默认 0.5，约杯子/笔记本量级）。
        large_min: 大于等于该值（米）判为 ``large``（默认 2.0，约车辆量级）。

    Raises:
        ValidationError: 阈值非法（``small_max <= 0`` 或 ``large_min <= small_max``）。
    """
    if small_max <= 0.0 or large_min <= small_max:
        raise ValidationError(
            f"size thresholds must satisfy 0 < small_max < large_min, "
            f"got small_max={small_max}, large_min={large_min}",
            field="size_thresholds",
            value=(small_max, large_min),
        )
    longest = max(float(width_m), float(height_m))
    if longest <= small_max:
        return "small"
    if longest >= large_min:
        return "large"
    return "medium"


def format_size(
    bbox: Sequence[float],
    pixels_per_meter: Optional[float],
    *,
    small_max: float = 0.5,
    large_min: float = 2.0,
) -> str:
    """生成用于标注框的尺寸文本，如 ``"0.42x0.55m medium"``。

    未标定时退化为像素文本 ``"50x60px"``。
    """
    meters = estimate_size_meters(bbox, pixels_per_meter)
    if meters is None:
        width_px, height_px, _ = pixel_size(bbox)
        return f"{width_px:.0f}x{height_px:.0f}px"
    width_m, height_m, _ = meters
    level = classify_size(width_m, height_m, small_max=small_max, large_min=large_min)
    return f"{width_m:.2f}x{height_m:.2f}m {level}"


__all__ = [
    "SIZE_CLASSES",
    "classify_size",
    "estimate_size_meters",
    "format_size",
    "pixel_size",
]
