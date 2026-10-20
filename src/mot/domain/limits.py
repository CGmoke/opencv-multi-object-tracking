"""领域层约束的单一真源：资源上限、阈值常量与边界校验器。

本模块集中定义所有 **边界常量** 与 **边界校验器**，供 ``application.config``
与 ``domain`` 各校验点共同引用，避免同一规则出现两份不一致的实现。

原则
----
* 这里只放 **上下界与取值范围**，不放“默认值”——默认值属于应用层配置
  （``application/config.py`` 与 ``configs/default.yaml``），以免出现双真源。
* 校验器统一抛出 :class:`~mot.domain.errors.ValidationError`；应用层再翻译为
  ``ConfigError``（退出码 2）。领域层只用裸字段名，应用层负责拼接成
  ``"tracking.iou_threshold"`` 之类的配置路径。
* 仅依赖标准库与 :mod:`mot.domain.errors`（不得 import ``cv2`` / ``yaml`` /
  ``csv`` / ``os``），保证核心约束可在无 OpenCV 环境下单元测试。

扩展方式（新增可调参数时）
--------------------------
定义 ``MAX_*`` 常量 → 增加对应的具名校验器 → 由 config schema 与相关领域构造器
共同引用。校验器保持“返回归一后的值、非法即抛异常”的统一契约。
"""

from __future__ import annotations

import math
from typing import Any, Optional, Tuple

from .errors import ValidationError

# ---------------------------------------------------------------------------
# 几何 / 边界常量
# ---------------------------------------------------------------------------

#: 任意包围盒（含检测框、手工框、跟踪器输出）允许的最小边长（像素）。
MIN_BOX_SIDE: float = 1.0

# ---------------------------------------------------------------------------
# 资源上限（面向不可信输入的内存 / CPU 防护）
# ---------------------------------------------------------------------------

#: ``max_missing`` 上界：目标丢失后最多保留多少帧的 id。
MAX_MISSING_LIMIT: int = 1_000

#: ``trail_length`` 上界：单个目标保留的轨迹点上限。
MAX_TRAIL_LENGTH: int = 10_000

#: ``max_tracks`` 上界：同时存活 id 的硬上限（默认值 256 在应用层配置中）。
MAX_TRACKS: int = 10_000

#: 单帧吸收的检测框数量上限，避免 O(n^2) 去重与建 id 风暴。
MAX_DETECTIONS: int = 256

#: ``seed_boxes`` 数量上限。
MAX_SEED_BOXES: int = 100

#: ``max_frames`` 上界：单次运行处理的帧数上限。
MAX_FRAMES: int = 10_000_000

# ---------------------------------------------------------------------------
# IoU 阈值范围：半开区间 (0, 1]
# ---------------------------------------------------------------------------

#: ``iou_threshold`` 下界，**开区间**：恰好取 0 会让任意检测劫持已有 id。
IOU_THRESHOLD_MIN: float = 0.0

#: ``iou_threshold`` 上界，**闭区间**：1 表示“完全重合才算同一目标”。
IOU_THRESHOLD_MAX: float = 1.0

# ---------------------------------------------------------------------------
# 跟踪器后端枚举（封闭集合）
# ---------------------------------------------------------------------------

#: 支持的 OpenCV 跟踪器后端名。
#:
#: 放在领域层是为了让 ``application.config`` 能校验后端名而**不必依赖 adapters**
#: （架构守卫禁止 ``application → adapters``）；``adapters/trackers.py`` 反向引用
#: 本常量实现这些后端，从而保持单一真源。
SUPPORTED_BACKENDS: Tuple[str, ...] = ("csrt", "kcf", "mil", "mosse")


# ---------------------------------------------------------------------------
# 校验器
# ---------------------------------------------------------------------------


def _coerce_int(
    value: Any,
    *,
    field: str,
    minimum: int,
    maximum: Optional[int] = None,
) -> int:
    """把一个输入归一为落在 ``[minimum, maximum]`` 内的整数。

    与既有配置层语义保持一致：拒绝布尔值（``True`` 不应被当成 ``1``），
    接受整值浮点（如 ``256.0``），拒绝非整值浮点（如 ``256.5``），
    对字符串尝试 ``int()`` 解析（适配 CLI 传入）。

    Args:
        value: 原始输入。
        field: 用于错误信息的裸字段名。
        minimum: 允许的下界（闭区间）。
        maximum: 允许的上界（闭区间）；``None`` 表示不设上界。

    Returns:
        归一后的 ``int``。

    Raises:
        ValidationError: 类型非法、非整数、非有限或越界。
    """
    if isinstance(value, bool):
        raise ValidationError(
            f"{field} must be an integer, got {value!r}", field=field, value=value
        )
    if isinstance(value, float):
        if not math.isfinite(value) or not value.is_integer():
            raise ValidationError(
                f"{field} must be an integer, got {value!r}", field=field, value=value
            )
        number = int(value)
    else:
        try:
            number = int(value)
        except (TypeError, ValueError):
            raise ValidationError(
                f"{field} must be an integer, got {value!r}", field=field, value=value
            ) from None

    if number < minimum:
        raise ValidationError(
            f"{field} must be >= {minimum}, got {number}", field=field, value=value
        )
    if maximum is not None and number > maximum:
        raise ValidationError(
            f"{field} must be within [{minimum}, {maximum}], got {number}",
            field=field,
            value=value,
        )
    return number


def validate_iou_threshold(value: Any) -> float:
    """校验 IoU 阈值，必须落在半开区间 ``(0, 1]``。

    这是关联与同帧去重的共用阈值：低于阈值视为不同目标，达到阈值
    （包含式 ``>=``）视为同一目标。**下界为开区间**——恰好为 0 会让任意
    检测与任意已有目标配对，从而劫持已有 id，因此被明确拒绝。

    Args:
        value: 原始输入（数字或可转数字的字符串）。

    Returns:
        归一后的 ``float``。

    Raises:
        ValidationError: 布尔值、不可转数字、非有限值或落在 ``(0, 1]`` 之外。
    """
    if isinstance(value, bool):
        raise ValidationError(
            f"iou_threshold must be a number in (0, 1], got {value!r}",
            field="iou_threshold",
            value=value,
        )
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ValidationError(
            f"iou_threshold must be a number in (0, 1], got {value!r}",
            field="iou_threshold",
            value=value,
        ) from None

    if not math.isfinite(number) or not (IOU_THRESHOLD_MIN < number <= IOU_THRESHOLD_MAX):
        raise ValidationError(
            f"iou_threshold must be within (0, 1], got {number!r} "
            "(0 would let any detection hijack an existing id)",
            field="iou_threshold",
            value=value,
        )
    return number


def validate_max_missing(value: Any) -> int:
    """校验 ``max_missing``：目标丢失后保留 id 的帧数，范围 ``[0, MAX_MISSING_LIMIT]``。

    ``0`` 表示目标首次丢失即销毁 id。
    """
    return _coerce_int(value, field="max_missing", minimum=0, maximum=MAX_MISSING_LIMIT)


def validate_trail_length(value: Any) -> int:
    """校验 ``trail_length``：单目标轨迹点上限，范围 ``[1, MAX_TRAIL_LENGTH]``。"""
    return _coerce_int(value, field="trail_length", minimum=1, maximum=MAX_TRAIL_LENGTH)


def validate_max_tracks(value: Any) -> int:
    """校验 ``max_tracks``：同时存活 id 的硬上限，范围 ``[1, MAX_TRACKS]``。"""
    return _coerce_int(value, field="max_tracks", minimum=1, maximum=MAX_TRACKS)


def validate_max_frames(value: Any) -> int:
    """校验 ``max_frames``：单次运行的帧数上限，范围 ``[1, MAX_FRAMES]``。"""
    return _coerce_int(value, field="max_frames", minimum=1, maximum=MAX_FRAMES)


def validate_seed_box_count(value: Any) -> int:
    """校验 ``seed_boxes`` 的**数量**，范围 ``[0, MAX_SEED_BOXES]``。

    仅校验数量上界（单框内容由 ``domain.models.normalize_bbox`` 校验），
    对应既有配置层 ``len(seed_boxes) > MAX_SEED_BOXES`` 的检查。
    """
    return _coerce_int(value, field="seed_boxes", minimum=0, maximum=MAX_SEED_BOXES)


def validate_tracker_backend(value: Any) -> str:
    """校验跟踪器后端名是否在 :data:`SUPPORTED_BACKENDS` 内，并归一为小写。

    Returns:
        归一（``strip().lower()``）后的后端名。

    Raises:
        ValidationError: 后端名不在支持集合内。
    """
    text = str(value).strip().lower()
    if text not in SUPPORTED_BACKENDS:
        raise ValidationError(
            f"tracker backend must be one of {SUPPORTED_BACKENDS}, got {value!r}",
            field="backend",
            value=value,
        )
    return text


__all__ = [
    "IOU_THRESHOLD_MAX",
    "IOU_THRESHOLD_MIN",
    "MAX_DETECTIONS",
    "MAX_FRAMES",
    "MAX_MISSING_LIMIT",
    "MAX_SEED_BOXES",
    "MAX_TRACKS",
    "MAX_TRAIL_LENGTH",
    "MIN_BOX_SIDE",
    "SUPPORTED_BACKENDS",
    "validate_iou_threshold",
    "validate_max_frames",
    "validate_max_missing",
    "validate_max_tracks",
    "validate_seed_box_count",
    "validate_tracker_backend",
    "validate_trail_length",
]
