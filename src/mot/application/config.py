"""应用层配置：YAML 加载、严格校验与 CLI 覆盖。

单一真源：所有数值范围/常量来自 :mod:`mot.domain.limits`，``match_method`` 来自
:mod:`mot.domain.association`，``backend`` 名字来自 ``domain.limits.SUPPORTED_BACKENDS``。
本层**不 import adapters**（架构守卫强制），因此后端可用性（取决于 OpenCV 构建）
由装配阶段交给跟踪器适配器做 eager 校验（``TrackerSetupError``）。

优先级：defaults < 配置文件 < 环境变量（未来）< CLI。未知键报错；越界报错；不静默
截断。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

import yaml

from mot.domain.association import MATCH_METHODS
from mot.domain.errors import ValidationError
from mot.domain.limits import (
    MAX_FRAMES,
    validate_iou_threshold,
    validate_max_frames,
    validate_max_missing,
    validate_max_tracks,
    validate_seed_box_count,
    validate_tracker_backend,
    validate_trail_length,
)
from mot.domain.models import BBox, normalize_bbox

from .errors import ConfigError

#: 配置文件体积上限（不可信输入的资源防护，属应用层 I/O 约束）。
MAX_CONFIG_BYTES = 1_000_000

#: 检测器后端枚举；``none`` 表示禁用检测（默认）。
DETECTOR_BACKENDS: Tuple[str, ...] = ("none", "yolo")

#: 运动检测的背景建模后端（对应 ``MotionDetector(method=...)``）。
BACKGROUND_METHODS: Tuple[str, ...] = ("mog2", "knn")


# ---------------------------------------------------------------------------
# 校验辅助
# ---------------------------------------------------------------------------


def _require_bool(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise ConfigError(f"{name} must be true/false, got {value!r}")
    return value


def _as_int(value: Any, name: str, minimum: int, maximum: int) -> int:
    """把输入归一为 ``[minimum, maximum]`` 内的整数（拒绝 bool）。"""
    if isinstance(value, bool):
        raise ConfigError(f"{name} must be an integer, got {value!r}")
    if isinstance(value, float):
        if not value.is_integer():
            raise ConfigError(f"{name} must be an integer, got {value!r}")
        number = int(value)
    else:
        try:
            number = int(value)
        except (TypeError, ValueError):
            raise ConfigError(f"{name} must be an integer, got {value!r}") from None
    if not minimum <= number <= maximum:
        raise ConfigError(f"{name} must be within [{minimum}, {maximum}], got {number}")
    return number


def _as_float(value: Any, name: str, minimum: float, maximum: float) -> float:
    """把输入归一为有限且落在 ``[minimum, maximum]`` 的浮点数。"""
    if isinstance(value, bool):
        raise ConfigError(f"{name} must be a number, got {value!r}")
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ConfigError(f"{name} must be a number, got {value!r}") from None
    if not math.isfinite(number):
        raise ConfigError(f"{name} must be finite, got {value!r}")
    if not minimum <= number <= maximum:
        raise ConfigError(f"{name} must be within [{minimum}, {maximum}], got {number}")
    return number


def _as_kernel(value: Any, name: str, maximum: int = 99) -> int:
    """滤波核边长：``0`` 关闭，否则必须是奇数（OpenCV 要求）。"""
    number = _as_int(value, name, 0, maximum)
    if number and number % 2 == 0:
        raise ConfigError(f"{name} must be 0 (off) or an odd number, got {number}")
    return number


def _translated(validator: Any, value: Any, *, field_prefix: str) -> Any:
    """调用领域校验器，把 ``ValidationError`` 翻译为带配置路径的 ``ConfigError``。"""
    try:
        return validator(value)
    except ValidationError as exc:
        raise ConfigError(f"{field_prefix}: {exc.message}") from None


# ---------------------------------------------------------------------------
# 配置段
# ---------------------------------------------------------------------------


@dataclass
class TrackingConfig:
    backend: str = "csrt"
    iou_threshold: float = 0.3
    max_missing: int = 15
    trail_length: int = 120
    match_method: str = "greedy"
    max_tracks: int = 256
    #: 框平滑系数（EMA alpha）：``1`` 表示不平滑（直接用检测框），越小框越稳但越滞后。
    box_smoothing: float = 1.0
    #: 轨迹二次校验：与已有轨迹重叠但尺寸突变超过该倍率的检测被丢弃（1.0 关闭）。
    max_box_change: float = 1.6

    def validate(self) -> None:
        self.backend = _translated(
            validate_tracker_backend, self.backend, field_prefix="tracking.backend"
        )
        self.iou_threshold = _translated(
            validate_iou_threshold, self.iou_threshold, field_prefix="tracking.iou_threshold"
        )
        self.max_missing = _translated(
            validate_max_missing, self.max_missing, field_prefix="tracking.max_missing"
        )
        self.trail_length = _translated(
            validate_trail_length, self.trail_length, field_prefix="tracking.trail_length"
        )
        self.max_tracks = _translated(
            validate_max_tracks, self.max_tracks, field_prefix="tracking.max_tracks"
        )
        if self.match_method not in MATCH_METHODS:
            raise ConfigError(
                f"tracking.match_method must be one of {MATCH_METHODS}, got {self.match_method!r}"
            )
        self.box_smoothing = _as_float(
            self.box_smoothing, "tracking.box_smoothing", 1e-6, 1.0
        )
        self.max_box_change = _as_float(
            self.max_box_change, "tracking.max_box_change", 1.0, 100.0
        )


@dataclass
class DetectorConfig:
    """可选检测器配置段（默认禁用，接线见 ARCHITECTURE_DESIGN §6.1 P5）。"""

    backend: str = "none"
    model: Optional[str] = None
    confidence: float = 0.35
    classes: List[str] = field(default_factory=list)
    interval: int = 1

    def validate(self) -> None:
        backend = str(self.backend).strip().lower()
        if backend not in DETECTOR_BACKENDS:
            raise ConfigError(
                f"detector.backend must be one of {DETECTOR_BACKENDS}, got {self.backend!r}"
            )
        self.backend = backend
        if self.model is not None:
            text = str(self.model).strip()
            if not text:
                raise ConfigError("detector.model must be null or a non-empty local path")
            self.model = text
        self.confidence = _as_float(self.confidence, "detector.confidence", 0.0, 1.0)
        self.interval = _as_int(self.interval, "detector.interval", 1, MAX_FRAMES)
        if not isinstance(self.classes, (list, tuple)):
            raise ConfigError(f"detector.classes must be a list, got {type(self.classes).__name__}")
        cleaned: List[str] = []
        for item in self.classes:
            text = str(item).strip()
            if not text:
                raise ConfigError("detector.classes entries must be non-empty strings")
            cleaned.append(text)
        self.classes = cleaned


@dataclass
class DisplayConfig:
    enabled: bool = True
    show_trails: bool = True
    show_labels: bool = True
    show_hud: bool = True
    trail_fade: bool = True
    window_name: str = "OpenCV MOT"

    def validate(self) -> None:
        for name in ("enabled", "show_trails", "show_labels", "show_hud", "trail_fade"):
            setattr(self, name, _require_bool(getattr(self, name), f"display.{name}"))
        if not str(self.window_name).strip():
            raise ConfigError("display.window_name must not be empty")


@dataclass
class OutputConfig:
    csv_path: Optional[str] = "outputs/trajectories.csv"
    video_path: Optional[str] = None
    append: bool = False

    def validate(self) -> None:
        for name in ("csv_path", "video_path"):
            value = getattr(self, name)
            if value is None:
                continue
            text = str(value).strip()
            if not text:
                raise ConfigError(f"output.{name} must be null or a non-empty path")
            if text.endswith(("\\", "/")):
                raise ConfigError(f"output.{name} must be a file path, not a directory: {text!r}")
            setattr(self, name, text)
        self.append = _require_bool(self.append, "output.append")


@dataclass
class SourceConfig:
    source: str = "0"
    max_frames: Optional[int] = None
    loop: bool = False
    seed_boxes: List[BBox] = field(default_factory=list)

    def validate(self) -> None:
        if not str(self.source).strip():
            raise ConfigError("source.source must be a camera index or a video path")
        self.source = str(self.source).strip()
        if self.max_frames is not None:
            self.max_frames = _translated(
                validate_max_frames, self.max_frames, field_prefix="source.max_frames"
            )
        self.loop = _require_bool(self.loop, "source.loop")

        _translated(validate_seed_box_count, len(self.seed_boxes), field_prefix="source.seed_boxes")
        boxes: List[BBox] = []
        for index, raw in enumerate(self.seed_boxes):
            if not isinstance(raw, (list, tuple)) or len(raw) != 4:
                raise ConfigError(f"source.seed_boxes[{index}] must be [x, y, w, h], got {raw!r}")
            try:
                boxes.append(normalize_bbox(raw))
            except ValidationError as exc:
                raise ConfigError(f"source.seed_boxes[{index}]: {exc.message}") from None
        self.seed_boxes = boxes


@dataclass
class MotionConfig:
    """自动运动检测配置（无需检测器权重，基于背景建模）。

    关键调参项：
        * ``method``：背景建模后端（``mog2`` / ``knn``），均关闭阴影检测；
        * ``var_threshold``：MOG2 灵敏度，越小越敏感（默认 25，高于 OpenCV 默认 16）；
        * ``dist2_threshold``：KNN 的平方距离阈值（仅 ``method="knn"`` 生效）；
        * ``min_area_px`` / ``min_width_px`` / ``min_height_px``：最小目标尺寸阈值；
        * ``min_motion_px``：位移门限，用于排除静止物体；
        * ``min_consecutive``：连续命中帧数，用于抑制噪点；
        * ``blur_kernel`` / ``median_kernel``：输入灰度图的高斯 / 中值滤波核（压噪）；
        * ``temporal_frames``：最近 N 帧掩码按位与，压掉单帧闪烁；
        * ``min_aspect`` / ``max_aspect``：长宽比范围，剔极细长纹理；
        * ``min_fill_ratio``：轮廓填充率下限，剔中空/破碎的局部特征；
        * ``merge_iou`` / ``merge_gap_px``：同一物体碎片框的合并条件（间隙默认 0，
          只合并重叠/相接的框，避免误合并相邻物体）；
        * ``nms_iou`` / ``nms_containment``：抑制被整体框包含的局部框；
        * ``pixels_per_meter``：尺寸标定（每米像素），用于换算实际尺寸。
    """

    enabled: bool = False
    method: str = "mog2"
    history: int = 200
    var_threshold: float = 25.0
    dist2_threshold: float = 400.0
    learning_rate: float = 0.0
    warmup_frames: int = 20
    warmup_rate: float = 0.05
    min_area_px: float = 200.0
    min_width_px: int = 8
    min_height_px: int = 8
    max_area_ratio: float = 0.6
    blur_kernel: int = 5
    median_kernel: int = 0
    temporal_frames: int = 2
    min_aspect: float = 0.1
    max_aspect: float = 10.0
    min_fill_ratio: float = 0.0
    merge_iou: float = 0.15
    merge_gap_px: float = 0.0
    nms_iou: float = 0.6
    nms_containment: float = 0.8
    min_consecutive: int = 2
    min_motion_px: float = 6.0
    morph_kernel: int = 3
    interval: int = 2
    pixels_per_meter: Optional[float] = None
    small_max_m: float = 0.5
    large_min_m: float = 2.0

    def validate(self) -> None:
        self.enabled = _require_bool(self.enabled, "motion.enabled")
        backend = str(self.method).strip().lower()
        if backend not in BACKGROUND_METHODS:
            raise ConfigError(
                f"motion.method must be one of {BACKGROUND_METHODS}, got {self.method!r}"
            )
        self.method = backend
        self.history = _as_int(self.history, "motion.history", 1, 100_000)
        self.var_threshold = _as_float(self.var_threshold, "motion.var_threshold", 1e-6, 255.0)
        self.dist2_threshold = _as_float(
            self.dist2_threshold, "motion.dist2_threshold", 1e-6, 1e9
        )
        self.learning_rate = _as_float(self.learning_rate, "motion.learning_rate", 0.0, 1.0)
        self.warmup_frames = _as_int(self.warmup_frames, "motion.warmup_frames", 0, 100_000)
        self.warmup_rate = _as_float(self.warmup_rate, "motion.warmup_rate", 1e-6, 1.0)
        self.min_area_px = _as_float(self.min_area_px, "motion.min_area_px", 0.0, 1e9)
        self.min_width_px = _as_int(self.min_width_px, "motion.min_width_px", 1, 100_000)
        self.min_height_px = _as_int(self.min_height_px, "motion.min_height_px", 1, 100_000)
        self.max_area_ratio = _as_float(self.max_area_ratio, "motion.max_area_ratio", 1e-6, 1.0)
        self.blur_kernel = _as_kernel(self.blur_kernel, "motion.blur_kernel")
        self.median_kernel = _as_kernel(self.median_kernel, "motion.median_kernel")
        self.temporal_frames = _as_int(self.temporal_frames, "motion.temporal_frames", 1, 9)
        self.min_aspect = _as_float(self.min_aspect, "motion.min_aspect", 1e-6, 1000.0)
        self.max_aspect = _as_float(self.max_aspect, "motion.max_aspect", 1e-6, 1000.0)
        if self.max_aspect < self.min_aspect:
            raise ConfigError(
                "motion.max_aspect must be >= motion.min_aspect, "
                f"got min_aspect={self.min_aspect}, max_aspect={self.max_aspect}"
            )
        self.min_fill_ratio = _as_float(self.min_fill_ratio, "motion.min_fill_ratio", 0.0, 1.0)
        self.merge_iou = _as_float(self.merge_iou, "motion.merge_iou", 0.0, 1.0)
        self.merge_gap_px = _as_float(self.merge_gap_px, "motion.merge_gap_px", 0.0, 1e9)
        self.nms_iou = _as_float(self.nms_iou, "motion.nms_iou", 0.0, 1.0)
        self.nms_containment = _as_float(
            self.nms_containment, "motion.nms_containment", 1e-6, 1.0
        )
        self.min_consecutive = _as_int(self.min_consecutive, "motion.min_consecutive", 1, 1000)
        self.min_motion_px = _as_float(self.min_motion_px, "motion.min_motion_px", 0.0, 100_000.0)
        self.morph_kernel = _as_int(self.morph_kernel, "motion.morph_kernel", 1, 99)
        self.interval = _as_int(self.interval, "motion.interval", 1, 1000)
        if self.pixels_per_meter is not None:
            self.pixels_per_meter = _as_float(
                self.pixels_per_meter, "motion.pixels_per_meter", 1e-6, 1e9
            )
        self.small_max_m = _as_float(self.small_max_m, "motion.small_max_m", 1e-6, 1e6)
        self.large_min_m = _as_float(self.large_min_m, "motion.large_min_m", 1e-6, 1e6)
        if self.large_min_m <= self.small_max_m:
            raise ConfigError(
                "motion.large_min_m must be greater than motion.small_max_m, "
                f"got small_max_m={self.small_max_m}, large_min_m={self.large_min_m}"
            )


@dataclass
class WebConfig:
    """本地 Web 监测界面配置（默认仅监听本机）。"""

    host: str = "127.0.0.1"
    port: int = 8000
    max_sessions: int = 4
    stream_fps: float = 15.0
    upload_dir: Optional[str] = None
    max_upload_bytes: int = 2_000_000_000

    def validate(self) -> None:
        if not str(self.host).strip():
            raise ConfigError("web.host must not be empty")
        self.host = str(self.host).strip()
        self.port = _as_int(self.port, "web.port", 1, 65535)
        self.max_sessions = _as_int(self.max_sessions, "web.max_sessions", 1, 64)
        self.stream_fps = _as_float(self.stream_fps, "web.stream_fps", 1.0, 60.0)
        if self.upload_dir is not None:
            text = str(self.upload_dir).strip()
            if not text:
                raise ConfigError("web.upload_dir must be null or a non-empty path")
            self.upload_dir = text
        self.max_upload_bytes = _as_int(
            self.max_upload_bytes, "web.max_upload_bytes", 1, 100_000_000_000
        )


@dataclass
class AppConfig:
    """顶层配置对象。"""

    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    motion: MotionConfig = field(default_factory=MotionConfig)
    display: DisplayConfig = field(default_factory=DisplayConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    source: SourceConfig = field(default_factory=SourceConfig)
    web: WebConfig = field(default_factory=WebConfig)

    @classmethod
    def defaults(cls) -> "AppConfig":
        config = cls()
        config.validate()
        return config

    @classmethod
    def from_file(cls, path: str | Path) -> "AppConfig":
        """在默认值之上加载 YAML 文件。

        Raises:
            ConfigError: 文件不可读、非 mapping、未知段/键或数值非法。
        """
        config_path = Path(path).expanduser()
        try:
            stat = config_path.stat()
        except OSError as exc:
            raise ConfigError(f"cannot read config file {config_path}: {exc}") from None
        if not config_path.is_file():
            raise ConfigError(f"config path is not a regular file: {config_path}")
        if stat.st_size > MAX_CONFIG_BYTES:
            raise ConfigError(
                f"config file is unreasonably large ({stat.st_size} bytes): {config_path}"
            )
        try:
            text = config_path.read_text(encoding="utf-8")
        except OSError as exc:
            raise ConfigError(f"cannot read config file {config_path}: {exc}") from None
        if len(text) > MAX_CONFIG_BYTES:
            raise ConfigError(f"config file is unreasonably large: {config_path}")

        try:
            data = yaml.safe_load(text)
        except yaml.YAMLError as exc:
            raise ConfigError(f"invalid YAML in {config_path}: {exc}") from None
        if data is None:
            data = {}
        if not isinstance(data, Mapping):
            raise ConfigError(f"config root must be a mapping, got {type(data).__name__}")

        config = cls()
        sections: Dict[str, Any] = {
            "tracking": config.tracking,
            "detector": config.detector,
            "motion": config.motion,
            "display": config.display,
            "output": config.output,
            "source": config.source,
            "web": config.web,
        }
        unknown_sections = set(data) - set(sections)
        if unknown_sections:
            raise ConfigError(
                f"unknown config section(s) {sorted(unknown_sections)}; expected {sorted(sections)}"
            )
        for name, section in sections.items():
            if name not in data:
                continue
            payload = data[name]
            if not isinstance(payload, Mapping):
                raise ConfigError(f"config section {name!r} must be a mapping")
            known = {item.name for item in fields(type(section))}
            unknown = set(payload) - known
            if unknown:
                raise ConfigError(
                    f"unknown key(s) {sorted(unknown)} in section {name!r}; "
                    f"expected {sorted(known)}"
                )
            for key, value in payload.items():
                setattr(section, key, value)

        config.validate()
        return config

    def validate(self) -> None:
        self.tracking.validate()
        self.detector.validate()
        self.motion.validate()
        self.display.validate()
        self.output.validate()
        self.source.validate()
        self.web.validate()

    def to_dict(self) -> Dict[str, Any]:
        def as_dict(obj: Any) -> Dict[str, Any]:
            out: Dict[str, Any] = {}
            for item in fields(obj):
                value = getattr(obj, item.name)
                out[item.name] = list(value) if isinstance(value, tuple) else value
            return out

        return {
            "tracking": as_dict(self.tracking),
            "detector": as_dict(self.detector),
            "motion": as_dict(self.motion),
            "display": as_dict(self.display),
            "output": as_dict(self.output),
            "source": as_dict(self.source),
            "web": as_dict(self.web),
        }

    def describe(self) -> Tuple[str, ...]:
        """启动日志用的人类可读摘要（三行）。"""
        return (
            f"backend={self.tracking.backend} iou={self.tracking.iou_threshold:.2f} "
            f"max_missing={self.tracking.max_missing} trail={self.tracking.trail_length} "
            f"max_tracks={self.tracking.max_tracks}",
            f"source={self.source.source} max_frames={self.source.max_frames} "
            f"loop={self.source.loop} seeds={len(self.source.seed_boxes)}",
            f"csv={self.output.csv_path} video={self.output.video_path} "
            f"append={self.output.append} display={self.display.enabled}",
        )


__all__ = [
    "BACKGROUND_METHODS",
    "DETECTOR_BACKENDS",
    "MAX_CONFIG_BYTES",
    "AppConfig",
    "ConfigError",
    "DetectorConfig",
    "DisplayConfig",
    "MotionConfig",
    "OutputConfig",
    "SourceConfig",
    "TrackingConfig",
    "WebConfig",
]
