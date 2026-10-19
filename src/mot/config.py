"""Configuration schema, YAML loading and validation.

The config file mirrors the dataclasses below (see ``configs/default.yaml``).
Unknown keys are rejected instead of silently ignored: a typo such as
``iou_treshold`` should fail loudly, not quietly fall back to a default.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

import yaml

from .matching import MATCH_METHODS
from .models import BBox, normalize_bbox
from .trackers import SUPPORTED_BACKENDS

#: Hard limits on untrusted input. They exist to bound memory/cpu usage when a
#: config file is written by something other than the demo author.
MAX_CONFIG_BYTES = 1_000_000
MAX_MISSING_LIMIT = 1_000
MAX_TRAIL_LENGTH = 10_000
MAX_TRACKS = 10_000
MAX_SEED_BOXES = 100
MAX_FRAMES = 10_000_000


class ConfigError(ValueError):
    """Raised for invalid configuration values or an unknown config key."""


def _require_bool(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise ConfigError(f"{name} must be true/false, got {value!r}")
    return value


def _as_int(value: Any, name: str, minimum: int, maximum: int) -> int:
    """Convert to int, rejecting bools, fractions and out-of-range values."""
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
    """Convert to a finite float within range, wrapping YAML type errors."""
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


@dataclass
class TrackingConfig:
    backend: str = "csrt"
    iou_threshold: float = 0.3
    max_missing: int = 15
    trail_length: int = 120
    match_method: str = "greedy"
    max_tracks: int = 256

    def validate(self) -> None:
        backend = str(self.backend).strip().lower()
        if backend not in SUPPORTED_BACKENDS:
            raise ConfigError(
                f"tracking.backend must be one of {SUPPORTED_BACKENDS}, got {self.backend!r}"
            )
        self.backend = backend
        self.iou_threshold = _as_float(
            self.iou_threshold, "tracking.iou_threshold", 0.0, 1.0
        )
        self.max_missing = _as_int(
            self.max_missing, "tracking.max_missing", 0, MAX_MISSING_LIMIT
        )
        self.trail_length = _as_int(
            self.trail_length, "tracking.trail_length", 1, MAX_TRAIL_LENGTH
        )
        self.max_tracks = _as_int(self.max_tracks, "tracking.max_tracks", 1, MAX_TRACKS)
        if self.match_method not in MATCH_METHODS:
            raise ConfigError(
                f"tracking.match_method must be one of {MATCH_METHODS}, got {self.match_method!r}"
            )


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
            self.max_frames = _as_int(self.max_frames, "source.max_frames", 1, MAX_FRAMES)
        self.loop = _require_bool(self.loop, "source.loop")

        if len(self.seed_boxes) > MAX_SEED_BOXES:
            raise ConfigError(
                f"source.seed_boxes holds {len(self.seed_boxes)} boxes; "
                f"at most {MAX_SEED_BOXES} are supported"
            )
        boxes: List[BBox] = []
        for index, raw in enumerate(self.seed_boxes):
            if not isinstance(raw, (list, tuple)) or len(raw) != 4:
                raise ConfigError(
                    f"source.seed_boxes[{index}] must be [x, y, w, h], got {raw!r}"
                )
            try:
                boxes.append(normalize_bbox(raw))
            except (TypeError, ValueError) as exc:
                raise ConfigError(f"source.seed_boxes[{index}]: {exc}") from exc
        self.seed_boxes = boxes


@dataclass
class AppConfig:
    """Top level configuration object."""

    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    display: DisplayConfig = field(default_factory=DisplayConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    source: SourceConfig = field(default_factory=SourceConfig)

    @classmethod
    def defaults(cls) -> "AppConfig":
        config = cls()
        config.validate()
        return config

    @classmethod
    def from_file(cls, path: str | Path) -> "AppConfig":
        """Load a YAML file over the defaults.

        Raises:
            ConfigError: on unreadable files, non-mapping YAML, or unknown keys.
        """
        config_path = Path(path).expanduser()
        try:
            stat = config_path.stat()
        except OSError as exc:
            raise ConfigError(f"cannot read config file {config_path}: {exc}") from exc
        if not config_path.is_file():
            raise ConfigError(f"config path is not a regular file: {config_path}")
        if stat.st_size > MAX_CONFIG_BYTES:
            raise ConfigError(
                f"config file is unreasonably large ({stat.st_size} bytes): {config_path}"
            )
        try:
            text = config_path.read_text(encoding="utf-8")
        except OSError as exc:
            raise ConfigError(f"cannot read config file {config_path}: {exc}") from exc
        if len(text) > MAX_CONFIG_BYTES:
            raise ConfigError(f"config file is unreasonably large: {config_path}")

        try:
            data = yaml.safe_load(text)
        except yaml.YAMLError as exc:
            raise ConfigError(f"invalid YAML in {config_path}: {exc}") from exc
        if data is None:
            data = {}
        if not isinstance(data, Mapping):
            raise ConfigError(f"config root must be a mapping, got {type(data).__name__}")

        config = cls()
        sections: Mapping[str, Any] = {
            "tracking": config.tracking,
            "display": config.display,
            "output": config.output,
            "source": config.source,
        }
        unknown_sections = set(data) - set(sections)
        if unknown_sections:
            raise ConfigError(
                f"unknown config section(s) {sorted(unknown_sections)}; "
                f"expected {sorted(sections)}"
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
                    f"unknown key(s) {sorted(unknown)} in section {name!r}; expected {sorted(known)}"
                )
            for key, value in payload.items():
                setattr(section, key, value)

        config.validate()
        return config

    def validate(self) -> None:
        self.tracking.validate()
        self.display.validate()
        self.output.validate()
        self.source.validate()

    def to_dict(self) -> Dict[str, Any]:
        def as_dict(obj: Any) -> Dict[str, Any]:
            out: Dict[str, Any] = {}
            for item in fields(obj):
                value = getattr(obj, item.name)
                out[item.name] = list(value) if isinstance(value, tuple) else value
            return out

        return {
            "tracking": as_dict(self.tracking),
            "display": as_dict(self.display),
            "output": as_dict(self.output),
            "source": as_dict(self.source),
        }

    def describe(self) -> Tuple[str, ...]:
        """Human readable summary used in the startup log."""
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
    "AppConfig",
    "ConfigError",
    "DisplayConfig",
    "OutputConfig",
    "SourceConfig",
    "TrackingConfig",
]
