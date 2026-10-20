"""接口层：CLI 与组合根（composition root）。

这是**唯一**知道“用哪个适配器”的地方：解析 CLI → 合并配置 → 装配
（source / tracker factory / detector / sinks / renderer / clock）→ 运行
:class:`~mot.application.session.TrackingSession` → 映射退出码。

退出码契约：``0`` 正常 / ``1`` 无帧 / ``2`` 配置或输入输出错误 / ``130`` 中断 /
``3`` 未预期异常。
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from typing import Any, List, Optional, Sequence, Tuple

import yaml

from mot.adapters.clock import MonotonicClock
from mot.adapters.sinks.csv import CsvTrajectoryWriter
from mot.adapters.sinks.video import VideoWriterSink
from mot.adapters.sources import VideoSource
from mot.adapters.trackers import (
    BACKEND_NOTES,
    init_tracker,
    make_tracker_factory,
    tracker_available,
    update_tracker,
)
from mot.adapters.ui import InteractiveSession
from mot.application.config import DETECTOR_BACKENDS, AppConfig
from mot.application.errors import ConfigError, MotError
from mot.application.pipeline import MOTPipeline
from mot.application.session import SessionStats, TrackingSession
from mot.domain.association import MATCH_METHODS
from mot.domain.errors import (
    DomainError,
    SinkError,
    SourceError,
    TrackerSetupError,
)
from mot.domain.lifecycle import TrackManager
from mot.domain.limits import SUPPORTED_BACKENDS
from mot.domain.models import normalize_bbox
from mot.interfaces.ui_window import WindowRenderer, display_available
from mot.utils.logging import setup_logging

LOGGER = logging.getLogger("mot.cli")
PROG = "mot-track"


# ---------------------------------------------------------------------------
# CLI 解析
# ---------------------------------------------------------------------------
def _parse_seed_box(text: str) -> Tuple[float, float, float, float]:
    parts = [part for part in text.replace(";", ",").split(",") if part.strip()]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError(f"expected x,y,w,h but got {text!r}")
    try:
        values = tuple(float(part) for part in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"seed box values must be numbers: {text!r}") from exc
    try:
        return normalize_bbox(values)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=PROG,
        description=(
            "Pure-OpenCV multi-object tracking: one CSRT/KCF tracker per target, "
            "IoU association, unique ids, trails and csv export."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--source",
        "-s",
        default=None,
        help="camera index (e.g. 0) or path to a video file; overrides the config file",
    )
    parser.add_argument(
        "--config", "-c", default=None, help="YAML config file (see configs/default.yaml)"
    )
    parser.add_argument(
        "--tracker",
        "-t",
        choices=list(SUPPORTED_BACKENDS),
        default=None,
        help="OpenCV tracker backend",
    )
    parser.add_argument(
        "--iou-threshold", type=float, default=None, help="minimum IoU to keep an id"
    )
    parser.add_argument(
        "--max-missing", type=int, default=None, help="frames a lost target may hold its id"
    )
    parser.add_argument(
        "--trail-length", type=int, default=None, help="maximum trail points per target"
    )
    parser.add_argument(
        "--max-tracks", type=int, default=None, help="hard cap on simultaneously live ids"
    )
    parser.add_argument(
        "--match-method",
        choices=list(MATCH_METHODS),
        default=None,
        help="association algorithm (hungarian needs scipy)",
    )
    parser.add_argument(
        "--detector",
        choices=list(DETECTOR_BACKENDS),
        default=None,
        help="optional detector backend",
    )
    parser.add_argument(
        "--detector-model",
        dest="detector_model",
        default=None,
        help="local model path for the detector",
    )
    parser.add_argument(
        "--detector-interval",
        dest="detector_interval",
        type=int,
        default=None,
        help="run the detector every N frames",
    )
    parser.add_argument(
        "--detector-conf",
        dest="detector_conf",
        type=float,
        default=None,
        help="detector confidence threshold",
    )
    parser.add_argument(
        "--csv",
        dest="csv_path",
        default=None,
        help="trajectory csv path; pass 'none' to disable export",
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="append to an existing trajectory csv instead of truncating it",
    )
    parser.add_argument(
        "--record",
        dest="video_path",
        default=None,
        help="write the annotated stream to this video file",
    )
    parser.add_argument(
        "--seed-box",
        action="append",
        type=_parse_seed_box,
        default=None,
        metavar="X,Y,W,H",
        help="initialise a target at startup (repeatable)",
    )
    parser.add_argument(
        "--max-frames", type=int, default=None, help="stop after N processed frames"
    )
    parser.add_argument("--loop", action="store_true", help="restart a video file when it ends")
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="run without preview windows (tests, servers, batch runs)",
    )
    parser.add_argument(
        "--list-trackers",
        action="store_true",
        help="print available OpenCV tracker backends and exit",
    )
    parser.add_argument(
        "--dump-config", action="store_true", help="print the resolved configuration and exit"
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="debug logging")
    return parser


def build_config(args: argparse.Namespace) -> AppConfig:
    """把 CLI 覆盖合并进 YAML/默认配置。"""
    config = AppConfig.from_file(args.config) if args.config else AppConfig.defaults()
    if args.source is not None:
        config.source.source = args.source
    if args.tracker is not None:
        config.tracking.backend = args.tracker
    if args.iou_threshold is not None:
        config.tracking.iou_threshold = args.iou_threshold
    if args.max_missing is not None:
        config.tracking.max_missing = args.max_missing
    if args.trail_length is not None:
        config.tracking.trail_length = args.trail_length
    if args.match_method is not None:
        config.tracking.match_method = args.match_method
    if getattr(args, "max_tracks", None) is not None:
        config.tracking.max_tracks = args.max_tracks
    if args.detector is not None:
        config.detector.backend = args.detector
    if args.detector_model is not None:
        config.detector.model = args.detector_model
    if args.detector_interval is not None:
        config.detector.interval = args.detector_interval
    if args.detector_conf is not None:
        config.detector.confidence = args.detector_conf
    if args.csv_path is not None:
        text = str(args.csv_path).strip()
        config.output.csv_path = None if text.lower() in ("none", "-", "") else text
    if args.video_path is not None:
        config.output.video_path = args.video_path or None
    if args.seed_box:
        config.source.seed_boxes = list(args.seed_box)
    if args.max_frames is not None:
        config.source.max_frames = args.max_frames
    if getattr(args, "append", False):
        config.output.append = True
    if args.loop:
        config.source.loop = True
    if args.no_display:
        config.display.enabled = False
    config.validate()
    return config


# ---------------------------------------------------------------------------
# 装配助手
# ---------------------------------------------------------------------------
def _print_trackers() -> int:
    print(f"{'backend':8} {'available':10} notes")
    for backend in SUPPORTED_BACKENDS:
        state = "yes" if tracker_available(backend) else "no"
        print(f"{backend:8} {state:10} {BACKEND_NOTES.get(backend, '')}")
    return 0


def _class_id(item: Any) -> int:
    try:
        return int(str(item).strip())
    except (TypeError, ValueError):
        raise ConfigError(f"detector.classes entries must be class ids, got {item!r}") from None


def build_detector(config: AppConfig) -> Optional[Any]:
    """按配置构造检测器；``none`` 返回 ``None``（手工模式）。"""
    backend = config.detector.backend
    if backend == "none":
        return None
    if backend == "yolo":
        from mot.adapters.detectors.yolo import DEFAULT_MODEL, YoloDetector

        classes = [_class_id(item) for item in config.detector.classes]
        return YoloDetector(
            model=config.detector.model or DEFAULT_MODEL,
            confidence=config.detector.confidence,
            classes=classes or None,
        )
    raise ConfigError(f"unsupported detector backend {backend!r}")


def _print_summary(
    stats: SessionStats,
    elapsed: float,
    csv_writer: Optional[CsvTrajectoryWriter],
    video_sink: Optional[VideoWriterSink],
) -> None:
    fps = stats.frames / elapsed
    print(
        f"MOT finished: {stats.frames} frames in {elapsed:.1f}s ({fps:.1f} fps), "
        f"{stats.created_ids} ids created, {stats.removed_ids} destroyed, "
        f"{stats.trail_points} trail points"
    )
    if csv_writer is not None:
        print(f"trajectories: {csv_writer.path} ({csv_writer.rows_written} rows)")
    if video_sink is not None:
        print(f"annotated video: {video_sink.path} [{video_sink.codec}]")
    if stats.spawn_failures:
        print(f"tracker init failures: {stats.spawn_failures}", file=sys.stderr)


# ---------------------------------------------------------------------------
# 运行
# ---------------------------------------------------------------------------
def run(config: AppConfig) -> int:
    """装配并运行一次跟踪会话，返回进程退出码。"""
    try:
        factory = make_tracker_factory(config.tracking.backend)
        manager = TrackManager(
            factory,
            init_fn=init_tracker,
            update_fn=update_tracker,
            iou_threshold=config.tracking.iou_threshold,
            max_missing=config.tracking.max_missing,
            trail_length=config.tracking.trail_length,
            match_method=config.tracking.match_method,
            max_tracks=config.tracking.max_tracks,
        )
        detector = build_detector(config)
    except (TrackerSetupError, ConfigError, ImportError) as exc:
        LOGGER.error("%s", exc)
        return 2

    pipeline = MOTPipeline(manager, detector=detector, detection_interval=config.detector.interval)

    try:
        source = VideoSource(config.source.source, loop=config.source.loop)
    except SourceError as exc:
        LOGGER.error("%s", exc)
        return 2

    csv_writer: Optional[CsvTrajectoryWriter] = None
    if config.output.csv_path:
        try:
            csv_writer = CsvTrajectoryWriter(config.output.csv_path, append=config.output.append)
        except SinkError as exc:
            LOGGER.error("cannot open trajectory csv: %s", exc)
            source.release()
            return 2

    sinks: List[Any] = [csv_writer] if csv_writer is not None else []
    session = TrackingSession(
        pipeline,
        source,
        MonotonicClock(),
        sinks=sinks,
        seed_boxes=config.source.seed_boxes,
        max_frames=config.source.max_frames,
    )

    video_sink: Optional[VideoWriterSink] = None
    renderer: Optional[WindowRenderer] = None
    try:
        if config.output.video_path:
            video_sink = VideoWriterSink(
                config.output.video_path, source.fps, (source.width, source.height)
            )
        show_display = config.display.enabled and display_available()
        if config.display.enabled and not show_display:
            LOGGER.warning("no GUI display detected; running without preview windows")
        if show_display or video_sink is not None:
            renderer = WindowRenderer(
                config.display.window_name,
                ui=InteractiveSession(),
                session=session,
                stats=pipeline.stats,
                backend=config.tracking.backend,
                source_name=source.name,
                show_trails=config.display.show_trails,
                show_labels=config.display.show_labels,
                show_hud=config.display.show_hud,
                trail_fade=config.display.trail_fade,
                show_display=show_display,
                video_sink=video_sink,
            )
            renderer.open()
            session.attach_renderer(renderer)
    except SinkError as exc:
        LOGGER.error("cannot record annotated video: %s", exc)
        if csv_writer is not None:
            csv_writer.close()
        source.release()
        return 2

    LOGGER.debug("configuration: %s", " | ".join(config.describe()))
    started = time.perf_counter()
    stats = session.run()
    elapsed = max(1e-9, time.perf_counter() - started)
    if renderer is not None:
        renderer.close()
    if stats.frames == 0:
        LOGGER.warning("no frames were read from the source")
    _print_summary(stats, elapsed, csv_writer, video_sink)
    return stats.exit_code


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    setup_logging(args.verbose)

    if args.list_trackers:
        return _print_trackers()

    try:
        config = build_config(args)
    except ConfigError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 2

    if args.dump_config:
        print(yaml.safe_dump(config.to_dict(), sort_keys=False, allow_unicode=True).rstrip())
        return 0

    try:
        return run(config)
    except KeyboardInterrupt:
        LOGGER.info("interrupted by user")
        return 130
    except MotError as exc:
        LOGGER.error("%s", exc)
        return exc.exit_code
    except DomainError as exc:
        LOGGER.error("%s", exc)
        return 2
    except Exception as exc:  # noqa: BLE001 - 接口层兜底
        LOGGER.error("unexpected error: %s", exc)
        return 3


__all__ = [
    "build_config",
    "build_detector",
    "build_parser",
    "main",
    "run",
]
