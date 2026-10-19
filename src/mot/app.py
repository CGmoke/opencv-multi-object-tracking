"""Application entry point: CLI, event loop, rendering and export."""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path
from typing import Optional, Sequence, Tuple

import cv2
import yaml

from .config import AppConfig, ConfigError
from .interactive import InteractiveSession
from .models import AppStats, TrackedObject, clip_bbox, normalize_bbox
from .pipeline import MOTPipeline
from .track_manager import TrackManager
from .trackers import (
    BACKEND_NOTES,
    SUPPORTED_BACKENDS,
    TrackerBackendError,
    make_tracker_factory,
    tracker_available,
)
from .trajectory import CsvTrajectoryWriter
from .utils import FpsMeter, sanitize_filename, setup_logging
from .video_source import VideoSource, VideoSourceError, VideoWriterSink
from .visualization import (
    draw_banner,
    draw_help,
    draw_hud,
    draw_selection,
    draw_tracks,
)

LOGGER = logging.getLogger("mot.app")
PROG = "mot-track"


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
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
        "--config",
        "-c",
        default=None,
        help="YAML config file (see configs/default.yaml)",
    )
    parser.add_argument(
        "--tracker",
        "-t",
        choices=list(SUPPORTED_BACKENDS),
        default=None,
        help="OpenCV tracker backend",
    )
    parser.add_argument("--iou-threshold", type=float, default=None, help="minimum IoU to keep an id")
    parser.add_argument(
        "--max-missing",
        type=int,
        default=None,
        help="frames a lost target may hold its id before the id is destroyed",
    )
    parser.add_argument(
        "--trail-length",
        type=int,
        default=None,
        help="maximum trail points kept per target",
    )
    parser.add_argument(
        "--max-tracks",
        type=int,
        default=None,
        help="hard cap on simultaneously live ids",
    )
    parser.add_argument(
        "--match-method",
        choices=("greedy", "hungarian"),
        default=None,
        help="association algorithm (hungarian needs scipy)",
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
        help="initialise a target at startup (repeatable); useful headless",
    )
    parser.add_argument("--max-frames", type=int, default=None, help="stop after N processed frames")
    parser.add_argument(
        "--loop",
        action="store_true",
        help="restart a video file when it ends",
    )
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
        "--dump-config",
        action="store_true",
        help="print the resolved configuration and exit",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="debug logging")
    return parser


def build_config(args: argparse.Namespace) -> AppConfig:
    """Merge CLI overrides into the YAML/default configuration."""
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


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _print_trackers() -> int:
    print(f"{'backend':8} {'available':10} notes")
    for backend in SUPPORTED_BACKENDS:
        state = "yes" if tracker_available(backend) else "no"
        print(f"{backend:8} {state:10} {BACKEND_NOTES[backend]}")
    return 0


def _display_available() -> bool:
    """Best-effort check for a usable GUI before creating windows."""
    if sys.platform.startswith("linux"):
        if not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY"):
            return False
    return True


def _mouse_callback(session: InteractiveSession):
    def callback(event: int, x: int, y: int, flags: int, _param: object) -> None:
        session.handle_mouse(event, x, y, flags)

    return callback


def save_snapshot(
    frame, directory: str | Path, source_name: str, frame_index: int
) -> Path:
    """Write the current annotated frame to ``directory`` and return the path.

    Repeated presses of ``s`` on one paused frame must not overwrite the
    previous screenshot, and a pre-planted symlink must not be followed into an
    unrelated file.
    """
    base = Path(directory) / (
        f"{sanitize_filename(source_name, 'source')}_frame_{int(frame_index):06d}"
    )
    path = base.with_name(base.name + ".png")
    if path.is_symlink():
        raise OSError(f"refusing to overwrite a symlink snapshot: {path}")
    counter = 1
    while path.exists():
        path = base.with_name(f"{base.name}_{counter}.png")
        counter += 1
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), frame):
        raise OSError(f"could not write snapshot {path}")
    return path


def _window_closed(window: str) -> bool:
    """True when the user closed the preview window with the window manager."""
    try:
        return cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1
    except cv2.error:
        return True


# --------------------------------------------------------------------------- #
# Main loop
# --------------------------------------------------------------------------- #
def run(config: AppConfig) -> int:
    """Run one tracking session. Returns a process exit code."""
    try:
        factory = make_tracker_factory(config.tracking.backend)
    except TrackerBackendError as exc:
        LOGGER.error("%s", exc)
        return 2

    manager = TrackManager(
        factory,
        iou_threshold=config.tracking.iou_threshold,
        max_missing=config.tracking.max_missing,
        trail_length=config.tracking.trail_length,
        match_method=config.tracking.match_method,
        max_tracks=config.tracking.max_tracks,
    )
    pipeline = MOTPipeline(manager)

    try:
        source = VideoSource(config.source.source, loop=config.source.loop)
    except VideoSourceError as exc:
        LOGGER.error("%s", exc)
        return 2

    display_enabled = config.display.enabled and _display_available()
    if config.display.enabled and not display_enabled:
        LOGGER.warning("no GUI display detected; running without preview windows")
    window = config.display.window_name
    session = InteractiveSession()
    if display_enabled:
        try:
            cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)
        except cv2.error as exc:
            LOGGER.warning("cannot create preview window (%s); continuing without it", exc)
            display_enabled = False
        else:
            cv2.setMouseCallback(window, _mouse_callback(session))

    csv_writer: Optional[CsvTrajectoryWriter] = None
    if config.output.csv_path:
        try:
            csv_writer = CsvTrajectoryWriter(
                config.output.csv_path, append=config.output.append
            )
        except OSError as exc:
            LOGGER.error("cannot open trajectory csv: %s", exc)
            source.release()
            if display_enabled:
                cv2.destroyAllWindows()
            return 2

    video_sink: Optional[VideoWriterSink] = None
    csv_path = csv_writer.path if csv_writer is not None else None
    fps_meter = FpsMeter()
    stats = AppStats()
    processed = 0
    seeded = False
    last_render = None
    fps_estimate = source.fps
    started = time.perf_counter()
    exit_code = 0
    LOGGER.debug("configuration: %s", " | ".join(config.describe()))

    def apply_event(event: Optional[str], current_frame) -> None:
        """Act on an interactive event; ``current_frame`` is the last canvas."""
        if event == "clear_trails":
            pipeline.clear_trails()
            session.set_notice("trails cleared")
        elif event == "reset":
            pipeline.reset()
            session.set_notice("all ids cleared; numbering continues")
        elif event == "save_frame":
            if current_frame is None:
                session.set_notice("nothing to save yet", 45)
                return
            try:
                path = save_snapshot(
                    current_frame, "outputs/snapshots", source.name, manager.frame_index
                )
            except OSError as exc:
                LOGGER.error("%s", exc)
                session.set_notice("snapshot failed", 60)
            else:
                session.set_notice(f"saved {path.name}")

    try:
        while not session.quit_requested:
            if session.paused:
                if display_enabled:
                    key = cv2.waitKey(30) & 0xFF
                    apply_event(session.handle_key(key), last_render)
                    if last_render is not None:
                        cv2.imshow(window, last_render)
                        if _window_closed(window):
                            break
                else:
                    time.sleep(0.05)
                continue

            ok, frame = source.read()
            if not ok or frame is None:
                LOGGER.info("end of stream after %d frames", processed)
                break

            if not seeded:
                seeded = True
                for index, box in enumerate(config.source.seed_boxes):
                    clipped = clip_bbox(box, frame.shape[1], frame.shape[0])
                    if clipped is None:
                        LOGGER.warning("seed box %d lies outside the frame; skipped", index)
                        continue
                    if pipeline.add_manual_box(frame, clipped) is None:
                        LOGGER.warning("seed box %d could not initialise a tracker", index)

            for box in session.consume_new_boxes():
                clipped = clip_bbox(box, frame.shape[1], frame.shape[0])
                if clipped is None:
                    session.set_notice("box outside the frame; ignored", 45)
                    continue
                if pipeline.add_manual_box(frame, clipped) is None:
                    session.set_notice("tracker init failed", 60)

            for point in session.consume_delete_points():
                removed_id = pipeline.delete_at(point)
                if removed_id is None:
                    session.set_notice("no target under the click", 45)
                else:
                    session.set_notice(f"deleted id {removed_id}")

            result = pipeline.step(frame)
            fps_meter.tick()
            timestamp = processed / fps_estimate if fps_estimate > 0 else 0.0
            if csv_writer is not None:
                active: Tuple[TrackedObject, ...] = tuple(
                    track for track in result.tracks if track.is_active
                )
                csv_writer.write_frame(result.frame_index, timestamp, active)

            if video_sink is None and config.output.video_path:
                try:
                    video_sink = VideoWriterSink(
                        config.output.video_path,
                        fps_estimate,
                        (frame.shape[1], frame.shape[0]),
                    )
                except VideoSourceError as exc:
                    LOGGER.error("cannot record annotated video: %s", exc)
                    config.output.video_path = None
                else:
                    LOGGER.info(
                        "recording annotated video to %s (%s)", video_sink.path, video_sink.codec
                    )

            if display_enabled or video_sink is not None:
                canvas = frame.copy()
                canvas = draw_tracks(
                    canvas,
                    result.tracks,
                    show_trails=config.display.show_trails,
                    show_labels=config.display.show_labels,
                    fade=config.display.trail_fade,
                )
                drag = session.drag_box
                if drag is not None:
                    canvas = draw_selection(canvas, drag[0], drag[1])
                if config.display.show_hud:
                    canvas = draw_hud(
                        canvas,
                        fps=fps_meter.fps,
                        n_active=len(manager.active_tracks),
                        total_ids=manager.total_created,
                        frame_index=result.frame_index,
                        paused=session.paused,
                        backend=config.tracking.backend,
                        source_name=source.name,
                    )
                if session.help_visible:
                    canvas = draw_help(canvas)
                if session.notice:
                    canvas = draw_banner(canvas, session.notice)
                if video_sink is not None:
                    video_sink.write(canvas)
                if display_enabled:
                    cv2.imshow(window, canvas)
                    last_render = canvas
                    if _window_closed(window):
                        break
                    apply_event(session.handle_key(cv2.waitKey(1) & 0xFF), canvas)

            session.tick()
            processed += 1
            stats.frames = processed
            stats.created_ids = manager.total_created
            stats.removed_ids = manager.total_removed
            stats.spawn_failures = manager.spawn_failures
            if config.source.max_frames and processed >= config.source.max_frames:
                LOGGER.info("reached --max-frames=%d", config.source.max_frames)
                break
    except KeyboardInterrupt:
        LOGGER.info("interrupted by user")
        exit_code = 130
    finally:
        if csv_writer is not None:
            csv_writer.close()
        if video_sink is not None:
            video_sink.release()
        source.release()
        if display_enabled:
            cv2.destroyAllWindows()

    if processed == 0 and exit_code == 0:
        LOGGER.warning("no frames were read from the source")
        exit_code = 1

    elapsed = max(1e-9, time.perf_counter() - started)
    summary = pipeline.stats()
    print(
        f"MOT finished: {processed} frames in {elapsed:.1f}s "
        f"({processed / elapsed:.1f} fps), {summary['created_ids']} ids created, "
        f"{summary['removed_ids']} destroyed, {summary['trail_points']} trail points"
    )
    if csv_path is not None:
        print(f"trajectories: {csv_path} ({csv_writer.rows_written if csv_writer else 0} rows)")
    if video_sink is not None:
        print(f"annotated video: {video_sink.path} [{video_sink.codec}]")
    if summary["spawn_failures"]:
        print(f"tracker init failures: {summary['spawn_failures']}", file=sys.stderr)
    return exit_code


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

    return run(config)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
