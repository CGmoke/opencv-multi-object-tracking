"""Rendering helpers: boxes, ids, trails, HUD and the interactive overlay.

All functions mutate and return the frame they are given, so the caller can
chain them without extra copies::

    frame = draw_tracks(frame, manager.tracks)
    frame = draw_hud(frame, fps=fps, ...)
"""

from __future__ import annotations

from typing import Iterable, Optional, Sequence, Tuple

import cv2
import numpy as np

from .models import BBox, Point, TrackedObject

Color = Tuple[int, int, int]

MIN_FONT_SCALE = 0.38
MAX_FONT_SCALE = 0.9


def color_for_id(track_id: int) -> Color:
    """Stable, high-contrast BGR colour for an id (golden-ratio hue walk)."""
    hue = int(((int(track_id) * 0.618033988749895) % 1.0) * 179)
    hsv = np.uint8([[[hue, 220, 255]]])
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0][0]
    return (int(bgr[0]), int(bgr[1]), int(bgr[2]))


def scale_color(color: Sequence[int], factor: float, background: Sequence[int] = (0, 0, 0)) -> Color:
    """Blend ``color`` toward ``background`` by ``factor`` (for trail fading)."""
    factor = max(0.0, min(1.0, float(factor)))
    return tuple(
        int(round(bg + (fg - bg) * factor)) for fg, bg in zip(color, background)
    )  # type: ignore[return-value]


def font_scale_for(frame: np.ndarray) -> float:
    height = frame.shape[0]
    return max(MIN_FONT_SCALE, min(MAX_FONT_SCALE, height / 720.0 * 0.6))


def thickness_for(frame: np.ndarray) -> int:
    height = frame.shape[0]
    return max(1, int(round(height / 480.0 * 2)))


def draw_box(
    frame: np.ndarray,
    bbox: BBox,
    color: Color,
    thickness: int = 2,
    label: Optional[str] = None,
    score: Optional[float] = None,
    show_label: bool = True,
) -> np.ndarray:
    """Draw one rectangle plus an optional id/score tag."""
    x, y, w, h = (int(round(v)) for v in bbox)
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, thickness, lineType=cv2.LINE_AA)
    if not show_label or not label:
        return frame

    text = f"#{label}" if not label.startswith("#") else label
    if score is not None:
        text += f" {score:.2f}"
    scale = font_scale_for(frame)
    (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
    ty = y - 6 if y - th - baseline - 6 >= 0 else y + h + th + 6
    top = max(0, ty - th - baseline)
    cv2.rectangle(frame, (x, top), (x + tw + 6, top + th + baseline + 4), color, -1)
    cv2.putText(
        frame,
        text,
        (x + 3, top + th + 2),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        (0, 0, 0),
        1,
        cv2.LINE_AA,
    )
    return frame


def draw_trail(
    frame: np.ndarray,
    trail: Iterable[Point],
    color: Color,
    thickness: int = 2,
    fade: bool = True,
) -> np.ndarray:
    """Draw a polyline trail, optionally fading old points into the frame."""
    points = [(int(round(px)), int(round(py))) for px, py in trail]
    if len(points) < 2:
        if len(points) == 1:
            cv2.circle(frame, points[0], max(2, thickness), color, -1, cv2.LINE_AA)
        return frame

    total = len(points) - 1
    for index in range(total):
        factor = (index + 1) / total if fade else 1.0
        segment_color = scale_color(color, factor) if fade else color
        cv2.line(
            frame,
            points[index],
            points[index + 1],
            segment_color,
            thickness,
            cv2.LINE_AA,
        )
    cv2.circle(frame, points[-1], max(2, thickness), color, -1, cv2.LINE_AA)
    return frame


def draw_tracks(
    frame: np.ndarray,
    tracks: Iterable[TrackedObject],
    show_trails: bool = True,
    show_labels: bool = True,
    show_lost: bool = True,
    fade: bool = True,
) -> np.ndarray:
    """Render boxes, ids and trails for every tracked object."""
    thickness = thickness_for(frame)
    for track in tracks:
        if not track.is_active and not show_lost:
            continue
        color = color_for_id(track.track_id)
        if not track.is_active:
            color = scale_color(color, 0.45)
        if show_trails and track.trail:
            draw_trail(frame, track.trail, color, thickness=max(1, thickness - 1), fade=fade)
        draw_box(
            frame,
            track.bbox,
            color,
            thickness=thickness,
            label=f"{track.track_id} {track.label}",
            score=track.confidence,
            show_label=show_labels,
        )
    return frame


def _draw_panel(
    frame: np.ndarray, origin: Tuple[int, int], lines: Sequence[str], scale: float
) -> np.ndarray:
    x0, y0 = origin
    line_height = int(22 * max(1.0, scale / 0.6))
    width = max(
        (cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)[0][0] for line in lines),
        default=0,
    ) + 16
    height = line_height * len(lines) + 12
    overlay = frame.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + width, y0 + height), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
    for index, line in enumerate(lines):
        cv2.putText(
            frame,
            line,
            (x0 + 8, y0 + line_height * (index + 1) - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            (235, 235, 235),
            1,
            cv2.LINE_AA,
        )
    return frame


def draw_hud(
    frame: np.ndarray,
    *,
    fps: float,
    n_active: int,
    total_ids: int,
    frame_index: int,
    paused: bool = False,
    backend: str = "csrt",
    source_name: str = "",
) -> np.ndarray:
    """Draw the status panel (FPS, ids, backend, source)."""
    scale = font_scale_for(frame)
    lines = [
        f"FPS {fps:5.1f}   frame {frame_index}",
        f"ids active {n_active} / created {total_ids}   tracker {backend.upper()}",
    ]
    if source_name:
        lines.append(f"source {source_name}")
    if paused:
        lines.append("PAUSED  (space to resume)")
    frame = _draw_panel(frame, (8, 8), lines, scale)
    if paused:
        h, w = frame.shape[:2]
        cv2.putText(
            frame,
            "PAUSED",
            (w // 2 - int(w * 0.11), 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            max(1.2, scale * 2.2),
            (0, 0, 255),
            3,
            cv2.LINE_AA,
        )
    return frame


def draw_selection(frame: np.ndarray, start: Point, end: Point) -> np.ndarray:
    """Draw the box currently being dragged to initialise a new target."""
    x1, y1 = int(round(start[0])), int(round(start[1]))
    x2, y2 = int(round(end[0])), int(round(end[1]))
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 1, cv2.LINE_AA)
    return frame


HELP_LINES: Tuple[str, ...] = (
    "n  arm new-target box, then drag with the left mouse button",
    "x  arm delete mode, then click a target",
    "space  pause / resume        s  save frame        c  clear trails",
    "r  reset all ids             h  toggle this help   q / esc  quit",
)


def draw_help(frame: np.ndarray) -> np.ndarray:
    """Draw the keyboard cheat sheet in the bottom-left corner."""
    scale = font_scale_for(frame) * 0.95
    line_height = int(20 * max(1.0, scale / 0.6))
    height = line_height * len(HELP_LINES) + 16
    _draw_panel(frame, (8, max(0, frame.shape[0] - height - 8)), list(HELP_LINES), scale)
    return frame


def draw_banner(frame: np.ndarray, text: str, color: Color = (0, 255, 255)) -> np.ndarray:
    """Draw a short-lived instruction banner at the bottom centre."""
    h, w = frame.shape[:2]
    scale = font_scale_for(frame)
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
    x0 = max(8, (w - tw) // 2 - 10)
    y0 = h - th - 30
    overlay = frame.copy()
    cv2.rectangle(overlay, (x0 - 6, y0 - 6), (x0 + tw + 10, y0 + th + 12), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
    cv2.putText(frame, text, (x0, y0 + th), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)
    return frame


__all__ = [
    "HELP_LINES",
    "color_for_id",
    "draw_banner",
    "draw_box",
    "draw_help",
    "draw_hud",
    "draw_selection",
    "draw_tracks",
    "draw_trail",
    "font_scale_for",
    "scale_color",
    "thickness_for",
]
