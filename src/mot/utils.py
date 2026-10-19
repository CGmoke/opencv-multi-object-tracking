"""Small utilities shared by the application layer."""

from __future__ import annotations

import logging
import re
import time
from collections import deque
from pathlib import Path
from typing import Deque, Optional

_SAFE_NAME = re.compile(r"[^A-Za-z0-9._-]+")


class FpsMeter:
    """Sliding-window FPS estimator.

    Uses the time between the first and last tick in the window instead of an
    exponential average, so the number is honest when frames are irregular.
    """

    def __init__(self, window: int = 30) -> None:
        if int(window) < 2:
            raise ValueError(f"window must be >= 2, got {window}")
        self._timestamps: Deque[float] = deque(maxlen=int(window))

    def tick(self, now: Optional[float] = None) -> float:
        self._timestamps.append(time.perf_counter() if now is None else float(now))
        return self.fps

    @property
    def fps(self) -> float:
        if len(self._timestamps) < 2:
            return 0.0
        elapsed = self._timestamps[-1] - self._timestamps[0]
        if elapsed <= 1e-9:
            return 0.0
        return (len(self._timestamps) - 1) / elapsed

    def reset(self) -> None:
        self._timestamps.clear()


def sanitize_filename(text: str, fallback: str = "frame") -> str:
    """Reduce arbitrary text (e.g. a source name) to a safe file name part."""
    cleaned = _SAFE_NAME.sub("_", str(text)).strip("._-")
    return cleaned[:80] or fallback


def ensure_parent_dir(path: str | Path) -> Path:
    """Create the parent directory of ``path`` and return the resolved path."""
    target = Path(path).expanduser()
    target.parent.mkdir(parents=True, exist_ok=True)
    return target


def setup_logging(verbose: bool = False) -> None:
    """Configure console logging once, at the requested verbosity."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(levelname)s %(message)s",
    )


__all__ = ["FpsMeter", "ensure_parent_dir", "sanitize_filename", "setup_logging"]
