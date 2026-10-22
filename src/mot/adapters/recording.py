"""适配层：运行期“预录”动像视频，结束后由用户决定保存或删除。

会话一开始就把**带轨迹标注**的画面写入临时视频（``outputs/.pending/<name>/``），
运行结束时只关闭文件、不落最终路径；随后由接口层调用 :meth:`commit` 保存到用户
指定的位置，或 :meth:`discard` 删除预录文件。

设计要点
--------
* 写入器**惰性创建**：首帧到达时才根据画面尺寸打开 ``cv2.VideoWriter``，因此
  空会话、假帧（``None``）不会留下空文件。
* 录像失败（磁盘满 / 编解码器不可用）只记录一次告警并停止录制，绝不影响跟踪主流程。
* 预录文件与最终文件都在同一文件系统（默认 ``outputs``）时，``commit`` 是原子移动。
"""

from __future__ import annotations

import logging
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np

from mot.adapters.rendering import draw_tracks
from mot.adapters.sinks.video import VideoWriterSink
from mot.domain.errors import SinkError

LOGGER = logging.getLogger(__name__)

#: ``commit`` 的路径参数。
PathLike = Union[str, "os.PathLike[str]", Path]


class PendingVideoRecorder:
    """把每帧（含轨迹标注）写入临时视频，等待保存或删除。

    Args:
        fps: 录像帧率（通常取输入源帧率）。
        pending_dir: 预录临时目录的父目录。
        name: 会话名（用于临时目录前缀）。
        show_trails / show_labels / fade: 标注绘制开关（与预览一致）。
        pixels_per_meter / small_max_m / large_min_m: 实际尺寸标注参数。
    """

    def __init__(
        self,
        *,
        fps: float = 25.0,
        pending_dir: PathLike = "outputs/.pending",
        name: str = "recording",
        show_trails: bool = True,
        show_labels: bool = True,
        fade: bool = True,
        pixels_per_meter: Optional[float] = None,
        small_max_m: float = 0.5,
        large_min_m: float = 2.0,
    ) -> None:
        self._fps = max(1.0, float(fps))
        self._pending_dir = Path(os.fspath(pending_dir)).expanduser()
        self._name = str(name)
        self._show_trails = bool(show_trails)
        self._show_labels = bool(show_labels)
        self._fade = bool(fade)
        self._pixels_per_meter = pixels_per_meter
        self._small_max_m = float(small_max_m)
        self._large_min_m = float(large_min_m)

        self._temp_dir: Optional[Path] = None
        self._video_path: Optional[Path] = None
        self._writer: Optional[VideoWriterSink] = None
        self._frames = 0
        self._closed = False
        self._failed = False
        self._committed = False
        self._discarded = False

    # -- 只读视图 --------------------------------------------------------
    @property
    def frames(self) -> int:
        """已写入的帧数。"""
        return self._frames

    @property
    def pending(self) -> bool:
        """是否有等待用户决定的预录视频。"""
        if self._frames <= 0 or self._committed or self._discarded:
            return False
        return True

    @property
    def path(self) -> Optional[Path]:
        """当前视频路径：预录阶段为临时文件，保存后为最终文件。"""
        return self._video_path

    @property
    def failed(self) -> bool:
        return self._failed

    # -- FrameRecorder ---------------------------------------------------
    def write(self, frame: Any, result: Any) -> None:
        """写入一帧；``frame`` 为 ``None`` 或非 BGR uint8 图像时跳过。"""
        if self._closed or self._failed:
            return
        if frame is None:
            return
        try:
            canvas = np.asarray(frame)
            if canvas.ndim != 3 or canvas.dtype != np.uint8 or canvas.size == 0:
                return
            canvas = draw_tracks(
                canvas.copy(),
                result.tracks,
                show_trails=self._show_trails,
                show_labels=self._show_labels,
                fade=self._fade,
                pixels_per_meter=self._pixels_per_meter,
                small_max_m=self._small_max_m,
                large_min_m=self._large_min_m,
            )
            if self._writer is None:
                self._open_writer(canvas)
            if self._writer is not None:
                self._writer.write(canvas)
                self._frames += 1
        except Exception as exc:  # noqa: BLE001 - 录像失败不得影响跟踪主流程
            self._failed = True
            self._release_writer()
            LOGGER.warning("video recording disabled for %s: %s", self._name, exc)

    def close(self) -> None:
        """结束写入（预录文件保留）。"""
        self._closed = True
        self._release_writer()

    # -- 结束动作 --------------------------------------------------------
    def commit(self, target: Optional[PathLike] = None) -> Path:
        """把预录视频保存到 ``target``；未指定时使用默认文件名。

        Raises:
            SinkError: 没有可保存的录像、会话仍在录制、目标已存在或移动失败。
        """
        if not self._closed:
            raise SinkError("stop the session before saving the recording")
        if self._frames <= 0 or self._video_path is None or not self._video_path.exists():
            raise SinkError("no recorded video is available")
        if self._committed:
            return self._video_path

        destination = Path(os.fspath(target)).expanduser() if target else self.default_target()
        if destination.exists() and destination.is_dir():
            destination = destination / self.default_target().name
        if destination.suffix == "":
            destination = destination.with_suffix(".mp4")
        if destination.exists():
            raise SinkError(f"output video already exists: {destination}")
        try:
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(self._video_path), str(destination))
        except OSError as exc:
            raise SinkError(f"cannot save the recording to {destination}: {exc}") from exc

        self._video_path = destination
        self._committed = True
        self._cleanup_temp_dir()
        return destination

    def discard(self) -> bool:
        """删除预录视频；没有可删除的文件时返回 ``False``。"""
        if not self._closed:
            raise SinkError("stop the session before discarding the recording")
        if self._committed or self._discarded:
            return False
        self._discarded = True
        self._release_writer()
        self._cleanup_temp_dir()
        self._video_path = None
        return True

    def default_target(self) -> Path:
        """默认保存位置：``pending_dir`` 的父目录下的 ``<name>_recording.mp4``。"""
        base = self._pending_dir.parent
        return base / f"{self._name}_recording.mp4"

    # -- 内部实现 --------------------------------------------------------
    def _open_writer(self, canvas: np.ndarray) -> None:
        height, width = int(canvas.shape[0]), int(canvas.shape[1])
        self._pending_dir.mkdir(parents=True, exist_ok=True)
        self._temp_dir = Path(
            tempfile.mkdtemp(prefix=f"{self._name}-", dir=str(self._pending_dir))
        )
        self._video_path = self._temp_dir / "recording.mp4"
        self._writer = VideoWriterSink(str(self._video_path), self._fps, (width, height))

    def _release_writer(self) -> None:
        if self._writer is not None:
            self._writer.release()
            self._writer = None

    def _cleanup_temp_dir(self) -> None:
        if self._temp_dir is not None:
            shutil.rmtree(self._temp_dir, ignore_errors=True)
            self._temp_dir = None


__all__ = ["PendingVideoRecorder"]
