"""接口层：本地 Web 监测界面（组合根 + Flask 路由）。

在浏览器里实时查看跟踪画面与统计，输入源支持四类：

* ``usb``      —— 本机摄像头索引（服务端 ``cv2.VideoCapture``）
* ``url``      —— RTSP / RTMP / HTTP 网络流（服务端采集，断流自动重连）
* ``file``     —— 本地上传的视频文件（服务端处理，处理完自动结束）
* ``browser``  —— 浏览器摄像头（前端 ``getUserMedia`` 定时 POST JPEG，服务端解码进队列）

每路会话在独立线程里跑 :class:`~mot.application.session.TrackingSession`；画面经
:class:`~mot.adapters.web_render.JpegRenderer` 编码成 JPEG，通过 MJPEG 推给浏览器；
轨迹写 CSV、事件写 JSONL，并同时打到控制台日志。

仅本地/局域网使用，**无多用户与鉴权**，默认只监听 ``127.0.0.1``。

本模块是组合根，允许同时 import ``application`` 与 ``adapters``。
"""

from __future__ import annotations

import atexit
import logging
import sys
import tempfile
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Deque, Dict, List, Optional

from flask import Flask, Response, jsonify, render_template, request, stream_with_context

from mot.adapters.clock import MonotonicClock
from mot.adapters.queue_source import QueueFrameSource
from mot.adapters.recording import PendingVideoRecorder
from mot.adapters.sinks.csv import CsvTrajectoryWriter
from mot.adapters.sinks.jsonl import JsonlEventSink
from mot.adapters.sources import open_source
from mot.adapters.trackers import init_tracker, make_tracker_factory, update_tracker
from mot.adapters.web_render import JpegRenderer, decode_jpeg, mjpeg_chunks
from mot.application.config import AppConfig
from mot.application.errors import ConfigError
from mot.application.pipeline import MOTPipeline
from mot.application.session import SessionStats, TrackingSession
from mot.domain.errors import DomainError, SinkError, SourceError, TrackerSetupError
from mot.domain.lifecycle import TrackManager
from mot.domain.limits import SUPPORTED_BACKENDS
from mot.domain.models import normalize_bbox
from mot.interfaces.cli import build_config, build_parser
from mot.utils.fps import FrameBudget
from mot.utils.logging import setup_logging
from mot.utils.paths import sanitize_filename

LOGGER = logging.getLogger("mot.web")

#: 允许上传的视频扩展名。
UPLOAD_SUFFIXES = (".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v")

#: 支持的会话类型。
SESSION_KINDS = ("usb", "url", "file", "browser")

#: 浏览器摄像头会话的“空闲即结束”秒数（客户端长时间不再推帧）。
BROWSER_IDLE_TIMEOUT = 15.0

#: HTTP 事件流保留的最近事件条数。
EVENT_BUFFER = 200


@dataclass
class _SessionRecord:
    """一帧事件的轻量记录（供页面轮询；完整事件见 JSONL 文件）。"""

    index: int
    frame: int
    timestamp: float
    created: List[int]
    removed: List[int]
    active: int


class SessionHandle:
    """一路跟踪会话：worker 线程 + 渲染器 + 状态/统计/事件缓冲。

    同时实现 ``domain.ports.EventSink``：每帧把轻量记录追加到环形缓冲，供
    ``GET /api/sessions/<id>/events`` 消费。
    """

    def __init__(
        self,
        session_id: str,
        name: str,
        kind: str,
        session: TrackingSession,
        source: Any,
        renderer: JpegRenderer,
        closers: List[Any],
        csv_path: Optional[Path],
        jsonl_path: Optional[Path],
        stats_fn: Callable[[], Dict[str, int]],
        stream_fps: float = 15.0,
        spec: Any = None,
        seed_boxes: List[Any] = (),
        recorder: Optional[PendingVideoRecorder] = None,
        recording_target: Optional[Path] = None,
    ) -> None:
        self.id = session_id
        self.name = name
        self.kind = kind
        self.spec = spec
        self.seed_boxes = list(seed_boxes)
        self._session = session
        self._source = source
        self._renderer = renderer
        self._recorder = recorder
        self.recording_target = recording_target
        self._closers = list(closers)
        self._stats_fn = stats_fn
        self._budget = FrameBudget(window=30, target_fps=stream_fps)
        self.csv_path = csv_path
        self.jsonl_path = jsonl_path
        self._events: Deque[_SessionRecord] = deque(maxlen=EVENT_BUFFER)
        self._event_count = 0
        self._last_active = 0
        self._stats: Optional[SessionStats] = None
        self._state = "starting"
        self._stop_requested = False
        self._error = ""
        self._started_at = time.time()
        self._thread = threading.Thread(target=self._run, name=f"mot-web-{session_id}", daemon=True)

    # -- 生命周期 --------------------------------------------------------
    def start(self) -> None:
        self._state = "running"
        self._thread.start()

    def stop(self) -> None:
        """请求停止：置 quit 事件；队列源还需唤醒阻塞中的 ``read()``。"""
        if self._state not in ("running", "starting"):
            return
        self._stop_requested = True
        self._session.handle_event("quit")
        if isinstance(self._source, QueueFrameSource):
            self._source.release()

    def queue_box(self, bbox: Any) -> None:
        """入队一个手工框（下一帧生效）；已停止的会话忽略。"""
        if self.stopped:
            return
        self._session.queue_box([float(value) for value in bbox])

    def join(self, timeout: Optional[float] = None) -> None:
        self._thread.join(timeout=timeout)

    # -- 预录视频 --------------------------------------------------------
    def save_recording(self, target: Optional[str] = None) -> Path:
        """把本次运行的预录视频保存到 ``target``（空则用默认路径）。

        Raises:
            SinkError: 会话仍在运行、没有录像或目标已存在。
        """
        if self._recorder is None:
            raise SinkError("recording is not enabled for this session")
        if not self.stopped:
            raise SinkError("stop the session before saving the recording")
        destination = target.strip() if isinstance(target, str) else None
        return self._recorder.commit(destination or self.recording_target)

    def discard_recording(self) -> bool:
        """删除本次运行的预录视频；会话运行中拒绝删除。"""
        if self._recorder is None:
            return False
        if not self.stopped:
            raise SinkError("stop the session before discarding the recording")
        return self._recorder.discard()

    def _run(self) -> None:
        try:
            self._stats = self._session.run()
            self._state = "done"
        except Exception as exc:  # noqa: BLE001 - 单路失败不得拖垮服务
            self._error = str(exc)
            self._state = "error"
            LOGGER.exception("session %s failed", self.id)
        finally:
            for closer in self._closers:
                try:
                    closer.close()
                except Exception:  # noqa: BLE001 - 关闭失败不影响其他清理
                    LOGGER.warning("failed to close a sink for session %s", self.id)

    # -- 帧预算 ----------------------------------------------------------
    def mark_capture(self, now: Optional[float] = None) -> None:
        """记录一帧被接收（browser 为 POST 入队成功）。"""
        self._budget.mark_capture(now)

    def mark_stream(self, now: Optional[float] = None) -> None:
        """记录一个 MJPEG 分块成功推给浏览器。"""
        self._budget.mark_stream(now)

    def should_send(self) -> bool:
        """推流跳帧决策：积压时只推最新帧，不影响处理线程。"""
        return self._budget.should_send()

    def rates_snapshot(self) -> Dict[str, Any]:
        return self._budget.snapshot()

    # -- EventSink -------------------------------------------------------
    def on_frame(self, result: Any) -> None:
        active = sum(1 for track in result.tracks if track.is_active)
        if not isinstance(self._source, QueueFrameSource):
            self._budget.mark_capture()  # 服务端源：read 与处理同拍
        self._budget.mark_process()
        self._last_active = active
        if self._event_count % 10 == 0:  # 控制台日志：每 10 帧一次，避免刷屏
            LOGGER.info("session %s frame=%s active=%s", self.id, result.frame_index, active)
        self._events.append(
            _SessionRecord(
                index=self._event_count,
                frame=int(result.frame_index),
                timestamp=float(result.timestamp),
                created=list(result.created),
                removed=list(result.removed),
                active=active,
            )
        )
        self._event_count += 1

    # -- 只读视图 --------------------------------------------------------
    @property
    def renderer(self) -> JpegRenderer:
        return self._renderer

    @property
    def source(self) -> Any:
        return self._source

    @property
    def state(self) -> str:
        return self._state

    @property
    def stopped(self) -> bool:
        return self._state in ("done", "error")

    @property
    def stopping(self) -> bool:
        """已请求停止但线程尚未退出（``restart`` 据此决定是否等待）。"""
        return self._stop_requested and not self.stopped

    def events(self, since: int) -> List[Dict[str, Any]]:
        return [vars(record) for record in self._events if record.index >= since]

    @property
    def event_count(self) -> int:
        return self._event_count

    def snapshot(self) -> Dict[str, Any]:
        """当前状态 + 实时统计（会话跑完后由权威 ``SessionStats`` 覆盖）。"""
        try:  # 实时读管道统计；并发下的结构性报错按“暂不可用”处理
            live = self._stats_fn()
        except Exception:  # noqa: BLE001 - 实时统计是尽力而为
            live = {}
        snapshot: Dict[str, Any] = {
            "id": self.id,
            "name": self.name,
            "kind": self.kind,
            "state": self._state,
            "error": self._error,
            "started_at": self._started_at,
            "events": self._event_count,
            "active": self._last_active,
            "frames": int(live.get("frames", 0)),
            "created_ids": int(live.get("created_ids", 0)),
            "removed_ids": int(live.get("removed_ids", 0)),
            "spawn_failures": int(live.get("spawn_failures", 0)),
            "trail_points": int(live.get("trail_points", 0)),
            "csv": str(self.csv_path) if self.csv_path else None,
            "jsonl": str(self.jsonl_path) if self.jsonl_path else None,
            "recording_pending": bool(self._recorder.pending) if self._recorder else False,
            "recording_frames": self._recorder.frames if self._recorder else 0,
            "recording_target": (
                str(self.recording_target) if self.recording_target is not None else None
            ),
        }
        if self._stats is not None:
            snapshot.update(
                {
                    "frames": self._stats.frames,
                    "created_ids": self._stats.created_ids,
                    "removed_ids": self._stats.removed_ids,
                    "spawn_failures": self._stats.spawn_failures,
                    "trail_points": self._stats.trail_points,
                    "exit_code": self._stats.exit_code,
                }
            )
        if isinstance(self._source, QueueFrameSource):
            snapshot["pushed"] = self._source.pushed
            snapshot["dropped"] = self._source.dropped
        snapshot.update(self._budget.snapshot())
        return snapshot


class SessionRegistry:
    """管理多路会话：创建、停止、查询；上限 ``max_sessions``。"""

    def __init__(self, config: AppConfig) -> None:
        self._config = config
        self._lock = threading.Lock()
        self._sessions: Dict[str, SessionHandle] = {}
        self._counter = 0
        self._upload_dir = (
            Path(config.web.upload_dir).expanduser()
            if config.web.upload_dir
            else Path(tempfile.mkdtemp(prefix="mot-uploads-"))
        )

    @property
    def config(self) -> AppConfig:
        return self._config

    @property
    def max_sessions(self) -> int:
        return int(self._config.web.max_sessions)

    @property
    def upload_dir(self) -> Path:
        return self._upload_dir

    def get(self, session_id: str) -> Optional[SessionHandle]:
        with self._lock:
            return self._sessions.get(session_id)

    def list(self) -> List[Dict[str, Any]]:
        with self._lock:
            handles = list(self._sessions.values())
        return [handle.snapshot() for handle in handles]

    def start(
        self,
        kind: str,
        spec: Any = None,
        *,
        name: Optional[str] = None,
        seed_boxes: Any = (),
    ) -> SessionHandle:
        """创建并启动一路会话。

        Args:
            kind: ``usb`` / ``url`` / ``file`` / ``browser``。
            spec: 摄像头索引、网络流地址或本地视频路径（``browser`` 忽略）。
            name: 显示名。
            seed_boxes: 启动时初始化的目标框 ``[[x, y, w, h], ...]``（无检测器时必需，
                否则画面里不会产生任何目标）。

        Raises:
            ConfigError: 未知 ``kind`` 或非法 ``spec``。
            ValidationError: 种子框非法。
            SourceError: 输入源打不开。
            TrackerSetupError: 跟踪器后端不可用。
            MemoryError: 达到 ``max_sessions`` 上限（路由映射为 429）。
        """
        kind = str(kind).strip().lower()
        if kind not in SESSION_KINDS:
            raise ConfigError(f"unknown session kind {kind!r}; expected one of {SESSION_KINDS}")
        boxes = [normalize_bbox(box) for box in (seed_boxes or ())]

        with self._lock:
            if len(self._sessions) >= self.max_sessions:
                raise MemoryError(f"too many sessions (max {self.max_sessions})")
            self._counter += 1
            session_id = f"s{self._counter}"
            handle = self._build(session_id, kind, spec, name, boxes)
            self._sessions[session_id] = handle
        handle.start()
        LOGGER.info("session %s started (kind=%s)", session_id, kind)
        return handle

    def stop(self, session_id: str) -> bool:
        handle = self.get(session_id)
        if handle is None:
            return False
        handle.stop()
        return True

    def remove(self, session_id: str) -> bool:
        """删除一个**已停止**的会话；运行中的会话需先停止（返回 ``False``）。"""
        with self._lock:
            handle = self._sessions.get(session_id)
            if handle is None or not handle.stopped:
                return False
            del self._sessions[session_id]
        try:  # 删除会话即放弃未保存的预录视频
            handle.discard_recording()
        except SinkError:  # pragma: no cover - 已停止的会话不应抛出
            LOGGER.warning("failed to discard the recording of session %s", session_id)
        LOGGER.info("session %s removed", session_id)
        return True

    def restart(self, session_id: str) -> Optional[SessionHandle]:
        """用**同一个 id** 重启一个已停止的会话（沿用原输入源与种子框）。

        「停止中」的会话（``stop()`` 只异步置 quit 事件，线程尚未退出）会先在锁外
        等待其退出，避免用户停止后立刻点重启时误报 409。

        Returns:
            新的会话句柄；会话不存在或仍在运行时返回 ``None``。

        Raises:
            SourceError / TrackerSetupError / ConfigError / ValidationError: 重建失败，
            此时原会话保持不变。
        """
        previous = self.get(session_id)
        if previous is None:
            return None
        if previous.stopping:
            previous.join(timeout=5.0)  # 锁外等待，不阻塞其它会话的操作
        with self._lock:
            previous = self._sessions.get(session_id)
            if previous is None or not previous.stopped:
                return None
            try:  # 重启视为放弃上一次运行未保存的预录视频
                previous.discard_recording()
            except SinkError:  # pragma: no cover - 已停止的会话不应抛出
                LOGGER.warning("failed to discard the recording of session %s", session_id)
            handle = self._build(
                session_id, previous.kind, previous.spec, previous.name, previous.seed_boxes
            )
            self._sessions[session_id] = handle
        handle.start()
        LOGGER.info("session %s restarted (kind=%s)", session_id, handle.kind)
        return handle

    def stop_all(self) -> None:
        with self._lock:
            handles = list(self._sessions.values())
        for handle in handles:
            handle.stop()
        for handle in handles:
            handle.join(timeout=2.0)

    def _build(
        self,
        session_id: str,
        kind: str,
        spec: Any,
        name: Optional[str],
        seed_boxes: List[Any],
    ) -> SessionHandle:
        config = self._config
        if kind == "browser":
            source: Any = QueueFrameSource(
                fps=config.web.stream_fps, idle_timeout=BROWSER_IDLE_TIMEOUT
            )
        elif kind == "usb":
            source = open_source(str(spec), loop=False) if spec is not None else open_source("0")
        elif kind == "url":
            if not spec:
                raise ConfigError("kind='url' requires a network stream address")
            source = open_source(str(spec), loop=False)
        else:  # file
            if not spec:
                raise ConfigError("kind='file' requires a local video path")
            source = open_source(str(spec), loop=False)

        manager = TrackManager(
            make_tracker_factory(config.tracking.backend),
            init_fn=init_tracker,
            update_fn=update_tracker,
            iou_threshold=config.tracking.iou_threshold,
            max_missing=config.tracking.max_missing,
            max_coast=config.tracking.max_coast,
            max_parallel=config.tracking.max_parallel,
            trail_length=config.tracking.trail_length,
            match_method=config.tracking.match_method,
            max_tracks=config.tracking.max_tracks,
            box_smoothing=config.tracking.box_smoothing,
            min_hits=config.tracking.min_hits,
            smooth_window=config.tracking.smooth_window,
            nest_containment=config.tracking.merge_track_containment,
        )
        detector = _build_detector(config)
        motion_mode = config.detector.backend == "none" and config.motion.enabled
        pipeline = MOTPipeline(
            manager,
            detector=detector,
            detection_interval=_detection_interval(config),
            max_box_change=config.tracking.max_box_change,
            static_iou=config.motion.static_iou,
            static_frames=config.motion.static_frames if motion_mode else 0,
            dedupe_tracks=config.tracking.dedupe_tracks,
            merge_track_iou=config.tracking.merge_track_iou,
            merge_track_frames=config.tracking.merge_track_frames,
            merge_track_containment=config.tracking.merge_track_containment,
        )

        out_dir = _output_dir(config)
        csv_path = out_dir / f"web_{session_id}.csv"
        jsonl_path = out_dir / f"web_{session_id}.jsonl"
        csv_writer = CsvTrajectoryWriter(csv_path)
        jsonl_writer = JsonlEventSink(jsonl_path)

        # 每次运行都自动预录；运行结束后由用户决定保存到目标路径或删除。
        recorder = PendingVideoRecorder(
            fps=float(getattr(source, "fps", 25.0) or 25.0),
            pending_dir=out_dir / ".pending",
            name=f"web_{session_id}",
            show_trails=config.display.show_trails,
            show_labels=config.display.show_labels,
            fade=config.display.trail_fade,
            pixels_per_meter=config.motion.pixels_per_meter,
            small_max_m=config.motion.small_max_m,
            large_min_m=config.motion.large_min_m,
        )
        recording_target = out_dir / (
            f"web_{session_id}_{time.strftime('%Y%m%d_%H%M%S')}.mp4"
        )

        renderer = JpegRenderer(
            stats_fn=pipeline.stats,
            source_name=source.name,
            backend=config.tracking.backend,
            show_hud=config.display.show_hud,
            show_trails=config.display.show_trails,
            show_labels=config.display.show_labels,
            trail_fade=config.display.trail_fade,
            max_fps=config.web.stream_fps,
            pixels_per_meter=config.motion.pixels_per_meter,
            small_max_m=config.motion.small_max_m,
            large_min_m=config.motion.large_min_m,
        )
        handle = SessionHandle(
            session_id,
            name or source.name,
            kind,
            session=None,  # type: ignore[arg-type]  # 占位，下面立刻赋值
            source=source,
            renderer=renderer,
            closers=[jsonl_writer],
            csv_path=csv_path,
            jsonl_path=jsonl_path,
            stats_fn=pipeline.stats,
            stream_fps=config.web.stream_fps,
            spec=spec,
            seed_boxes=seed_boxes,
            recorder=recorder,
            recording_target=recording_target,
        )
        renderer.set_rates_fn(handle.rates_snapshot)
        session = TrackingSession(
            pipeline,
            source,
            MonotonicClock(),
            sinks=[csv_writer],
            event_sinks=[jsonl_writer, handle],
            renderer=renderer,
            recorder=recorder,
            seed_boxes=seed_boxes,
        )
        handle._session = session
        return handle


def _output_dir(config: AppConfig) -> Path:
    if config.output.csv_path:
        return Path(config.output.csv_path).expanduser().parent
    return Path("outputs")


def _build_detector(config: AppConfig) -> Optional[Any]:
    from mot.interfaces.cli import _build_motion, build_detector

    detector = build_detector(config)
    if detector is None and config.motion.enabled:
        detector = _build_motion(config)
    return detector


def _detection_interval(config: AppConfig) -> int:
    from mot.interfaces.cli import _detection_interval as interval

    return interval(config)


def create_app(registry: SessionRegistry) -> Flask:
    """构造 Flask 应用（页面 + JSON API + MJPEG 流）。"""
    app = Flask(__name__, template_folder="templates", static_folder="static")
    app.config["JSON_SORT_KEYS"] = False

    @app.get("/")
    def index() -> Any:
        return render_template("index.html")

    @app.get("/api/sources")
    def api_sources() -> Any:
        return jsonify(
            {
                "kinds": list(SESSION_KINDS),
                "trackers": list(SUPPORTED_BACKENDS),
                "max_sessions": registry.max_sessions,
            }
        )

    @app.get("/api/sessions")
    def api_list() -> Any:
        return jsonify({"sessions": registry.list()})

    @app.post("/api/sessions")
    def api_start() -> Any:
        payload = request.get_json(silent=True) or {}
        try:
            handle = registry.start(
                payload.get("kind"),
                payload.get("spec"),
                name=payload.get("name"),
                seed_boxes=payload.get("seed_boxes") or (),
            )
        except MemoryError as exc:
            return jsonify({"error": str(exc)}), 429
        except ConfigError as exc:
            return jsonify({"error": str(exc)}), 400
        except (SourceError, TrackerSetupError) as exc:
            return jsonify({"error": str(exc)}), 400
        except DomainError as exc:
            return jsonify({"error": str(exc)}), 400
        return jsonify(handle.snapshot()), 201

    @app.post("/api/sessions/<session_id>/box")
    def api_box(session_id: str) -> Any:
        handle = registry.get(session_id)
        if handle is None:
            return jsonify({"error": "unknown session"}), 404
        payload = request.get_json(silent=True) or {}
        bbox = payload.get("bbox")
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            return jsonify({"error": "bbox must be [x, y, w, h]"}), 400
        try:
            normalize_bbox(bbox)
        except DomainError as exc:
            return jsonify({"error": str(exc)}), 400
        handle.queue_box(bbox)
        return jsonify({"queued": True})

    @app.post("/api/sessions/<session_id>/recording/save")
    def api_save_recording(session_id: str) -> Any:
        handle = registry.get(session_id)
        if handle is None:
            return jsonify({"error": "unknown session"}), 404
        payload = request.get_json(silent=True) or {}
        try:
            saved = handle.save_recording(payload.get("path"))
        except SinkError as exc:
            return jsonify({"error": str(exc)}), 400
        LOGGER.info("session %s recording saved to %s", session_id, saved)
        return jsonify({"saved": str(saved), "bytes": saved.stat().st_size})

    @app.post("/api/sessions/<session_id>/recording/discard")
    def api_discard_recording(session_id: str) -> Any:
        handle = registry.get(session_id)
        if handle is None:
            return jsonify({"error": "unknown session"}), 404
        try:
            discarded = handle.discard_recording()
        except SinkError as exc:
            return jsonify({"error": str(exc)}), 400
        LOGGER.info("session %s recording discarded=%s", session_id, discarded)
        return jsonify({"discarded": discarded})

    @app.post("/api/upload")
    def api_upload() -> Any:
        upload = request.files.get("file")
        if upload is None or not upload.filename:
            return jsonify({"error": "no file part named 'file'"}), 400
        filename = sanitize_filename(Path(upload.filename).name, "upload")
        suffix = Path(filename).suffix.lower()
        if suffix not in UPLOAD_SUFFIXES:
            return jsonify({"error": f"unsupported file type {suffix!r}"}), 400
        registry.upload_dir.mkdir(parents=True, exist_ok=True)
        target = registry.upload_dir / filename
        upload.save(str(target))
        size = target.stat().st_size
        if size > registry.config.web.max_upload_bytes:
            target.unlink(missing_ok=True)
            return jsonify({"error": "file too large"}), 413
        try:
            handle = registry.start("file", str(target), name=filename)
        except MemoryError as exc:
            return jsonify({"error": str(exc)}), 429
        except (ConfigError, SourceError, TrackerSetupError, DomainError) as exc:
            return jsonify({"error": str(exc)}), 400
        return jsonify(handle.snapshot()), 201

    @app.get("/api/sessions/<session_id>")
    def api_status(session_id: str) -> Any:
        handle = registry.get(session_id)
        if handle is None:
            return jsonify({"error": "unknown session"}), 404
        return jsonify(handle.snapshot())

    @app.post("/api/sessions/<session_id>/stop")
    def api_stop(session_id: str) -> Any:
        if not registry.stop(session_id):
            return jsonify({"error": "unknown session"}), 404
        return jsonify({"stopped": session_id})

    @app.post("/api/sessions/<session_id>/restart")
    def api_restart(session_id: str) -> Any:
        if registry.get(session_id) is None:
            return jsonify({"error": "unknown session"}), 404
        try:
            handle = registry.restart(session_id)
        except MemoryError as exc:
            return jsonify({"error": str(exc)}), 429
        except (ConfigError, SourceError, TrackerSetupError, DomainError) as exc:
            return jsonify({"error": str(exc)}), 400
        if handle is None:
            return jsonify({"error": "session is still running; stop it first"}), 409
        return jsonify(handle.snapshot())

    @app.delete("/api/sessions/<session_id>")
    def api_remove(session_id: str) -> Any:
        if registry.get(session_id) is None:
            return jsonify({"error": "unknown session"}), 404
        if not registry.remove(session_id):
            return jsonify({"error": "stop the session before removing it"}), 409
        return jsonify({"removed": session_id})

    @app.get("/api/sessions/<session_id>/events")
    def api_events(session_id: str) -> Any:
        handle = registry.get(session_id)
        if handle is None:
            return jsonify({"error": "unknown session"}), 404
        since = request.args.get("since", 0, type=int)
        return jsonify({"events": handle.events(since), "next": handle.event_count})

    @app.get("/stream/<session_id>.mjpeg")
    def api_stream(session_id: str) -> Any:
        handle = registry.get(session_id)
        if handle is None:
            return jsonify({"error": "unknown session"}), 404
        chunks = mjpeg_chunks(
            handle.renderer.holder,
            fps=registry.config.web.stream_fps,
            should_stop=lambda: handle.stopped,
            should_send=handle.should_send,
            on_send=handle.mark_stream,
        )
        return Response(
            stream_with_context(chunks),
            mimetype="multipart/x-mixed-replace; boundary=motframe",
        )

    @app.post("/api/sessions/<session_id>/frame")
    def api_frame(session_id: str) -> Any:
        handle = registry.get(session_id)
        if handle is None:
            return jsonify({"error": "unknown session"}), 404
        source = handle.source
        if not isinstance(source, QueueFrameSource):
            return jsonify({"error": "session does not accept pushed frames"}), 409
        frame = decode_jpeg(request.get_data())
        if frame is None:
            return jsonify({"error": "body must be a JPEG image"}), 400
        source.push(frame)
        handle.mark_capture()
        return jsonify({"ok": True, "pushed": source.pushed})

    return app


def serve(config: AppConfig) -> int:
    """启动 Web 服务（阻塞直到 Ctrl-C）；端口占用返回退出码 2。"""
    registry = SessionRegistry(config)
    atexit.register(registry.stop_all)
    app = create_app(registry)
    address = f"http://{config.web.host}:{config.web.port}"
    LOGGER.info("web UI: %s", address)
    print(f"Web 监测界面已启动：{address}    (Ctrl-C 退出)")
    try:
        app.run(
            host=config.web.host,
            port=config.web.port,
            threaded=True,
            use_reloader=False,
        )
    except KeyboardInterrupt:
        LOGGER.info("interrupted by user")
        return 130
    except OSError as exc:
        LOGGER.error("cannot start web server on %s: %s", address, exc)
        return 2
    finally:
        registry.stop_all()
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    """``mot-web`` 入口：解析 CLI、合并配置、启动 Web 服务。"""
    parser = build_parser()
    parser.prog = "mot-web"
    args = parser.parse_args(argv)
    setup_logging(args.verbose)
    try:
        config = build_config(args)
    except ConfigError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 2
    return serve(config)


__all__ = [
    "BROWSER_IDLE_TIMEOUT",
    "FrameBudget",
    "UPLOAD_SUFFIXES",
    "SessionHandle",
    "SessionRegistry",
    "create_app",
    "main",
    "serve",
]
