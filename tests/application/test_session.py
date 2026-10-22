"""应用层会话编排（步骤 5 验收测试）。

用无 cv2 的假件（``FakeSource`` / ``FakeClock`` / 内存接收器 / 假渲染器，见
``tests/conftest.py``）驱动 ``TrackingSession.run``，覆盖无头运行、种子、交互入队、
事件与退出码。
"""

from __future__ import annotations

from conftest import (
    FakeClock,
    FakeSource,
    RecordingEventSink,
    RecordingTrajectorySink,
    ScriptedRenderer,
)
from mot.application.pipeline import MOTPipeline
from mot.application.session import TrackingSession
from mot.domain.models import Detection

BOX = (10.0, 10.0, 20.0, 20.0)
# session 把 ``frame is None`` 视为流结束，故假帧必须是非 None 对象。
FRAME = object()


def make_session(make_manager, pool, frames, **kwargs):
    pipeline = MOTPipeline(make_manager(pool))
    source = FakeSource(frames)
    clock = FakeClock(step=0.5)
    return TrackingSession(pipeline, source, clock, **kwargs), source


class _FrameCollector:
    """记录会话喂入的帧，并记录是否被关闭。"""

    def __init__(self) -> None:
        self.frames = []
        self.closed = False

    def write(self, frame, result) -> None:
        self.frames.append((frame, result.frame_index))

    def close(self) -> None:
        self.closed = True


class TestHeadlessRun:
    def test_processes_all_frames_and_writes_sink(self, make_manager, pool):
        sink = RecordingTrajectorySink()
        session, source = make_session(
            make_manager, pool, [FRAME, FRAME, FRAME], sinks=[sink], seed_boxes=[BOX]
        )
        stats = session.run()
        assert stats.frames == 3
        assert stats.created_ids == 1
        assert stats.exit_code == 0
        assert len(sink.rows) == 3
        assert sink.closed is True
        assert source.released is True

    def test_recorder_receives_every_frame_and_is_closed(self, make_manager, pool):
        recorder = _FrameCollector()
        session, _ = make_session(
            make_manager, pool, [FRAME, FRAME, FRAME], recorder=recorder
        )
        session.run()
        assert recorder.frames == [(FRAME, 1), (FRAME, 2), (FRAME, 3)]
        assert recorder.closed is True

    def test_sink_receives_frame_index_and_timestamp(self, make_manager, pool):
        sink = RecordingTrajectorySink()
        session, _ = make_session(make_manager, pool, [FRAME], sinks=[sink], seed_boxes=[BOX])
        session.run()
        frame_index, timestamp, ids = sink.rows[0]
        assert frame_index == 1
        assert timestamp >= 0.0
        assert ids == (1,)

    def test_no_frames_sets_exit_code_one(self, make_manager, pool):
        session, _ = make_session(make_manager, pool, [])
        stats = session.run()
        assert stats.frames == 0
        assert stats.exit_code == 1

    def test_max_frames_limits_processing(self, make_manager, pool):
        session, _ = make_session(make_manager, pool, [FRAME] * 5, max_frames=2)
        stats = session.run()
        assert stats.frames == 2

    def test_event_sink_receives_frame_results(self, make_manager, pool):
        events = RecordingEventSink()
        session, _ = make_session(
            make_manager, pool, [FRAME, FRAME], event_sinks=[events], seed_boxes=[BOX]
        )
        session.run()
        assert len(events.results) == 2
        assert events.results[0].frame_index == 1
        assert events.results[0].timestamp >= 0.0

    def test_headless_pause_is_ignored(self, make_manager, pool):
        # 无渲染器时无法恢复，pause 不生效，避免死循环。
        session, _ = make_session(make_manager, pool, [FRAME, FRAME])
        session.handle_event("pause")
        stats = session.run()
        assert stats.frames == 2


class TestSeedingAndQueue:
    def test_seed_boxes_initialise_on_first_frame(self, make_manager, pool):
        session, _ = make_session(make_manager, pool, [FRAME], seed_boxes=[BOX])
        stats = session.run()
        assert stats.created_ids == 1

    def test_queued_box_is_applied(self, make_manager, pool):
        session, _ = make_session(make_manager, pool, [FRAME])
        session.queue_box(BOX)
        stats = session.run()
        assert stats.created_ids == 1

    def test_queued_delete_removes_a_target(self, make_manager, pool):
        session, _ = make_session(make_manager, pool, [FRAME])
        session.queue_box(BOX)
        session.queue_delete((20.0, 20.0))  # BOX 的中心
        stats = session.run()
        assert stats.created_ids == 1
        assert stats.removed_ids == 1


class TestEvents:
    def test_quit_event_stops_after_current_frame(self, make_manager, pool):
        renderer = ScriptedRenderer(["quit"])
        session, source = make_session(make_manager, pool, [FRAME] * 5, renderer=renderer)
        stats = session.run()
        assert stats.frames == 1
        assert len(renderer.presented) == 1

    def test_pause_then_quit_via_renderer(self, make_manager, pool):
        renderer = ScriptedRenderer(["pause", "quit"])
        session, _ = make_session(make_manager, pool, [FRAME] * 5, renderer=renderer)
        stats = session.run()
        # 第 1 帧后暂停，随后重呈现最后一帧收到 quit。
        assert stats.frames == 1
        assert len(renderer.presented) == 2

    def test_reset_event_clears_tracks(self, make_manager, pool):
        session, _ = make_session(make_manager, pool, [FRAME])
        session._pipeline.add_manual_box(None, BOX)
        assert session._pipeline.tracks != ()
        session.handle_event("reset")
        assert session._pipeline.tracks == ()

    def test_unknown_event_is_ignored(self, make_manager, pool):
        session, _ = make_session(make_manager, pool, [FRAME])
        session.handle_event("nonsense")  # 不应抛异常
        assert session.run().frames == 1


def test_detection_import_is_not_shadowed():
    # 保证本模块的导入（含 Detection）可用，便于后续扩展用例。
    assert Detection(BOX).bbox == (10.0, 10.0, 20.0, 20.0)
