"""接口层 Web 监测界面（P2 验收测试）。

依赖 ``flask`` 与 ``cv2``；缺少任一即整文件跳过。会话使用真实跟踪器（csrt），
在无可用后端的构建上跳过相关用例。
"""

from __future__ import annotations

import io
import time
from pathlib import Path
from typing import Any, Callable

import pytest

flask = pytest.importorskip("flask")
cv2 = pytest.importorskip("cv2")

import numpy as np  # noqa: E402

from mot.adapters.trackers import tracker_available  # noqa: E402
from mot.application.config import AppConfig  # noqa: E402
from mot.interfaces.web import SessionRegistry, create_app  # noqa: E402


def require_csrt() -> None:
    if not tracker_available("csrt"):
        pytest.skip("csrt tracker not available in this OpenCV build")


def make_client(tmp_path: Path, **overrides: Any):
    config = AppConfig()
    config.output.csv_path = str(tmp_path / "trajectories.csv")
    config.web.upload_dir = str(tmp_path / "uploads")
    for key, value in overrides.items():
        setattr(config.web, key, value)
    config.validate()
    registry = SessionRegistry(config)
    app = create_app(registry)
    app.config.update(TESTING=True)
    return registry, app.test_client(), app


@pytest.fixture
def web(tmp_path):
    registry, client, _app = make_client(tmp_path)
    try:
        yield registry, client
    finally:
        registry.stop_all()


def jpeg_bytes(value: int = 128, height: int = 240, width: int = 320) -> bytes:
    frame = np.full((height, width, 3), value, dtype=np.uint8)
    ok, buffer = cv2.imencode(".jpg", frame)
    assert ok
    return buffer.tobytes()


def wait_for(predicate: Callable[[], bool], timeout: float = 3.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return predicate()


def test_index_page_renders(web):
    _, client = web
    response = client.get("/")
    assert response.status_code == 200
    assert b"<html" in response.data.lower()


def test_api_sources(web):
    _, client = web
    data = client.get("/api/sources").get_json()
    assert data["kinds"] == ["usb", "url", "file", "browser"]
    assert "csrt" in data["trackers"]
    assert data["max_sessions"] == 4


def test_unknown_kind_is_rejected(web):
    _, client = web
    assert client.post("/api/sessions", json={"kind": "magic"}).status_code == 400


def test_unknown_session_is_404(web):
    _, client = web
    assert client.get("/api/sessions/nope").status_code == 404
    assert client.post("/api/sessions/nope/stop").status_code == 404


def test_browser_session_accepts_pushed_frames(web):
    registry, client = web
    response = client.post("/api/sessions", json={"kind": "browser"})
    assert response.status_code == 201
    session_id = response.get_json()["id"]

    pushed = client.post(
        f"/api/sessions/{session_id}/frame",
        data=jpeg_bytes(),
        content_type="image/jpeg",
    )
    assert pushed.status_code == 200

    handle = registry.get(session_id)
    assert wait_for(lambda: handle.renderer.holder.get() is not None)
    assert client.get(f"/api/sessions/{session_id}").get_json()["kind"] == "browser"

    assert client.post(f"/api/sessions/{session_id}/stop").status_code == 200
    assert wait_for(lambda: handle.stopped)


def test_non_browser_session_rejects_pushed_frames(web, sample_video):
    registry, client = web
    response = client.post("/api/sessions", json={"kind": "file", "spec": str(sample_video)})
    assert response.status_code == 201
    session_id = response.get_json()["id"]

    rejected = client.post(
        f"/api/sessions/{session_id}/frame",
        data=jpeg_bytes(),
        content_type="image/jpeg",
    )
    assert rejected.status_code == 409
    registry.stop(session_id)


def test_bad_frame_body_is_rejected(web):
    registry, client = web
    session_id = client.post("/api/sessions", json={"kind": "browser"}).get_json()["id"]
    response = client.post(
        f"/api/sessions/{session_id}/frame",
        data=b"not an image",
        content_type="image/jpeg",
    )
    assert response.status_code == 400
    registry.stop(session_id)


def test_events_endpoint(web):
    registry, client = web
    session_id = client.post("/api/sessions", json={"kind": "browser"}).get_json()["id"]
    client.post(f"/api/sessions/{session_id}/frame", data=jpeg_bytes(), content_type="image/jpeg")
    handle = registry.get(session_id)
    assert wait_for(lambda: handle.event_count > 0)

    data = client.get(f"/api/sessions/{session_id}/events").get_json()
    assert data["events"]
    assert data["next"] == handle.event_count
    registry.stop(session_id)


def test_file_session_writes_csv_and_jsonl(web, sample_video):
    require_csrt()
    registry, client = web
    session_id = client.post(
        "/api/sessions", json={"kind": "file", "spec": str(sample_video)}
    ).get_json()["id"]
    handle = registry.get(session_id)

    assert wait_for(lambda: handle.stopped, timeout=10.0)
    assert handle.state == "done"

    snapshot = client.get(f"/api/sessions/{session_id}").get_json()
    assert snapshot["frames"] == 20
    assert Path(snapshot["csv"]).exists()
    assert Path(snapshot["jsonl"]).exists()


def _run_file_session(registry, client, sample_video):
    session_id = client.post(
        "/api/sessions", json={"kind": "file", "spec": str(sample_video)}
    ).get_json()["id"]
    handle = registry.get(session_id)
    assert wait_for(lambda: handle.stopped, timeout=10.0)
    return session_id, handle


def test_recording_can_be_saved_after_the_session_stops(web, sample_video, tmp_path):
    require_csrt()
    registry, client = web
    session_id, _handle = _run_file_session(registry, client, sample_video)
    snapshot = client.get(f"/api/sessions/{session_id}").get_json()
    if snapshot["recording_frames"] == 0:
        pytest.skip("video writer unavailable in this OpenCV build")

    target = tmp_path / "saved" / "recording.mp4"
    response = client.post(
        f"/api/sessions/{session_id}/recording/save", json={"path": str(target)}
    )
    assert response.status_code == 200
    payload = response.get_json()
    assert Path(payload["saved"]) == target
    assert payload["bytes"] > 0 and target.exists()
    assert client.get(f"/api/sessions/{session_id}").get_json()["recording_pending"] is False


def test_recording_can_be_discarded_after_the_session_stops(web, sample_video):
    require_csrt()
    registry, client = web
    session_id, _handle = _run_file_session(registry, client, sample_video)
    snapshot = client.get(f"/api/sessions/{session_id}").get_json()
    if snapshot["recording_frames"] == 0:
        pytest.skip("video writer unavailable in this OpenCV build")

    response = client.post(f"/api/sessions/{session_id}/recording/discard")
    assert response.status_code == 200
    assert response.get_json()["discarded"] is True
    assert client.get(f"/api/sessions/{session_id}").get_json()["recording_pending"] is False


def test_recording_cannot_be_saved_while_running(web):
    registry, client = web
    session_id = client.post("/api/sessions", json={"kind": "browser"}).get_json()["id"]
    response = client.post(f"/api/sessions/{session_id}/recording/save", json={})
    assert response.status_code == 400
    assert "stop the session" in response.get_json()["error"]
    registry.stop(session_id)


def test_upload_accepts_a_video(web, sample_video):
    require_csrt()
    registry, client = web
    data = {"file": (io.BytesIO(sample_video.read_bytes()), "demo.mp4")}
    response = client.post("/api/upload", data=data, content_type="multipart/form-data")
    assert response.status_code == 201
    registry.stop(response.get_json()["id"])


def test_upload_rejects_bad_extension(web):
    _, client = web
    data = {"file": (io.BytesIO(b"x"), "notes.txt")}
    assert (
        client.post("/api/upload", data=data, content_type="multipart/form-data").status_code == 400
    )


def test_upload_requires_a_file(web):
    _, client = web
    assert client.post("/api/upload", data={}).status_code == 400


def test_stream_route_returns_mjpeg(web):
    registry, client = web
    session_id = client.post("/api/sessions", json={"kind": "browser"}).get_json()["id"]
    client.post(f"/api/sessions/{session_id}/frame", data=jpeg_bytes(), content_type="image/jpeg")
    handle = registry.get(session_id)
    assert wait_for(lambda: handle.renderer.holder.get() is not None)

    response = client.get(f"/stream/{session_id}.mjpeg", buffered=False)
    assert response.status_code == 200
    assert response.mimetype == "multipart/x-mixed-replace"
    response.close()
    registry.stop(session_id)


def test_max_sessions_is_enforced(tmp_path):
    registry, client, _app = make_client(tmp_path, max_sessions=1)
    try:
        assert client.post("/api/sessions", json={"kind": "browser"}).status_code == 201
        assert client.post("/api/sessions", json={"kind": "browser"}).status_code == 429
    finally:
        registry.stop_all()


def test_seed_boxes_create_tracks(web, sample_video):
    require_csrt()
    registry, client = web
    response = client.post(
        "/api/sessions",
        json={"kind": "file", "spec": str(sample_video), "seed_boxes": [[20, 40, 40, 40]]},
    )
    assert response.status_code == 201
    session_id = response.get_json()["id"]
    handle = registry.get(session_id)
    assert wait_for(lambda: handle.stopped, timeout=10.0)

    snapshot = client.get(f"/api/sessions/{session_id}").get_json()
    assert snapshot["created_ids"] == 1
    assert Path(snapshot["csv"]).read_text(encoding="utf-8").count("\n") > 1


def test_invalid_seed_box_is_rejected(web, sample_video):
    _, client = web
    response = client.post(
        "/api/sessions",
        json={"kind": "file", "spec": str(sample_video), "seed_boxes": [[0, 0, 0, 10]]},
    )
    assert response.status_code == 400


def test_box_endpoint_queues_a_manual_box(web):
    registry, client = web
    session_id = client.post("/api/sessions", json={"kind": "browser"}).get_json()["id"]
    client.post(f"/api/sessions/{session_id}/frame", data=jpeg_bytes(), content_type="image/jpeg")
    response = client.post(f"/api/sessions/{session_id}/box", json={"bbox": [30, 90, 50, 60]})
    assert response.status_code == 200
    assert response.get_json()["queued"] is True

    # 手工框在“下一帧”生效：再推一帧以触发应用。
    client.post(
        f"/api/sessions/{session_id}/frame", data=jpeg_bytes(200), content_type="image/jpeg"
    )
    handle = registry.get(session_id)
    assert wait_for(lambda: handle.snapshot()["created_ids"] >= 1)
    registry.stop(session_id)


def test_box_endpoint_rejects_bad_bbox(web):
    registry, client = web
    session_id = client.post("/api/sessions", json={"kind": "browser"}).get_json()["id"]
    assert (
        client.post(f"/api/sessions/{session_id}/box", json={"bbox": [1, 2, 3]}).status_code == 400
    )
    assert client.post(f"/api/sessions/{session_id}/box", json={}).status_code == 400
    assert client.post("/api/sessions/nope/box", json={"bbox": [1, 2, 3, 4]}).status_code == 404
    registry.stop(session_id)


class TestRemoveAndRestart:
    def test_unknown_session_is_404(self, web):
        _, client = web
        assert client.delete("/api/sessions/nope").status_code == 404
        assert client.post("/api/sessions/nope/restart").status_code == 404

    def test_running_session_cannot_be_removed(self, web):
        registry, client = web
        session_id = client.post("/api/sessions", json={"kind": "browser"}).get_json()["id"]
        assert client.delete(f"/api/sessions/{session_id}").status_code == 409
        assert registry.get(session_id) is not None
        registry.stop(session_id)

    def test_stopped_session_can_be_removed(self, web):
        registry, client = web
        session_id = client.post("/api/sessions", json={"kind": "browser"}).get_json()["id"]
        registry.stop(session_id)
        assert wait_for(lambda: registry.get(session_id).stopped)
        assert client.delete(f"/api/sessions/{session_id}").status_code == 200
        assert registry.get(session_id) is None
        assert client.get(f"/api/sessions/{session_id}").status_code == 404

    def test_running_session_cannot_be_restarted(self, web):
        registry, client = web
        session_id = client.post("/api/sessions", json={"kind": "browser"}).get_json()["id"]
        assert client.post(f"/api/sessions/{session_id}/restart").status_code == 409
        registry.stop(session_id)

    def test_stopping_session_restarts_without_manual_wait(self, web):
        """停止后立刻重启：restart 内部等待线程退出，不得误报 409。

        回归背景：stop() 只异步置 quit 事件，此前 restart 要求会话已完全停止，
        用户在浏览器点「停止」后马上点「重启」会拿到 409，界面无任何反应。
        """
        registry, client = web
        session_id = client.post("/api/sessions", json={"kind": "browser"}).get_json()["id"]
        assert client.post(f"/api/sessions/{session_id}/stop").status_code == 200
        # 不 wait_for：restart 自身负责等待退出中的线程
        response = client.post(f"/api/sessions/{session_id}/restart")
        assert response.status_code == 200
        assert response.get_json()["id"] == session_id
        assert response.get_json()["state"] == "running"
        registry.stop(session_id)

    def test_stopped_file_session_restarts_with_the_same_id(self, web, sample_video):
        require_csrt()
        registry, client = web
        session_id = client.post(
            "/api/sessions",
            json={"kind": "file", "spec": str(sample_video), "seed_boxes": [[20, 40, 40, 40]]},
        ).get_json()["id"]
        first = registry.get(session_id)
        assert wait_for(lambda: first.stopped, timeout=10.0)

        response = client.post(f"/api/sessions/{session_id}/restart")
        assert response.status_code == 200
        assert response.get_json()["id"] == session_id
        assert response.get_json()["state"] == "running"

        second = registry.get(session_id)
        assert second is not first  # 新的句柄
        assert wait_for(lambda: second.stopped, timeout=10.0)
        assert second.snapshot()["created_ids"] == 1
