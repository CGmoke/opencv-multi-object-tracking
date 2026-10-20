"""接口层 CLI 与端到端（步骤 6b 验收测试）。

移植自基线 v0.2.0 的 ``tests/test_app.py``：导入路径改到 ``mot.interfaces.cli`` 与
``mot.interfaces.ui_window``，日志 logger 改为 ``mot.cli``。
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys

import pytest

cv2 = pytest.importorskip("cv2")

import numpy as np  # noqa: E402 - importorskip 必须先行

from mot.adapters.trackers import tracker_available  # noqa: E402
from mot.interfaces.cli import (  # noqa: E402
    _parse_seed_box,
    build_config,
    build_detector,
    build_parser,
    main,
)
from mot.interfaces.ui_window import display_available, save_snapshot  # noqa: E402


def require_csrt() -> None:
    if not tracker_available("csrt"):
        pytest.skip("csrt tracker not available in this OpenCV build")


class TestSeedBoxParser:
    def test_parses_four_numbers(self):
        assert _parse_seed_box("10,20,30,40") == (10.0, 20.0, 30.0, 40.0)

    def test_accepts_semicolons_and_spaces(self):
        assert _parse_seed_box(" 10 ; 20 , 30 ,40 ") == (10.0, 20.0, 30.0, 40.0)

    def test_flips_negative_sizes(self):
        assert _parse_seed_box("50,60,-10,-20") == (40.0, 40.0, 10.0, 20.0)

    @pytest.mark.parametrize("text", ["1,2,3", "1,2,3,4,5", ""])
    def test_rejects_wrong_arity(self, text):
        with pytest.raises(argparse.ArgumentTypeError):
            _parse_seed_box(text)

    def test_rejects_non_numeric(self):
        with pytest.raises(argparse.ArgumentTypeError):
            _parse_seed_box("a,b,c,d")

    def test_rejects_zero_area(self):
        with pytest.raises(argparse.ArgumentTypeError):
            _parse_seed_box("0,0,0,10")


class TestBuildConfig:
    def test_defaults_when_no_flags(self):
        config = build_config(build_parser().parse_args([]))
        assert config.tracking.backend == "csrt"
        assert config.source.source == "0"
        assert config.display.enabled is True
        assert config.detector.backend == "none"

    def test_cli_overrides(self):
        args = build_parser().parse_args(
            [
                "--source",
                "clip.mp4",
                "--tracker",
                "kcf",
                "--iou-threshold",
                "0.5",
                "--max-missing",
                "3",
                "--trail-length",
                "40",
                "--match-method",
                "hungarian",
                "--csv",
                "none",
                "--record",
                "annotated.mp4",
                "--seed-box",
                "1,2,3,4",
                "--max-frames",
                "7",
                "--loop",
                "--no-display",
            ]
        )
        config = build_config(args)
        assert config.source.source == "clip.mp4"
        assert config.tracking.backend == "kcf"
        assert config.tracking.iou_threshold == 0.5
        assert config.tracking.max_missing == 3
        assert config.tracking.trail_length == 40
        assert config.tracking.match_method == "hungarian"
        assert config.output.csv_path is None
        assert config.output.video_path == "annotated.mp4"
        assert config.source.seed_boxes == [(1.0, 2.0, 3.0, 4.0)]
        assert config.source.max_frames == 7
        assert config.source.loop is True
        assert config.display.enabled is False

    def test_detector_flags(self):
        args = build_parser().parse_args(
            [
                "--detector",
                "yolo",
                "--detector-model",
                "w.pt",
                "--detector-interval",
                "5",
                "--detector-conf",
                "0.6",
            ]
        )
        config = build_config(args)
        assert config.detector.backend == "yolo"
        assert config.detector.model == "w.pt"
        assert config.detector.interval == 5
        assert config.detector.confidence == 0.6

    def test_config_file_is_overridden_by_cli(self, tmp_path):
        config_file = tmp_path / "config.yaml"
        config_file.write_text("tracking:\n  backend: kcf\n  max_missing: 9\n", encoding="utf-8")
        args = build_parser().parse_args(
            ["--config", str(config_file), "--tracker", "csrt", "--max-missing", "2"]
        )
        config = build_config(args)
        assert config.tracking.backend == "csrt"
        assert config.tracking.max_missing == 2
        assert config.source.source == "0"

    def test_dash_also_disables_csv(self):
        args = build_parser().parse_args(["--csv", "-"])
        assert build_config(args).output.csv_path is None


class TestBuildDetector:
    def test_none_backend_returns_none(self):
        assert build_detector(build_config(build_parser().parse_args([]))) is None

    def test_yolo_without_ultralytics_raises_import_error(self):
        from mot.adapters.detectors.yolo import yolo_available

        if yolo_available():
            pytest.skip("ultralytics installed; the ImportError path is not exercised")
        config = build_config(build_parser().parse_args(["--detector", "yolo"]))
        with pytest.raises(ImportError):
            build_detector(config)


class TestMainEntryPoints:
    def test_dump_config(self, capsys):
        assert main(["--dump-config"]) == 0
        out = capsys.readouterr().out
        assert "tracking:" in out
        assert "backend: csrt" in out
        assert "detector:" in out

    def test_list_trackers(self, capsys):
        assert main(["--list-trackers"]) == 0
        out = capsys.readouterr().out
        assert "csrt" in out
        assert "kcf" in out

    def test_missing_config_returns_two(self, tmp_path, capsys):
        assert main(["--config", str(tmp_path / "missing.yaml")]) == 2
        assert "config error" in capsys.readouterr().err

    def test_invalid_cli_value_returns_argparse_error(self):
        with pytest.raises(SystemExit) as excinfo:
            main(["--tracker", "yolo"])
        assert excinfo.value.code == 2


class TestHelpers:
    def test_display_available_respects_environment(self, monkeypatch):
        if not sys.platform.startswith("linux"):
            pytest.skip("Linux-specific probe")
        monkeypatch.delenv("DISPLAY", raising=False)
        monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
        assert display_available() is False
        monkeypatch.setenv("DISPLAY", ":99")
        assert display_available() is True

    def test_save_snapshot_names_the_file(self, tmp_path):
        path = save_snapshot(np.zeros((40, 60, 3), dtype=np.uint8), tmp_path, "clip.mp4", 7)
        assert path.exists()
        assert path.name == "clip.mp4_frame_000007.png"

    def test_sanitizes_source_name(self, tmp_path):
        path = save_snapshot(np.zeros((40, 60, 3), dtype=np.uint8), tmp_path, "../a b", 1)
        assert path.name == "a_b_frame_000001.png"


class TestEndToEnd:
    def test_tracks_a_synthetic_video_headless(self, sample_video, tmp_path, capsys):
        require_csrt()
        csv_path = tmp_path / "out" / "trajectories.csv"
        code = main(
            [
                "--source",
                str(sample_video),
                "--no-display",
                "--csv",
                str(csv_path),
                "--seed-box",
                "20,40,40,40",
                "--max-frames",
                "10",
                "--tracker",
                "csrt",
            ]
        )
        assert code == 0
        assert "MOT finished: 10 frames" in capsys.readouterr().out
        assert csv_path.exists()

        with csv_path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        assert len(rows) >= 5
        assert {row["track_id"] for row in rows} == {"1"}
        assert max(int(row["frame"]) for row in rows) <= 10

    def test_csv_can_be_disabled(self, sample_video, tmp_path, capsys):
        require_csrt()
        code = main(
            [
                "--source",
                str(sample_video),
                "--no-display",
                "--csv",
                "none",
                "--seed-box",
                "20,40,40,40",
                "--max-frames",
                "3",
            ]
        )
        assert code == 0
        assert not (tmp_path / "trajectories.csv").exists()

    def test_missing_source_returns_two(self, tmp_path, caplog):
        with caplog.at_level(logging.ERROR, logger="mot.cli"):
            code = main(["--source", str(tmp_path / "nope.mp4"), "--no-display"])
        assert code == 2
        assert any("does not exist" in record.getMessage() for record in caplog.records)

    def test_recording_produces_a_video(self, sample_video, tmp_path, capsys):
        require_csrt()
        video_path = tmp_path / "annotated.mp4"
        code = main(
            [
                "--source",
                str(sample_video),
                "--no-display",
                "--csv",
                "none",
                "--record",
                str(video_path),
                "--seed-box",
                "20,40,40,40",
                "--max-frames",
                "5",
            ]
        )
        assert code == 0
        assert video_path.exists()
        assert video_path.stat().st_size > 0
