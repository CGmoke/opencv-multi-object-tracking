"""应用层配置加载与校验（步骤 5 验收测试）。

移植自基线 v0.2.0 的 ``tests/test_config.py``，导入路径改到
``mot.application.config``；并补充新增 ``detector`` 段与 ``backend`` 校验用例。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from mot.application.config import (
    AppConfig,
    ConfigError,
    DetectorConfig,
    DisplayConfig,
    OutputConfig,
    SourceConfig,
    TrackingConfig,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def write(tmp_path: Path, text: str, name: str = "config.yaml") -> Path:
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


class TestDefaults:
    def test_defaults_are_valid(self):
        config = AppConfig.defaults()
        assert config.tracking.backend == "csrt"
        assert config.tracking.iou_threshold == 0.3
        assert config.tracking.max_missing == 15
        assert config.source.source == "0"
        assert config.display.enabled is True
        assert config.output.csv_path == "outputs/trajectories.csv"
        assert config.detector.backend == "none"

    def test_shipped_config_matches_defaults(self):
        config = AppConfig.from_file(REPO_ROOT / "configs" / "default.yaml")
        assert config.to_dict() == AppConfig.defaults().to_dict()

    def test_describe_returns_three_lines(self):
        lines = AppConfig.defaults().describe()
        assert len(lines) == 3
        assert any("backend=csrt" in line for line in lines)


class TestFromFile:
    def test_empty_file_yields_defaults(self, tmp_path):
        config = AppConfig.from_file(write(tmp_path, ""))
        assert config.to_dict() == AppConfig.defaults().to_dict()

    def test_partial_file_keeps_other_defaults(self, tmp_path):
        path = write(tmp_path, "tracking:\n  backend: kcf\n")
        config = AppConfig.from_file(path)
        assert config.tracking.backend == "kcf"
        assert config.tracking.iou_threshold == 0.3

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(ConfigError, match="cannot read config file"):
            AppConfig.from_file(tmp_path / "nope.yaml")

    def test_unknown_section_raises(self, tmp_path):
        with pytest.raises(ConfigError, match="unknown config section"):
            AppConfig.from_file(write(tmp_path, "trackng:\n  backend: kcf\n"))

    def test_unknown_key_raises(self, tmp_path):
        with pytest.raises(ConfigError, match="unknown key"):
            AppConfig.from_file(write(tmp_path, "tracking:\n  iou_treshold: 0.4\n"))

    def test_non_mapping_root_raises(self, tmp_path):
        with pytest.raises(ConfigError, match="must be a mapping"):
            AppConfig.from_file(write(tmp_path, "- a\n- b\n"))

    def test_section_must_be_a_mapping(self, tmp_path):
        with pytest.raises(ConfigError, match="must be a mapping"):
            AppConfig.from_file(write(tmp_path, "tracking: 5\n"))

    def test_invalid_yaml_raises(self, tmp_path):
        with pytest.raises(ConfigError, match="invalid YAML"):
            AppConfig.from_file(write(tmp_path, "tracking: [unclosed\n"))

    def test_huge_file_is_rejected(self, tmp_path):
        path = write(tmp_path, "#" + "x" * 1_000_001)
        with pytest.raises(ConfigError, match="unreasonably large"):
            AppConfig.from_file(path)


class TestValidation:
    @pytest.mark.parametrize(
        "section,payload",
        [
            ("tracking", "  backend: yolo\n"),
            ("tracking", "  iou_threshold: 1.5\n"),
            ("tracking", "  iou_threshold: -0.2\n"),
            ("tracking", "  max_missing: -1\n"),
            ("tracking", "  trail_length: 0\n"),
            ("tracking", "  match_method: magic\n"),
            ("tracking", "  box_smoothing: 0\n"),
            ("tracking", "  box_smoothing: 1.5\n"),
            ("source", "  max_frames: 0\n"),
            ("source", "  loop: 'yes'\n"),
            ("source", "  source: ''\n"),
            ("display", "  enabled: 'true'\n"),
            ("display", "  window_name: ''\n"),
            ("output", "  csv_path: outputs/\n"),
            ("output", "  csv_path: ''\n"),
        ],
    )
    def test_invalid_values_are_rejected(self, tmp_path, section, payload):
        with pytest.raises(ConfigError):
            AppConfig.from_file(write(tmp_path, f"{section}:\n{payload}"))

    def test_seed_boxes_are_normalised(self, tmp_path):
        path = write(tmp_path, "source:\n  seed_boxes:\n    - [10, 20, -30, 40]\n")
        config = AppConfig.from_file(path)
        assert config.source.seed_boxes == [(-20.0, 20.0, 30.0, 40.0)]

    def test_bad_seed_box_shape_is_rejected(self, tmp_path):
        path = write(tmp_path, "source:\n  seed_boxes:\n    - [1, 2, 3]\n")
        with pytest.raises(ConfigError, match="seed_boxes"):
            AppConfig.from_file(path)

    def test_zero_area_seed_box_is_rejected(self, tmp_path):
        path = write(tmp_path, "source:\n  seed_boxes:\n    - [1, 2, 0, 3]\n")
        with pytest.raises(ConfigError, match="seed_boxes"):
            AppConfig.from_file(path)

    def test_csv_path_can_be_null(self, tmp_path):
        config = AppConfig.from_file(write(tmp_path, "output:\n  csv_path: null\n"))
        assert config.output.csv_path is None

    def test_validate_normalises_types(self):
        tracking = TrackingConfig(backend=" CSRT ", iou_threshold=0.5, max_missing=3)
        tracking.validate()
        assert tracking.backend == "csrt"
        assert isinstance(tracking.iou_threshold, float)

        output = OutputConfig(csv_path=" out/traj.csv ")
        output.validate()
        assert output.csv_path == "out/traj.csv"

        source = SourceConfig(source=" clip.mp4 ", max_frames=10)
        source.validate()
        assert source.source == "clip.mp4"
        assert source.max_frames == 10

        display = DisplayConfig(window_name="win")
        display.validate()
        assert display.window_name == "win"


class TestDetectorSection:
    def test_default_is_disabled(self):
        assert DetectorConfig().backend == "none"

    def test_yolo_backend_is_accepted(self):
        detector = DetectorConfig(backend=" YOLO ", model="weights/y.pt", confidence=0.5)
        detector.validate()
        assert detector.backend == "yolo"
        assert detector.model == "weights/y.pt"

    @pytest.mark.parametrize(
        "payload",
        [
            "  backend: magic\n",
            "  confidence: 1.5\n",
            "  confidence: -0.1\n",
            "  interval: 0\n",
            "  model: ''\n",
            "  classes: ['']\n",
        ],
    )
    def test_invalid_values_are_rejected(self, tmp_path, payload):
        with pytest.raises(ConfigError):
            AppConfig.from_file(write(tmp_path, f"detector:\n{payload}"))

    def test_unknown_detector_key_is_rejected(self, tmp_path):
        with pytest.raises(ConfigError, match="unknown key"):
            AppConfig.from_file(write(tmp_path, "detector:\n  weight: y.pt\n"))


class TestMotionSection:
    def test_defaults_are_disabled(self):
        motion = AppConfig.defaults().motion
        assert motion.enabled is False
        assert motion.method == "mog2"
        assert motion.var_threshold == 25.0
        assert motion.dist2_threshold == 400.0
        assert motion.min_area_px == 200.0
        assert motion.pixels_per_meter is None

    @pytest.mark.parametrize(
        "payload",
        [
            "  var_threshold: 0\n",
            "  method: yolo\n",
            "  dist2_threshold: 0\n",
            "  min_area_px: -1\n",
            "  min_width_px: 0\n",
            "  min_consecutive: 0\n",
            "  min_motion_px: -1\n",
            "  max_area_ratio: 0\n",
            "  blur_kernel: 4\n",
            "  median_kernel: 2\n",
            "  blur_kernel: -1\n",
            "  min_fill_ratio: -0.1\n",
            "  merge_iou: 1.1\n",
            "  merge_gap_px: -1\n",
            "  nms_iou: -0.1\n",
            "  nms_containment: 0\n",
            "  interval: 0\n",
            "  enabled: 'yes'\n",
            "  large_min_m: 0.1\n",
            "  pixels_per_meter: 0\n",
        ],
    )
    def test_invalid_values_are_rejected(self, tmp_path, payload):
        with pytest.raises(ConfigError):
            AppConfig.from_file(write(tmp_path, f"motion:\n{payload}"))

    def test_unknown_motion_key_is_rejected(self, tmp_path):
        with pytest.raises(ConfigError, match="unknown key"):
            AppConfig.from_file(write(tmp_path, "motion:\n  thresholds: 5\n"))


class TestWebSection:
    def test_defaults(self):
        web = AppConfig.defaults().web
        assert web.host == "127.0.0.1"
        assert web.port == 8000
        assert web.max_sessions == 4

    @pytest.mark.parametrize(
        "payload",
        [
            "  port: 0\n",
            "  port: 70000\n",
            "  max_sessions: 0\n",
            "  stream_fps: 0\n",
            "  host: ''\n",
            "  max_upload_bytes: 0\n",
            "  upload_dir: ''\n",
        ],
    )
    def test_invalid_values_are_rejected(self, tmp_path, payload):
        with pytest.raises(ConfigError):
            AppConfig.from_file(write(tmp_path, f"web:\n{payload}"))

    def test_unknown_web_key_is_rejected(self, tmp_path):
        with pytest.raises(ConfigError, match="unknown key"):
            AppConfig.from_file(write(tmp_path, "web:\n  threads: 8\n"))


class TestRoundTrip:
    def test_to_dict_contains_every_section(self):
        data = AppConfig.defaults().to_dict()
        assert set(data) == {
            "tracking",
            "detector",
            "motion",
            "display",
            "output",
            "source",
            "web",
        }
        assert data["source"]["seed_boxes"] == []

    def test_from_file_accepts_its_own_output(self, tmp_path):
        import yaml

        config = AppConfig.defaults()
        path = write(tmp_path, yaml.safe_dump(config.to_dict()))
        assert AppConfig.from_file(path).to_dict() == config.to_dict()
