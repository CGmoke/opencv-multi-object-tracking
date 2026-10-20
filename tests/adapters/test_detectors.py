"""适配层检测器（步骤 6a 验收测试）。"""

from __future__ import annotations

import pytest

from mot.adapters.detectors.base import FixedDetector, NullDetector
from mot.adapters.detectors.yolo import DEFAULT_MODEL, YoloDetector, yolo_available
from mot.domain.models import Detection
from mot.domain.ports import Detector


def test_null_detector_returns_empty():
    assert NullDetector()(None, 0.0) == []


def test_fixed_detector_returns_configured_detections():
    dets = [Detection(bbox=(0, 0, 10, 10), label="car")]
    detector = FixedDetector(dets)
    assert detector(None, 1.0) == dets
    assert detector(None, 2.0) == dets  # 每次返回副本，内容一致


def test_fake_detectors_satisfy_the_port():
    assert isinstance(NullDetector(), Detector)
    assert isinstance(FixedDetector(), Detector)


def test_yolo_available_reports_a_bool():
    assert isinstance(yolo_available(), bool)


def test_default_model_name():
    assert DEFAULT_MODEL.endswith(".pt")


def test_yolo_detector_requires_ultralytics():
    if yolo_available():
        pytest.skip("ultralytics installed; the ImportError path is not exercised")
    with pytest.raises(ImportError):
        YoloDetector("yolov8n.pt")
