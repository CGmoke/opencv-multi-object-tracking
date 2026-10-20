"""适配层运动检测器（验收测试）。

依赖 cv2；用合成帧驱动 MOG2 背景建模。注意 MOG2 会把**静止**内容迅速学成背景，
因此“能被检测到”的用例必须让物体真正移动（这也正是运动检测的语义）。
"""

from __future__ import annotations

from typing import List, Sequence

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from mot.adapters.motion import MotionDetector  # noqa: E402
from mot.domain.ports import Detector  # noqa: E402

HEIGHT, WIDTH = 240, 320


def blank() -> np.ndarray:
    return np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)


def with_box(x: int, y: int, w: int = 41, h: int = 51) -> np.ndarray:
    frame = blank()
    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), -1)
    return frame


def warm_up(detector: MotionDetector, frames: int = 20) -> None:
    for _ in range(frames):
        detector(blank(), 0.0)


def feed_moving(detector: MotionDetector, xs: Sequence[int] = (60, 75, 90)):
    detections: List = []
    for x in xs:
        detections = detector(with_box(x, 90), 0.0)
    return detections


def test_implements_the_detector_port():
    assert isinstance(MotionDetector(), Detector)


def test_opaque_frames_are_ignored():
    detector = MotionDetector()
    assert detector(None, 0.0) == []
    assert detector(object(), 0.0) == []


def test_detects_a_moving_object():
    detector = MotionDetector(warmup_frames=20)
    warm_up(detector)
    detections = feed_moving(detector)
    assert detections, "moving object should be detected"
    box = detections[0].bbox
    assert 50 <= box[0] <= 120
    assert 70 <= box[1] <= 110
    assert 30 <= box[2] <= 60
    assert 40 <= box[3] <= 70
    assert detections[0].label == "motion"


def test_static_object_is_not_detected():
    """静止物体虽留在前景中，但位移门限使其始终不输出。"""
    detector = MotionDetector(warmup_frames=20, min_consecutive=1)
    warm_up(detector)
    frame = with_box(100, 90)
    samples = [detector(frame, 0.0) for _ in range(12)]
    assert all(sample == [] for sample in samples)


def test_single_frame_blip_is_filtered():
    detector = MotionDetector(warmup_frames=20, min_consecutive=2)
    warm_up(detector)
    assert detector(with_box(100, 90), 0.0) == []  # 首次出现 → 未达连续命中
    assert detector(blank(), 0.0) == []  # 立刻消失


def test_tiny_moving_blobs_are_filtered():
    """面积/边长过滤与持续性无关：小噪点即使连续出现也不输出。"""
    detector = MotionDetector(warmup_frames=20, min_area_px=200)
    warm_up(detector)
    for x in (100, 110, 120):
        frame = blank()
        cv2.rectangle(frame, (x, 90), (x + 3, 93), (255, 255, 255), -1)  # 4x4=16px
        assert detector(frame, 0.0) == []


def test_max_area_ratio_rejects_full_frame_candidate():
    detector = MotionDetector(warmup_frames=0)
    full = np.full((HEIGHT, WIDTH), 255, dtype=np.uint8)
    assert detector._find_candidates(full, WIDTH, HEIGHT) == []
    partial = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
    cv2.rectangle(partial, (100, 90), (140, 140), 255, -1)
    assert len(detector._find_candidates(partial, WIDTH, HEIGHT)) == 1


def test_full_frame_change_is_ignored():
    detector = MotionDetector(warmup_frames=20)
    warm_up(detector)
    bright = np.full((HEIGHT, WIDTH, 3), 255, dtype=np.uint8)
    assert detector(bright, 0.0) == []


def test_multiple_moving_objects_get_separate_boxes():
    detector = MotionDetector(warmup_frames=20, min_consecutive=2)
    warm_up(detector)
    detections: List = []
    for step in (0, 15, 30):
        frame = blank()
        cv2.rectangle(frame, (40 + step, 60), (80 + step, 139), (255, 255, 255), -1)
        cv2.rectangle(frame, (200 - step, 100), (240 - step, 179), (255, 255, 255), -1)
        detections = detector(frame, 0.0)
    assert len(detections) == 2
    centers = sorted(box.bbox[0] + box.bbox[2] / 2 for box in detections)
    assert centers[0] < WIDTH / 2 < centers[1]


def test_resolution_change_rebuilds_the_model():
    detector = MotionDetector(warmup_frames=0)
    detector(blank(), 0.0)
    assert detector._size == (WIDTH, HEIGHT)
    detector(np.zeros((120, 160, 3), dtype=np.uint8), 0.0)
    assert detector._size == (160, 120)


def test_reset_clears_candidates():
    detector = MotionDetector(warmup_frames=20)
    warm_up(detector)
    detector(with_box(60, 90), 0.0)  # 首次出现即建立候选（尚未输出）
    assert detector._candidates
    detector.reset()
    assert detector._candidates == []


def test_fragments_of_one_object_are_merged_into_one_whole_box():
    """同一物体被切成两块（bbox 明显重叠）→ 只输出一个覆盖整体的框。"""
    detector = MotionDetector(warmup_frames=0, min_area_px=200)
    mask = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
    cv2.fillPoly(mask, [np.array([[40, 40], [120, 40], [120, 110]])], 255)
    cv2.fillPoly(mask, [np.array([[40, 44], [40, 120], [113, 120]])], 255)
    candidates = detector._find_candidates(mask, WIDTH, HEIGHT)
    assert len(candidates) == 1
    box = candidates[0][2]
    assert 38 <= box[0] <= 42
    assert 38 <= box[1] <= 42
    assert box[2] >= 74
    assert box[3] >= 74


def test_adjacent_objects_keep_their_own_boxes():
    """两个有间隙的相邻物体必须保持两个候选，不能被合并成一个。"""
    detector = MotionDetector(warmup_frames=0, min_area_px=200)
    mask = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
    cv2.rectangle(mask, (40, 60), (90, 140), 255, -1)
    cv2.rectangle(mask, (100, 60), (150, 140), 255, -1)  # 10px gap
    candidates = detector._find_candidates(mask, WIDTH, HEIGHT)
    assert len(candidates) == 2
    xs = sorted(candidate[2][0] for candidate in candidates)
    assert xs[0] < 60 and xs[1] > 90


def test_local_box_contained_in_a_whole_box_is_suppressed():
    """大框（环）内部的小连通域不应成为独立候选。"""
    detector = MotionDetector(warmup_frames=0, min_area_px=100)
    mask = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
    cv2.rectangle(mask, (60, 50), (170, 150), 255, 4)
    cv2.rectangle(mask, (100, 90), (120, 110), 255, -1)
    candidates = detector._find_candidates(mask, WIDTH, HEIGHT)
    assert len(candidates) == 1
    box = candidates[0][2]
    assert box[2] >= 100 and box[3] >= 95  # 保留的是覆盖整体的大框


def test_min_fill_ratio_rejects_thin_fragments():
    mask = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
    cv2.line(mask, (20, 20), (140, 140), 255, 5)
    loose = MotionDetector(warmup_frames=0, min_area_px=200, min_fill_ratio=0.0)
    strict = MotionDetector(warmup_frames=0, min_area_px=200, min_fill_ratio=0.3)
    assert len(loose._find_candidates(mask, WIDTH, HEIGHT)) == 1
    assert strict._find_candidates(mask, WIDTH, HEIGHT) == []


@pytest.mark.parametrize(
    "kwargs",
    [
        {"history": 0},
        {"var_threshold": 0},
        {"min_area_px": -1},
        {"min_width_px": 0},
        {"max_area_ratio": 0.0},
        {"min_fill_ratio": -0.1},
        {"min_fill_ratio": 1.1},
        {"merge_iou": -0.1},
        {"merge_iou": 1.1},
        {"merge_gap_px": -1},
        {"nms_iou": 1.1},
        {"nms_containment": 0.0},
        {"min_consecutive": 0},
        {"min_motion_px": -1},
        {"warmup_frames": -1},
        {"warmup_rate": 0.0},
    ],
)
def test_invalid_parameters_are_rejected(kwargs):
    with pytest.raises(ValueError):
        MotionDetector(**kwargs)
