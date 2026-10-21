"""适配层运动检测器（验收测试）。

依赖 cv2；用合成帧驱动 MOG2 背景建模。注意 MOG2 会把**静止**内容迅速学成背景，
因此“能被检测到”的用例必须让物体真正移动（这也正是运动检测的语义）。
"""

from __future__ import annotations

from collections import deque
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


@pytest.mark.parametrize("method", ["mog2", "knn"])
def test_both_background_methods_detect_a_moving_object(method):
    detector = MotionDetector(warmup_frames=20, method=method)
    warm_up(detector)
    detections = feed_moving(detector)
    assert detections, f"{method} should detect the moving object"
    assert detections[0].bbox[2] >= 20


def test_method_is_normalised_to_lower_case():
    assert MotionDetector(method="KNN").method == "knn"


def test_reset_rebuilds_the_same_backend():
    detector = MotionDetector(method="knn", dist2_threshold=625)
    detector.reset()
    assert detector.method == "knn"
    assert detector._background.getDist2Threshold() == pytest.approx(625)


def test_shadow_detection_is_disabled_by_default():
    assert MotionDetector()._background.getDetectShadows() is False


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
    detector = MotionDetector(warmup_frames=20, temporal_frames=1)
    warm_up(detector)
    detector(with_box(60, 90), 0.0)  # 首次出现即建立候选（尚未输出）
    assert detector._candidates
    detector.reset()
    assert detector._candidates == []
    assert detector._masks == deque(maxlen=1)


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


def test_preprocessing_converts_to_greyscale():
    detector = MotionDetector(warmup_frames=0)
    prepared = detector._prepare(with_box(10, 10))
    assert prepared.ndim == 2
    assert prepared.dtype == np.uint8
    assert prepared.shape == (HEIGHT, WIDTH)


def test_gaussian_blur_reduces_pixel_level_noise():
    rng = np.random.default_rng(0)
    frame = np.full((HEIGHT, WIDTH, 3), 120, dtype=np.uint8)
    frame = np.clip(frame + rng.normal(0, 20, frame.shape), 0, 255).astype(np.uint8)
    raw = MotionDetector(warmup_frames=0, blur_kernel=0)._prepare(frame)
    smooth = MotionDetector(warmup_frames=0, blur_kernel=5)._prepare(frame)
    assert cv2.Laplacian(smooth, cv2.CV_32F).var() < cv2.Laplacian(raw, cv2.CV_32F).var()


def test_median_filter_removes_salt_and_pepper():
    frame = np.full((HEIGHT, WIDTH, 3), 100, dtype=np.uint8)
    frame[::7, ::5] = 255  # 孤立亮点
    frame[::11, ::9] = 0  # 孤立暗点
    raw = MotionDetector(warmup_frames=0, blur_kernel=0)._prepare(frame)
    median = MotionDetector(warmup_frames=0, blur_kernel=0, median_kernel=3)._prepare(frame)
    assert int(np.count_nonzero(median == 255)) < int(np.count_nonzero(raw == 255))
    assert int(np.count_nonzero(median == 0)) < int(np.count_nonzero(raw == 0))


def test_zero_kernels_disable_the_filters():
    frame = with_box(10, 10)
    prepared = MotionDetector(warmup_frames=0, blur_kernel=0, median_kernel=0)._prepare(frame)
    assert np.array_equal(prepared, cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))


def test_temporal_and_rejects_a_single_frame_flicker():
    """只在 1 帧出现的噪点斑块不得进入候选（窗口=2 的掩码交集为空）。"""
    detector = MotionDetector(warmup_frames=20, temporal_frames=2, min_consecutive=1)
    warm_up(detector)
    blank_mask = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
    blob = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
    cv2.rectangle(blob, (200, 40), (280, 120), 255, -1)

    assert np.count_nonzero(detector._stabilize(blank_mask)) == 0
    assert np.count_nonzero(detector._stabilize(blob)) == 0  # 与上一帧无交集
    assert np.count_nonzero(detector._stabilize(blob)) > 0  # 连续出现才保留


def test_temporal_and_waits_for_a_full_window():
    detector = MotionDetector(warmup_frames=0, temporal_frames=3)
    mask = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
    cv2.rectangle(mask, (40, 40), (120, 120), 255, -1)
    assert np.count_nonzero(detector._stabilize(mask)) == 0
    assert np.count_nonzero(detector._stabilize(mask)) == 0
    assert np.count_nonzero(detector._stabilize(mask)) > 0


def test_temporal_window_is_cleared_on_reset():
    detector = MotionDetector(warmup_frames=0, temporal_frames=3)
    detector._stabilize(np.full((HEIGHT, WIDTH), 255, dtype=np.uint8))
    detector.reset()
    assert len(detector._masks) == 0


def test_morphology_closing_fills_pinholes():
    detector = MotionDetector(warmup_frames=0, morph_kernel=3)
    mask = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
    cv2.rectangle(mask, (40, 40), (120, 120), 255, -1)
    mask[80, 80] = 0  # 1px 孔洞
    cleaned = detector._clean_mask(mask)
    assert cleaned[80, 80] == 255


def test_extreme_aspect_ratio_is_rejected():
    mask = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
    cv2.rectangle(mask, (20, 100), (300, 120), 255, -1)  # 281x21，横向长条
    loose = MotionDetector(warmup_frames=0, min_area_px=200, max_aspect=20.0)
    strict = MotionDetector(warmup_frames=0, min_area_px=200)  # 默认 max_aspect=10
    assert len(loose._find_candidates(mask, WIDTH, HEIGHT)) == 1
    assert strict._find_candidates(mask, WIDTH, HEIGHT) == []


def test_vertical_strip_is_rejected_by_min_aspect():
    mask = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
    cv2.rectangle(mask, (150, 10), (170, 230), 255, -1)  # 21x221，纵向细条
    detector = MotionDetector(warmup_frames=0, min_area_px=200)
    assert detector._find_candidates(mask, WIDTH, HEIGHT) == []


@pytest.mark.parametrize(
    "kwargs",
    [
        {"history": 0},
        {"method": "yolo"},
        {"var_threshold": 0},
        {"dist2_threshold": 0},
        {"min_area_px": -1},
        {"min_width_px": 0},
        {"max_area_ratio": 0.0},
        {"blur_kernel": -1},
        {"blur_kernel": 4},
        {"median_kernel": 2},
        {"temporal_frames": 0},
        {"min_aspect": 0.0},
        {"min_aspect": 5.0, "max_aspect": 2.0},
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
