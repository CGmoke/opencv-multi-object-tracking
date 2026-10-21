"""适配层：运动目标检测器（实现 ``domain.ports.Detector``）。

用 OpenCV 背景建模（MOG2）从视频流里**自动发现移动物体**，替代手工框选或需要外部
权重的 YOLO：无需模型文件，开箱即用。多层过滤用于排除静态干扰与噪点：

0. **输入预处理**：灰度化 + 小核高斯模糊（可选中值滤波）。传感器颗粒、压缩块噪声与
   微小纹理是后续误检的放大器，在进入背景模型前先压掉，成本极低；
1. 背景建模本身把静止内容学成背景（静止物体不产生前景）；
2. **时序一致性**：最近 N 帧掩码按位与，只保留持续存在的前景（压制闪烁噪点）；
3. **形态学开运算 + 闭运算**：开运算去孤立噪点，闭运算填补目标内部孔洞并连接裂口；
4. **最小面积 / 最小边长 / 长宽比 / 填充率**：滤掉小噪点、极细长条纹与中空碎片；
5. **最大面积占比**：滤掉整屏或大面积光照突变；
6. **碎片合并**：把重叠/相接的局部框合并成一个整体框（避免一个物体被拆成多个）；
7. **NMS**：抑制被整体框包含的局部框，优先输出覆盖整体的框；
8. **连续命中 + 位移门限**：单帧闪烁与静止物体不进入管道。

检测器只负责“发现候选”，ID 生命周期、轨迹记录与绘制仍由
``TrackManager`` / ``TrackingSession`` 承担，因此本模块与跟踪逻辑正交。

使用注意
--------
背景建模只能看到“与背景不同”的区域：**在预热期内已经在移动的物体**会被当成背景，
之后只剩很窄的前沿条（检测框偏小）。因此实际部署时：

* 让预热期覆盖“场景相对静止”的起始阶段（``warmup_frames``，默认 20 帧）；
* 或先启动服务、再让物体进入画面（物体不曾在背景里出现，即可拿到完整框）。
"""

from __future__ import annotations

from collections import deque
from typing import Any, Deque, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from mot.domain.models import Detection, clip_bbox
from mot.domain.postprocess import merge_fragments, nms

#: 支持的背景建模后端。
BACKGROUND_METHODS: Tuple[str, ...] = ("mog2", "knn")


class MotionDetector:
    """基于背景建模（MOG2 / KNN）的运动检测器。

    Args:
        method: 背景建模算法：``"mog2"``（默认，快）或 ``"knn"``（对高频纹理更稳一些）。
        history: 背景模型的历史帧数（越长越稳，越短适应越快）。
        var_threshold: MOG2 的方差阈值；**越小越敏感**。默认 ``25``（比 OpenCV 默认
            16 高），配合输入模糊可压掉纹理上的细小波动而不丢掉真实目标。
        dist2_threshold: KNN 的平方距离阈值（仅 ``method="knn"`` 生效），默认 ``400``，
            噪点多时调高到 ``625`` 左右。
        learning_rate: 背景学习率（默认 0.0，即预热后**冻结**背景）。冻结的好处是运动
            物体在移动期间**整体**保持为前景，从而得到接近真实的包围盒与尺寸；静止
            物体同样会残留前景，但会被 ``min_motion_px`` 位移门限滤掉。场景缓慢变化
            时可调大（如 0.0005）。
        min_area_px: 最小前景面积（像素）；小于该值视为噪点。
        min_width_px / min_height_px: 包围框最小边长（像素）。
        max_area_ratio: 包围框面积 / 画面面积的上限；超过视为整屏突变。
        blur_kernel: 高斯模糊核边长（奇数，默认 ``5``；``0`` 表示关闭）。在灰度图上
            平滑像素级高频噪声，避免传感器颗粒被背景模型判成前景。
        median_kernel: 中值滤波核边长（奇数，默认 ``0`` 关闭）。对椒盐/压缩噪点比高斯
            更强，代价是轻微抹掉细小纹理；``3`` 通常足够。
        temporal_frames: 掩码时序一致性窗口（默认 ``2``）：最近 N 帧前景按位与后才进入
            后续处理，可压掉只在单帧闪烁的噪点。``1`` 表示关闭。窗口越大越干净，但
            快速移动目标的框会被交集裁窄，通常不要超过 ``3``。
        min_aspect / max_aspect: 包围框长宽比（``w / h``）允许范围，默认 ``0.1~10``；
            用于剔除极细长的纹理条纹（如灯管、百叶窗缝）。
        min_fill_ratio: 轮廓面积 / 包围框面积的下限；用于剔除中空、破碎的局部特征
            （``0`` 表示关闭）。
        merge_iou: 碎片合并的 IoU 阈值；重叠超过该值的局部框合并为一个整体框。
        merge_gap_px: 碎片合并的最大边缘间隙（像素）；默认 ``0`` 关闭间隙合并，只合并
            IoU 足够的重叠框；调到 ``2~3`` 可修复被细噪声线切开的碎片，同时仍不会
            合并有正常间隙的相邻物体。
        nms_iou: NMS 的 IoU 阈值，超过该值的重叠框只保留较大的一个。
        nms_containment: 被大框包含的比例阈值；达到该比例的局部框会被抑制。
        min_consecutive: 需要连续命中的帧数（``1`` 表示不要求持续性）。
        min_motion_px: **位移门限**：候选中心相对首次出现位置的位移小于该值（像素）时
            不输出，用于排除静止物体与轻微抖动；``0`` 表示关闭该过滤。
        morph_kernel: 形态学核边长（``<=1`` 表示跳过形态学）。
        persist_radius_px: 连续命中的中心点匹配半径。
        max_candidates: 保留的候选上限（防止异常帧爆量）。
        label: 输出检测的标签（写入 CSV）。
        detect_shadows: 是否把阴影也标为前景（默认关闭，避免一半噪声）。
        warmup_frames: 前若干帧只用于建立初始背景、**不输出检测**（默认 20）。
        warmup_rate: 预热期的学习率（默认 0.05）。**必须用温和值**：用 1.0 会把模型
            重置成单分量状态，导致之后运动物体一帧就被吸收、只剩很窄的前沿条。

    Raises:
        ValueError: 关键参数非法。
    """

    def __init__(
        self,
        *,
        method: str = "mog2",
        history: int = 200,
        var_threshold: float = 25.0,
        dist2_threshold: float = 400.0,
        learning_rate: float = 0.0,
        min_area_px: float = 200.0,
        min_width_px: int = 8,
        min_height_px: int = 8,
        max_area_ratio: float = 0.6,
        blur_kernel: int = 5,
        median_kernel: int = 0,
        temporal_frames: int = 2,
        min_aspect: float = 0.1,
        max_aspect: float = 10.0,
        min_fill_ratio: float = 0.0,
        merge_iou: float = 0.15,
        merge_gap_px: float = 0.0,
        nms_iou: float = 0.6,
        nms_containment: float = 0.8,
        min_consecutive: int = 2,
        min_motion_px: float = 6.0,
        morph_kernel: int = 3,
        persist_radius_px: float = 40.0,
        max_candidates: int = 200,
        label: str = "motion",
        detect_shadows: bool = False,
        warmup_frames: int = 20,
        warmup_rate: float = 0.05,
    ) -> None:
        if int(history) < 1:
            raise ValueError(f"history must be >= 1, got {history}")
        backend = str(method).strip().lower()
        if backend not in BACKGROUND_METHODS:
            raise ValueError(f"method must be one of {BACKGROUND_METHODS}, got {method!r}")
        if float(var_threshold) <= 0:
            raise ValueError(f"var_threshold must be > 0, got {var_threshold}")
        if float(dist2_threshold) <= 0:
            raise ValueError(f"dist2_threshold must be > 0, got {dist2_threshold}")
        if float(min_area_px) < 0:
            raise ValueError(f"min_area_px must be >= 0, got {min_area_px}")
        if int(min_width_px) < 1 or int(min_height_px) < 1:
            raise ValueError(f"min box sides must be >= 1, got {min_width_px}x{min_height_px}")
        if not 0.0 < float(max_area_ratio) <= 1.0:
            raise ValueError(f"max_area_ratio must be within (0, 1], got {max_area_ratio}")
        for name, value in (("blur_kernel", blur_kernel), ("median_kernel", median_kernel)):
            if int(value) < 0 or (int(value) != 0 and int(value) % 2 == 0):
                raise ValueError(f"{name} must be 0 or a positive odd number, got {value}")
        if int(temporal_frames) < 1:
            raise ValueError(f"temporal_frames must be >= 1, got {temporal_frames}")
        if not 0.0 < float(min_aspect) <= float(max_aspect):
            raise ValueError(
                f"min_aspect must be > 0 and <= max_aspect, got {min_aspect}/{max_aspect}"
            )
        if not 0.0 <= float(min_fill_ratio) <= 1.0:
            raise ValueError(f"min_fill_ratio must be within [0, 1], got {min_fill_ratio}")
        if not 0.0 <= float(merge_iou) <= 1.0:
            raise ValueError(f"merge_iou must be within [0, 1], got {merge_iou}")
        if float(merge_gap_px) < 0:
            raise ValueError(f"merge_gap_px must be >= 0, got {merge_gap_px}")
        if not 0.0 <= float(nms_iou) <= 1.0:
            raise ValueError(f"nms_iou must be within [0, 1], got {nms_iou}")
        if not 0.0 < float(nms_containment) <= 1.0:
            raise ValueError(f"nms_containment must be within (0, 1], got {nms_containment}")
        if int(min_consecutive) < 1:
            raise ValueError(f"min_consecutive must be >= 1, got {min_consecutive}")
        if float(min_motion_px) < 0:
            raise ValueError(f"min_motion_px must be >= 0, got {min_motion_px}")
        if int(warmup_frames) < 0:
            raise ValueError(f"warmup_frames must be >= 0, got {warmup_frames}")
        if not 0.0 < float(warmup_rate) <= 1.0:
            raise ValueError(f"warmup_rate must be within (0, 1], got {warmup_rate}")

        self.min_area_px = float(min_area_px)
        self.min_width_px = int(min_width_px)
        self.min_height_px = int(min_height_px)
        self.max_area_ratio = float(max_area_ratio)
        self.blur_kernel = int(blur_kernel)
        self.median_kernel = int(median_kernel)
        self.temporal_frames = int(temporal_frames)
        self.min_aspect = float(min_aspect)
        self.max_aspect = float(max_aspect)
        self.min_fill_ratio = float(min_fill_ratio)
        self.merge_iou = float(merge_iou)
        self.merge_gap_px = float(merge_gap_px)
        self.nms_iou = float(nms_iou)
        self.nms_containment = float(nms_containment)
        self.min_consecutive = int(min_consecutive)
        self.min_motion_px = float(min_motion_px)
        self.label = str(label)
        self.method = backend
        self.var_threshold = float(var_threshold)
        self.dist2_threshold = float(dist2_threshold)
        self._history = int(history)
        self._detect_shadows = bool(detect_shadows)
        self._morph_kernel = int(morph_kernel)
        self._persist_radius = float(persist_radius_px)
        self._max_candidates = max(1, int(max_candidates))
        self._warmup_frames = int(warmup_frames)
        self._warmup_rate = float(warmup_rate)
        self._frame_count = 0
        self._size: Optional[Tuple[int, int]] = None  # (width, height)
        self._candidates: List[Tuple[float, float, int, float, float]] = []
        self._masks: Deque[np.ndarray] = deque(maxlen=self.temporal_frames)
        self._background = self._make_background()
        self._learning_rate = float(learning_rate)

    def _make_background(self) -> Any:
        """按 ``method`` 新建背景模型；``reset`` / 分辨率变化时复用同一组参数。"""
        if self.method == "knn":
            return cv2.createBackgroundSubtractorKNN(
                history=self._history,
                dist2Threshold=self.dist2_threshold,
                detectShadows=self._detect_shadows,
            )
        return cv2.createBackgroundSubtractorMOG2(
            history=self._history,
            varThreshold=self.var_threshold,
            detectShadows=self._detect_shadows,
        )

    # -- Detector 端口 ---------------------------------------------------
    def __call__(self, frame: Any, timestamp: float = 0.0) -> List[Detection]:
        """返回该帧中“持续出现的运动目标”检测。

        对缺少 ``shape`` 的假帧（单测）直接返回空列表，保持与应用层假件兼容。
        """
        shape = getattr(frame, "shape", None)
        if shape is None or len(shape) < 2:
            return []
        height, width = int(shape[0]), int(shape[1])
        if width <= 0 or height <= 0:
            return []
        if self._size != (width, height):  # 分辨率变化 → 重建背景模型
            self._size = (width, height)
            self.reset(keep_background=False)

        self._frame_count += 1
        warming = bool(self._warmup_frames) and self._frame_count <= self._warmup_frames
        # 预热期用温和学习率建立初始背景，且不输出检测；预热后按 learning_rate 更新
        # （默认冻结），使运动物体整体保持为前景。
        rate = self._warmup_rate if warming else self._learning_rate
        # 先压噪再进背景模型：灰度化 → 高斯（→ 中值）。预热期也用同样的预处理，
        # 保证背景与检测阶段的输入分布一致。
        prepared = self._prepare(frame)
        # 必须用关键字传 learningRate：第二个位置参数在部分 OpenCV 版本里被当作
        # fgmask，导致学习率静默失效（物体躯干被吸收，只剩前沿窄条）。
        mask = self._background.apply(prepared, learningRate=rate)
        if warming:
            self._masks.append(mask)  # 预热期也维护时序窗口，首帧检测即有历史
            return []

        mask = self._stabilize(mask)
        mask = self._clean_mask(mask)
        candidates = self._find_candidates(mask, width, height)
        matched = self._apply_persistence(candidates)

        detections: List[Detection] = []
        for box, area, hits, moved in matched:
            if hits < self.min_consecutive:
                continue
            if moved < self.min_motion_px:  # 位移不足 → 静止物体 / 抖动
                continue
            confidence = min(1.0, area / max(1.0, self.min_area_px * 4.0))
            detections.append(Detection(bbox=box, confidence=confidence, label=self.label))
        return detections

    def reset(self, *, keep_background: bool = False) -> None:
        """清空候选与时序窗口；``keep_background=False`` 时同时重建背景模型。"""
        self._candidates.clear()
        self._masks.clear()
        if not keep_background:
            self._background = self._make_background()
            self._frame_count = 0

    # -- 内部实现 --------------------------------------------------------
    def _prepare(self, frame: np.ndarray) -> np.ndarray:
        """输入预处理：灰度化 + 高斯模糊（可选 + 中值滤波）。

        只压高频噪声，不做尺寸缩放：缩小会让小目标消失，放大则无收益。
        """
        gray = frame
        if gray.ndim == 3:
            gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
        if self.blur_kernel > 1:
            gray = cv2.GaussianBlur(gray, (self.blur_kernel, self.blur_kernel), 0)
        if self.median_kernel > 1:
            gray = cv2.medianBlur(gray, self.median_kernel)
        return gray

    def _stabilize(self, mask: np.ndarray) -> np.ndarray:
        """时序一致性：最近 ``temporal_frames`` 帧掩码按位与。

        只在单帧出现的噪点、边缘抖动与压缩闪烁无法通过多帧交集，从源头被压掉；
        真实目标连续存在，交集后仍保留（快速目标的框会略被裁窄，故窗口默认仅 2）。
        历史不足时返回全零掩码，等价于“延后 N-1 帧再输出”。
        """
        self._masks.append(mask)
        if self.temporal_frames <= 1:
            return mask
        if len(self._masks) < self.temporal_frames:
            return np.zeros_like(mask)
        stable = self._masks[0]
        for previous in list(self._masks)[1:]:
            stable = cv2.bitwise_and(stable, previous)
        return stable

    def _clean_mask(self, mask: np.ndarray) -> np.ndarray:
        """形态学清理：开运算去孤立噪点，闭运算填补孔洞、连接裂口。"""
        if self._morph_kernel <= 1:
            return mask
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (self._morph_kernel, self._morph_kernel)
        )
        opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        return cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)

    def _find_candidates(
        self, mask: np.ndarray, width: int, height: int
    ) -> List[Tuple[float, float, Tuple[float, float, float, float], float]]:
        """过滤连通域 → 碎片合并 → NMS，返回 ``(cx, cy, bbox, area)``。

        处理顺序有意如此：先按单个连通域过滤噪点，再把属于同一物体的碎片合并成
        整体框，最后用 NMS 抑制仍残留的“整体框 + 局部框”重复。若先 NMS 再合并，
        局部框会各自成组，反而把单个物体拆成多个目标。
        """
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        frame_area = float(max(1, width * height))
        raw: List[Tuple[Tuple[float, float, float, float], float]] = []
        for contour in contours:
            area = float(cv2.contourArea(contour))
            if area < self.min_area_px:
                continue
            x, y, w, h = cv2.boundingRect(contour)
            if w < self.min_width_px or h < self.min_height_px:
                continue
            aspect = float(w) / float(h)
            if aspect < self.min_aspect or aspect > self.max_aspect:
                continue  # 极细长条纹：灯管 / 百叶窗缝 / 地平线纹理
            if (float(w) * float(h)) / frame_area > self.max_area_ratio:
                continue
            if self.min_fill_ratio > 0.0 and area / float(w * h) < self.min_fill_ratio:
                continue
            box = clip_bbox((float(x), float(y), float(w), float(h)), width, height)
            if box is None:
                continue
            raw.append((box, area))
            if len(raw) >= self._max_candidates:
                break
        if not raw:
            return []

        merged = merge_fragments(
            (box for box, _area in raw),
            iou_threshold=self.merge_iou,
            gap_px=self.merge_gap_px,
        )
        kept = nms(
            merged,
            iou_threshold=self.nms_iou,
            containment_threshold=self.nms_containment,
        )

        candidates: List[Tuple[float, float, Tuple[float, float, float, float], float]] = []
        for box in kept:
            x, y, w, h = box
            if (float(w) * float(h)) / frame_area > self.max_area_ratio:
                continue  # 合并后仍可能是整屏突变
            if h > 0 and not self.min_aspect <= float(w) / float(h) <= self.max_aspect:
                continue  # 两个碎片拼接后可能变成极细长条，需复查长宽比
            area = float(w) * float(h)
            candidates.append((x + w / 2.0, y + h / 2.0, box, area))
        candidates.sort(key=lambda item: (item[1], item[0]))
        return candidates[: self._max_candidates]

    def _apply_persistence(
        self, candidates: Sequence[Tuple[float, float, Any, float]]
    ) -> List[Tuple[Any, float, int, float]]:
        """就近匹配上一帧候选，累计连续命中次数并计算相对首次出现位置的位移。

        Returns:
            ``(box, area, hits, moved_px)`` 列表；``moved_px`` 用于排除静止物体。
        """
        previous = self._candidates
        used = [False] * len(previous)
        matched: List[Tuple[Any, float, int, float]] = []
        current: List[Tuple[float, float, int, float, float]] = []
        for cx, cy, box, area in candidates:
            best_index = -1
            best_distance = self._persist_radius * self._persist_radius
            for index, (px, py, _hits, _fx, _fy) in enumerate(previous):
                if used[index]:
                    continue
                distance = (cx - px) ** 2 + (cy - py) ** 2
                if distance <= best_distance:
                    best_distance = distance
                    best_index = index
            if best_index >= 0:
                used[best_index] = True
                hits = previous[best_index][2] + 1
                first_x, first_y = previous[best_index][3], previous[best_index][4]
            else:
                hits = 1
                first_x, first_y = cx, cy
            moved = ((cx - first_x) ** 2 + (cy - first_y) ** 2) ** 0.5
            current.append((cx, cy, hits, first_x, first_y))
            matched.append((box, area, hits, moved))
        self._candidates = current[: self._max_candidates]
        return matched


__all__ = ["BACKGROUND_METHODS", "MotionDetector"]
