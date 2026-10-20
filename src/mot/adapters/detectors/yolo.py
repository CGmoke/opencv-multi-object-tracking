"""适配层：Ultralytics YOLO 检测器（可选扩展点）。

这是一个**扩展点**，不属于核心功能：

* ``ultralytics`` 为可选依赖，惰性导入，故无它时跟踪包照常工作；
* 本仓库不内置权重，故无自动化测试，需对本地模型人工冒烟；
* 框被裁剪到画面并转成 :class:`mot.domain.models.Detection`，使管道与检测器无关。

安全提示：``.pt`` 权重经 ``torch.load`` 反序列化，可能执行任意代码——只加载可信来
源的模型，且本包默认不自动下载权重（见 ``docs/ROADMAP.md``）。
"""

from __future__ import annotations

from typing import Any, List, Optional, Sequence

from mot.domain.models import Detection, clip_bbox

DEFAULT_MODEL = "yolov8n.pt"


def yolo_available() -> bool:
    """返回可选依赖 ``ultralytics`` 是否已安装。"""
    try:
        import ultralytics  # noqa: F401
    except Exception:  # noqa: BLE001 - 任何导入问题都视为“不可用”
        return False
    return True


class YoloDetector:
    """由 Ultralytics YOLO 模型驱动的可调用检测器。

    Args:
        model: 模型名或路径（``yolov8n.pt``、``yolo11s.pt`` ...）。
        confidence: 采用检测的最低置信度，须落在 ``(0, 1]``。
        classes: 可选保留的类别 id 列表（如 ``[0]`` 表示人）。
        device: 可选推理设备字符串（``"cpu"``、``"cuda:0"``）。

    Raises:
        ImportError: 未安装 ``ultralytics``。
        ValidationError: ``confidence`` 超出 ``(0, 1]``。
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        confidence: float = 0.35,
        classes: Optional[Sequence[int]] = None,
        device: Optional[str] = None,
    ) -> None:
        try:
            from ultralytics import YOLO
        except ImportError as exc:  # pragma: no cover - 可选依赖
            raise ImportError(
                "YoloDetector requires the optional 'ultralytics' package: pip install ultralytics"
            ) from exc

        from mot.domain.errors import ValidationError

        if not 0.0 < float(confidence) <= 1.0:
            raise ValidationError(
                f"confidence must be within (0, 1], got {confidence}",
                field="confidence",
                value=confidence,
            )
        self.model = model
        self.confidence = float(confidence)
        self.classes = list(classes) if classes else None
        self.device = device
        self._net = YOLO(model)

    def __call__(self, frame: Any, timestamp: float = 0.0) -> List[Detection]:
        """对单帧 BGR 图推理并返回已校验的检测。"""
        height, width = frame.shape[:2]
        results = self._net.predict(
            frame,
            conf=self.confidence,
            classes=self.classes,
            device=self.device,
            verbose=False,
        )
        detections: List[Detection] = []
        for result in results:
            boxes = getattr(result, "boxes", None)
            if boxes is None:
                continue
            names = getattr(result, "names", {}) or {}
            for box in boxes:
                x, y, w, h = (float(value) for value in box.xywh[0].tolist())
                clipped = clip_bbox((x - w / 2.0, y - h / 2.0, w, h), width, height)
                if clipped is None:
                    continue
                class_id = int(box.cls[0]) if box.cls is not None else -1
                label = str(names.get(class_id, f"class_{class_id}"))
                score = float(box.conf[0]) if box.conf is not None else self.confidence
                detections.append(Detection(bbox=clipped, confidence=score, label=label))
        return detections


__all__ = ["DEFAULT_MODEL", "YoloDetector", "yolo_available"]
