"""领域层：数据模型、几何度量、IoU 关联、ID 生命周期与端口契约。

约束：本层不得 import ``cv2`` / ``yaml`` / ``csv`` / ``os``，
仅允许标准库与 ``numpy``，以保证核心逻辑可在无 OpenCV 环境下单元测试。

待落地模块（见 docs/ARCHITECTURE_DESIGN.md §3.1）：
``limits`` · ``errors`` · ``models`` · ``records`` · ``geometry`` ·
``association`` · ``ports`` · ``lifecycle``。
"""

from __future__ import annotations

__all__: list[str] = []
