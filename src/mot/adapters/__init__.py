"""适配层：以端口（``domain.ports``）的实现对接外部框架。

约束：本层是唯一允许 import ``cv2`` 的一层；不得反向依赖
``application`` / ``interfaces``。

待落地模块（见 docs/ARCHITECTURE_DESIGN.md §3.3）：
``sources`` · ``trackers`` · ``detectors`` · ``sinks`` · ``rendering`` · ``ui``。
"""

from __future__ import annotations

__all__: list[str] = []
