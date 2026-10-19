"""应用层：单帧用例编排、会话状态、配置解析与错误翻译。

约束：本层不得 import ``cv2`` / ``argparse``，仅可依赖 ``domain``。

待落地模块（见 docs/ARCHITECTURE_DESIGN.md §3.2）：
``errors`` · ``config`` · ``pipeline`` · ``session``。
"""

from __future__ import annotations

__all__: list[str] = []
