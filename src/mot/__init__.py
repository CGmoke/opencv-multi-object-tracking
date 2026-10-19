"""纯 OpenCV 多目标跟踪框架（目标架构 v0.2.0）。

分层（依赖方向自上而下单向）：``domain`` → ``application`` → ``adapters`` → ``interfaces``。

公开 API 将在各层模块落地后于此处统一再导出，见
``docs/ARCHITECTURE_DESIGN.md`` §6.1（阶段 P4 / 步骤 E23）。
"""

from __future__ import annotations

__version__ = "0.2.0"

__all__: list[str] = []
