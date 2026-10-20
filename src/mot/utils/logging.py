"""无状态工具：日志配置。"""

from __future__ import annotations

import logging


def setup_logging(verbose: bool = False) -> None:
    """按指定详细度配置一次控制台日志。"""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(levelname)s %(message)s",
    )


__all__ = ["setup_logging"]
