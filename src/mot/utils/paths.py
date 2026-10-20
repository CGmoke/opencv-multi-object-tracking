"""无状态工具：文件名清洗与父目录创建。"""

from __future__ import annotations

import re
from pathlib import Path

_SAFE_NAME = re.compile(r"[^A-Za-z0-9._-]+")


def sanitize_filename(text: str, fallback: str = "frame") -> str:
    """把任意文本（如源名）降为安全的文件名片段。"""
    cleaned = _SAFE_NAME.sub("_", str(text)).strip("._-")
    return cleaned[:80] or fallback


def ensure_parent_dir(path: str | Path) -> Path:
    """创建 ``path`` 的父目录并返回展开后的路径。"""
    target = Path(path).expanduser()
    target.parent.mkdir(parents=True, exist_ok=True)
    return target


__all__ = ["ensure_parent_dir", "sanitize_filename"]
