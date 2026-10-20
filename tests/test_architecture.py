"""架构守卫：强制分层依赖方向（见 ``docs/ARCHITECTURE_DESIGN.md`` §2.5）。

以 AST 静态扫描 ``src/mot``，确保每一层不 import 被禁止的模块。这是把分层约束
“纳入 CI”的强制手段，防止后续模块引入反向依赖或框架泄漏。

被守护的规则（允许方向）::

    interfaces ──► application ──► domain
         │                │           ▲
         └────────────────┴──► adapters ┘   # adapters 只依赖 domain

要点
----
* 与设计文档 §2.5 的参考实现相比，这里**正确解析了相对导入**：``from ..application
  import x`` 会被还原为绝对模块 ``mot.application`` 再判定，避免漏判。
* 领域层额外禁止文件/进程 I/O 模块（``pathlib`` / ``tempfile`` / ``shutil`` /
  ``subprocess`` / ``socket``），落实“领域层零 I/O”约束。
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Iterator, List, Set

#: 项目根目录下的源码目录。
SRC = Path(__file__).resolve().parents[1] / "src"

#: 每层禁止导入的模块前缀（``x`` 或 ``x.<sub>`` 均视为违规）。
FORBIDDEN: dict[str, Set[str]] = {
    "mot.domain": {
        "cv2",
        "yaml",
        "csv",
        "os",
        "argparse",
        "pathlib",
        "tempfile",
        "shutil",
        "subprocess",
        "socket",
        "mot.adapters",
        "mot.application",
        "mot.interfaces",
    },
    "mot.application": {
        "cv2",
        "argparse",
        "mot.adapters",
        "mot.interfaces",
    },
    "mot.adapters": {
        "mot.application",
        "mot.interfaces",
    },
}


def _layer_files(layer: str) -> List[Path]:
    """返回某一层的全部 ``.py`` 文件（``mot.domain`` -> ``src/mot/domain``）。"""
    directory = SRC.joinpath(*layer.split("."))
    return sorted(directory.rglob("*.py"))


def _module_name(path: Path) -> str:
    """由文件路径得到其点分模块名，``__init__.py`` 映射到包名。"""
    parts = list(path.relative_to(SRC).with_suffix("").parts)
    if parts and parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def _resolve_relative(node: ast.ImportFrom, package: str) -> str:
    """把 ``ImportFrom``（含相对形式）还原为绝对模块名。"""
    if not node.level:
        return node.module or ""
    base_parts = package.split(".") if package else []
    drop = node.level - 1
    if drop:
        base_parts = base_parts[: len(base_parts) - drop]
    base = ".".join(base_parts)
    if not node.module:
        return base
    return f"{base}.{node.module}" if base else node.module


def _iter_imports(tree: ast.AST, module_name: str) -> Iterator[str]:
    """产出该文件所有被导入的绝对模块名。"""
    package = module_name.rsplit(".", 1)[0] if "." in module_name else ""
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            resolved = _resolve_relative(node, package)
            if resolved:
                yield resolved
        elif isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name


def _is_forbidden(module: str, banned: Set[str]) -> bool:
    return any(module == b or module.startswith(b + ".") for b in banned)


def _violations() -> List[str]:
    offenders: List[str] = []
    for layer, banned in FORBIDDEN.items():
        for path in _layer_files(layer):
            module_name = _module_name(path)
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for imported in _iter_imports(tree, module_name):
                if _is_forbidden(imported, banned):
                    offenders.append(
                        f"{path.relative_to(SRC.parent)} imports {imported!r} (in {layer})"
                    )
    return offenders


def test_guard_is_not_vacuous():
    """每一层都必须真的有源码文件，否则守卫会静默通过。"""
    for layer in FORBIDDEN:
        assert _layer_files(layer), f"no .py files found for {layer}"


def test_import_directions_are_respected():
    offenders = _violations()
    assert not offenders, "architecture violations:\n" + "\n".join(offenders)


def test_relative_imports_are_resolved():
    """自测守卫：相对导入必须被还原成绝对名，否则漏判跨层反向依赖。"""
    tree = ast.parse(
        "from .errors import ValidationError\n"
        "from ..application import pipeline\n"
        "import cv2\n"
        "from cv2 import VideoCapture\n"
    )
    mods = set(_iter_imports(tree, "mot.domain.sample"))
    assert mods == {"mot.domain.errors", "mot.application", "cv2"}

    banned = FORBIDDEN["mot.domain"]
    assert _is_forbidden("mot.application", banned)
    assert _is_forbidden("mot.adapters.sinks.csv", banned)
    assert _is_forbidden("cv2", banned)
    assert not _is_forbidden("mot.domain.errors", banned)
    assert not _is_forbidden("numpy", banned)
