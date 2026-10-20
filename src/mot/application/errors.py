"""应用层错误基类与退出码映射。

分层约定（见 ``docs/ARCHITECTURE_DESIGN.md`` §4.2）：接口层兜底并把
``MotError.exit_code`` 作为进程退出码。

退出码契约：``0`` 正常 / ``1`` 无帧 / ``2`` 配置或输入输出错误 / ``130`` 中断。

**端口边界错误不在本模块**：``SourceError`` / ``SinkError`` / ``TrackerSetupError``
由适配器抛出，而架构守卫禁止 ``adapters → application``，因此它们定义在
:mod:`mot.domain.errors`（领域层拥有端口契约）。接口层在兜底时同时映射
``MotError`` 与 ``DomainError`` 两类。
"""

from __future__ import annotations


class MotError(Exception):
    """应用层可预期错误的基类；``exit_code`` 供接口层映射为进程退出码。"""

    exit_code: int = 2


class ConfigError(MotError):
    """配置或 CLI 参数非法。"""


__all__ = ["ConfigError", "MotError"]
