"""领域层异常定义与错误规范。

本模块只依赖标准库，集中定义 ``domain`` 层统一的异常继承体系与标准字段
（``code`` / ``message`` / ``context``），供所有领域校验点复用。

设计约定
--------
* 领域层只做“边界/致命”判断：违反约束立即抛出，由上层翻译与兜底
  （见 ``docs/ARCHITECTURE_DESIGN.md`` §4.2）。
* 进程退出码是 **接口层** 的职责，不在本层定义；应用层的 ``MotError`` 家族
  负责把领域异常翻译为退出码（0/1/2/130）。
* :class:`ValidationError` 同时继承 :class:`ValueError`，以兼容既有“非法入参用
  ``ValueError`` 表达”的调用方（含 ``pytest.raises(ValueError)``）。

继承体系::

    Exception
    └── DomainError                    # 统一字段：code / message / context
        ├── ValidationError            # 边界校验失败（同时也是 ValueError）
        ├── SourceError                # 端口：输入源打开/读取失败
        ├── SinkError                  # 端口：输出目标打开/写入失败
        └── TrackerSetupError          # 端口：跟踪器后端不可用/构造失败

**端口边界错误放在领域层的原因**：适配器（``adapters/``）需要抛出可被接口层统一
映射为退出码的错误，但架构守卫禁止 ``adapters → application``。领域层已拥有端口
契约（``ports``），故在此定义端口失败类型；退出码仍由接口层决定（见 §4.2）。

扩展方式
--------
新增领域错误时继承 :class:`DomainError`，覆写类属性 ``code``，并在 ``__init__``
中把结构化信息写入 ``context``。这样 ``code`` / ``message`` / ``context``
三个标准字段对所有领域异常保持一致。
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

#: 哨兵：用于区分“调用方未提供 value”与“value 恰好为 None”。
_MISSING: Any = object()


class DomainError(Exception):
    """领域层所有异常的基类。

    统一携带三个标准字段：

    - ``code``：稳定、可检索的错误码（类级默认值，可在实例上覆盖）。
    - ``message``：面向人的错误描述（也是 ``Exception.args`` 的唯一元素）。
    - ``context``：结构化上下文（字段名、非法值等），便于日志与测试断言。

    Args:
        message: 错误描述；非字符串会被 ``str()`` 归一。
        code: 覆盖类级默认错误码；``None`` 表示沿用子类的 ``code``。
        context: 结构化上下文字典；``None`` 视为空，内部会做浅拷贝，
            调用方后续修改原字典不会影响异常对象。
    """

    #: 该类错误的默认错误码；子类应覆盖为更具体的值。
    code: str = "MOT-DOMAIN-ERROR"

    def __init__(
        self,
        message: str,
        *,
        code: Optional[str] = None,
        context: Optional[Mapping[str, Any]] = None,
        field: Optional[str] = None,
        value: Any = _MISSING,
    ) -> None:
        merged: Dict[str, Any] = dict(context) if context else {}
        if field is not None:
            merged.setdefault("field", field)
        if value is not _MISSING:
            merged.setdefault("value", value)
        super().__init__(str(message))
        self.message: str = str(message)
        if code is not None:
            self.code = str(code)
        self.context: Dict[str, Any] = merged
        self.field: Optional[str] = field
        self.value: Any = None if value is _MISSING else value

    def to_dict(self) -> Dict[str, Any]:
        """返回可序列化的错误表示（供日志、事件输出与测试断言使用）。

        说明：``context`` 内部可能含有非 JSON 友好值（例如 numpy 标量），
        本方法只做结构拷贝，不做强制序列化，由调用方按需处理。
        """
        return {"code": self.code, "message": self.message, "context": dict(self.context)}

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(code={self.code!r}, "
            f"message={self.message!r}, context={self.context!r})"
        )


class ValidationError(DomainError, ValueError):
    """边界校验失败：某个输入值违反了领域约束。

    继承 :class:`DomainError`（标准化字段）与 :class:`ValueError`（向后兼容），
    使既有 ``except ValueError`` / ``pytest.raises(ValueError, match=...)``
    的调用方在迁移到新架构后仍可正常工作。

    额外携带两个便捷字段：

    - ``field``：出错的字段名（领域层使用裸名，如 ``"iou_threshold"``；
      应用层负责拼接成 ``"tracking.iou_threshold"`` 这样的配置路径）。
    - ``value``：被拒绝的原始值。

    两者都会同步写入 ``context``（``{"field": ..., "value": ...}``），
    满足设计文档 §4.2“携带字段名与非法值”的契约。

    Args:
        message: 错误描述。
        field: 出错的字段名；``None`` 表示不指定。
        value: 被拒绝的原始值；省略时不写入 ``context``（与显式传入
            ``None`` 区分）。支持任意类型，不做序列化约束。
        code: 覆盖默认错误码。
        context: 附加结构化上下文；与 ``field`` / ``value`` 合并，
            已在 ``context`` 中的同名键不会被覆盖。
    """

    code: str = "MOT-DOMAIN-VALIDATION"


class SourceError(DomainError):
    """输入源（视频文件 / 摄像头）打开或读取失败。"""

    code: str = "MOT-SOURCE"


class SinkError(DomainError):
    """输出目标（轨迹 CSV / 录像文件）打开或写入失败。"""

    code: str = "MOT-SINK"


class TrackerSetupError(DomainError):
    """跟踪器后端不可用或构造失败（例如 OpenCV 构建缺少对应跟踪器）。"""

    code: str = "MOT-TRACKER"


__all__ = [
    "DomainError",
    "SinkError",
    "SourceError",
    "TrackerSetupError",
    "ValidationError",
]
