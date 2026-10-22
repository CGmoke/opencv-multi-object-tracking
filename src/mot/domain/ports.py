"""领域层端口契约（Protocol）。

核心（领域）定义端口，外部框架以适配器实现端口，接口层作为唯一的组合根装配
（见 ``docs/ARCHITECTURE_DESIGN.md`` §2.1 / §3.1）。本模块是领域层对外部世界的
唯一接口声明，**不 import 任何框架 / I/O 库**，只依赖标准库与领域内模型。

端口清单
--------
* ``Frame`` / ``TrackerFactory``：类型别名。
* ``SelfHostedTracker``：单个目标的跟踪器（OpenCV ``init`` / ``update`` 契约的抽象）。
* ``FrameSource``：帧输入源（视频文件 / 摄像头 / 未来的 ROS 图像源）。
* ``Detector``：可选检测器，``(frame, timestamp) -> Sequence[Detection]``。
* ``TrajectorySink``：轨迹落盘（CSV / 未来的 Parquet）。
* ``EventSink``：帧事件输出（日志 / ROS / 测试）。
* ``Clock``：时间来源（单调秒）。

约定
----
* 端口一律用 ``typing.Protocol`` 声明；适配器按结构（structural）满足即可，
  不要求显式继承。
* 帧对象对领域层**不透明**：``Frame`` 即 ``Any``，具体类型由采集适配器决定。
* ``EventSink`` 的载荷是应用层 ``FrameResult``，但领域层不感知该类型，故用
  ``Any`` 声明（域内不得引用应用层符号）。
"""

from __future__ import annotations

from typing import Any, Callable, Optional, Protocol, Sequence, Tuple, runtime_checkable

from .models import Detection, TrackedObject

#: 帧类型别名：领域层把它当作不透明对象处理。
Frame = Any


@runtime_checkable
class SelfHostedTracker(Protocol):
    """单个目标的跟踪器契约（每目标一个实例）。"""

    def init(self, frame: Frame, bbox: Sequence[float]) -> bool:
        """用 ``bbox`` 初始化该目标；返回是否初始化成功。"""
        ...

    def update(self, frame: Frame) -> Tuple[bool, Sequence[float]]:
        """推进一帧；返回 ``(success, bbox)``。失败时 ``bbox`` 可忽略。"""
        ...


#: 跟踪器工厂：零参调用返回一个全新的跟踪器。
TrackerFactory = Callable[[], SelfHostedTracker]


@runtime_checkable
class FrameSource(Protocol):
    """帧输入源契约（视频文件 / 摄像头 / ROS 图像源）。"""

    @property
    def fps(self) -> float:
        """源帧率（用于时间戳估算）。"""
        ...

    @property
    def name(self) -> str:
        """源的短名称（日志 / 文件名清洗用）。"""
        ...

    def read(self) -> Tuple[bool, Optional[Frame]]:
        """读一帧；返回 ``(ok, frame)``，``ok=False`` 表示流结束或不可读。"""
        ...

    def release(self) -> None:
        """释放底层资源。"""
        ...


@runtime_checkable
class Detector(Protocol):
    """可选检测器契约：一帧进，检测序列出。"""

    def __call__(self, frame: Frame, timestamp: float) -> Sequence[Detection]:
        """返回该帧的检测结果（可空序列）。"""
        ...


@runtime_checkable
class TrajectorySink(Protocol):
    """轨迹落盘契约（CSV / 未来的 Parquet 等）。"""

    def write_frame(
        self, frame_index: int, timestamp: float, tracks: Sequence[TrackedObject]
    ) -> int:
        """写入一帧的轨迹行；返回写入的行数。"""
        ...

    def close(self) -> None:
        """冲刷并关闭落盘资源。"""
        ...


@runtime_checkable
class EventSink(Protocol):
    """帧事件输出契约（日志 / ROS / 测试）。

    载荷为应用层 ``FrameResult``；领域层不感知其具体类型，故声明为 ``Any``。
    """

    def on_frame(self, result: Any) -> None:
        """消费一帧的结果事件。"""
        ...


@runtime_checkable
class FrameRecorder(Protocol):
    """逐帧录像端口：写入一帧（可带轨迹标注），会话结束时关闭。

    录像内容（是否绘制轨迹、写到哪里）由适配器决定；应用层只负责按帧喂入。
    """

    def write(self, frame: Frame, result: Any) -> None:
        """写入一帧；``result`` 是应用层 ``FrameResult``（领域层不感知其类型）。"""
        ...

    def close(self) -> None:
        """结束写入并冲刷文件（不删除已录内容）。"""
        ...


@runtime_checkable
class Clock(Protocol):
    """时间来源契约：返回单调递增的秒数。

    用于替代“按 FPS 估算时间戳”，修复摄像头漂移与 ``--loop`` 失准（G6）。
    """

    def now(self) -> float:
        """当前单调时间（秒）。"""
        ...


@runtime_checkable
class Renderer(Protocol):
    """渲染/交互端口：绘制并呈现一帧，返回交互事件名。

    适配器（如基于 OpenCV 高亮窗口的实现）同时负责“绘制 + 呈现 + 轮询输入”，
    使应用层主循环无需接触 GUI。``result`` 是应用层 ``FrameResult``；领域层不感知
    其类型，故声明为 ``Any``。
    """

    def present(self, frame: Frame, result: Any) -> Optional[str]:
        """绘制并呈现一帧，返回一个事件名（如 ``"quit"`` / ``"pause"``）或 ``None``。"""
        ...


__all__ = [
    "Clock",
    "Detector",
    "EventSink",
    "Frame",
    "FrameRecorder",
    "FrameSource",
    "Renderer",
    "SelfHostedTracker",
    "TrackerFactory",
    "TrajectorySink",
]
