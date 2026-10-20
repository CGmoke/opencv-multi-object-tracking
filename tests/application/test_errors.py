"""应用层错误基类与退出码映射（步骤 5 验收测试）。

端口边界错误（``SourceError`` / ``SinkError`` / ``TrackerSetupError``）定义在领域层
（``mot.domain.errors``），因为适配器抛出它们且不得依赖应用层。
"""

from __future__ import annotations

import pytest

from mot.application.errors import ConfigError, MotError
from mot.domain.errors import DomainError, SinkError, SourceError, TrackerSetupError


def test_base_error_exit_code_is_two():
    assert MotError.exit_code == 2


def test_config_error_is_a_mot_error():
    assert issubclass(ConfigError, MotError)
    assert ConfigError.exit_code == 2


def test_message_is_preserved():
    with pytest.raises(ConfigError) as info:
        raise ConfigError("tracking.iou_threshold: bad")
    assert "tracking.iou_threshold" in str(info.value)


@pytest.mark.parametrize("cls", [SourceError, SinkError, TrackerSetupError])
def test_port_errors_live_in_the_domain_layer(cls):
    assert issubclass(cls, DomainError)
    assert issubclass(cls, Exception)
    assert not issubclass(cls, MotError)  # 适配器不得依赖应用层
