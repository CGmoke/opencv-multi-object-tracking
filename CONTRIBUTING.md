# Contributing

感谢参与 `opencv-multi-object-tracking`！本文说明开发环境、代码约定与 PR 要求。

## 开发环境

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"        # pytest + ruff
# 可选：匈牙利匹配与 YOLO 适配器
pip install -e ".[dev,hungarian,detect]"
```

确认 OpenCV 跟踪后端可用：

```bash
python -m mot --list-trackers
```

## 常用命令

```bash
make test     # python -m pytest
make lint     # ruff check src tests
make fmt      # ruff format（可选）
make build    # python -m build + twine check
```

提交前至少跑通 `make lint && make test`。

## 代码风格

- 目标 Python 3.9+；运行时注解使用 `from __future__ import annotations`，不要在运行时代码里写 `X | None` 之外的新语法而不加 future import。
- ruff 规则为 `E, F, W, I, B`（行宽 100，`E501` 关闭）。请勿扩大规则集或全局忽略来“修”告警。
- 模块 docstring 用英文；用户可见文档用中文。
- 保持分层：领域层（`models` / `iou` / `matching` / `track_manager` / `pipeline` / `trajectory`）不得导入 `cv2`；OpenCV 只出现在 `trackers`、`video_source`、`visualization`、`app` 与 `detectors` 中。
- 新的输入路径都要校验（类型、范围、上界），错误信息要能指导用户修正。

## 测试要求

- 领域层改动用假跟踪器（`tests/conftest.py`）覆盖，不依赖 OpenCV。
- 涉及 cv2 的测试在模块顶层 `pytest.importorskip("cv2")`，并加 `@pytest.mark.opencv` 或 `@pytest.mark.slow` 标记。
- 修复 bug 时先补一个能复现的用例。
- 不修改既有测试的期望值来“通过”——如果语义确实要变，请同时更新 `docs/ARCHITECTURE.md` 并在 PR 里说明。

## 新增跟踪后端

1. 在 `src/mot/trackers.py` 的 `SUPPORTED_BACKENDS` 中添加名称，并补候选工厂链（`cv2.TrackerX_create` → `cv2.legacy.TrackerX_create` → `cv2.TrackerX.create`）。
2. 在 `BACKEND_NOTES` 写一句特点说明。
3. 补 `--list-trackers` 的可用性测试（存在性检查，不要求所有平台都可用）。
4. 更新 README 的“多种跟踪后端”表。

## 新增检测器

1. 实现 `Callable[[np.ndarray], Sequence[Detection]]`，异常请自行处理或允许管道兜底。
2. 放在 `src/mot/detectors/`，保持可选依赖懒加载（构造时再 `import`）。
3. 需要暴露到 CLI 时，同步修改 `config.py`、`app.py` 与 `configs/default.yaml`，并在 README 标注“实验性”。
4. 注意安全：不可信模型权重可能执行代码，文档必须说明来源要求。

## 提交与 PR

- 提交信息使用 Conventional Commits：`feat: ...`、`fix: ...`、`docs: ...`、`test: ...`、`refactor: ...`、`chore: ...`。
- 一个 PR 只做一件事；行为变更附测试；文档变更附截图或命令输出。
- PR 描述里写清：动机、改动范围、验证方式（命令 + 结果）、兼容性影响、是否更新文档。
- 涉及 ID 生命周期、CSV 契约、配置 schema 的改动，请先开 Issue 讨论。

## 报告问题

- 缺陷请用 Bug Report 模板，附最小复现、配置、环境（Python / OpenCV 版本、平台）。
- 安全问题请**不要**公开提 Issue，见 [SECURITY.md](SECURITY.md)。
