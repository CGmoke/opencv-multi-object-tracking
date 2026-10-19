# Changelog

格式参考 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/)，版本号遵循 [Semantic Versioning](https://semver.org/lang/zh-CN/)。

## [Unreleased]

### Planned

- 修正 `r` 重置后 CSV `frame` 列回绕（A1）
- `TrackManager._refresh` 异常保护（A5）
- YOLO 检测周期刷新接入 config / CLI（A2）

## [0.1.0] - 2026-10-19

首个公开版本：仓库框架 + 可运行参考实现。

### Added

- 纯 OpenCV 多目标跟踪：每目标一个 CSRT / KCF / MIL / MOSSE 跟踪器，IoU 关联维持唯一 ID
- ID 生命周期：丢失后保留 `max_missing` 帧、超时销毁、新目标自动分配；`max_tracks` 上限
- 视频 / 摄像头输入、循环播放、录像输出、`--no-display` 无界面运行
- 鼠标新增 / 删除目标、暂停、重置、清空轨迹、截图、帮助浮层、HUD
- 轨迹绘制（带衰减）与 CSV 导出：中心点、时间戳、标签、`--append`、原子写入
- 贪心 / 匈牙利两种关联算法（后者需 scipy，缺失时告警回退）
- 严格 YAML 配置校验（未知键、类型、范围、上界）与 `--dump-config`
- 可选 YOLO 检测器适配器（实验性，未接入 CLI）
- 279 个 pytest 用例；ruff 静态检查；GitHub Actions CI（Python 3.9/3.11/3.12）；打包校验
- 文档：README、架构说明、Roadmap、贡献指南、安全策略、行为准则、Issue/PR 模板

### Security

- 拒绝通过符号链接写 CSV / 录像 / 截图（CWE-59）
- CSV 公式注入转义，覆盖 BOM 与零宽字符绕过
- 配置文件大小与字段上界校验，拒绝 FIFO / 设备文件
- 检测器异常容错；无 scipy 回退时告警；空输入返回非零退出码

[Unreleased]: https://github.com/CGmoke/opencv-multi-object-tracking/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/CGmoke/opencv-multi-object-tracking/releases/tag/v0.1.0
