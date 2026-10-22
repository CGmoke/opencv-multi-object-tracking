# OpenCV Multi-Object Tracking

[![CI](https://github.com/CGmoke/opencv-multi-object-tracking/actions/workflows/ci.yml/badge.svg)](https://github.com/CGmoke/opencv-multi-object-tracking/actions/workflows/ci.yml)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)](pyproject.toml)
[![OpenCV](https://img.shields.io/badge/opencv-4.5.1%2B-green.svg)](pyproject.toml)

一个**纯 OpenCV** 的多目标跟踪框架：为每个目标各持有一个 CSRT / KCF 跟踪器，用 IoU 做帧间关联来维持唯一 ID，目标丢失超过容忍帧数即销毁 ID，新目标自动分配新 ID，并绘制包围框、ID 与运动轨迹，轨迹点可导出 CSV。目标来源支持三种方式：手工种子框 / 运行时框选、内置 MOG2/KNN 运动检测（无需模型权重）、可选 YOLO 检测器。

仓库定位是**教学与二次开发骨架**：依赖少、分层清楚（domain → application → adapters → interfaces 端口-适配器架构），核心逻辑可脱离 OpenCV 单测。它不是生产级 MOT 系统（没有 ReID 与卡尔曼滤波）。

> 状态：`v0.2.0`（alpha）。仓库框架、参考实现与测试已就位；真实摄像头、GUI 交互与扩展方向见 [已知限制](#已知限制)。

---

## 目录

- [特性](#特性)
- [目录结构](#目录结构)
- [快速开始](#快速开始)
- [命令行](#命令行)
- [Web 界面](#web-界面)
- [自动运动检测](#自动运动检测)
- [交互按键](#交互按键)
- [配置文件](#配置文件)
- [轨迹 CSV](#轨迹-csv)
- [架构总览](#架构总览)
- [测试与 CI](#测试与-ci)
- [已验证环境](#已验证环境)
- [扩展方向](#扩展方向)
- [已知限制](#已知限制)
- [安全与隐私](#安全与隐私)
- [参与贡献](#参与贡献)
- [许可证](#许可证)

## 特性

| 能力 | 说明 |
| --- | --- |
| 视频 / 摄像头输入 | `--source 0` 读摄像头，`--source path/to/video.mp4` 读文件；`--loop` 循环播放 |
| 跟踪器初始化 | 启动时用 `--seed-box x,y,w,h` 初始化目标，或运行时用鼠标框选新增 |
| 帧间跟踪 + IoU 关联 | 每个目标一个跟踪器；检测框与已有目标按 IoU 关联，支持 `greedy` / `hungarian` 两种匹配 |
| ID 生命周期 | 目标丢失后保留 ID `max_missing` 帧等待恢复；超时销毁；无主检测自动分配新 ID |
| 轨迹绘制与导出 | 绘制包围框、ID、带衰减的历史轨迹；轨迹点写入 CSV（含中心点、时间戳、标签） |
| 多种跟踪后端 | `csrt`（默认，精度优先）、`kcf`（速度优先）、`mil`、`mosse` |
| 轨迹生命周期过滤 | `min_hits` 确认制挡掉 1~2 帧误检；`max_coast` 限制无检测支持的滑行时长，防漂移僵尸轨迹 |
| 轨迹去重 | 长期描述同一物理目标的多条轨迹按 IoU + 连续帧数条件自动合并，仅保留更早、更可信的轨迹 |
| 并行跟踪 | `--max-parallel` 用线程池并行执行跟踪器 update 纯计算阶段（默认串行，保证结果可复现） |
| 运动检测守护 | 光照突变整帧丢弃 + 背景快速重学、复合置信度评分、碎片合并面积护栏，防 ID 爆发与巨型框 |
| YOLO 检测（可选） | `--detector yolo` 接入 ultralytics 权重，类别过滤 / 置信度 / 检测间隔可配 |
| Web 帧率监控 | 捕获 / 处理 / 推流三路滑动窗口速率统计，处理积压时自动跳帧避免阻塞 |
| 无显示运行 | `--no-display` 用于服务器 / CI / 批量处理 |
| 录像 | `--record out.mp4` 保存带标注的视频 |
| 安全默认值 | 拒绝符号链接写入、CSV 公式注入转义、配置项严格类型与上界校验、`--append` 显式追加 |

## 目录结构

代码按严格的四层端口-适配器架构组织，依赖方向单向（外层依赖内层，反向禁止）：

```text
.
├── configs/
│   ├── default.yaml            # 完整配置示例（含注释）
│   ├── indoor-640.yaml         # 640x480 固定机位室内场景预设
│   └── upload-720.yaml         # 720x1280 竖屏手持上传视频场景预设
├── src/mot/
│   ├── __init__.py             # 公开 API（惰性加载，无 OpenCV 也可导入核心）
│   ├── __main__.py             # python3 -m mot 入口
│   ├── domain/                 # 纯领域层：零 OpenCV / I/O 依赖
│   │   ├── models.py           #   BBox、Detection、TrackedObject、IdAllocator
│   │   ├── lifecycle.py        #   ID 生命周期状态机（tentative/active/lost）
│   │   ├── association.py      #   IoU 重叠矩阵、贪心与匈牙利关联
│   │   ├── postprocess.py      #   检测后处理：碎片合并、NMS、置信度复合评分
│   │   ├── sizing.py           #   像素 → 实际尺寸换算（纯函数）
│   │   ├── limits.py           #   全部数值阈值与边界校验（单一真源）
│   │   ├── errors.py           #   ValidationError 等领域错误
│   │   └── ports.py            #   Tracker/Detector 等端口契约
│   ├── application/            # 用例编排层：不依赖 cv2 / argparse
│   │   ├── pipeline.py         #   检测 → 后处理 → 去重 → 跟踪 的单帧管道
│   │   ├── session.py          #   会话主循环（种子框、手工框、事件分发）
│   │   └── config.py           #   YAML 配置加载与严格校验
│   ├── adapters/               # 适配层：唯一允许导入 OpenCV 的层级
│   │   ├── trackers.py         #   CSRT/KCF/MIL/MOSSE 兼容封装（新旧 API 回退）
│   │   ├── motion.py           #   MOG2/KNN 背景建模 + 预处理 + 守护逻辑
│   │   ├── detectors/          #   检测器适配（base.py 假检测器、yolo.py 可选 YOLO）
│   │   ├── sources.py          #   摄像头 / 文件 / RTSP 输入
│   │   ├── queue_source.py     #   Web 上传与浏览器摄像头的队列帧源
│   │   ├── sinks/              #   轨迹 CSV / 事件 JSONL / 录像输出
│   │   └── rendering.py        #   包围框、ID、轨迹绘制
│   ├── interfaces/             # 入口层：组合根与用户界面
│   │   ├── cli.py              #   CLI 参数解析与适配器装配（唯一组合根）
│   │   ├── ui_window.py        #   OpenCV 窗口与鼠标 / 键盘交互
│   │   └── web.py              #   Flask Web 监测界面（会话注册表、MJPEG 推流）
│   └── utils/                  # FPS 计（含 FrameBudget）、日志、路径工具
├── tests/                      # pytest 用例（核心逻辑无需 OpenCV）
├── docs/                       # 架构设计 / 优化计划 / 路线图
├── .github/                    # CI、Issue/PR 模板
├── pyproject.toml
├── requirements.txt
├── requirements-dev.txt
├── Makefile
└── CONTRIBUTING.md · SECURITY.md · CODE_OF_CONDUCT.md · CHANGELOG.md · LICENSE
```

## 快速开始

要求 Python 3.9+，以及带 tracking 模块的 OpenCV（`opencv-contrib-python`）。

```bash
git clone https://github.com/CGmoke/opencv-multi-object-tracking.git
cd opencv-multi-object-tracking

python3 -m venv <venv-dir>
source <venv-dir>/bin/activate   # Windows: <venv-dir>\Scripts\activate
pip install -e ".[dev]"          # 或：pip install -r requirements-dev.txt
```

确认当前环境有哪些跟踪后端：

```bash
mot-track --list-trackers        # 等价于 python -m mot --list-trackers
```

用摄像头跑（需要图形界面）：

```bash
mot-track --source 0
```

用视频文件跑，并在第一帧用给定框初始化一个目标：

```bash
mot-track --source demo.mp4 --seed-box 120,80,60,140 --csv trails.csv
```

无界面 / 服务器上运行，只看统计结果：

```bash
mot-track --source demo.mp4 --seed-box 120,80,60,140 \
          --no-display --csv trails.csv --max-frames 200
```

查看合并 CLI 覆盖后的最终配置：

```bash
mot-track --dump-config --iou-threshold 0.4 --tracker kcf
```

## 命令行

| 参数 | 默认 | 说明 |
| --- | --- | --- |
| `--source, -s` | `0` | 摄像头索引或视频路径 |
| `--config, -c` | 无 | YAML 配置文件，见 [`configs/default.yaml`](configs/default.yaml) |
| `--tracker, -t` | `csrt` | `csrt` / `kcf` / `mil` / `mosse` |
| `--iou-threshold` | `0.3` | 保持 ID 的最小 IoU，取值 `(0, 1]` |
| `--max-missing` | `15` | 丢失目标最多保留 ID 的帧数 |
| `--trail-length` | `120` | 每个目标保留的轨迹点上限 |
| `--max-tracks` | `256` | 同时存活 ID 的硬上限 |
| `--box-smoothing` | `1.0` | 检测框 EMA 平滑系数（`1`=不平滑，越小越稳但略滞后） |
| `--min-hits` | `3` | 轨迹确认所需命中数；未确认轨迹不输出且一次丢失即销毁 |
| `--smooth-window` | `5` | 包围盒/轨迹滑动平均窗口（`<=1` 关闭，优先于 `--box-smoothing`） |
| `--max-box-change` | `1.6` | 轨迹二次校验：与已有轨迹重叠但尺寸突变超此倍率的检测被丢弃（`1` 关闭） |
| `--max-coast` | `8` | 无检测支持（滑行）连续超过该帧数即销毁 ID，防跟踪器自滑行产生漂移轨迹 |
| `--dedupe-tracks` / `--no-dedupe-tracks` | 开 | 轨迹级去重：长期描述同一物体（连续多帧同时满足 IoU、面积比、中心距离条件）的轨迹合并，保留更早的一条 |
| `--merge-track-iou` | `0.5` | 两条轨迹判定为同一物体的 IoU 下限（配合 `--merge-track-frames`） |
| `--merge-track-frames` | `5` | 连续满足合并条件的帧数下限 |
| `--max-parallel` | `0` | 线程池并行执行的跟踪器 update 数量（`0` 串行；仅并行纯计算阶段，关联与状态机仍在主线程，保证可复现） |
| `--match-method` | `greedy` | `greedy` 或 `hungarian`（后者需要 scipy） |
| `--detector` | `none` | 检测后端：`none`（种子框 / 框选 / 运动检测）或 `yolo`（需 `pip install ultralytics`；与 `--motion` 同开时检测器优先） |
| `--detector-model` | 无 | YOLO 本地权重路径（不自动下载，见 [SECURITY.md](SECURITY.md)） |
| `--detector-interval` | `1` | 每 N 帧运行一次检测器（首帧总是检测，间隙由跟踪器推进） |
| `--detector-conf` | `0.35` | 检测置信度阈值 |
| `--csv` | `outputs/trajectories.csv` | 轨迹输出路径，传 `none` 关闭 |
| `--append` | 关 | 追加写而不是截断已有 CSV |
| `--record` | 无 | 保存带标注视频的路径 |
| `--seed-box x,y,w,h` | 无 | 启动时初始化目标，可重复 |
| `--max-frames` | 无 | 处理 N 帧后停止 |
| `--loop` | 关 | 视频结束后循环（配合 `--max-frames` 使用） |
| `--no-display` | 关 | 不创建窗口 |
| `--list-trackers` | — | 打印可用后端并退出 |
| `--dump-config` | — | 打印最终配置并退出 |
| `--verbose, -v` | 关 | 调试日志 |
| `--motion` | 关 | 自动检测移动物体（背景建模，无需模型权重） |
| `--motion-method` | `mog2` | 背景建模后端：`mog2`（快）或 `knn`（纹理场景更稳） |
| `--motion-sensitivity` | `25` | MOG2 方差阈值，**越小越敏感** |
| `--motion-knn-dist2` | `400` | KNN 平方距离阈值（仅 `--motion-method knn`） |
| `--motion-min-area` | `200` | 最小前景面积（像素，噪点过滤） |
| `--motion-min-size` | `8` | 最小目标边长（像素，尺寸阈值） |
| `--motion-min-motion` | `6` | 最小位移（像素，排除静止物体） |
| `--motion-static-iou` | `0.95` | 相邻检测框 IoU 达到该值即视为“未移动” |
| `--motion-static-frames` | `30` | 连续静止多少次后剔除该目标（`0` 关闭） |
| `--motion-min-consecutive` | `2` | 连续命中帧数（抑制单帧噪点） |
| `--motion-blur` | `5` | 灰度图高斯模糊核（奇数，`0` 关闭；压制传感器颗粒） |
| `--motion-median` | `0` | 中值滤波核（奇数，`0` 关闭；椒盐噪声明显时设 `3`） |
| `--motion-clahe` | 关 | 开启 CLAHE 局部光照均衡（放大噪声，建议搭配 `--motion-median 3`） |
| `--motion-clahe-clip` | `2.0` | CLAHE 限幅，越大对比越强、噪声也越多 |
| `--motion-brightness-step` | `8` | 帧亮度突变门控阈值（灰度级，`0` 关闭） |
| `--motion-temporal` | `2` | 最近 N 帧掩码按位与（`1` 关闭；快目标建议 `1~2`） |
| `--motion-aspect` | `0.1,10` | 包围框长宽比允许范围（剔除极细长纹理） |
| `--motion-min-fill` | `0` | 轮廓填充率下限（剔除中空/破碎碎片，`0` 关闭） |
| `--motion-merge-iou` | `0.15` | 碎片框合并的 IoU 阈值（同一物体拆成多块时合并） |
| `--motion-merge-gap` | `0` | 碎片合并的最大正间隙（像素，`0` 关闭间隙合并） |
| `--motion-nms-iou` | `0.6` | NMS IoU 阈值（重叠框只留整体框） |
| `--motion-nms-containment` | `0.8` | 局部框被大框覆盖比例达到该值时抑制 |
| `--motion-interval` | `2` | 每 N 帧做一次运动检测（首帧总是检测） |
| `--motion-ppm` | — | 尺寸标定：每米像素数（用于换算实际尺寸） |
| `--web` | 关 | 启动本地 Web 监测界面（需 `pip install -e ".[web]"`） |
| `--web-host` | `127.0.0.1` | Web 界面监听地址（局域网用 `0.0.0.0`） |
| `--web-port` | `8000` | Web 界面端口 |

退出码：`0` 正常，`1` 未读取到任何帧，`2` 参数 / 配置 / 跟踪器 / 输入输出错误，`130` 被 `Ctrl-C` 中断。

轨迹 CSV、录像与截图等运行时产物默认写入仓库内的产物目录（已被 `.gitignore` 忽略，不会上传），上表中出现的默认路径只描述程序行为，不是仓库内容。

## Web 界面

在浏览器里实时查看跟踪画面与统计，输入源覆盖**本机摄像头 / RTSP-IP 网络流 / 上传视频 / 浏览器摄像头**四类。

```bash
# Debian/Ubuntu 的系统 Python 受 PEP 668 保护，需先建虚拟环境
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[web]"          # 安装包与可选依赖（Flask）
mot-web                          # 或：python3 -m mot --web
# 浏览器打开 http://127.0.0.1:8000
```

| 输入源 | 用法 |
| --- | --- |
| 本机摄像头 | 类型选「本机摄像头」，填索引（如 `0`） |
| RTSP / IP 网络流 | 类型选「网络流」，填 `rtsp://user:pass@host:554/stream`（断流自动重连） |
| 上传视频 | 点「上传视频」；或类型选「本地视频路径」填绝对路径 |
| 浏览器摄像头 | 点「使用浏览器摄像头」（需授权，约 8 fps 回传至服务端） |

每张卡片在**停止后**会出现「重启」与「删除」按钮：重启沿用原输入源（同一 ID，产物另写
新文件）；删除只移除页面条目，已生成的 CSV/JSONL 保留。开启 `--motion` 启动 `mot-web`
后，会话无需种子框即可自动发现移动目标。

**放大查看**：单击卡片画面即全屏放大（单击、`Esc` 或右上角 `×` 退出，回到原卡片布局）；
按住拖拽仍是框选目标，两者按是否发生拖动区分。

**帧率监控**：每路会话用滑动窗口分别统计**捕获 / 处理 / 推流**三路速率（`capture_fps` /
`process_fps` / `stream_fps`）与跳帧数（`stream_skipped`、`dropped`），随会话状态接口一并
返回。当处理速度跟不上输入时会自动跳帧，只处理最新帧，避免积压阻塞 pipeline；若发现
`process_fps` 明显低于 `capture_fps`，优先换 `--tracker kcf`、增大 `--motion-interval` 或
降低 `stream_fps`。

> 没有检测器时，请用**种子框**或在实时画面上**拖拽框选**目标（下一帧生效）——否则画面里不会产生任何目标。

HTTP API（同源，便于脚本与二次开发）：

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| GET | `/` | 监测页面 |
| GET | `/api/sources` | 支持的输入源类型 / 跟踪器后端 / 会话上限 |
| POST | `/api/sessions` | 启动会话：`{"kind":"usb\|url\|file\|browser","spec":...,"seed_boxes":[[x,y,w,h]]}` |
| GET | `/api/sessions` | 列出全部会话与实时统计 |
| GET | `/api/sessions/<id>` | 单会话状态 + 统计 |
| POST | `/api/sessions/<id>/stop` | 停止会话 |
| POST | `/api/sessions/<id>/restart` | 重启**已停止**的会话（沿用原输入源与种子框；运行中返回 409） |
| DELETE | `/api/sessions/<id>` | 删除**已停止**的会话（运行中返回 409） |
| POST | `/api/sessions/<id>/box` | 入队手工目标框：`{"bbox":[x,y,w,h]}` |
| POST | `/api/sessions/<id>/frame` | 浏览器摄像头推 JPEG 帧（body 为 JPEG） |
| GET | `/api/sessions/<id>/events` | 最近事件（`?since=N`） |
| GET | `/stream/<id>.mjpeg` | MJPEG 实时画面 |
| POST | `/api/upload` | 上传视频（multipart `file`），成功后自动建会话 |

产物：每路会话写 `outputs/web_<id>.csv`（轨迹）与 `outputs/web_<id>.jsonl`（事件），并同时打控制台日志。

配置见 [`configs/default.yaml`](configs/default.yaml) 的 `web:` 段（`host` / `port` / `max_sessions` / `stream_fps` / `upload_dir` / `max_upload_bytes`），可用 `--web-host` / `--web-port` 覆盖。

> 安全：仅供本机 / 局域网使用，**无多用户与鉴权**，默认只监听 `127.0.0.1`；需要局域网访问时用 `--web-host 0.0.0.0`，并自行做好网络隔离。上传仅接受视频扩展名，且限制单文件大小。

## 自动运动检测

不提供模型权重也能自动发现移动物体：基于 MOG2 背景建模 + 形态学 + 多重过滤，输出
包围框交给现有跟踪器维护 ID 与轨迹。

```bash
# 摄像头：自动检测移动物体
python3 -m mot --source 0 --motion

# 视频文件 + 尺寸标定（每米 120 像素 → 框上显示实际尺寸与大小分级）
python3 -m mot --source clip.mp4 --motion --motion-ppm 120

# 调灵敏度：越小越敏感；min-area / min-size 控制最小目标
python3 -m mot --source 0 --motion --motion-sensitivity 10 --motion-min-area 400 --motion-min-size 12
```

**过滤逻辑**（排除静态物体与噪点）

| 过滤 | 参数 | 作用 |
| --- | --- | --- |
| 输入预处理 | `blur_kernel` / `median_kernel` | 灰度化后高斯（默认 5×5）/ 中值滤波，压掉传感器颗粒与压缩噪点 |
| 光照均衡 | `clahe` / `clahe_clip` | CLAHE 提升局部对比，低对比暗区目标更易检出（会放大噪声） |
| 亮度突变门控 | `brightness_step` / `brightness_rate` | 帧亮度中位数跳变即判定曝光/开关灯/云影，丢弃该帧并重新适应背景 |
| 背景建模 | `method` / `var_threshold` | MOG2（或 KNN，均 `detectShadows=False`）把静止内容学成背景；阈值默认 25 比 OpenCV 默认 16 更抗纹理抖动 |
| 时序一致性 | `temporal_frames` | 最近 N 帧掩码按位与，单帧闪烁不进轮廓阶段 |
| 形态学 | `morph_kernel` | 开运算去孤立噪点 + 闭运算补孔洞/连裂口 |
| 最小面积 | `min_area_px` | 去掉小噪点 |
| 最小边长 | `min_width_px` / `min_height_px` | 去掉细长纹理 |
| 长宽比 | `min_aspect` / `max_aspect` | 去掉灯管、百叶窗缝等极细长条纹 |
| 最大占比 | `max_area_ratio` | 去掉整屏光照突变 |
| 填充率 | `min_fill_ratio` | 去掉中空、破碎的局部特征（`0` 关闭） |
| 碎片合并 | `merge_iou` / `merge_gap_px` | 同一物体的多个局部框合并为覆盖整体的外接框 |
| NMS | `nms_iou` / `nms_containment` | 抑制重叠框与被整体框包含的局部框 |
| 连续命中 | `min_consecutive` | 单帧闪烁不入管道 |
| 位移门限 | `min_motion_px` | 中心位移不足视为静止，不输出 |
| 静止剔除 | `static_iou` / `static_frames` | IoU 时间连续性：移动后停下的目标连续 `static_frames` 次不移动后剔除 |

**颗粒度与稳定性**：默认只合并 IoU ≥ 0.15 的重叠碎片（`merge_gap_px=0` 关闭间隙合并），
因此一个物体只会得到**一个完整框**；有间隙的相邻物体保持各自独立的框与 ID。若目标被
1~2px 的噪声线切成两块，可用 `--motion-merge-gap 2` 修复。检测框抖动明显时可加
`--box-smoothing 0.5`（EMA，`1` 表示不平滑）。

**轨迹二次校验**：已建立 ID 的目标若收到与自身框高度重叠、但宽或高突变超过
`--max-box-change`（默认 `1.6` 倍）的检测，该检测会被丢弃，目标继续由自身跟踪器推进，
避免局部框/碎片框顶替整体框（`1` 关闭校验）。

**静止剔除**：目标移动后停下时，背景冻结会让它一直留在前景里，`min_motion_px` 无法
识别。此时用 IoU 时间连续性判定——相邻检测框 IoU ≥ `static_iou`（默认 `0.95`）连续
`static_frames`（默认 `30`）次即判定静止：检测端停止输出该候选，跟踪端同时销毁该 ID；
目标再次移动后重新获得新 ID。`--motion-static-frames 0` 可关闭。

**轨迹生命周期过滤与平滑**：轨迹需累计 `min_hits`（默认 `3`）次命中才被“确认”并进入
渲染 / CSV / JSONL；未确认的暂定轨迹（通常 1~2 帧的误检）一次丢失就立即销毁，不产生
移除事件。所有包围盒（跟踪器输出与检测重锚）都会经过 `smooth_window`（默认 `5`）滑动
平均，轨迹点取平滑后框的中心，因此抖动量明显下降。
`--smooth-window 1` 关闭滑动平均，此时若 `--box-smoothing < 1` 则退回 EMA 平滑。

**稳健性守护**（配置项，位于 `motion:` 段）：针对自动检测常见的三类失控设有护栏——

| 守护 | 配置项 | 机制 |
| --- | --- | --- |
| 光照突变 ID 爆发 | `max_fg_ratio` / `burst_recover_frames` | 前景占比超过阈值（默认 `0.25`）时判定为全局突变，丢弃本帧全部候选，并以 `warmup_rate` 快速重学背景 `burst_recover_frames` 帧 |
| 置信度饱和 | `min_quality` | 候选置信度采用面积、填充率、长宽比等复合评分（不随面积单调递增），低于 `min_quality`（默认 `0.45`）的候选不进入管道 |
| 巨型合并框 | `merge_growth` | 每次碎片合并前校验并集面积不超过较大输入框的 `merge_growth` 倍（默认 `3.0`），防止链式合并出覆盖半屏的巨型框 |

此外 `max_drift_px`（默认 `0` 关闭）可限制无检测支持轨迹的累计漂移距离；`learning_rate`
设为小正值（如 `0.01`）可让背景缓慢吸收“移动过的静态区域”，手持拍摄场景能显著减少
ghost 残留（冻结背景 `0.0` 则保持框完整覆盖物体）。典型场景可直接用预设：
[`configs/indoor-640.yaml`](configs/indoor-640.yaml)（固定机位）与
[`configs/upload-720.yaml`](configs/upload-720.yaml)（竖屏手持上传视频）。

**实际尺寸**：给出标定 `pixels_per_meter`（现场用已知尺寸参考物换算）后，每个框的标签
会显示如 `0.42x0.55m medium`（`small` ≤ 0.5 m，`large` ≥ 2.0 m，阈值可配）；未标定时
显示像素尺寸 `50x60px`。尺寸换算为纯函数 `mot.domain.sizing`，可独立复用。

> 背景建模只能看到“与背景不同”的区域：预热期内已经在移动的物体可能被学成背景。
> 实际部署时让 `warmup_frames`（默认 20 次检测调用）覆盖场景相对静止的起始阶段，或先
> 启动服务再让目标进入画面。

**帧率实测**（640×480、两目标、含绘制与 JPEG 编码）：

| 后端 | 每帧检测 | 每 2 帧检测 |
| --- | --- | --- |
| KCF | 41.6 fps | 52.8 fps |
| CSRT | 31.4 fps | 32.5 fps |

均高于 25 fps；纯运动检测约 320 fps，瓶颈在跟踪器。低配机器优先 `--tracker kcf`，
并适当增大 `--motion-interval`。

## 交互按键

| 按键 | 作用 |
| --- | --- |
| `q` / `Esc` | 退出 |
| `空格` | 暂停 / 继续 |
| `n` | 进入新增模式，鼠标拖拽画框，松开后创建目标 |
| `x` | 进入删除模式，点击目标将其删除（命中面积最小的目标） |
| `c` | 清空所有轨迹线（保留目标） |
| `r` | 重置会话（销毁所有目标，ID 计数继续） |
| `s` | 保存当前帧截图（写入默认输出目录，同名自动加序号） |
| `h` | 显示 / 隐藏帮助 |

新增与删除在暂停状态下同样可用，方便逐帧标注。

## 配置文件

完整示例见 [`configs/default.yaml`](configs/default.yaml)，另有
[`indoor-640.yaml`](configs/indoor-640.yaml)（固定机位室内）与
[`upload-720.yaml`](configs/upload-720.yaml)（竖屏手持上传视频）两个场景预设：

```yaml
source:
  source: "0"          # 摄像头索引或视频路径
  max_frames: null
  loop: false
  seed_boxes: []       # [[x, y, w, h], ...]

tracking:
  backend: csrt
  iou_threshold: 0.3
  max_missing: 15
  trail_length: 120
  match_method: greedy
  max_tracks: 256
  min_hits: 3          # 轨迹确认所需命中数
  smooth_window: 5     # 包围盒/轨迹滑动平均（<=1 关闭）
  max_box_change: 1.6  # 尺寸突变二次校验（1 关闭）
  max_coast: 8         # 无检测支持滑行上限
  dedupe_tracks: true  # 轨迹级去重开关
  merge_track_iou: 0.5      # 去重：同物体 IoU 下限
  merge_track_frames: 5     # 去重：需连续满足的帧数
  max_parallel: 0      # 并行跟踪器数（0 串行）

detector:
  backend: none        # none 或 yolo（需 pip install ultralytics）
  model: null          # 本地权重路径，不自动下载
  confidence: 0.35
  classes: []          # 空 = 全部类别
  interval: 1          # 每 N 帧检测一次

motion:
  enabled: false       # --motion 等价
  method: mog2         # mog2 / knn
  var_threshold: 25.0
  learning_rate: 0.0   # 0 冻结背景；手持场景建议 0.01
  merge_iou: 0.15      # 碎片合并
  merge_growth: 3.0    # 合并面积护栏（并集 <= 较大框 x 此倍数）
  max_fg_ratio: 0.25   # 光照突变守护：前景占比上限
  min_quality: 0.45    # 复合置信度下限

display:
  enabled: true
  show_trails: true
  show_labels: true
  show_hud: true
  trail_fade: true
  window_name: "OpenCV MOT"

output:
  csv_path: outputs/trajectories.csv
  video_path: null
  append: false

web:
  host: "127.0.0.1"
  port: 8000
  max_sessions: 4
  stream_fps: 15.0
  upload_dir: null
  max_upload_bytes: 2000000000
```

校验是严格的：未知字段、类型不符、越界（如 `iou_threshold > 1`、`max_missing < 0`）都会以退出码 `2` 报错，而不是静默截断。CLI 参数优先于配置文件。

## 轨迹 CSV

每帧为每个处于 `ACTIVE` 状态的目标写一行：

```csv
track_id,frame,timestamp,x,y,w,h,cx,cy,label
1,1,0.033333,120.000,80.000,60.000,140.000,150.000,150.000,object
1,2,0.066667,121.500,80.000,60.000,140.000,151.500,150.000,object
```

| 列 | 含义 |
| --- | --- |
| `track_id` | 会话内唯一的自增 ID |
| `frame` | 会话内帧号（`r` 重置后从 1 重新计数，ID 继续） |
| `timestamp` | 秒，按视频 FPS 估算（假定恒定帧率） |
| `x, y, w, h` | 包围框，左上角 + 宽高 |
| `cx, cy` | 包围框中心 |
| `label` | 标签（手工框为 `object`，检测器输出沿用其类别名） |

默认覆盖写；需要追加时显式加 `--append`。以 `= + - @` 开头的标签会被加上 `'` 前缀，防止表格软件把轨迹文件当公式执行。

## 架构总览

```text
            ┌────────────┐   frame    ┌──────────────┐  detections ┌──────────────────┐
 video/ ───►│  adapters  ├───────────►│ application  ├────────────►│  domain          │
 camera     │  sources   │            │  MOTPipeline │  (后处理)    │  关联 + ID 生命周期│
 web        └────────────┘            │  (可选检测器) │             │  轨迹去重 + trail │
                                      └──────┬───────┘             └───────┬──────────┘
                                             │  FrameResult                │ Tracker 端口
                                             ▼                             ▼
                                    ┌────────────────┐            ┌──────────────────┐
                                    │  session.py    │◄───────────│ adapters         │
                                    │  主循环 / 事件  │   端口注入  │ CSRT/KCF、MOG2、 │
                                    │  CSV / 录像    │            │ YOLO、渲染、录像  │
                                    └────────────────┘            └──────────────────┘
```

关键设计：

- **四层端口-适配器架构**：`domain → application → adapters → interfaces` 依赖方向单向；领域层零 OpenCV / I/O 依赖，跟踪器与检测器通过端口（`domain/ports.py`）注入，由 adapters 层提供实现，因此可用假跟踪器完整单测 ID 生命周期。架构守卫测试强制执行依赖方向约束。
- **单帧管道是纯函数式的**：`MOTPipeline.step(frame, timestamp)` 返回 `FrameResult`（目标、新建、销毁、匹配、检测），便于替换 UI。
- **ID 语义**：ID 单调自增、会话内不复用；目标丢失后保留 `max_missing` 帧供 IoU 恢复，连续无检测支持超过 `max_coast` 帧即销毁；同一位置的新目标拿到新 ID。
- **同帧去重 + 轨迹去重**：手工框与检测框同帧按 IoU 合并，避免“一物两 ID”；跨帧长期重叠的重复轨迹由轨迹级去重合并。
- **可复现优先**：`--max-parallel` 只并行跟踪器 update 的纯计算阶段，关联与状态机始终在主线程按固定顺序执行；默认串行。

细节（状态机、匹配算法、设计取舍、性能热点）见源码与 `tests/` 下的对应用例。

## 测试与 CI

```bash
make test      # python -m pytest
make lint      # ruff check src tests
```

- 核心逻辑（IoU、匹配、ID 生命周期、管道、后处理、CSV、配置）用假跟踪器测试，不需要 OpenCV。
- 依赖 cv2 的用例在模块级 `pytest.importorskip("cv2")`，无 OpenCV 环境自动跳过。
- 架构守卫测试（`tests/test_architecture.py`）强制执行四层依赖方向约束。
- 当前：**609 个用例**，覆盖 CLI 端到端（合成视频 → 跟踪 → CSV / 录像 / 错误路径）与 Web 会话。

CI（[`.github/workflows/ci.yml`](.github/workflows/ci.yml)）在 Ubuntu 上跑 Python 3.9 / 3.11 / 3.12 矩阵：ruff 静态检查、pytest、`python -m build` + `twine check` 打包校验。

## 已验证环境

| 组件 | 已验证 | 声明支持 |
| --- | --- | --- |
| Python | 3.12.3 | >= 3.9 |
| OpenCV | 4.6.0（contrib，含 tracking） | >= 4.5.1 |
| NumPy | 1.26.4 | >= 1.21 |
| 平台 | Linux（无显示器，`--no-display`） | 跨平台，macOS/Windows 未实测 |

跟踪器构造按 `cv2.legacy.TrackerX_create` → `cv2.TrackerX_create` → `cv2.TrackerX.create()` 的顺序探测（**legacy 优先**）：OpenCV 5.0.0 的新实现 `cv2.TrackerX_create` 在“运动检测 + 大量实例创建/销毁”负载下存在堆损坏缺陷，`cv2.legacy` 版本稳定；旧构建没有 `cv2.legacy` 时自动回退到新实现。GUI、真实摄像头与高版本 OpenCV 尚未在 CI 中验证。

## 扩展方向

- **结合 YOLO 做检测 + 跟踪**：`--detector yolo --detector-model yolo11n.pt --detector-conf 0.35 --detector-interval 3` 即可让检测驱动跟踪（需 `pip install ultralytics`；类别过滤暂经配置文件的 `detector.classes` 列表设置）。权重不随仓库分发、不自动下载，加载不可信 `.pt` 权重存在代码执行风险，见 [SECURITY.md](SECURITY.md)。
- **换检测器**：实现检测器端口 `(frame, timestamp) -> Sequence[Detection]`（见 `mot.domain.ports.Detector` 与 `adapters/detectors/base.py` 的假检测器）即可注入；`adapters/detectors/yolo.py` 是参考实现。
- **换匹配算法**：`--match-method hungarian` 改用全局最优匹配（需要 `pip install scipy`；缺失时自动回退 greedy 并告警）。
- **替换跟踪器**：实现 `init(frame, bbox) -> bool` / `update(frame) -> (bool, bbox)` 的对象即可通过 Tracker 端口注入。
- **机器人 / ROS 复用**：`domain` 与 `application` 层无 cv2、无全局状态，可复用；`interfaces/cli.py` 的装配需要改造（时间戳随帧传入、事件化输出）。

## 已知限制

- 摄像头输入、真实视频与 GUI 交互未在 CI 中验证；本仓库的自动化证据来自合成视频。
- `r` 重置后 `frame` 列从 1 重新计数，而 `track_id` 继续递增；把同一 CSV 按 `frame` 关联视频时需要留意。
- 时间戳基于 FPS 估算，假定恒定帧率；`--loop` 回绕后与真实时间脱钩。
- `matched` 字段返回的是“当帧目标列表下标对”，不是 `track_id`。
- `--loop` 且不指定 `--max-frames` 时会无限运行。
- YOLO 检测依赖本地 `.pt` 权重，权重不随仓库分发、不自动下载；加载不可信权重存在代码执行风险，见 [SECURITY.md](SECURITY.md)。
- 检测间隔（`--motion-interval` / `detector.interval`）的语义是“第 interval 帧起每 interval 帧检测一次”，短片段可能一次都不检测。

## 安全与隐私

- 轨迹 CSV 可能包含人员位置等敏感数据，默认写入 `outputs/`（已被 `.gitignore` 忽略），发布演示前请脱敏。
- CSV 标签做了公式注入转义（含 BOM / 零宽字符绕过）；配置文件限制 1 MB 并拒绝 FIFO / 设备文件。
- 写入轨迹与录像前会拒绝符号链接，避免通过预置链接覆盖任意文件。
- 详细威胁模型与报告方式见 [SECURITY.md](SECURITY.md)。

## 参与贡献

欢迎 Issue / PR。开发环境、代码风格、测试要求与提交规范见 [CONTRIBUTING.md](CONTRIBUTING.md)。参与即表示同意遵守 [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md)。

## 许可证

[Apache License 2.0](LICENSE) © 2026 CGmoke
