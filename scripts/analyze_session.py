#!/usr/bin/env python3
"""对一次跟踪会话做“噪点 / 精准度 / 帧率”体检。

输入是 Web 界面或 CLI 产出的 JSONL 事件流（推荐，含 removed 事件），CSV 仅作为
几何统计的回退来源。全部指标都是**无标注**的：不需要人工画 ground truth，就能量化
“ID 比真实目标多了几倍”“同帧重复框多少对”“框几何有多离谱”。

用法::

    python3 scripts/analyze_session.py outputs/web_s2.jsonl
    python3 scripts/analyze_session.py outputs/web_s2.jsonl --frame 640x480
    python3 scripts/analyze_session.py outputs/a.jsonl outputs/b.jsonl   # A/B 对比

判定阈值集中在 ``THRESHOLDS``，可自行调整；退出码 1 表示有 FAIL 项，便于接到 CI。
"""

from __future__ import annotations

import argparse
import json
import math
import statistics as st
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

BBox = Tuple[float, float, float, float]

#: 判定阈值：(warn, fail)。指标越小越好（``higher_is_better`` 反转）。
THRESHOLDS: Dict[str, Tuple[float, float]] = {
    # 同帧并发 ID / 空间簇数。1.0 = 每个真实目标恰好一个 ID。
    "id_inflation": (1.4, 2.0),
    # 同帧 IoU>0.45 的重复框对数，占帧比例。
    "dup_pair_rate": (0.05, 0.15),
    # 几何离谱框（面积占比超界 / 极细长）占全部行的比例。
    "bad_geom_rate": (0.05, 0.15),
    # 贴边（距画面边界 <4px）框比例，典型是编码/黑边伪影。
    "border_rate": (0.02, 0.06),
    # 每帧被检测器确认之外的“自由漂移”速度，px/帧。
    "drift_px": (3.0, 6.0),
    # 短命 ID（<=5 帧）比例。
    "short_lived_rate": (0.10, 0.25),
}

HIGHER_IS_BETTER = {"fps"}


# --------------------------------------------------------------------------
# 几何工具
# --------------------------------------------------------------------------
def _inter(a: BBox, b: BBox) -> float:
    left, top = max(a[0], b[0]), max(a[1], b[1])
    right = min(a[0] + a[2], b[0] + b[2])
    bottom = min(a[1] + a[3], b[1] + b[3])
    if right <= left or bottom <= top:
        return 0.0
    return float((right - left) * (bottom - top))


def iou(a: BBox, b: BBox) -> float:
    inter = _inter(a, b)
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0.0


def center(b: BBox) -> Tuple[float, float]:
    return (b[0] + b[2] / 2.0, b[1] + b[3] / 2.0)


def cluster_count(boxes: Sequence[BBox], radius: float) -> int:
    """按中心距离做单链聚类，返回簇数（= 估计的真实目标数）。"""
    n = len(boxes)
    if n == 0:
        return 0
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    centers = [center(box) for box in boxes]
    for i in range(n):
        for j in range(i + 1, n):
            dx = centers[i][0] - centers[j][0]
            dy = centers[i][1] - centers[j][1]
            if dx * dx + dy * dy < radius * radius:
                parent[find(i)] = find(j)
    return len({find(i) for i in range(n)})


# --------------------------------------------------------------------------
# 读取
# --------------------------------------------------------------------------
def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def frames_from_jsonl(records: Iterable[Dict[str, Any]]) -> List[Tuple[float, float, List[BBox]]]:
    """返回 (frame, timestamp, boxes)。"""
    out = []
    for record in records:
        boxes = [tuple(float(v) for v in item["bbox"]) for item in (record.get("tracks") or [])]
        out.append((float(record.get("frame", 0)), float(record.get("timestamp", 0.0)), boxes))
    return out


# --------------------------------------------------------------------------
# 指标
# --------------------------------------------------------------------------
def analyze(
    frames: Sequence[Tuple[float, float, List[BBox]]],
    frame_size: Tuple[int, int],
    cluster_radius: float = 60.0,
) -> Dict[str, float]:
    width, height = frame_size
    frame_area = float(width * height)
    metrics: Dict[str, float] = {}

    non_empty = [item for item in frames if item[2]]
    metrics["frames"] = float(len(frames))
    metrics["frames_with_tracks"] = float(len(non_empty))
    if not frames:
        return metrics

    stamps = [item[1] for item in frames if item[1] > 0]
    if len(stamps) > 1:
        span = stamps[-1] - stamps[0]
        metrics["fps"] = (len(stamps) - 1) / span if span > 0 else 0.0

    counts = [len(item[2]) for item in frames]
    metrics["tracks_peak"] = float(max(counts))
    metrics["tracks_mean"] = float(st.mean([c for c in counts if c])) if non_empty else 0.0

    # 空间簇膨胀比
    ratios: List[float] = []
    for _frame, _ts, boxes in non_empty:
        if len(boxes) < 2:
            continue
        clusters = cluster_count(boxes, cluster_radius)
        if clusters:
            ratios.append(len(boxes) / clusters)
    metrics["id_inflation"] = float(st.mean(ratios)) if ratios else 1.0
    metrics["clusters_mean"] = (
        float(st.mean([cluster_count(item[2], cluster_radius) for item in non_empty]))
        if non_empty
        else 0.0
    )

    # 同帧重复对
    dup_pairs = 0
    for _frame, _ts, boxes in non_empty:
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                if iou(boxes[i], boxes[j]) > 0.45:
                    dup_pairs += 1
    metrics["dup_pairs"] = float(dup_pairs)
    metrics["dup_pair_rate"] = dup_pairs / max(1, len(non_empty))

    # 几何
    areas: List[float] = []
    bad_geom = 0
    border = 0
    rows = 0
    for _frame, _ts, boxes in non_empty:
        for box in boxes:
            x, y, w, h = box
            rows += 1
            area_ratio = (w * h) / frame_area
            areas.append(w * h)
            aspect = w / h if h > 0 else 0.0
            if not (0.002 <= area_ratio <= 0.08) or not (0.3 <= aspect <= 3.0):
                bad_geom += 1
            if x < 4 or y < 4 or x + w > width - 4 or y + h > height - 4:
                border += 1
    metrics["bad_geom_rate"] = bad_geom / max(1, rows)
    metrics["border_rate"] = border / max(1, rows)
    if areas:
        metrics["area_p50_pct"] = 100.0 * st.median(areas) / frame_area
        metrics["area_p10_pct"] = 100.0 * st.quantiles(areas, n=10)[0] / frame_area
        metrics["area_p90_pct"] = 100.0 * st.quantiles(areas, n=10)[8] / frame_area

    # 逐 ID 轨迹：寿命、漂移、几何稳定性
    tracks: Dict[int, List[Tuple[float, BBox]]] = defaultdict(list)
    for record_frame, _ts, boxes in frames:
        for box in boxes:
            tracks.setdefault(id(box), []).append((record_frame, box))
    # 上面按对象身份分组不可靠，改为由调用方提供 id 版；这里退化为帧内统计。
    metrics["_track_groups"] = float(len(tracks))
    return metrics


def analyze_with_ids(
    records: Sequence[Dict[str, Any]],
    frame_size: Tuple[int, int],
    cluster_radius: float = 60.0,
) -> Dict[str, float]:
    """带 track_id 的完整分析（JSONL 路径专用）。"""
    width, height = frame_size
    frame_area = float(width * height)

    frames: List[Tuple[float, float, List[Tuple[int, BBox]]]] = []
    for record in records:
        items = []
        for item in record.get("tracks") or []:
            items.append((int(item["id"]), tuple(float(v) for v in item["bbox"])))
        frames.append((float(record.get("frame", 0)), float(record.get("timestamp", 0.0)), items))

    metrics: Dict[str, float] = {}
    non_empty = [item for item in frames if item[2]]
    metrics["frames"] = float(len(frames))
    metrics["frames_with_tracks"] = float(len(non_empty))
    stamps = [item[1] for item in frames if item[1] > 0]
    if len(stamps) > 1:
        span = stamps[-1] - stamps[0]
        metrics["fps"] = (len(stamps) - 1) / span if span > 0 else 0.0

    counts = [len(item[2]) for item in frames]
    metrics["tracks_peak"] = float(max(counts)) if counts else 0.0
    metrics["tracks_mean"] = float(st.mean([c for c in counts if c])) if non_empty else 0.0
    metrics["ids_total"] = float(len({tid for _f, _t, items in frames for tid, _b in items}))

    # 空间簇
    ratios: List[float] = []
    clusters: List[int] = []
    for _frame, _ts, items in non_empty:
        boxes = [b for _tid, b in items]
        if len(boxes) < 2:
            continue
        n = cluster_count(boxes, cluster_radius)
        clusters.append(n)
        if n:
            ratios.append(len(boxes) / n)
    metrics["id_inflation"] = float(st.mean(ratios)) if ratios else 1.0
    metrics["clusters_mean"] = float(st.mean(clusters)) if clusters else 0.0

    # 重复对
    dup_pairs = 0
    for _frame, _ts, items in non_empty:
        boxes = [b for _tid, b in items]
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                if iou(boxes[i], boxes[j]) > 0.45:
                    dup_pairs += 1
    metrics["dup_pairs"] = float(dup_pairs)
    metrics["dup_pair_rate"] = dup_pairs / max(1, len(non_empty))

    # 几何
    bad_geom = border = rows = 0
    areas: List[float] = []
    for _frame, _ts, items in non_empty:
        for _tid, box in items:
            x, y, w, h = box
            rows += 1
            areas.append(w * h)
            aspect = w / h if h > 0 else 0.0
            if not (0.002 <= (w * h) / frame_area <= 0.08) or not (0.3 <= aspect <= 3.0):
                bad_geom += 1
            if x < 4 or y < 4 or x + w > width - 4 or y + h > height - 4:
                border += 1
    metrics["bad_geom_rate"] = bad_geom / max(1, rows)
    metrics["border_rate"] = border / max(1, rows)
    if areas:
        metrics["area_p50_pct"] = 100.0 * st.median(areas) / frame_area
        metrics["area_p10_pct"] = 100.0 * st.quantiles(areas, n=10)[0] / frame_area
        metrics["area_p90_pct"] = 100.0 * st.quantiles(areas, n=10)[8] / frame_area

    # 逐 ID
    tracks: Dict[int, List[Tuple[float, BBox]]] = defaultdict(list)
    for record_frame, _ts, items in frames:
        for tid, box in items:
            tracks[tid].append((record_frame, box))

    lives = [len(v) for v in tracks.values()]
    metrics["life_median"] = float(st.median(lives)) if lives else 0.0
    metrics["life_max"] = float(max(lives)) if lives else 0.0
    metrics["short_lived_rate"] = (
        sum(1 for life in lives if life <= 5) / max(1, len(lives))
    )
    metrics["ids_short"] = float(sum(1 for life in lives if life <= 5))

    # 单帧最大位移（漂移/追踪失败的代理指标）
    steps: List[float] = []
    for history in tracks.values():
        for (f0, b0), (f1, b1) in zip(history, history[1:]):
            gap = max(1.0, f1 - f0)
            c0, c1 = center(b0), center(b1)
            steps.append(math.dist(c0, c1) / gap)
    if steps:
        steps.sort()
        metrics["drift_p50"] = steps[len(steps) // 2]
        metrics["drift_p95"] = steps[min(len(steps) - 1, int(len(steps) * 0.95))]
        metrics["drift_px"] = metrics["drift_p95"]
    metrics["tracks"] = float(len(tracks))
    return metrics


# --------------------------------------------------------------------------
# 报告
# --------------------------------------------------------------------------
def _verdict(name: str, value: float) -> str:
    warn, fail = THRESHOLDS[name]
    if name in HIGHER_IS_BETTER:
        if value <= fail:
            return "FAIL"
        return "WARN" if value <= warn else "OK"
    if value >= fail:
        return "FAIL"
    return "WARN" if value >= warn else "OK"


def report(label: str, metrics: Dict[str, float]) -> List[str]:
    lines = [f"=== {label} ==="]
    fps = metrics.get("fps")
    lines.append(
        "帧: {frames:.0f} (含目标 {frames_with_tracks:.0f})   "
        "时长: {dur:.1f}s   会话平均帧率: {fps}".format(
            frames=metrics["frames"],
            frames_with_tracks=metrics["frames_with_tracks"],
            dur=(
                metrics["frames"] / fps if fps else 0.0
            ),
            fps=f"{fps:.1f}" if fps else "n/a",
        )
    )
    lines.append(
        "ID: 总数 {ids:.0f}  峰值并发 {peak:.0f}  均值 {mean:.1f}   "
        "空间簇均值 {clusters:.1f}  →  ID 膨胀比 {infl:.2f}x  [{v}]".format(
            ids=metrics.get("ids_total", metrics.get("tracks", 0)),
            peak=metrics["tracks_peak"],
            mean=metrics["tracks_mean"],
            clusters=metrics.get("clusters_mean", 0.0),
            infl=metrics.get("id_inflation", 1.0),
            v=_verdict("id_inflation", metrics.get("id_inflation", 1.0)),
        )
    )
    lines.append(
        "重复框: 同帧 IoU>0.45 共 {dup:.0f} 对，{rate:.3f}/帧  [{v}]".format(
            dup=metrics.get("dup_pairs", 0.0),
            rate=metrics.get("dup_pair_rate", 0.0),
            v=_verdict("dup_pair_rate", metrics.get("dup_pair_rate", 0.0)),
        )
    )
    lines.append(
        "几何: 面积中位 {p50} / p10 {p10} / p90 {p90} (%画面)   "
        "离谱框 {bad:.1%}  [{v}]   贴边框 {border:.1%}  [{v2}]".format(
            p50=f"{metrics.get('area_p50_pct', 0):.2f}",
            p10=f"{metrics.get('area_p10_pct', 0):.2f}",
            p90=f"{metrics.get('area_p90_pct', 0):.2f}",
            bad=metrics.get("bad_geom_rate", 0.0),
            v=_verdict("bad_geom_rate", metrics.get("bad_geom_rate", 0.0)),
            border=metrics.get("border_rate", 0.0),
            v2=_verdict("border_rate", metrics.get("border_rate", 0.0)),
        )
    )
    if "life_median" in metrics:
        lines.append(
            "寿命: 中位 {med:.0f} 帧 最长 {mx:.0f}  短命(<=5帧) {short:.0f} 个 "
            "({rate:.1%})  [{v}]".format(
                med=metrics["life_median"],
                mx=metrics["life_max"],
                short=metrics.get("ids_short", 0.0),
                rate=metrics.get("short_lived_rate", 0.0),
                v=_verdict("short_lived_rate", metrics.get("short_lived_rate", 0.0)),
            )
        )
    if "drift_px" in metrics:
        lines.append(
            "漂移: 中心位移 p50 {p50:.1f} px/帧  p95 {p95:.1f} px/帧  [{v}]".format(
                p50=metrics["drift_p50"],
                p95=metrics["drift_p95"],
                v=_verdict("drift_px", metrics["drift_px"]),
            )
        )
    fails = [
        name
        for name in THRESHOLDS
        if name in metrics and _verdict(name, metrics[name]) == "FAIL"
    ]
    warns = [
        name
        for name in THRESHOLDS
        if name in metrics and _verdict(name, metrics[name]) == "WARN"
    ]
    lines.append(
        "结论: "
        + ("FAIL " + ", ".join(fails) if fails else "无 FAIL")
        + ("  |  WARN " + ", ".join(warns) if warns else "")
    )
    return lines


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def _parse_frame(text: str) -> Tuple[int, int]:
    try:
        w, h = text.lower().split("x")
        return int(w), int(h)
    except Exception:  # noqa: BLE001
        raise argparse.ArgumentTypeError(f"expected WIDTHxHEIGHT, got {text!r}") from None


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("paths", nargs="+", type=Path, help="JSONL 事件流（支持多个做 A/B）")
    parser.add_argument(
        "--frame",
        type=_parse_frame,
        default=(640, 480),
        help="采集分辨率，默认 640x480（用于把框面积换算成画面占比）",
    )
    parser.add_argument(
        "--cluster-radius",
        type=float,
        default=60.0,
        help="空间簇半径（像素）；应略大于目标典型尺寸，默认 60",
    )
    args = parser.parse_args(argv)

    exit_code = 0
    for path in args.paths:
        if not path.is_file():
            print(f"跳过不存在的文件: {path}", file=sys.stderr)
            exit_code = 2
            continue
        records = load_jsonl(path)
        metrics = analyze_with_ids(records, args.frame, args.cluster_radius)
        for line in report(path.as_posix(), metrics):
            print(line)
        if any(
            name in metrics and _verdict(name, metrics[name]) == "FAIL"
            for name in THRESHOLDS
        ):
            exit_code = 1
        print()
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
