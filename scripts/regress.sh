#!/usr/bin/env bash
# 跟踪质量回归：单测 + 会话体检 + 与基线对比。
#
# 用法:
#   scripts/regress.sh                          # 用 outputs/ 里最新的会话对比基线
#   scripts/regress.sh outputs/web_s7.jsonl     # 指定新会话
#   BASELINE=outputs/web_s1.jsonl scripts/regress.sh
#
# 退出码: 0 全部通过；1 单测失败；2 指标回归；3 文件缺失。
set -uo pipefail

cd "$(dirname "$0")/.." || exit 3
PY="${PY:-.venv/bin/python}"
[ -x "$PY" ] || PY=python3
FRAME="${FRAME:-640x480}"
BASELINE="${BASELINE:-outputs/web_s2.jsonl}"

fail=0

echo "==> 1/3 单元测试"
# pytest 可能只装在系统环境而不是 .venv 里，两种都试一下。
if PYTHONPATH=src "$PY" -c "import pytest" >/dev/null 2>&1; then
    runner=("$PY" -m pytest -q)
elif command -v pytest >/dev/null 2>&1; then
    runner=(pytest -q)
else
    runner=()
fi
if [ "${#runner[@]}" -eq 0 ]; then
    echo "   （未找到 pytest，跳过单测；pip install -e '.[dev]' 可安装）"
elif ! PYTHONPATH=src "${runner[@]}"; then
    echo "!! 单元测试失败" >&2
    fail=1
fi

echo
echo "==> 2/3 会话体检"
if [ "$#" -gt 0 ]; then
    targets=("$@")
else
    # 默认取 outputs/ 下最新的 web_*.jsonl（mtime 倒序第一）
    latest=$(ls -t outputs/web_*.jsonl 2>/dev/null | head -1)
    if [ -z "$latest" ]; then
        echo "!! outputs/ 下没有 web_*.jsonl，先跑一段会话再回归" >&2
        exit 3
    fi
    targets=("$latest")
fi

for target in "${targets[@]}"; do
    if [ ! -f "$target" ]; then
        echo "!! 文件不存在: $target" >&2
        exit 3
    fi
    "$PY" scripts/analyze_session.py "$target" --frame "$FRAME"
    # FAIL 时退出码 1
    if [ $? -ne 0 ]; then
        fail=2
    fi
done

echo "==> 3/3 与基线对比: $BASELINE"
skip_base=0
for target in "${targets[@]}"; do
    [ "$(readlink -f "$target")" = "$(readlink -f "$BASELINE" 2>/dev/null)" ] && skip_base=1
done
if [ "$skip_base" -eq 1 ]; then
    echo "   （本次会话即基线，跳过重复对比）"
elif [ -f "$BASELINE" ]; then
    "$PY" scripts/analyze_session.py "$BASELINE" "${targets[@]}" --frame "$FRAME"
else
    echo "   （基线 $BASELINE 不存在，跳过）"
fi

if [ "$fail" -eq 0 ]; then
    echo
    echo "==> PASS"
else
    echo
    echo "==> FAIL (code=$fail)"
fi
exit "$fail"
