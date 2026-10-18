#!/usr/bin/env bash
# 线程扩展性基准：同一台机器上跑 1/2/4/6/8 线程，输出写 logs/phone_bench_<主机名>.txt。
#
# 这是唯一能产出「可引用数字」的入口——文档里的 tok/s 都必须来自这里或手机上的
# phone_chat --bench，不能凭估算写。
#
# 用法：./bench.sh [model_export 目录] [生成 token 数]
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
MODEL="${1:-$ROOT/board/model_export_4.0-embed}"
NGEN="${2:-32}"
BIN="$HERE/build/phone_chat"
LOGDIR="$ROOT/logs"
OUT="$LOGDIR/phone_bench_$(hostname).txt"

[ -x "$BIN" ] || { echo "先跑 build.sh 生成 $BIN" >&2; exit 1; }
[ -f "$MODEL/model.bin" ] || { echo "找不到 $MODEL/model.bin" >&2; exit 1; }

mkdir -p "$LOGDIR"
{
    echo "# feng-30m 手机引擎基准 — $(date '+%Y-%m-%d %H:%M:%S')"
    echo "# 主机 $(hostname) ｜ $( (nproc 2>/dev/null || echo '?') ) 逻辑核 ｜ 模型 $MODEL"
    echo "# ctx 2048 ｜ q2/block8 ｜ 每档生成 $NGEN token（含 8 token 热身）"
    echo "# 注意：这是手机引擎在**本机**的数字，不是手机的。真机数字请在 Termux 里跑同样的命令。"
    for t in 1 2 4 6 8; do
        echo "--- threads=$t ---"
        FENG_PHONE_THREADS=$t "$BIN" --bench "$NGEN" "$MODEL" 2>&1 | grep -E 'bench:'
    done
} | tee "$OUT"

echo
echo "已写 $OUT"
