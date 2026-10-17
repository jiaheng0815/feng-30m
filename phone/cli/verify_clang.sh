#!/usr/bin/env bash
# 用 clang 重建一份，跑 32 题矩阵，与 gcc 版逐字节比对。
#
# 为什么单独验：Termux 装的是 clang，而开发机默认用 gcc；两边编译出的浮点代码
# 有可能不同（尤其 -O2 下的向量化），"换编译器结果还一样" 是手机端能跑的前提。
#
# 用法：./verify_clang.sh [model_export 目录]
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
MODEL="${1:-$ROOT/board/model_export_v3_19b6}"
FILLER="$ROOT/pc/prompt_long.txt"
LOGDIR="$ROOT/logs"

command -v clang >/dev/null 2>&1 || { echo "没有 clang（Ubuntu: apt install clang；Termux: pkg install clang）" >&2; exit 1; }
[ -x "$HERE/build/pc_kv_suite_phone" ] || { echo "先跑 build.sh（gcc 版）" >&2; exit 1; }

OUT="$HERE/build-clang"
echo "== 用 clang 重建 -> $OUT"
CC=clang CXX=clang++ OUT="$OUT" "$HERE/build.sh" >/dev/null

echo "== 跑 32 题矩阵"
FENG_PHONE_THREADS=4 "$OUT/pc_kv_suite_phone" "$MODEL" "$FILLER" 5200 > "$LOGDIR/phone_clang_kv.txt" 2>&1
tail -1 "$LOGDIR/phone_clang_kv.txt"

# 与 gcc 版对比（gcc 版的结果在 verify.sh 里刚跑过；没有就先跑一次）
if [ ! -f "$LOGDIR/phone_4t_kv.txt" ]; then
    echo "== gcc 版结果缺失，补跑一次"
    FENG_PHONE_THREADS=4 "$HERE/build/pc_kv_suite_phone" "$MODEL" "$FILLER" 5200 > "$LOGDIR/phone_4t_kv.txt" 2>&1
fi

if diff <(tr -d '\r' < "$LOGDIR/phone_4t_kv.txt") <(tr -d '\r' < "$LOGDIR/phone_clang_kv.txt") >/dev/null; then
    echo "  [PASS] clang 版与 gcc 版输出逐字节相同（日志 logs/phone_clang_kv.txt）"
else
    echo "  [FAIL] clang 与 gcc 输出不同："
    diff <(tr -d '\r' < "$LOGDIR/phone_4t_kv.txt") <(tr -d '\r' < "$LOGDIR/phone_clang_kv.txt") | head -20
    exit 1
fi
