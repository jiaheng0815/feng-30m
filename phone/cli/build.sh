#!/usr/bin/env bash
# 手机端 CLI 构建脚本（Termux 设备上直接跑；WSL / 桌面 Linux 上跑同一份源码做验证）。
#
#   ./build.sh                    # 用环境里默认的编译器，产物进 build/
#   CC=clang ./build.sh           # 指定 C 编译器
#   OUT=out JOBS=8 ./build.sh     # 指定产物目录与并行度
#
# 依赖：
#   Termux  : pkg install -y clang
#   Ubuntu  : apt install -y gcc-14（-std=c23 需要 gcc 14+；Ubuntu 24.04 的默认 gcc-13
#             只认 -std=c2x，不认 -std=c23）
#
# 产出：build/phone_chat（手机端 CLI）
#       build/pc_kv_suite_phone、build/pc_mt_suite_phone（验收套件，需要 C++ 编译器）
#
# 编译口径与板端固件一致（v4.0 起引擎全面 C23）：
#   核心 C23 + 热点 -O2 / 非热点 -Os；验收套件是 pc/ 下的 C++23 驱动，原样复用。
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
ENG="$HERE/../engine"
SUITE_DIR="$ROOT/esp32s3-feng-llm/pc"

# -std=c23 需要 gcc 14+ / clang 18+。Ubuntu 24.04 的默认 cc 是 gcc-13（只认 -std=c2x），
# Termux 的 clang 够新。没显式指定 CC 时逐个探测，省得用户踩这个坑。
pick_cc() {
    local want="$1"; shift
    for c in "$@"; do
        if command -v "$c" >/dev/null 2>&1 && echo | "$c" "$want" -E - >/dev/null 2>&1; then
            echo "$c"; return 0
        fi
    done
    echo ""
}
CC="${CC:-$(pick_cc -std=c23 cc gcc-14 clang gcc)}"
CXX="${CXX:-$(pick_cc -std=c++23 c++ g++-14 clang++ g++)}"
OUT="${OUT:-$HERE/build}"
JOBS="${JOBS:-$( (nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 2) )}"

if [ -z "$CC" ]; then
    echo "找不到支持 -std=c23 的 C 编译器（需要 gcc 14+ 或 clang 18+）" >&2
    echo "  Ubuntu: apt install gcc-14     Termux: pkg install clang" >&2
    exit 1
fi

# KV 配置：与板端发布配置一致（q2 / block8）
KVFLAGS="-DFENG_KV_Q2=1 -DFENG_KV_Q2_BLOCK=8"
CFLAGS="-std=c23 -pthread -I$ENG $KVFLAGS -Wall"
CXXFLAGS="-std=c++23 -pthread -I$ENG $KVFLAGS"

# 热点：每 token 内层循环所在（与 main/CMakeLists.txt 的分组一致）+ 手机端新增的三个文件
HOT="feng_llm.c feng_quant.c feng_smp.c feng_sample.c feng_phone.c feng_neon.c"
COLD="feng_model.c feng_tokenizer.c feng_calc.c feng_tools.c feng_memory.c"

mkdir -p "$OUT/obj"

echo "== CC=$CC  OUT=$OUT"
"$CC" --version | head -1

OBJS=""
for f in $HOT; do
    o="$OUT/obj/${f%.c}.o"
    echo "   CC $f  [-O2]"
    "$CC" $CFLAGS -O2 -c "$ENG/$f" -o "$o"
    OBJS="$OBJS $o"
done
for f in $COLD; do
    o="$OUT/obj/${f%.c}.o"
    echo "   CC $f  [-Os]"
    "$CC" $CFLAGS -Os -c "$ENG/$f" -o "$o"
    OBJS="$OBJS $o"
done

echo "   LD phone_chat"
"$CC" $CFLAGS -O2 "$HERE/phone_chat.c" $OBJS -o "$OUT/phone_chat" -lm

# 验收套件：直接复用板端 pc/ 那两份套件源码，保证判定口径完全一致（不另写一份题表）。
# 它们是 C++23 驱动（与板端 PC 侧同一份文件），链上面的 C 目标文件即可。
if [ -f "$SUITE_DIR/pc_kv_suite.cpp" ] && command -v "$CXX" >/dev/null 2>&1; then
    echo "   LD pc_kv_suite_phone / pc_mt_suite_phone（复用 esp32s3-feng-llm/pc 的套件源码）"
    "$CXX" $CXXFLAGS -O2 "$SUITE_DIR/pc_kv_suite.cpp" $OBJS -o "$OUT/pc_kv_suite_phone" -lm
    "$CXX" $CXXFLAGS -O2 "$SUITE_DIR/pc_mt_suite.cpp" $OBJS -o "$OUT/pc_mt_suite_phone" -lm
else
    echo "!! 跳过验收套件（找不到 $SUITE_DIR/pc_kv_suite.cpp 或没有 C++ 编译器）"
fi

echo
echo "完成 -> $OUT"
ls -la "$OUT" | grep -v '^d' || true
echo
echo "下一步（<MODEL> 指向含 model.bin / tokenizer.bin 的目录）："
echo "  $OUT/phone_chat --bench 32 <MODEL>                       # tok/s"
echo "  $OUT/pc_kv_suite_phone <MODEL> $SUITE_DIR/prompt_long.txt 5200"
echo "  $OUT/pc_mt_suite_phone <MODEL>"
