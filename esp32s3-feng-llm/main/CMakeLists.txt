# ---- 源文件清单 --------------------------------------------------------------
# 默认构建只编「99.9% 场景要用」的模块；排障与实验性的做成开关，默认不进固件。
#
# 体积账（实测，q2/2048 发布配置）：
#   gbk_table.c        46,384 B  ← 3865 字的双向映射表，老式终端才用得到
#   feng_gemv_bench.c   4,994 B  ← 周期级剖析
#   feng_pie_*.S          155 B  ← PIE 实验（已结案：1.05×）
#
# 注意 feng_gemv_bench.c 之前是**无条件编译**的（只有 ESP_PLATFORM 的 include 保护），
# 所以那 4,994 B 一直在固件里，即使 FENG_BENCH_GEMV 默认关。现在按开关进出 SRCS。
set(FENG_SRCS
    "main.c" "feng_model.c" "feng_llm.c" "feng_quant.c" "feng_smp.c"
    "feng_tokenizer.c" "feng_sample.c"
    "feng_calc.c" "feng_tools.c" "feng_memory.c"
    "gbk.c"            # 退化实现只有几十字节；46 KB 的映射表见 FENG_GBK
    "feng_bonus.c"     # 彩蛋（关掉用 -DFENG_NO_BONUS=ON）
)

# 串口 GBK 转码。只有老式终端（SuperCom/XCOM 的 ANSI 模式）会发 GBK 字节；
# 本项目所有脚本、VS Code 串口、以及绝大多数现代终端都是 UTF-8。
# 关掉后固件按 UTF-8 直通收发，`\gbk` 会提示未编译。
#
# 只影响固件：PC 构建（build_pc_chat.ps1）的清单里没有 gbk.c——pc_chat 不做
# 串口编码转换；pc/gbk_selftest.cpp 是验真实转换的独立测试，单独编。
option(FENG_GBK "串口 GBK 转码（老式终端；+46 KB 映射表）" OFF)
if(FENG_GBK)
    list(APPEND FENG_SRCS "gbk_table.c")
endif()

# 周期级剖析：把 7.52 周期/权重的 GEMV 停顿归因到 LUT / 激活 / flash / FPU。
# 排障才用，平时白占 5 KB。
option(FENG_BENCH_GEMV "cycle-level Q4 GEMV inner-loop profiling on boot" OFF)
if(FENG_BENCH_GEMV)
    list(APPEND FENG_SRCS "feng_gemv_bench.c")
endif()

# PIE（128 位 SIMD）实验。已实测结案（正确的整块内核只做到 1.05×，
# 缘由见 docs/CHANGELOG.md v3.16-embed 附录），代码留着复现，默认不编。
option(FENG_BENCH_PIE "benchmark raw PIE (128-bit SIMD) throughput on boot" OFF)
option(FENG_GEMV_PIE "PIE (128-bit SIMD) Q4 GEMV kernel" OFF)
if(FENG_BENCH_PIE OR FENG_GEMV_PIE)
    list(APPEND FENG_SRCS "feng_pie_bench.S" "feng_pie_gemv.S")
endif()

idf_component_register(SRCS ${FENG_SRCS} INCLUDE_DIRS ".")

# 全组件严格 C23（v4.0 起，嵌入式侧从 C++23 迁到 C23）。
# C_EXTENSIONS OFF = -std=c23 而不是 -std=gnu23，与 PC 侧口径一致。
#
# 注意：原来的 -fno-exceptions / -fno-rtti / -fno-threadsafe-statics 是 C++ 专属选项，
# gcc 对 .c 文件会报 "valid for C++ but not for C"；而且它们同时会命中 .S 汇编文件。
# 迁到 C 之后这些约束天然成立（C 没有异常/RTTI），直接删掉。
set_target_properties(${COMPONENT_LIB} PROPERTIES
    C_STANDARD 23 C_STANDARD_REQUIRED ON C_EXTENSIONS OFF)
# the inference kernels are the hot path: -O2 + FPU on the XL7
target_compile_options(${COMPONENT_LIB} PRIVATE -O2)
# 非热点模块用 -Os 压体积：它们不在每 token 的内层循环里（tool 秒回、
# tokenizer 只在加载/解码时用），对推理 tok/s 无可测影响；
# 推理热点 feng_llm/feng_quant/feng_smp/feng_sample 保持上面的 -O2。
set(FENG_OS_SOURCES
    "main.c" "feng_model.c" "feng_tokenizer.c"
    "feng_tools.c" "feng_memory.c" "gbk.c" "gbk_table.c")
set_source_files_properties(${FENG_OS_SOURCES} PROPERTIES COMPILE_OPTIONS "-Os")
# feng_calc 里既有的 snprintf 拼接在 -Os 的值域分析下会报保守的 format-truncation
# （snprintf 本身有界，行为不变），只关掉这一条 -Werror。
# 迁到 C23 后 gcc 的值域分析会变，feng_memory / feng_tools / main 里串联的
# snprintf 也可能触发同类误报，一并抑制（snprintf 都带长度上限，行为不变）。
set_source_files_properties("feng_calc.c" PROPERTIES
    COMPILE_OPTIONS "-Os;-Wno-error=format-truncation")
set_source_files_properties("feng_memory.c" "feng_tools.c" "main.c" PROPERTIES
    COMPILE_OPTIONS "-Os;-Wno-error=format-truncation")
# KV cache mode: default int8 (1024 ctx). -DFENG_USE_Q2_KV=ON switches to the
# 2-bit block-8 cache (2048 ctx in the same PSRAM budget; quality study in
# scripts/kv_quant_experiment.py, PC regression exes pc_kv_test_*).
option(FENG_USE_Q2_KV "use 2-bit block-8 KV cache (2048 ctx on 16MB PSRAM)" OFF)
# 测试用：-DFENG_MAX_CTX_OVERRIDE=256 把上下文缩到 256，快速验证"写满自动开新对话"
set(FENG_MAX_CTX_OVERRIDE 0 CACHE STRING "override MAX_CTX for testing (0 = default)")
if(NOT FENG_MAX_CTX_OVERRIDE EQUAL 0)
    target_compile_definitions(${COMPONENT_LIB} PRIVATE FENG_MAX_CTX_OVERRIDE=${FENG_MAX_CTX_OVERRIDE})
endif()
# 彩蛋：固件里藏了一句话（见 feng_bonus.c）。默认开着，
# `-DFENG_NO_BONUS=ON` 关掉——那时连混淆数据都不会进二进制。
# 触发方式故意不写在 \help 里，README 的小字也只提到这个选项本身。
option(FENG_NO_BONUS "关闭固件里的那个小彩蛋" OFF)
if(FENG_NO_BONUS)
    target_compile_definitions(${COMPONENT_LIB} PRIVATE FENG_NO_BONUS=1)
endif()
# 测试用：-DFENG_BENCH_CTX=1 在开机时测 256/1024/2048 上下文的单次 forward 耗时
option(FENG_BENCH_CTX "benchmark attention at 256/1024/2048 ctx on boot" OFF)
if(FENG_BENCH_CTX)
    target_compile_definitions(${COMPONENT_LIB} PRIVATE FENG_BENCH_CTX=1)
endif()
# 测试用：-DFENG_Q2_LUT=OFF 关闭 q2 注意力内层的字节 LUT（A/B 基准）
option(FENG_Q2_LUT "use a byte LUT in q2 attention inner loops" ON)
if(NOT FENG_Q2_LUT)
    target_compile_definitions(${COMPONENT_LIB} PRIVATE FENG_Q2_LUT=0)
endif()
# 测试用：-DFENG_ATTN_PROF=ON 在基准里分段落 K / softmax / V 的周期
option(FENG_ATTN_PROF "profile q2 attention stages (K/softmax/V cycles)" OFF)
if(FENG_ATTN_PROF)
    target_compile_definitions(${COMPONENT_LIB} PRIVATE FENG_ATTN_PROF=1)
endif()
# 测试用：-DFENG_GEMV_A8=ON 用 int8 激活 + 整数点积的 Q4 GEMV（激活量化每个 GEMV 只做一次）
option(FENG_GEMV_A8 "int8 activation Q4 GEMV (quantize activations once per GEMV)" OFF)
if(FENG_GEMV_A8)
    target_compile_definitions(${COMPONENT_LIB} PRIVATE FENG_GEMV_A8=1)
endif()
# 测试用：-DFENG_GEMV_MADD=ON 用纯 madd 链累加（FP 运算少 1/3，末位取整会变）
option(FENG_GEMV_MADD "pure madd accumulation chain in Q4 GEMV" OFF)
if(FENG_GEMV_MADD)
    target_compile_definitions(${COMPONENT_LIB} PRIVATE FENG_GEMV_MADD=1)
endif()
# 测试用：-DFENG_FAST_EXP=ON 用多项式快速 exp 替换 softmax/silu 的 newlib expf
option(FENG_FAST_EXP "polynomial fast exp in softmax/silu" OFF)
if(FENG_FAST_EXP)
    target_compile_definitions(${COMPONENT_LIB} PRIVATE FENG_FAST_EXP=1)
endif()
# 测试用：-DFENG_Q2_VFOLD=ON 把 (scores*vscale) 先乘好，V 段内层只剩 madd
option(FENG_Q2_VFOLD "fold scores*vscale before the q2 V inner loop" OFF)
if(FENG_Q2_VFOLD)
    target_compile_definitions(${COMPONENT_LIB} PRIVATE FENG_Q2_VFOLD=1)
endif()
if(FENG_USE_Q2_KV)
    target_compile_definitions(${COMPONENT_LIB} PRIVATE FENG_KV_Q2=1 FENG_KV_Q2_BLOCK=8)
else()
    target_compile_definitions(${COMPONENT_LIB} PRIVATE FENG_KV_INT8=1)
endif()
# FENG_GBK 传给编译单元（gbk.c 靠它选真实实现还是退化实现）
if(FENG_GBK)
    target_compile_definitions(${COMPONENT_LIB} PRIVATE FENG_GBK=1)
endif()
