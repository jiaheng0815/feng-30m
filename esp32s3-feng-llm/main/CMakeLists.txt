idf_component_register(
    SRCS "main.c" "feng_model.c" "feng_llm.c" "feng_quant.c" "feng_smp.c" "feng_tokenizer.c"
         "feng_calc.c" "feng_tools.c" "gbk.c" "gbk_table.c"
    INCLUDE_DIRS "."
)

# the inference kernels are the hot path: -O2 + FPU on the XL7
target_compile_options(${COMPONENT_LIB} PRIVATE -O2)
# KV cache mode: default int8 (1024 ctx). -DFENG_USE_Q2_KV=ON switches to the
# 2-bit block-8 cache (2048 ctx in the same PSRAM budget; quality study in
# scripts/kv_quant_experiment.py, PC regression exes pc_kv_test_*).
option(FENG_USE_Q2_KV "use 2-bit block-8 KV cache (2048 ctx on 16MB PSRAM)" OFF)
# 测试用：-DFENG_MAX_CTX_OVERRIDE=256 把上下文缩到 256，快速验证"写满自动开新对话"
set(FENG_MAX_CTX_OVERRIDE 0 CACHE STRING "override MAX_CTX for testing (0 = default)")
if(NOT FENG_MAX_CTX_OVERRIDE EQUAL 0)
    target_compile_definitions(${COMPONENT_LIB} PRIVATE FENG_MAX_CTX_OVERRIDE=${FENG_MAX_CTX_OVERRIDE})
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
if(FENG_USE_Q2_KV)
    target_compile_definitions(${COMPONENT_LIB} PRIVATE FENG_KV_Q2=1 FENG_KV_Q2_BLOCK=8)
else()
    target_compile_definitions(${COMPONENT_LIB} PRIVATE FENG_KV_INT8=1)
endif()
