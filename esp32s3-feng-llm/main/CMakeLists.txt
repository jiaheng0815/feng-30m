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
if(FENG_USE_Q2_KV)
    target_compile_definitions(${COMPONENT_LIB} PRIVATE FENG_KV_Q2=1 FENG_KV_Q2_BLOCK=8)
else()
    target_compile_definitions(${COMPONENT_LIB} PRIVATE FENG_KV_INT8=1)
endif()
