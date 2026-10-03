idf_component_register(
    SRCS "main.c" "feng_model.c" "feng_llm.c" "feng_quant.c" "feng_smp.c" "feng_tokenizer.c"
         "gbk.c" "gbk_table.c"
    INCLUDE_DIRS "."
)

# the inference kernels are the hot path: -O2 + FPU on the XL7
target_compile_options(${COMPONENT_LIB} PRIVATE -O2)
# int8 KV cache (4x longer context) on the device; the PC build keeps fp32 for parity
target_compile_definitions(${COMPONENT_LIB} PRIVATE FENG_KV_INT8=1)
