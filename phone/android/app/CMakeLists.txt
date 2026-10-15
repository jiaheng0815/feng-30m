# Android 端原生库：把 phone/engine 那份 C23 引擎 + JNI 薄壳编成 libfeng.so。
#
# 引擎源码不在本目录下（在 phone/engine/），直接从那里取——只有一份源码，
# Termux CLI 与 APK 用的是同一批 .c 文件，不做二次拷贝。
cmake_minimum_required(VERSION 3.22)
project(feng30m C)

set(CMAKE_C_STANDARD 23)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)

set(ENGINE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../engine)

if(NOT EXISTS ${ENGINE_DIR}/feng_llm.c)
    message(FATAL_ERROR "找不到引擎源码 ${ENGINE_DIR}（应当在仓库的 phone/engine/ 下）")
endif()

set(FENG_HOT
    ${ENGINE_DIR}/feng_llm.c
    ${ENGINE_DIR}/feng_quant.c
    ${ENGINE_DIR}/feng_smp.c
    ${ENGINE_DIR}/feng_sample.c
    ${ENGINE_DIR}/feng_phone.c
    ${ENGINE_DIR}/feng_neon.c)

set(FENG_COLD
    ${ENGINE_DIR}/feng_model.c
    ${ENGINE_DIR}/feng_tokenizer.c
    ${ENGINE_DIR}/feng_calc.c
    ${ENGINE_DIR}/feng_tools.c
    ${ENGINE_DIR}/feng_memory.c)

add_library(feng SHARED
    feng_jni.c
    ${FENG_HOT}
    ${FENG_COLD})

target_include_directories(feng PRIVATE ${ENGINE_DIR})

# 与板端发布配置一致：q2 KV / block8（2048 上下文约 9.6MB）
target_compile_definitions(feng PRIVATE FENG_KV_Q2=1 FENG_KV_Q2_BLOCK=8)

# 热点 -O2、非热点 -Os（与 main/CMakeLists.txt 的分组一致，控制 .so 体积）
set_source_files_properties(${FENG_HOT} PROPERTIES COMPILE_OPTIONS "-O2")
set_source_files_properties(${FENG_COLD} PROPERTIES COMPILE_OPTIONS "-Os")

target_link_libraries(feng PRIVATE log)
