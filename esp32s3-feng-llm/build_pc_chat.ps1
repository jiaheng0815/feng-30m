# 编译 PC 端 C++23 引擎运行时（带 tool）：pc_chat / pc_kv_suite / pc_calc_test
# 用法： .\build_pc_chat.ps1        （可选 $env:FENG_GXX 指定 g++ 路径）
$ErrorActionPreference = "Stop"
$gxx = $env:FENG_GXX
if (-not $gxx) { $gxx = "g++" }          # 例如 <你的 MSYS2>\ucrt64\bin\g++.exe

Push-Location $PSScriptRoot
try {
    # C++23 严格模式 + 零开销约定：无异常、无 RTTI、无函数内静态对象守卫
    $flags = @("-std=c++23", "-fno-exceptions", "-fno-rtti", "-fno-threadsafe-statics", "-O2")
    $base = @(
        "main\feng_model.cpp", "main\feng_llm.cpp", "main\feng_quant.cpp",
        "main\feng_smp.cpp", "main\feng_tokenizer.cpp", "main\feng_sample.cpp",
        "main\feng_calc.cpp", "main\feng_tools.cpp", "main\feng_memory.cpp"
    )
    # 1) 聊天运行时（q2 block8 / 2048 上下文）
    & $gxx @flags -DFENG_KV_Q2=1 -DFENG_KV_Q2_BLOCK=8 -o pc\pc_chat_q2b8.exe `
        pc\pc_chat.cpp @base -Imain -lm
    # 2) 板端代理套件（32 题矩阵，q2 block8；算式题走 tool）
    & $gxx @flags -DFENG_KV_Q2=1 -DFENG_KV_Q2_BLOCK=8 -o pc\pc_kv_suite_q2b8.exe `
        pc\pc_kv_suite.cpp @base -Imain -lm
    & $gxx @flags -DFENG_KV_INT8=1 -o pc\pc_kv_suite_i8.exe `
        pc\pc_kv_suite.cpp @base -Imain -lm
    # 3) 多轮回归套件（报名字→问身份/问名字、同类取新、12 题连续记忆）
    & $gxx @flags -DFENG_KV_Q2=1 -DFENG_KV_Q2_BLOCK=8 -o pc\pc_mt_suite.exe `
        pc\pc_mt_suite.cpp @base -Imain -lm
    # 4) tool 单测（算式识别/求值/中文句式 + 时间/随机数语义）
    & $gxx @flags -o pc\pc_calc_test.exe pc\pc_calc_test.cpp main\feng_calc.cpp -Imain -lm
    & $gxx @flags -o pc\pc_tools_test.exe pc\pc_tools_test.cpp main\feng_tools.cpp -Imain -lm
    & $gxx @flags -o pc\pc_mem_test.exe pc\pc_mem_test.cpp main\feng_memory.cpp -Imain -lm
    & $gxx @flags -o pc\pc_sample_test.exe pc\pc_sample_test.cpp main\feng_sample.cpp main\feng_quant.cpp -Imain -lm
    & $gxx @flags -o pc\pc_calc_ask.exe pc\pc_calc_ask.cpp main\feng_calc.cpp -Imain -lm
    & $gxx @flags -o pc\pc_mem_ask.exe pc\pc_mem_ask.cpp main\feng_memory.cpp -Imain -lm
    & $gxx @flags -o pc\pc_rand_ask.exe pc\pc_rand_ask.cpp main\feng_tools.cpp -Imain -lm
    & $gxx @flags -o pc\pc_time_ask.exe pc\pc_time_ask.cpp main\feng_tools.cpp -Imain -lm
    # 5) fp32 路径一致性检查（对 PyTorch 参考 logits；默认配置，不带 KV 量化）
    & $gxx @flags -o pc\pc_check.exe pc\pc_check.cpp main\feng_model.cpp main\feng_llm.cpp `
        main\feng_quant.cpp main\feng_smp.cpp main\feng_tokenizer.cpp -Imain -lm
    if ($LASTEXITCODE -ne 0) { throw "g++ 编译失败" }
    Write-Host "已生成 pc\pc_chat_q2b8.exe / pc_kv_suite_q2b8.exe / pc_kv_suite_i8.exe / pc_mt_suite.exe / pc_check.exe / pc_calc_test.exe / pc_tools_test.exe / pc_mem_test.exe / pc_sample_test.exe / pc_calc_ask.exe / pc_mem_ask.exe / pc_rand_ask.exe / pc_time_ask.exe" -ForegroundColor Green
    Write-Host "跑一下 tool 单测：" -ForegroundColor Cyan
    & pc\pc_calc_test.exe | Select-Object -Last 2
    & pc\pc_tools_test.exe | Select-Object -Last 2
    & pc\pc_mem_test.exe | Select-Object -Last 2
    & pc\pc_sample_test.exe | Select-Object -Last 2
}
finally {
    Pop-Location
}
