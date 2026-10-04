# 编译 PC 端 C 引擎运行时（带 tool）：pc_chat / pc_kv_suite / pc_calc_test
# 用法： .\build_pc_chat.ps1        （可选 $env:FENG_GCC 指定 gcc 路径）
$ErrorActionPreference = "Stop"
$gcc = $env:FENG_GCC
if (-not $gcc) { $gcc = "gcc" }          # 例如 <你的 MSYS2>\ucrt64\bin\gcc.exe

Push-Location $PSScriptRoot
try {
    $base = @(
        "main\feng_model.c", "main\feng_llm.c", "main\feng_quant.c",
        "main\feng_smp.c", "main\feng_tokenizer.c", "main\feng_sample.c",
        "main\feng_calc.c", "main\feng_tools.c", "main\feng_memory.c"
    )
    # 1) 聊天运行时（q2 block8 / 2048 上下文）
    & $gcc -O2 -DFENG_KV_Q2=1 -DFENG_KV_Q2_BLOCK=8 -o pc\pc_chat_q2b8.exe `
        pc\pc_chat.c @base -Imain -lm
    # 2) 板端代理套件（32 题矩阵，q2 block8；算式题走 tool）
    & $gcc -O2 -DFENG_KV_Q2=1 -DFENG_KV_Q2_BLOCK=8 -o pc\pc_kv_suite_q2b8.exe `
        pc\pc_kv_suite.c @base -Imain -lm
    & $gcc -O2 -DFENG_KV_INT8=1 -o pc\pc_kv_suite_i8.exe `
        pc\pc_kv_suite.c @base -Imain -lm
    # 3) 多轮回归套件（报名字→问身份/问名字、同类取新、12 题连续记忆）
    & $gcc -O2 -DFENG_KV_Q2=1 -DFENG_KV_Q2_BLOCK=8 -o pc\pc_mt_suite.exe `
        pc\pc_mt_suite.c @base -Imain -lm
    # 4) tool 单测（算式识别/求值/中文句式 + 时间/随机数语义）
    & $gcc -O2 -o pc\pc_calc_test.exe pc\pc_calc_test.c main\feng_calc.c -Imain -lm
    & $gcc -O2 -o pc\pc_tools_test.exe pc\pc_tools_test.c main\feng_tools.c -Imain -lm
    & $gcc -O2 -o pc\pc_mem_test.exe pc\pc_mem_test.c main\feng_memory.c -Imain -lm
    & $gcc -O2 -o pc\pc_sample_test.exe pc\pc_sample_test.c main\feng_sample.c main\feng_quant.c -Imain -lm
    & $gcc -O2 -o pc\pc_calc_ask.exe pc\pc_calc_ask.c main\feng_calc.c -Imain -lm
    & $gcc -O2 -o pc\pc_mem_ask.exe pc\pc_mem_ask.c main\feng_memory.c -Imain -lm
    & $gcc -O2 -o pc\pc_rand_ask.exe pc\pc_rand_ask.c main\feng_tools.c -Imain -lm
    & $gcc -O2 -o pc\pc_time_ask.exe pc\pc_time_ask.c main\feng_tools.c -Imain -lm
    # 5) fp32 路径一致性检查（对 PyTorch 参考 logits；默认配置，不带 KV 量化）
    & $gcc -O2 -o pc\pc_check.exe pc\pc_check.c main\feng_model.c main\feng_llm.c `
        main\feng_quant.c main\feng_smp.c main\feng_tokenizer.c -Imain -lm
    if ($LASTEXITCODE -ne 0) { throw "gcc 编译失败" }
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
