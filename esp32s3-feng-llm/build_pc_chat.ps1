# 编译 PC 端 C 引擎运行时（带 tool）：pc_chat / pc_kv_suite / pc_calc_test
# 用法： .\build_pc_chat.ps1        （可选 $env:FENG_GCC 指定 gcc 路径）
$ErrorActionPreference = "Stop"
$gcc = $env:FENG_GCC
if (-not $gcc) { $gcc = "gcc" }          # 例如 F:\msys2\ucrt64\bin\gcc.exe

Push-Location $PSScriptRoot
try {
    $base = @(
        "main\feng_model.c", "main\feng_llm.c", "main\feng_quant.c",
        "main\feng_smp.c", "main\feng_tokenizer.c",
        "main\feng_calc.c", "main\feng_tools.c"
    )
    # 1) 聊天运行时（q2 block8 / 2048 上下文）
    & $gcc -O2 -DFENG_KV_Q2=1 -DFENG_KV_Q2_BLOCK=8 -o pc\pc_chat_q2b8.exe `
        pc\pc_chat.c @base -Imain -lm
    # 2) 板端代理套件（32 题矩阵，q2 block8；算式题走 tool）
    & $gcc -O2 -DFENG_KV_Q2=1 -DFENG_KV_Q2_BLOCK=8 -o pc\pc_kv_suite_q2b8.exe `
        pc\pc_kv_suite.c @base -Imain -lm
    & $gcc -O2 -DFENG_KV_INT8=1 -o pc\pc_kv_suite_i8.exe `
        pc\pc_kv_suite.c @base -Imain -lm
    # 3) tool 单测（算式识别/求值/中文句式 + 时间/随机数语义）
    & $gcc -O2 -o pc\pc_calc_test.exe pc\pc_calc_test.c main\feng_calc.c -Imain -lm
    & $gcc -O2 -o pc\pc_tools_test.exe pc\pc_tools_test.c main\feng_tools.c -Imain -lm
    if ($LASTEXITCODE -ne 0) { throw "gcc 编译失败" }
    Write-Host "已生成 pc\pc_chat_q2b8.exe / pc_kv_suite_q2b8.exe / pc_kv_suite_i8.exe / pc_calc_test.exe / pc_tools_test.exe" -ForegroundColor Green
    Write-Host "跑一下 tool 单测：" -ForegroundColor Cyan
    & pc\pc_calc_test.exe | Select-Object -Last 2
    & pc\pc_tools_test.exe | Select-Object -Last 2
}
finally {
    Pop-Location
}
