# 编译 PC 端引擎运行时（带 tool）。
#
# Windows 默认用 **MSVC + C++23**（cl，/std:c++latest；MSVC 没有单独的 /std:c++23
# 选项，c++latest 就是它的 C++23 模式）：
#   .\build_pc_chat.ps1
# 备用路径（MinGW g++ -std=c++23）：
#   .\build_pc_chat.ps1 -MinGW
# CUDA 版见 build_pc_cuda.ps1（同样走 MSVC + nvcc）。
#
# 产物（两种工具链输出同名 exe）：
#   pc_chat_q2b8 / pc_kv_suite_q2b8 / pc_kv_suite_i8 / pc_mt_suite / pc_check
#   pc_calc_test / pc_tools_test / pc_mem_test / pc_sample_test
#   pc_calc_ask / pc_mem_ask / pc_rand_ask / pc_time_ask
param([switch]$MinGW)
$ErrorActionPreference = "Stop"

Push-Location $PSScriptRoot
try {
    $base = @(
        "main\feng_model.cpp", "main\feng_llm.cpp", "main\feng_quant.cpp",
        "main\feng_smp.cpp", "main\feng_tokenizer.cpp", "main\feng_sample.cpp",
        "main\feng_calc.cpp", "main\feng_tools.cpp", "main\feng_memory.cpp"
    )
    $q2Drivers = @(
        "pc\pc_chat.cpp", "pc\pc_kv_suite.cpp", "pc\pc_mt_suite.cpp",
        "pc\pc_bench.cpp",
        "pc\pc_calc_test.cpp", "pc\pc_tools_test.cpp", "pc\pc_mem_test.cpp",
        "pc\pc_sample_test.cpp", "pc\pc_calc_ask.cpp", "pc\pc_mem_ask.cpp",
        "pc\pc_rand_ask.cpp", "pc\pc_time_ask.cpp"
    )

    function Invoke-Native([string]$exe, [string[]]$argv) {
        $prev = $ErrorActionPreference
        $ErrorActionPreference = "Continue"     # 原生命令的 stderr 不中断脚本
        & $exe @argv 2>&1 | ForEach-Object { Write-Host $_ }   # 转发编译器输出
        $rc = $LASTEXITCODE
        $ErrorActionPreference = $prev
        return $rc
    }

    if (-not $MinGW) {
        # ---------------- MSVC + C++23（默认） ----------------
        if ($env:VSCMD_VER) { } else {
            $vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
            if (-not (Test-Path $vswhere)) {
                throw "找不到 vswhere（需要 Visual Studio Build Tools）；或改用 -MinGW"
            }
            $vsPath = & $vswhere -latest -products * -property installationPath | Select-Object -First 1
            $vcvars = Join-Path $vsPath "VC\Auxiliary\Build\vcvars64.bat"
            if (-not (Test-Path $vcvars)) { throw "找不到 vcvars64.bat：$vcvars" }
            $lines = & cmd /c "`"$vcvars`" >nul 2>&1 && set"
            foreach ($line in $lines) {
                $i = $line.IndexOf('=')
                if ($i -gt 0) { Set-Item -Path ("env:" + $line.Substring(0, $i)) -Value $line.Substring($i + 1) }
            }
            Write-Host "MSVC 环境：$vsPath" -ForegroundColor DarkGray
        }
        $objRoot = Join-Path $PSScriptRoot "pc\msvc_obj"
        $objQ2 = Join-Path $objRoot "q2"
        $objFp32 = Join-Path $objRoot "fp32"
        $objI8 = Join-Path $objRoot "i8"
        New-Item -ItemType Directory -Force -Path $objQ2, $objFp32, $objI8 | Out-Null

        $cxxFlags = @(
            "/nologo", "/c", "/O2", "/std:c++latest", "/utf-8",
            "/EHs-c-", "/GR-", "/D_HAS_EXCEPTIONS=0", "/wd4996", "/Imain",
            "/DFENG_USE_OMP=1", "/openmp"
        )
        $kvQ2 = @("/DFENG_KV_Q2=1", "/DFENG_KV_Q2_BLOCK=8")

        Write-Host "[1/4] cl 编译 q2 组（核心 + 11 个驱动）..." -ForegroundColor Cyan
        if ((Invoke-Native "cl" ($cxxFlags + $kvQ2 + $base + $q2Drivers + "/Fo$objQ2\")) -ne 0) {
            throw "cl 编译失败（q2 组）"
        }
        Write-Host "[2/4] cl 编译 fp32 组（核心 + pc_check）..." -ForegroundColor Cyan
        if ((Invoke-Native "cl" ($cxxFlags + $base + @("pc\pc_check.cpp") + "/Fo$objFp32\")) -ne 0) {
            throw "cl 编译失败（fp32 组）"
        }
        Write-Host "[3/4] cl 编译 i8 组（核心 + pc_kv_suite）..." -ForegroundColor Cyan
        if ((Invoke-Native "cl" ($cxxFlags + @("/DFENG_KV_INT8=1") + $base + @("pc\pc_kv_suite.cpp") +
                "/Fo$objI8\")) -ne 0) {
            throw "cl 编译失败（i8 组）"
        }

        Write-Host "[4/4] 链接 13 个产物 ..." -ForegroundColor Cyan
        function Get-Objs([string]$dir, [string[]]$sources) {
            $sources | ForEach-Object { Join-Path $dir ([IO.Path]::GetFileNameWithoutExtension($_) + ".obj") }
        }
        $q2Core = Get-Objs $objQ2 $base
        foreach ($d in $q2Drivers) {
            $name = [IO.Path]::GetFileNameWithoutExtension($d)
            $out = switch ($name) {
                "pc_chat"    { "pc\pc_chat_q2b8.exe" }
                "pc_kv_suite" { "pc\pc_kv_suite_q2b8.exe" }
                "pc_mt_suite" { "pc\pc_mt_suite.exe" }
                default      { "pc\$name.exe" }
            }
            if ((Invoke-Native "link" (@("/nologo", ("/OUT:" + (Join-Path $PSScriptRoot $out)),
                    (Join-Path $objQ2 ("{0}.obj" -f $name)) ) + $q2Core)) -ne 0) {
                throw "链接失败：$name"
            }
        }
        if ((Invoke-Native "link" (@("/nologo",
                ("/OUT:" + (Join-Path $PSScriptRoot "pc\pc_check.exe")),
                (Join-Path $objFp32 "pc_check.obj")) + (Get-Objs $objFp32 $base))) -ne 0) {
            throw "链接失败：pc_check"
        }
        if ((Invoke-Native "link" (@("/nologo",
                ("/OUT:" + (Join-Path $PSScriptRoot "pc\pc_kv_suite_i8.exe")),
                (Join-Path $objI8 "pc_kv_suite.obj")) + (Get-Objs $objI8 $base))) -ne 0) {
            throw "链接失败：pc_kv_suite_i8"
        }
        Write-Host "已生成 13 个产物（MSVC /std:c++latest）" -ForegroundColor Green
    } else {
        # ---------------- MinGW g++ -std=c++23（备用） ----------------
        $gxx = $env:FENG_GXX
        if (-not $gxx) { $gxx = "g++" }
        $flags = @("-std=c++23", "-fno-exceptions", "-fno-rtti", "-fno-threadsafe-statics",
                   "-O2", "-DFENG_USE_OMP=1", "-fopenmp")
        & $gxx @flags -DFENG_KV_Q2=1 -DFENG_KV_Q2_BLOCK=8 -o pc\pc_bench.exe pc\pc_bench.cpp @base -Imain -lm
        & $gxx @flags -DFENG_KV_Q2=1 -DFENG_KV_Q2_BLOCK=8 -o pc\pc_chat_q2b8.exe pc\pc_chat.cpp @base -Imain -lm
        & $gxx @flags -DFENG_KV_Q2=1 -DFENG_KV_Q2_BLOCK=8 -o pc\pc_kv_suite_q2b8.exe pc\pc_kv_suite.cpp @base -Imain -lm
        & $gxx @flags -DFENG_KV_INT8=1 -o pc\pc_kv_suite_i8.exe pc\pc_kv_suite.cpp @base -Imain -lm
        & $gxx @flags -DFENG_KV_Q2=1 -DFENG_KV_Q2_BLOCK=8 -o pc\pc_mt_suite.exe pc\pc_mt_suite.cpp @base -Imain -lm
        & $gxx @flags -o pc\pc_calc_test.exe pc\pc_calc_test.cpp main\feng_calc.cpp -Imain -lm
        & $gxx @flags -o pc\pc_tools_test.exe pc\pc_tools_test.cpp main\feng_tools.cpp -Imain -lm
        & $gxx @flags -o pc\pc_mem_test.exe pc\pc_mem_test.cpp main\feng_memory.cpp -Imain -lm
        & $gxx @flags -o pc\pc_sample_test.exe pc\pc_sample_test.cpp main\feng_sample.cpp main\feng_quant.cpp -Imain -lm
        & $gxx @flags -o pc\pc_calc_ask.exe pc\pc_calc_ask.cpp main\feng_calc.cpp -Imain -lm
        & $gxx @flags -o pc\pc_mem_ask.exe pc\pc_mem_ask.cpp main\feng_memory.cpp -Imain -lm
        & $gxx @flags -o pc\pc_rand_ask.exe pc\pc_rand_ask.cpp main\feng_tools.cpp -Imain -lm
        & $gxx @flags -o pc\pc_time_ask.exe pc\pc_time_ask.cpp main\feng_tools.cpp -Imain -lm
        & $gxx @flags -o pc\pc_check.exe pc\pc_check.cpp main\feng_model.cpp main\feng_llm.cpp `
            main\feng_quant.cpp main\feng_smp.cpp main\feng_tokenizer.cpp -Imain -lm
        if ($LASTEXITCODE -ne 0) { throw "g++ 编译失败" }
        Write-Host "已生成 13 个产物（MinGW g++ -std=c++23）" -ForegroundColor Green
    }

    Write-Host "跑一下 tool 单测：" -ForegroundColor Cyan
    & pc\pc_calc_test.exe | Select-Object -Last 1
    & pc\pc_tools_test.exe | Select-Object -Last 1
    & pc\pc_mem_test.exe | Select-Object -Last 1
    & pc\pc_sample_test.exe | Select-Object -Last 1
} finally {
    Pop-Location
}
