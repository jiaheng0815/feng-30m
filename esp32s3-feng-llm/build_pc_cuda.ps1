# 构建 CUDA 版 PC 引擎（MSVC cl/link + nvcc）。只在装好 VS Build Tools 与 CUDA Toolkit
# 的机器上使用；不带 CUDA 的 MinGW/g++ 构建见 build_pc_chat.ps1。
#
# 产出（不覆盖 MinGW 版产物）：
#   pc\pc_chat_cuda.exe      聊天运行时（q2 KV，GEMV 走 GPU）
#   pc\pc_bench_cuda.exe     速度基准（q2 KV）
#   pc\pc_kv_suite_cuda.exe  32 题矩阵套件（q2 KV）
#   pc\pc_mt_suite_cuda.exe  多轮回归套件（q2 KV）
#   pc\pc_check_cuda.exe     fp32 KV 一致性自检（与 CPU 版 pc_check 同口径）
#
# 用法： .\build_pc_cuda.ps1 [-CudaPath <CUDA 根目录>] [-Arch sm_120]
param(
    [string]$CudaPath = $env:CUDA_PATH,
    [string]$Arch = "sm_120"
)
$ErrorActionPreference = "Stop"

# cl/link 需要 VS 开发者环境（INCLUDE/LIB/PATH）。若尚未加载，用 vcvars64.bat 自动导入。
function Import-VcVars {
    if ($env:VSCMD_VER) { return }        # 已在 VS 开发者终端里
    $vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
    if (-not (Test-Path $vswhere)) { throw "找不到 vswhere（需要 Visual Studio Build Tools）" }
    $vsPath = & $vswhere -latest -products * -property installationPath | Select-Object -First 1
    $vcvars = Join-Path $vsPath "VC\Auxiliary\Build\vcvars64.bat"
    if (-not (Test-Path $vcvars)) { throw "找不到 vcvars64.bat：$vcvars" }
    $lines = & cmd /c "`"$vcvars`" >nul 2>&1 && set"
    foreach ($line in $lines) {
        $i = $line.IndexOf('=')
        if ($i -gt 0) { Set-Item -Path ("env:" + $line.Substring(0, $i)) -Value $line.Substring($i + 1) }
    }
    Write-Host "已加载 VS 开发者环境：$vsPath" -ForegroundColor DarkGray
}
Import-VcVars

if (-not $CudaPath -or -not (Test-Path (Join-Path $CudaPath "bin\nvcc.exe"))) {
    $root = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA"
    if (Test-Path $root) {
        $CudaPath = (Get-ChildItem $root -Directory | Sort-Object Name -Descending |
                     Select-Object -First 1).FullName
    }
}
$nvcc = Join-Path $CudaPath "bin\nvcc.exe"
if (-not (Test-Path $nvcc)) { throw "找不到 nvcc：$nvcc（用 -CudaPath 指定 CUDA 安装目录）" }
foreach ($tool in @("cl", "link")) {
    if (-not (Get-Command $tool -ErrorAction SilentlyContinue)) {
        throw "找不到 $tool（请在 VS 开发者环境 / 已配置 PATH 的终端里运行）"
    }
}

Push-Location $PSScriptRoot
try {
    $objRoot = Join-Path $PSScriptRoot "pc\cuda_obj"
    $objQ2 = Join-Path $objRoot "q2"
    $objFp32 = Join-Path $objRoot "fp32"
    New-Item -ItemType Directory -Force -Path $objQ2, $objFp32 | Out-Null

    $baseFlags = @(
        "/nologo", "/c", "/O2", "/std:c++latest", "/utf-8",
        "/EHs-c-", "/GR-", "/D_HAS_EXCEPTIONS=0", "/wd4996",
        "/DFENG_USE_CUDA=1", "/Imain"
    )
    $kvQ2 = @("/DFENG_KV_Q2=1", "/DFENG_KV_Q2_BLOCK=8")
    $core = @(
        "main\feng_model.cpp", "main\feng_llm.cpp", "main\feng_quant.cpp",
        "main\feng_smp.cpp", "main\feng_tokenizer.cpp", "main\feng_sample.cpp",
        "main\feng_calc.cpp", "main\feng_tools.cpp", "main\feng_memory.cpp"
    )
    $driversQ2 = @(
        "pc\pc_chat.cpp", "pc\pc_bench.cpp", "pc\pc_kv_suite.cpp", "pc\pc_mt_suite.cpp"
    )
    $continue = $ErrorActionPreference
    $ErrorActionPreference = "Continue"     # 原生命令的 stderr 不能中断脚本

    Write-Host "[1/4] cl 编译 q2 组（核心 + 4 个驱动）..." -ForegroundColor Cyan
    & cl @baseFlags @kvQ2 @core @driversQ2 "/Fo$objQ2\"
    if ($LASTEXITCODE -ne 0) { throw "cl 编译失败（q2 组）" }

    Write-Host "[2/4] cl 编译 fp32 组（核心 + pc_check）..." -ForegroundColor Cyan
    & cl @baseFlags @core "pc\pc_check.cpp" "/Fo$objFp32\"
    if ($LASTEXITCODE -ne 0) { throw "cl 编译失败（fp32 组）" }

    Write-Host "[3/4] nvcc 编译 feng_cuda.cu（-fmad=false 保证与 CPU 同序）..." -ForegroundColor Cyan
    $cudaObj = Join-Path $objRoot "feng_cuda.obj"
    & $nvcc "-arch=$Arch" -O3 -std=c++20 -fmad=false `
        -Xcompiler "/utf-8 /O2 /EHs-c-" -DFENG_USE_CUDA=1 -Imain `
        -c main\feng_cuda.cu -o $cudaObj
    if ($LASTEXITCODE -ne 0) { throw "nvcc 编译失败" }

    Write-Host "[4/4] 链接 CUDA 版运行时 ..." -ForegroundColor Cyan
    $cudaLibDir = Join-Path $CudaPath "lib\x64"
    $cudart = Join-Path $cudaLibDir "cudart.lib"
    $linkCommon = @("/nologo", $cudart, ("/LIBPATH:" + $cudaLibDir))
    function Get-Objs([string]$dir, [string[]]$sources) {
        $sources | ForEach-Object {
            Join-Path $dir ([IO.Path]::GetFileNameWithoutExtension($_) + ".obj")
        }
    }
    foreach ($d in $driversQ2) {
        $name = [IO.Path]::GetFileNameWithoutExtension($d)
        $out = Join-Path $PSScriptRoot ("pc\{0}_cuda.exe" -f $name)
        & link @linkCommon ("/OUT:" + $out) (Join-Path $objQ2 ("{0}.obj" -f $name)) `
            @(Get-Objs $objQ2 $core) $cudaObj
        if ($LASTEXITCODE -ne 0) { throw "链接失败：$name" }
        Write-Host ("  已生成 " + (Split-Path $out -Leaf)) -ForegroundColor Green
    }
    $out = Join-Path $PSScriptRoot "pc\pc_check_cuda.exe"
    & link @linkCommon ("/OUT:" + $out) (Join-Path $objFp32 "pc_check.obj") `
        @(Get-Objs $objFp32 $core) $cudaObj
    if ($LASTEXITCODE -ne 0) { throw "链接失败：pc_check" }
    Write-Host "  已生成 pc_check_cuda.exe" -ForegroundColor Green

    $ErrorActionPreference = $continue
} finally {
    Pop-Location
}
