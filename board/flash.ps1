<#
  一键烧录 feng-30m 到 ESP32-S3 (N32R16V)
  用法:  .\flash.ps1 -Port COM20 [-SkipBuild] [-Monitor] [-ModelDir .\model_export_4.0-embed]
         [-KvMode q2|int8] [-EspIdfPath <esp-idf 目录>] [-EspToolPy <python.exe>]

  工具链不再写死盘符：优先用参数，其次用环境变量 IDF_PATH / IDF_TOOLS_PATH /
  ESPTOOL_PY；都没有时给出提示（若当前终端已经跑过 export.ps1 可忽略）。
#>
param(
    [Parameter(Mandatory = $true)][string]$Port,
    [switch]$SkipBuild,
    [switch]$Monitor,
    [string]$ModelDir = "",
    [ValidateSet("q2", "int8")]
    [string]$KvMode = "q2",
    [string]$EspIdfPath = $env:IDF_PATH,
    [string]$EspToolPy = $env:ESPTOOL_PY
)

# 分区偏移/大小必须与 partitions.csv 一致（model / tokdata）。
$ModelOffset = 0x110000
$ModelSize   = 0xEF0000
$TokOffset   = 0x1000000
$TokSize     = 0x80000

$ErrorActionPreference = "Stop"

function Invoke-Checked {
    param([string]$Exe, [string[]]$Argv, [string]$What)
    & $Exe @Argv
    if ($LASTEXITCODE -ne 0) {
        throw "$What 失败（exit $LASTEXITCODE）：$Exe $($Argv -join ' ')"
    }
}

if ($ModelDir -eq "") {
    $candidates = @(
        (Join-Path $PSScriptRoot "model_export_4.0-embed"),
        (Join-Path $PSScriptRoot "model_export_v4"),
        (Join-Path $PSScriptRoot "model_export_20261017D")
    )
    foreach ($c in $candidates) {
        if (Test-Path (Join-Path $c "model.bin")) { $ModelDir = $c; break }
    }
    if ($ModelDir -eq "") {
        throw "找不到模型导出目录：请用 -ModelDir 指定（board\tools\export_model.py 的产出）"
    }
}

# 先校验文件与大小，再动 flash：避免刷完固件才发现模型缺失/超分区。
$model = Join-Path $ModelDir "model.bin"
$tok   = Join-Path $ModelDir "tokenizer.bin"
if (-not (Test-Path $model) -or -not (Test-Path $tok)) {
    throw "model.bin / tokenizer.bin not found in $ModelDir（先跑 board\tools\export_model.py）"
}
$modelLen = (Get-Item $model).Length
$tokLen   = (Get-Item $tok).Length
if ($modelLen -gt $ModelSize) {
    throw "model.bin $modelLen B 超过分区 0xEF0000（$ModelSize B），会覆盖 tokdata"
}
if ($tokLen -gt $TokSize) {
    throw "tokenizer.bin $tokLen B 超过 tokdata 分区 0x80000（$TokSize B）"
}
Write-Host ("model.bin {0:N0} B（余 {1:N0} B）；tokenizer.bin {2:N0} B（余 {3:N0} B）" -f `
    $modelLen, ($ModelSize - $modelLen), $tokLen, ($TokSize - $tokLen)) -ForegroundColor DarkGray

if (Test-Path $EspIdfPath) {
    & (Join-Path $EspIdfPath "export.ps1") | Out-Null
}
else {
    Write-Warning "未找到 esp-idf（-EspIdfPath 或环境变量 IDF_PATH）；若当前 shell 已执行过 export.ps1 可忽略。"
}

$py = if ($EspToolPy) { $EspToolPy } else { "python" }
$kvArg = if ($KvMode -eq "q2") { "-DFENG_USE_Q2_KV=ON" } else { "-DFENG_USE_Q2_KV=OFF" }

Push-Location $PSScriptRoot
try {
    if (-not $SkipBuild) {
        Write-Host "== build（KV=$KvMode）==" -ForegroundColor Cyan
        Invoke-Checked "idf.py" @($kvArg, "build") "idf.py build"
    }
    Write-Host "== flash firmware + partition table ==" -ForegroundColor Cyan
    Invoke-Checked "idf.py" @("--port", $Port, "flash") "idf.py flash"

    Write-Host "== flash model partitions ==" -ForegroundColor Cyan
    Invoke-Checked $py @("-m", "esptool", "--chip", "esp32s3", "--port", $Port,
        "--baud", "921600", "write_flash", "$ModelOffset", "$model", "$TokOffset", "$tok") `
        "esptool write_flash"

    Write-Host "== done ==" -ForegroundColor Green
    if ($Monitor) {
        Invoke-Checked "idf.py" @("--port", $Port, "monitor") "idf.py monitor"
    }
}
finally {
    Pop-Location
}
