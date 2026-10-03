<#
  一键烧录 feng-30m 到 ESP32-S3 (R16N32)
  用法:  .\flash.ps1 -Port COM20 [-SkipBuild] [-Monitor] [-ModelDir .\model_export_v3]
         [-EspIdfPath <esp-idf 目录>] [-EspToolPy <python.exe>]

  工具链不再写死盘符：优先用参数，其次用环境变量 IDF_PATH / IDF_TOOLS_PATH /
  ESPTOOL_PY；都没有时给出提示（若当前终端已经跑过 export.ps1 可忽略）。
#>
param(
    [Parameter(Mandatory = $true)][string]$Port,
    [switch]$SkipBuild,
    [switch]$Monitor,
    [string]$ModelDir = "$PSScriptRoot\model_export_v3",
    [string]$EspIdfPath = $env:IDF_PATH,
    [string]$EspToolPy = $env:ESPTOOL_PY
)

# 分区偏移必须与 partitions.csv 一致（model / tokdata），否则会出现 MMU fault 或读不到模型
$ModelOffset = 0x110000
$TokOffset   = 0x1000000

$ErrorActionPreference = "Stop"

if (Test-Path $EspIdfPath) {
    & (Join-Path $EspIdfPath "export.ps1") | Out-Null
}
else {
    Write-Warning "未找到 esp-idf（-EspIdfPath 或环境变量 IDF_PATH）；若当前 shell 已执行过 export.ps1 可忽略。"
}

$py = if ($EspToolPy) { $EspToolPy } else { "python" }

Push-Location $PSScriptRoot
try {
    if (-not $SkipBuild) {
        Write-Host "== build ==" -ForegroundColor Cyan
        idf.py build
    }
    Write-Host "== flash firmware + partition table ==" -ForegroundColor Cyan
    idf.py -p $Port flash

    Write-Host "== flash model partitions ==" -ForegroundColor Cyan
    $model = Join-Path $ModelDir "model.bin"
    $tok   = Join-Path $ModelDir "tokenizer.bin"
    if (-not (Test-Path $model) -or -not (Test-Path $tok)) {
        throw "model.bin / tokenizer.bin not found in $ModelDir (run tools\export_model.py first)"
    }
    & $py -m esptool --chip esp32s3 -p $Port -b 921600 write_flash `
        $ModelOffset $model `
        $TokOffset $tok

    Write-Host "== done ==" -ForegroundColor Green
    if ($Monitor) {
        idf.py -p $Port monitor
    }
}
finally {
    Pop-Location
}
