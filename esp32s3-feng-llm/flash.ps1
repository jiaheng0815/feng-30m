<#
  一键烧录 feng-30m 到 ESP32-S3 (R16N32)
  用法:  .\flash.ps1 -Port COM5 [-SkipBuild] [-ModelDir .\model_export]
#>
param(
    [Parameter(Mandatory = $true)][string]$Port,
    [switch]$SkipBuild,
    [string]$ModelDir = "$PSScriptRoot\model_export"
)

$ErrorActionPreference = "Stop"
$env:IDF_TOOLS_PATH = "F:\Espressif"
& "F:\esp\v5.5.5\esp-idf\export.ps1" | Out-Null

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
    python -m esptool --chip esp32s3 -p $Port -b 921600 write_flash `
        0x310000 $model `
        0x1A10000 $tok

    Write-Host "== done: open monitor ==" -ForegroundColor Green
    idf.py -p $Port monitor
}
finally {
    Pop-Location
}
