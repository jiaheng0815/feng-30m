#!/usr/bin/env pwsh
<#
.SYNOPSIS
  数据生成 → 模型训练的显卡交接：停服务、等显存真的回来、给出放行判定。

.DESCRIPTION
  为什么值得单独写一个脚本：Windows 用的是 WDDM 显存模型，「进程退出」和
  「显存归还」之间有一段时间差（驱动异步回收，实测几秒到几十秒）。杀完进程
  立刻开训练，可能撞上还没释放的显存——表现是 PyTorch OOM，或者更隐蔽地
  自己降配跑出跟预期不一样的结果。

  本脚本按顺序做四件事：

    1. 确认生成器已经跑完（data/ 下两份 jsonl 加起来达到提示词池规模），
       避免误杀还在写的任务。没跑完时除非 -Force，否则直接退出。
    2. 停掉生成器与教师服务（ninfer-serve）。
    3. 轮询 nvidia-smi，等显存回落到阈值以下。
    4. 明确报告「可以开训练」或「还有残留，别开」，并列出还在占卡的进程。

  注意：Windows 桌面本身常驻 1~2 GB 显存（dwm / 浏览器 / 微信 / 壁纸引擎…），
  所以目标是「回落到桌面水平」而不是 0。

.EXAMPLE
  pwsh tools/switch_to_train.ps1
      生成器已跑完 → 停服务 → 等显存回落 → 报告

.EXAMPLE
  pwsh tools/switch_to_train.ps1 -Force -TimeoutSec 180
      不检查数据是否跑完，直接停（调试用），最多等 3 分钟
#>
param(
    # 跳过"数据是否跑完"的检查（默认跑完才能切）
    [switch]$Force,
    # 等显存回落的最长秒数
    [int]$TimeoutSec = 120,
    # 已用显存降到这个值以下就算干净（桌面常驻约 1.5~2 GB）
    [int]$IdleLevelMiB = 2500
)

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$prompts = Join-Path $root 'data/v4_prompts.jsonl'
$outJsonl = Join-Path $root 'data/v4_teacher.jsonl'
$rejJsonl = Join-Path $root 'data/v4_rejects.jsonl'

function Get-UsedMiB {
    $v = (& nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits) -join ''
    return [int]$v.Trim()
}

function Get-GpuProcs {
    # 只列"真在算"的进程（Windows 图形进程的显存读数是 N/A，不参与判断）
    & nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader |
        Where-Object { $_ -notmatch 'N/A' -and $_.Trim() }
}

Write-Host '=== 显卡交接：生成 → 训练 ===' -ForegroundColor Cyan
$before = Get-UsedMiB
Write-Host ("  当前已用 {0:N0} MiB" -f $before)

# ---------- 1. 生成器跑完了吗 ----------
$gen = Get-CimInstance Win32_Process -Filter "Name='python.exe'" -ErrorAction SilentlyContinue |
       Where-Object { $_.CommandLine -and $_.CommandLine -like '*v4_teacher_gen*' }

if ($gen) {
    $nPrompt = (Get-Content $prompts -ErrorAction SilentlyContinue | Measure-Object -Line).Lines
    $nOk = (Get-Content $outJsonl -ErrorAction SilentlyContinue | Measure-Object -Line).Lines
    $nRej = (Get-Content $rejJsonl -ErrorAction SilentlyContinue | Measure-Object -Line).Lines
    $done = $nOk + $nRej
    $pct = if ($nPrompt) { $done / $nPrompt * 100 } else { 0 }

    Write-Host ("  生成器 PID {0} 仍在运行：已完成 {1:N0}/{2:N0}（{3:N1}%）" -f `
                $gen.ProcessId, $done, $nPrompt, $pct)

    if ($done -lt $nPrompt -and -not $Force) {
        Write-Host '  ✗ 数据还没跑完——现在切会丢掉剩下的提示词。' -ForegroundColor Yellow
        Write-Host '    确实要强行切（比如已决定提前收工）：加 -Force' -ForegroundColor Yellow
        exit 1
    }
    if ($Force -and $done -lt $nPrompt) {
        Write-Host '  ! -Force：数据未跑完，仍按你的要求继续' -ForegroundColor Yellow
    }

    Write-Host ("  停止生成器 {0} …" -f $gen.ProcessId)
    Stop-Process -Id $gen.ProcessId -Force -ErrorAction SilentlyContinue
} else {
    Write-Host '  生成器：未在运行（可能已经跑完）'
}

# ---------- 2. 停教师服务 ----------
$nin = Get-Process -Name 'ninfer-serve' -ErrorAction SilentlyContinue
if ($nin) {
    foreach ($p in $nin) {
        Write-Host ("  停止教师服务 {0}…" -f $p.Id)
        Stop-Process -Id $p.Id -Force -ErrorAction SilentlyContinue
    }
} else {
    Write-Host '  教师服务：未在运行'
}

# ---------- 3. 等显存真的回来 ----------
Write-Host '  等待驱动回收显存（WDDM 是异步的，别急着开训练）…'
$sw = [Diagnostics.Stopwatch]::StartNew()
$used = Get-UsedMiB
while ($sw.Elapsed.TotalSeconds -lt $TimeoutSec) {
    Start-Sleep -Milliseconds 1500
    $used = Get-UsedMiB
    if ($used -le $IdleLevelMiB) { break }
    Write-Host ("    {0,5:N0} MiB …（已等 {1:N0}s）" -f $used, $sw.Elapsed.TotalSeconds)
}
$sw.Stop()

# ---------- 4. 判定 ----------
Write-Host ''
Write-Host ("  已用 {0:N0} MiB（交接前 {1:N0}，释放 {2:N0}），耗时 {3:N1}s" -f `
            $used, $before, ($before - $used), $sw.Elapsed.TotalSeconds)

$residual = Get-GpuProcs
if ($residual) {
    Write-Host '  仍在用显存的进程：' -ForegroundColor Yellow
    $residual | ForEach-Object { Write-Host "    $_" -ForegroundColor Yellow }
}

if ($used -le $IdleLevelMiB) {
    Write-Host ''
    Write-Host '  ✓ 可以开训练了。' -ForegroundColor Green
    Write-Host '    下一步：python scripts\v4_build_sft.py --out data\v4_sft.jsonl `'
    Write-Host '                                --val-out data\v4_val.jsonl --val-ratio 0.02'
    exit 0
} else {
    Write-Host ''
    Write-Host ("  ✗ 仍有 {0:N0} MiB 未释放（阈值 {1:N0}）——先别开训练。" -f $used, $IdleLevelMiB) -ForegroundColor Red
    Write-Host '    排查：' -ForegroundColor Red
    Write-Host '      · 上面列出的残留进程，逐个确认是不是该退的' -ForegroundColor Red
    Write-Host '      · 浏览器 / 壁纸引擎 / 视频播放器很吃显存，可以先关掉' -ForegroundColor Red
    Write-Host '      · 再等一会儿，WDDM 偶尔要几十秒才彻底回收' -ForegroundColor Red
    Write-Host '      · 重跑本脚本复查：pwsh tools\switch_to_train.ps1 -Force' -ForegroundColor Red
    exit 2
}
