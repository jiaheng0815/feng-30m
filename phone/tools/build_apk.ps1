# 一键构建 feng-30m 的 Android APK。
#
# 它做四件事：
#   1) 找到 JDK（AGP 8.x 只支持 17~21）与 Android SDK，缺 SDK 就下 cmdline-tools 装；
#   2) 把模型（model.bin / tokenizer.bin）拷进 app/src/main/assets/ —— 仓库里不入库任何权
#      重，所以每次构建都从模型导出目录现拷；
#   3) 按需下载 Gradle 并写 local.properties（SDK 路径是本机路径，不入库）；
#   4) 跑 gradle assembleRelease，最后打印 APK 路径。
#
# 路径解析优先级（与仓库其它脚本一致）：命令行参数 > 环境变量 > scripts/local_paths.json > 脚本位置推导。
#
# 用法：
#   .\build_apk.ps1                                  # 用默认模型导出目录
#   .\build_apk.ps1 -ModelDir <导出目录>              # 指定模型目录
#   .\build_apk.ps1 -SdkDir <SDK> -JdkHome <JDK>     # 手动指定工具链
#   .\build_apk.ps1 -Debug                           # 出 debug 包（可直接 adb install）
param(
    [string]$ModelDir = "",
    [string]$SdkDir = "",
    [string]$JdkHome = "",
    [string]$GradleVersion = "8.9",
    [switch]$Debug,
    [switch]$SkipModel
)

$ErrorActionPreference = "Stop"
$PhoneDir = Split-Path -Parent $PSScriptRoot
$Root = Split-Path -Parent $PhoneDir
$AndroidDir = Join-Path $PhoneDir "android"

function Info($m) { Write-Host "  $m" }
function Step($m) { Write-Host "== $m" -ForegroundColor Cyan }
function Die($m)  { Write-Host "错误：$m" -ForegroundColor Red; exit 1 }

# ---------- 读 local_paths.json（可选，不入库） ----------
$LocalJson = @{}
$lp = Join-Path $Root "scripts\local_paths.json"
if (Test-Path $lp) {
    try { (Get-Content $lp -Raw | ConvertFrom-Json).PSObject.Properties | ForEach-Object { $LocalJson[$_.Name] = $_.Value } }
    catch { Info "local_paths.json 解析失败，忽略" }
}

# ---------- 1) JDK ----------
Step "解析 JDK"
$jdkCandidates = @()
if ($JdkHome) { $jdkCandidates += $JdkHome }
if ($env:FENG_JDK) { $jdkCandidates += $env:FENG_JDK }
if ($LocalJson["jdk_home"]) { $jdkCandidates += $LocalJson["jdk_home"] }
if ($env:JAVA_HOME) { $jdkCandidates += $env:JAVA_HOME }
$jdkCandidates += @(
    "$env:LOCALAPPDATA\Programs\Android Studio\jbr",
    "C:\Program Files\Android\Android Studio\jbr",
    "$env:ProgramFiles\Eclipse Adoptium\jdk-21*",
    "$env:ProgramFiles\Eclipse Adoptium\jdk-17*",
    "$env:ProgramFiles\Microsoft\jdk-21*"
)

$jdk = $null
foreach ($c in $jdkCandidates) {
    if (-not $c) { continue }
    $expanded = Get-Item $c -ErrorAction SilentlyContinue | Select-Object -First 1
    if (-not $expanded) { continue }
    $jhome = $expanded.FullName
    $javac = Join-Path $jhome "bin\javac.exe"
    if (-not (Test-Path $javac)) { continue }
    $ver = (& $javac -version 2>&1) -join " "
    if ($ver -match 'javac (\d+)') {
        $major = [int]$Matches[1]
        if ($major -ge 17 -and $major -le 21) { $jdk = $jhome; Info "用 $jhome（$ver）"; break }
        else { Info "跳过 $jhome（$ver）：AGP 8.x 只支持 JDK 17~21" }
    }
}
if (-not $jdk) {
    Die @"
找不到可用的 JDK 17~21。AGP 8.7 不支持 JDK 24/25/26（你系统里的 java 可能是 26）。
任选一种装上再重跑：
  - 装 Android Studio（自带 JBR 21，本脚本会自动找到）
  - 或装 Temurin 21：https://adoptium.net/temurin/releases/?version=21
  - 或用参数指定：.\build_apk.ps1 -JdkHome <JDK 17~21 目录>
"@
}

# ---------- 2) Android SDK ----------
Step "解析 Android SDK"
$sdkCandidates = @()
if ($SdkDir) { $sdkCandidates += $SdkDir }
if ($env:ANDROID_HOME) { $sdkCandidates += $env:ANDROID_HOME }
if ($env:ANDROID_SDK_ROOT) { $sdkCandidates += $env:ANDROID_SDK_ROOT }
if ($LocalJson["android_sdk"]) { $sdkCandidates += $LocalJson["android_sdk"] }
$sdkCandidates += "$env:LOCALAPPDATA\Android\Sdk"

$sdk = $null
foreach ($c in $sdkCandidates) {
    if (-not $c) { continue }
    if (Test-Path (Join-Path $c "platform-tools")) { $sdk = $c; break }
    if (Test-Path (Join-Path $c "cmdline-tools")) { $sdk = $c; break }
}

if (-not $sdk) {
    $sdk = "$env:LOCALAPPDATA\Android\Sdk"
    Info "没找到 SDK，准备下载 cmdline-tools 到 $sdk"
    $ctDir = Join-Path $sdk "cmdline-tools"
    $zip = Join-Path $env:TEMP "cmdline-tools.zip"
    $urls = @(
        "https://dl.google.com/android/repository/commandlinetools-win-11076708_latest.zip",
        "https://dl.google.com/android/repository/commandlinetools-win-10406996_latest.zip"
    )
    $got = $false
    foreach ($u in $urls) {
        try {
            Info "下载 $u"
            Invoke-WebRequest -Uri $u -OutFile $zip -UseBasicParsing
            $got = $true; break
        } catch { Info "失败，换下一个镜像" }
    }
    if (-not $got) {
        Die "cmdline-tools 下载失败（网络受限？）。请手装 Android Studio，或把已有 SDK 用 -SdkDir 传进来。"
    }
    New-Item -ItemType Directory -Force -Path $ctDir | Out-Null
    Expand-Archive -Path $zip -DestinationPath $ctDir -Force
    Remove-Item $zip -Force
    # 包内是 cmdline-tools/，sdkmanager 要求放在 cmdline-tools/latest/
    $inner = Join-Path $ctDir "cmdline-tools"
    if (Test-Path $inner) {
        $latest = Join-Path $ctDir "latest"
        if (Test-Path $latest) { Remove-Item $latest -Recurse -Force }
        Move-Item $inner $latest
    }
}
Info "SDK = $sdk"

$sdkmanager = Join-Path $sdk "cmdline-tools\latest\bin\sdkmanager.bat"
if (-not (Test-Path $sdkmanager)) {
    $sdkmanager = (Get-ChildItem (Join-Path $sdk "cmdline-tools") -Recurse -Filter "sdkmanager.bat" -ErrorAction SilentlyContinue | Select-Object -First 1).FullName
}
if (-not $sdkmanager) { Die "找不到 sdkmanager.bat（$sdk 里没有 cmdline-tools）" }

Step "安装 SDK 组件（platform 35 / build-tools 35 / ndk 27.3 / cmake / platform-tools）"
$env:JAVA_HOME = $jdk
$env:ANDROID_HOME = $sdk
$env:ANDROID_SDK_ROOT = $sdk
# sdkmanager 第一次会问是否接受许可，用 yes 管道喂进去
& cmd /c "echo y| `"$sdkmanager`" --sdk_root=`"$sdk`" --licenses" 2>&1 | Out-Null
# cmake 必须一起装：app/build.gradle.kts 里 externalNativeBuild 指定了 "3.22.1"，
# 没装的话 AGP 会在配置阶段直接报错。
& cmd /c "echo y| `"$sdkmanager`" --sdk_root=`"$sdk`" `"platform-tools`" `"platforms;android-35`" `"build-tools;35.0.0`" `"ndk;27.3.13750724`" `"cmake;3.22.1`"" 2>&1 |
    Where-Object { $_ -match 'done|Warning|Error|error' } | ForEach-Object { Info $_ }

# ---------- 3) 模型文件 ----------
Step "准备 assets（模型不入库，构建时现拷）"
$assets = Join-Path $AndroidDir "app\src\main\assets"
New-Item -ItemType Directory -Force -Path $assets | Out-Null

if (-not $SkipModel) {
    if (-not $ModelDir) {
        if ($LocalJson["model_export"]) { $ModelDir = $LocalJson["model_export"] }
        else { $ModelDir = Join-Path $Root "esp32s3-feng-llm\model_export_v3_19b6" }
    }
    foreach ($f in @("model.bin", "tokenizer.bin")) {
        $src = Join-Path $ModelDir $f
        if (-not (Test-Path $src)) {
            Die "找不到 $src。用 -ModelDir 指定含 model.bin / tokenizer.bin 的目录（见 phone/README.md）。"
        }
        Copy-Item $src (Join-Path $assets $f) -Force
        $mb = [math]::Round((Get-Item $src).Length / 1MB, 2)
        Info "$f  ($mb MB)"
    }
} else {
    Info "跳过模型拷贝（-SkipModel）"
}

# ---------- 4) Gradle ----------
Step "准备 Gradle $GradleVersion"
$gradleHome = Join-Path $env:USERPROFILE ".gradle-dist\gradle-$GradleVersion"
$gradleBat = Join-Path $gradleHome "bin\gradle.bat"
if (-not (Test-Path $gradleBat)) {
    $zip = Join-Path $env:TEMP "gradle-$GradleVersion-bin.zip"
    $url = "https://services.gradle.org/distributions/gradle-$GradleVersion-bin.zip"
    Info "下载 $url"
    Invoke-WebRequest -Uri $url -OutFile $zip -UseBasicParsing
    New-Item -ItemType Directory -Force -Path (Split-Path $gradleHome) | Out-Null
    Expand-Archive -Path $zip -DestinationPath (Split-Path $gradleHome) -Force
    Remove-Item $zip -Force
}
if (-not (Test-Path $gradleBat)) { Die "Gradle 准备失败（$gradleBat）" }
Info "Gradle = $gradleBat"

# local.properties 是本机路径，.gitignore 已排除
"# 由 phone/tools/build_apk.ps1 生成，不入库`nsdk.dir=$($sdk -replace '\\','\\')" |
    Set-Content -Path (Join-Path $AndroidDir "local.properties") -Encoding ASCII

# ---------- 5) 构建 ----------
$task = if ($Debug) { "assembleDebug" } else { "assembleRelease" }
Step "gradle $task"
Push-Location $AndroidDir
try {
    & $gradleBat --no-daemon "-Dorg.gradle.java.home=$jdk" $task
    if ($LASTEXITCODE -ne 0) { Die "gradle $task 失败（看上面输出的第一条错误）" }
} finally { Pop-Location }

$apk = Get-ChildItem (Join-Path $AndroidDir "app\build\outputs\apk") -Recurse -Filter "*.apk" |
       Sort-Object LastWriteTime -Descending | Select-Object -First 1
if ($apk) {
    Write-Host ""
    Write-Host "APK 就绪：" -ForegroundColor Green
    Info $apk.FullName
    Info "$([math]::Round($apk.Length / 1MB, 2)) MB"
    Write-Host ""
    Write-Host "装到手机：adb install -r `"$($apk.FullName)`"" -ForegroundColor Green
} else {
    Die "构建结束但没找到 APK"
}
