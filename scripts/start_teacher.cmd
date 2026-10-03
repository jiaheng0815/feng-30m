@echo off
rem 启动 NInfer 27B 教师服务（v2/v3 蒸馏用，thinking off，8 并发槽位）。
rem 路径不写死，先设置两个环境变量再运行本脚本：
rem   set NINFER_EXE=<ninfer-serve.exe 路径>
rem   set NINFER_MODEL=<.ninfer 模型路径>
setlocal

if "%NINFER_EXE%"=="" (
  echo [start_teacher] 未设置 NINFER_EXE，请先指向 ninfer-serve.exe
  exit /b 1
)
if "%NINFER_MODEL%"=="" (
  echo [start_teacher] 未设置 NINFER_MODEL，请先指向 .ninfer 模型
  exit /b 1
)

"%NINFER_EXE%" "%NINFER_MODEL%" ^
  --host 127.0.0.1 --port 8080 --model-id bonsai2-27b ^
  --max-context 16384 --kv-capacity auto --kv-dtype fp8 ^
  --max-concurrency 8 --no-cuda-graph --spec mtp --draft-tokens 2 ^
  --host-kv-mib 0 --host-state-slots 0 --device-state-slots 8 ^
  --max-private-continuations 8 --max-shared-prefixes 4 ^
  --max-pending-requests 32 ^
  --cors

endlocal
