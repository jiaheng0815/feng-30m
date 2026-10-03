@echo off
rem Start the NInfer 27B teacher exactly as the user specified (teacher mode, 8 slots).
"D:\ai-ninfer\build\b1\apps\ninfer-serve.exe" "D:\ai-ninfer\artifacts\bonsai2-27b.ninfer" ^
  --host 127.0.0.1 --port 8080 --model-id bonsai2-27b ^
  --max-context 16384 --kv-capacity auto --kv-dtype fp8 ^
  --max-concurrency 8 --no-cuda-graph --spec mtp --draft-tokens 2 ^
  --host-kv-mib 0 --host-state-slots 0 --device-state-slots 8 ^
  --max-private-continuations 8 --max-shared-prefixes 4 ^
  --max-pending-requests 32 ^
  --cors
