# esp32s3-feng-llm —— 固件工程与可移植推理引擎

> **v4.0 起全部说明文档归到 [`docs/`](../docs/)：**
> 本工程的完整说明（硬件要求、引擎结构、编译烧录、串口协议、实测结果）
> 见 **[`docs/esp32-firmware.md`](../docs/esp32-firmware.md)**。
> 版本演进见 [`docs/CHANGELOG.md`](../docs/CHANGELOG.md)。

## 这个目录里有什么

```
main/          推理核心（严格 C23，板端与 PC 共用）+ ESP-IDF 应用
  feng.h         模型容器 / 推理 API / KV 模式与优化开关（带 extern "C" 守卫）
  feng_model.c   model.bin 解析、张量定位
  feng_llm.c     RMSNorm / QK-norm / RoPE / 注意力 / SwiGLU / tied head / KV
  feng_quant.c   Q4 查表 GEMV、fp16 路径、A8/PIE 实验分支
  feng_smp.c     双核 GEMV（PC 上退化为单核直通）
  feng_sample.c  统一采样器（重复惩罚 1.15 + no-repeat 3-gram）
  feng_tokenizer.c  设备端 BPE
  feng_calc.c / feng_tools.c / feng_memory.c   四个 tool
  feng_gemv_bench.c  GEMV 内层循环的周期级剖析（-DFENG_BENCH_GEMV=ON）
  feng_cuda.cu   PC 侧 CUDA 后端（nvcc，不进固件）
pc/            PC 侧 C++23 驱动：一致性检查、32 题矩阵、多轮套件、基准
tools/         export_model.py（HF → model.bin/tokenizer.bin）、gen_gbk_table.py
```

## 构建（两趟：核心 C23 + 驱动 C++23）

```powershell
.\build_pc_chat.ps1          # MSVC：核心 /std:clatest /TC、驱动 /std:c++latest /TP
.\build_pc_cuda.ps1          # 同上 + nvcc 编译 feng_cuda.cu
idf.py -DFENG_USE_Q2_KV=ON build    # 固件（发布配置 q2/2048）
```

为什么必须分两趟、以及 C23 引擎的全部约定，见根 [`AGENTS.md`](../AGENTS.md)。
