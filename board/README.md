# board —— ESP32-S3 固件工程（推理核心在仓库根的 `core/`）

> **v4.0 起全部说明文档归到 [`docs/`](../docs/)：**
> 本工程的完整说明（硬件要求、引擎结构、编译烧录、串口协议、实测结果）
> 见 **[`docs/esp32-firmware.md`](../docs/esp32-firmware.md)**。
> 版本演进见 [`docs/CHANGELOG.md`](../docs/CHANGELOG.md)。

## 这个目录里有什么

```
main/           ESP-IDF 应用层（板端专有）
  main.c         串口 REPL、上下文管理、tool 调度、GBK 切换
  feng_bonus.c   彩蛋（-DFENG_NO_BONUS=ON 关闭）
  gbk*.c         GBK↔UTF-8 转换（默认不编译，-DFENG_GBK=ON 开启）
  CMakeLists.txt IDF 组件定义；核心源码用绝对路径引用仓库根的 core/
  feng_gemv_bench.c / feng_pie_*.S  可选的周期剖析 / PIE 实验
tools/           export_model.py（HF → model.bin/tokenizer.bin）、gen_gbk_table.py
partitions.csv   model 0x110000/0xEF0000、tokdata 0x1000000/0x80000
```

推理核心与 PC/手机共用，位于仓库根：

```
core/           严格 C23（零堆、零全局构造）
  feng.h / feng_model.c / feng_llm.c / feng_quant.c / feng_smp.c
  feng_sample.c / feng_tokenizer.c / feng_calc.c / feng_tools.c / feng_memory.c
pc/             PC 侧 C++23 驱动：pc_chat、32 题矩阵、多轮套件、pc_check、CUDA
```

## 构建

```powershell
# PC 引擎（两趟：核心 C23 + 驱动 C++23）
.\pc\build_pc_chat.ps1
.\pc\build_pc_cuda.ps1        # 可选 CUDA 后端

# 固件（发布配置 q2/2048）
idf.py -DFENG_USE_Q2_KV=ON build
```

为什么必须分两趟、以及 C23 引擎的全部约定，见根 [`AGENTS.md`](../AGENTS.md)。
