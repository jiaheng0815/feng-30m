# feng-30m

[![CI](https://github.com/jiaheng0815/feng-30m/actions/workflows/ci.yml/badge.svg)](https://github.com/jiaheng0815/feng-30m/actions/workflows/ci.yml)

**一个 29.43M 参数的中文对话模型：从零训练、原生 32k 上下文；Q4 量化后运行在 ESP32-S3 上，
算术 / 时间 / 随机数 / 记忆四个 tool 由板内 C++23 引擎确定性作答。**

> **当前发布（2026-10-05）**
>
> - 权重：PC = **v3.19**（`v3_19/pc4`）｜ 板端 = **v3.19-embed**（`v3_19/board6`，Q4 权重 + q2 KV 双 QAT）
> - 引擎：**v3.20 / C++23**（严格模式、零堆、无异常/RTTI、无全局构造）；板端固件 **304,576 B**，比 C 版还小 2,816 B
> - 板端实测：q2/2048 矩阵 **27/27 + 4/4**、工具 **13/13**、记忆 **12/12**、**2.26 tok/s**（938 ms/token）
> - 身份自述：**「我是 feng，一个由个人开发者 jiaheng 独立开发训练的 AI」**
> - 代码与权重均 **Apache-2.0**；**GGUF 自 v3.14 起不再发行**（llama.cpp 路径没有 tool）

## 目录

- [模型规格](#模型规格)
- [现在能做什么（实测）](#现在能做什么实测)
- [快速开始](#快速开始)
- [标准基准（英文，0-shot）](#标准基准英文0-shot)
- [下载](#下载)
- [版本速览](#版本速览)
- [验证与 CI](#验证与-ci)
- [训练怎么做的](#训练怎么做的)
- [仓库结构](#仓库结构)
- [配置](#配置)
- [已知限制](#已知限制)
- [许可证](#许可证)

## 模型规格

| 项目 | 值 |
|---|---|
| 架构 | Qwen3 结构（RMSNorm / QK-norm / RoPE / SwiGLU / tied embedding） |
| 层数 / 宽度 | 11 层 / hidden 448 / FFN 896 |
| 注意力 | 7 头 MHA（7 Q 头 = 7 KV 头），head_dim 64 |
| 词表 | 16384（自训 BPE），特殊符 `<|im_start|>` `<|im_end|>` `<|endoftext|>` `<|pad|>` |
| 参数量 | 29.43M（tied embedding） |
| 上下文 | 训练 32768（`rope_theta=1e6`，无 RoPE 插值）；板端 2048（q2 KV，9.62 MB PSRAM） |
| 量化 | 板端 Q4 块64（4.25 bpw）权重 + q2 block8 KV；历史 GGUF 支持 Q4_K_M / Q8_0 / f16 |
| 板端模型 | `model.bin` 14.93 MB + `tokenizer.bin` 413 KB（flash mmap 流式读） |

定位：**中文日常对话 + 嵌入式离线部署**。它不追求通用能力——复杂推理、专业问答、
英文任务都是容量边界内的弱项（见[已知限制](#已知限制)与[标准基准](#标准基准英文0-shot)）。

## 现在能做什么（实测）

### 对话与长文（v3.19 / PC）

| 评测 | 结果 | 出处 |
|---|---|---|
| 42 题广谱日常探针 | **42/42**（情绪 8/8，7 轮多轮不同回答比例 1.00） | `eval/v3_19pc4_probe42.json` |
| 身份 / 范围评测 | **12/12** / **10/10** | `eval/v3_19pc4_scope.json` |
| 针检索（每长度 32 题，单类别） | @4k/8k/16k/32k = **28/30/26/26 = 110/128** | `eval/longctx32*_v3_19pc4.json` |
| 针检索（多类别） | **105/128** | 同上 |
| "文中没有"拒答 | **61/64**（单类别） | `eval/longctx32*_v3_19pc4.json` |
| 记忆评测（24 题） | **23/24** | `eval/v3_19pc4_memory24.json` |
| 留出 30 题（开发集） | 19/30；**独立留出（未参与调参）18/30** | `eval/heldout2_*.json` |

### 四个 tool（板端固件 / PC C++ 引擎 / Python，三处同口径）

| tool | 能力 | 实测 |
|---|---|---|
| 算式 | 多位数 / 小数 / 括号 / 中文数字 / 百分号，以及 **序列数数**（"把 1 到 5 倒着数一遍" → `5、4、3、2、1。`） | 0.5 s 秒回，`4854+4411=9265` |
| 时间 | 宿主 SNTP 对时 + `\settime`；UTC+8 日历、时间戳、N 天前后、时钟推算 | 板端与 NTP 差 ≤3 s |
| 随机数 | xorshift64*，seed = 运行时间×1.54×1000，丢第一个取第二个；骰子 / 硬币 / 区间 | 连续 3 次不重复 |
| 记忆 | 从 `我叫X / 最喜欢Y / 住在Z / 养了W` 与通用键值槽抽事实，追问确定性作答；`\mem` 查看、`\reset` 清空 | 板端 **12/12**，身份永不串名 |

板端 tool 专项 **13/13**、记忆 12 题 **12/12**、7 轮换名身份序列 **7/7**
（`logs/board_tools_cpp23_q2_final.txt`、`logs/board_memory_cpp23_q2_final12.txt`）。
模型不再学算术（训练数据用 tool 识别器过滤纯算式样本）。

### 嵌入式（v3.19-embed / q2 2048 固件）

| 项目 | 结果 |
|---|---|
| 32 题矩阵 + 长文召回 | **27/27 + 4/4**（1963 token 输入） |
| 算术子集 | **21/21**（算式走 tool） |
| 速度 | **2.26 tok/s**（938 ms/token，ctx 19）/ 973 ms @ ctx 38 / 1291 ms @ ctx 54 |
| 内存 | 权重 flash mmap（不占 PSRAM）；KV + 激活 9.62 MB PSRAM；内部 SRAM 仅留内核与工作区 |

### 引擎（v3.20 / C++23）

- **严格 C++23**（`-std=c++23`，PC 与 ESP-IDF 双工具链），`-fno-exceptions -fno-rtti -fno-threadsafe-statics`
- **零堆**：8 个核心对象无 `malloc/free/new/delete`、无异常/RTTI 符号；唯一分配是 tokenizer
  **启动期**一次性 PSRAM 加载；全组件 `.init_array` 为空（无全局构造）
- **体积**：304,576 B，比 C 版（307,392 B）小 2,816 B；非热点模块 `-Os`、推理热点保持 `-O2`
- **性能**：板端 ms/token 与 C 版逐项相同；C++/Python tool 一致性 **173 条** + 4 套单测全过

## 快速开始

### 1) PC：C++ 引擎（推荐，自带四个 tool）

下载 `feng-30m-v3.19-engine.zip`（引擎源码）与 `feng-30m-c-engine-model-v3.19-embed.zip`
（预导出模型），解压到相邻目录后：

```powershell
.\build_pc_chat.ps1                      # MSYS2 g++，C++23 严格模式
.\pc_chat_q2b8.exe ..\feng-30m-c-engine-model
# you> 4854+4411          -> [tool] 4854 加 4411 等于 9265。
# you> 现在的时间戳是多少？  -> [tool] 时间戳：1791134671 —— 2026年10月05日 01:24:31（周一，UTC+8）。
# you> 我叫小明，请记住     -> 模型正常回应
# you> 我叫什么名字？       -> [tool] 你叫小明。（0.5 秒秒回）
```

> 引擎的 `model.bin` 是 **Q4 权重 + q2 KV 双 QAT** 格式，必须用板端 v3.19-embed 权重导出
> （预导出包已做好）。**PC 的 HF 权重没做量化感知训练，导出给引擎会明显退化**
> （同套 32 题矩阵实测 22/27 vs 27/27，`logs/pc_kv_suite32_v3_14pc2_q2b8.txt`）。

### 2) HF 权重 + transformers

```python
import torch
from transformers import AutoTokenizer, Qwen3ForCausalLM

path = "weights/hf"          # 解压后的目录
tok = AutoTokenizer.from_pretrained(path)
model = Qwen3ForCausalLM.from_pretrained(path, dtype=torch.bfloat16, device_map="cuda").eval()

ids = tok.apply_chat_template([{"role": "user", "content": "你是谁？"}],
                              add_generation_prompt=True, return_tensors="pt").to(model.device)
out = model.generate(ids, max_new_tokens=96, do_sample=False,
                     repetition_penalty=1.25, no_repeat_ngram_size=6,
                     pad_token_id=3, eos_token_id=0)
print(tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True))
```

### 3) ESP32-S3（WROOM-2-N32R16V）

编译、烧录、串口协议的完整步骤见 [`USAGE.md`](USAGE.md) 与
[`esp32s3-feng-llm/README.md`](esp32s3-feng-llm/README.md)；免装 IDF 的一包到底固件
见 Release 的 `feng-30m-v3.19-embed-firmware.zip`。

## 标准基准（英文，0-shot）

用 EleutherAI lm-evaluation-harness v0.4.13 测 8 个内置任务 + 自定义 SciCloze-900
（`eval/lm_eval_tasks/scicloze_900.yaml`），模型 v3.19/pc4（fp32，`prefix_token_id=0`——
自训 tokenizer 没有 bos/eos），共 76,706 个 loglikelihood 请求 / 3 分 29 秒（RTX 5060 Ti）。
复现：`python scripts/bench_standard.py --model v3_19/pc4 --tag v3_19_pc4`；
结果 `eval/lm_eval_feng_v3_19_pc4.json`。

| 基准 | 指标 | 结果 |
|---|---|---|
| SciCloze-900 | Accuracy | **25.56%** ±1.45 |
| SciQ | Accuracy / Normalized | **71.00%** ±1.44 / **73.80%** ±1.39 |
| PIQA | Normalized Accuracy | **53.37%** ±1.16 |
| ARC-Easy | Normalized Accuracy | **26.60%** ±0.91 |
| ARC-Challenge | Normalized Accuracy | **21.93%** ±1.21 |
| HellaSwag | Normalized Accuracy | **28.99%** ±0.45 |
| Winogrande | Accuracy | **49.33%** ±1.41 |
| OpenBookQA | Accuracy / Normalized | **14.00%** ±1.55 / **24.00%** ±1.91 |
| BoolQ | Accuracy | **37.83%** ±0.85 |

> 这些是**英文**基准，而 feng-30m 是中文教师蒸馏的 30M 模型：除 SciQ（明显高于 4 选 1
> 随机线）和 PIQA（略高于 2 选 1 随机线）外，其余任务基本贴近随机水平（ARC / HellaSwag /
> OpenBookQA / SciCloze ≈ 25%，Winogrande ≈ 50%，BoolQ 37.8% 低于随机线，yes/no 偏置）。
> 这是能力边界的如实呈现，不构成对中文对话能力的评价。

## 下载

权重（fp32 / 板端模型）与蒸馏数据集打包在
**[Releases](https://github.com/jiaheng0815/feng-30m/releases)**：

| 附件 | 内容 |
|---|---|
| `feng-30m-v3.19-release.zip` | PC 版：HF 权重 + 蒸馏数据集 |
| `feng-30m-v3.19-embed-release.zip` | 板端版：Q4+q2 双 QAT 权重 + 可直接烧录的 `model.bin` / `tokenizer.bin` |
| `feng-30m-v3.19-engine.zip` | PC 引擎源码（C++23） |
| `feng-30m-c-engine-model-v3.19-embed.zip` | 引擎预导出模型（免装 torch 直接跑） |
| `feng-30m-v3.19-embed-firmware.zip` | 板端一包到底：预编译固件 + 模型 + 烧录说明（免装 ESP-IDF） |

**本仓库只放代码与文档，训练数据与权重不入库**；旧附件（v3.14 / v3.16-embed）保留在对应 Release 页。

## 版本速览

| | v3.10 | v3.11 | v3.12 | v3.13 | v3.14 | **v3.19（当前）** |
|---|---|---|---|---|---|---|
| 累计训练量 | v3.9 + 约 6M KV-QAT | v3.10 + 约 14M 双 QAT（权重+KV） | v3.9 + 约 12M 末层算术微调 | v3.12/v3.11 + 约 12M 记忆混训 | v3.13/v3.9 底座 + **约 14M 无算术混训**（tool 接管算术） | v3.14/pc2 + **440 条定向教师数据**（末 2 层 15ep） |
| 身份自述 | 同（12/12） | 同（12/12） | 同（12/12） | 同（12/12） | 同（12/12） | **同（12/12）** |
| 范围内评测 | 10/10 | 10/10 | 10/10 | 10/10 | 10/10 | **10/10** |
| 日常对话探针（42 题） | 42/42 | 42/42 | 42/42 | 42/42 | 42/42 | **42/42** |
| 针检索·单类别（4k/8k/16k/32k） | 28/30/28/21 | 29/30/27/20 | 28/29/26/27 | 28/29/29/27 | 28/28/25/27 | **28/30/26/26 = 110** |
| 针检索·多类别 | 94 | 90 | 108 | 108 | 107 | **105** |
| 多轮记忆（PC 24 题 / 板端 12 题抽测） | — | — | 5/24 | 21/24 ｜ 板端 6/6 抽查 | 24/24 ｜ 板端 10/12 | **23/24 ｜ 板端 12/12** |
| 算式 / 时间 / 随机数 | 模型硬算 | 模型硬算 | 模型硬算 | 模型硬算 | 引擎 tool：0.5s 全对（含多位数/小数） | **同（tool）** |
| 嵌入式 32 题矩阵（q2 KV） | 27/27+4/4 | 27/27+4/4 | — | 27/27+4/4 | 27/27+4/4 | **27/27+4/4** |
| 板端算术子集 21 题（q2） | 12/21 | 21/21 | — | 21/21 | 21/21（数学题由 tool 回答） | **21/21** |
| ESP32-S3 实机 | 10/10 + 10/10 | 30/30 + 记忆 6/6 | — | 30/30 + 记忆 6/6 | 工具 8/8 + 30/30 + 记忆 10/12，1.80 tok/s | ✅ **工具 13/13 + 记忆 12/12 + 身份 7/7** |

> 同口径复测记录（结果 JSON 均在 `eval/`）：范围内评测 v3.7/v3.9~v3.12 = 10/10、v3.8 = 8/10、
> v3.6 = 8/10、v3.5 = 7/10、v3.4 = 9/10；42 题探针 v3.4 = 27/42、v3.5 = 29/42、v3.6 起 42/42。
> **留出 30 题是开发集**（被多轮实验使用过），只作版本间对比；**独立留出 30 题**上
> v3.14/pc2、v3.19/pc4、v3.19/board6 均为 **18/30**——v3.19 的开发集增益（16→19）
> 在独立集上未复现，如实标注。历史最好：单类别 113（v3.4 / v3.13 并列）。

## 验证与 CI

本地一键验收（文档自检 → PC C++23 引擎构建 + 四套单测 → 32 题矩阵 + 算术子集 → fp32 参考 logits）：

```powershell
python tools\check_all.py
```

CI（GitHub Actions，每次 push/PR）跑 9 步，全部在**干净 clone** 上执行——不依赖本机权重/日志：

1. Python 语法检查（全部脚本）
2. C++23 单测四套：算式 41 / 时间随机数 53 / 记忆 69 / 采样器 4
3. C++/Python 算式+序列一致性 29 条（`tools/check_tool_parity.py`）
4. C++/Python 记忆 tool 一致性 36 回合（`tools/check_mem_parity.py`）
5. C++/Python 随机数一致性 56 组（`tools/check_rand_parity.py`）
6. C++/Python 时间 tool 一致性 52 条（`tools/check_time_parity.py`）
7. C++23 全引擎编译（g++ `-std=c++23`、pc_chat / pc_kv_suite / pc_mt_suite / pc_check + 9 个核心源）
8. 文档自检（`tools/check_md.py` + `tools/check_docs.py`，本机产物路径允许缺并计数）

> 四套 C++/Python 交叉验证合计 **173 条**逐条对比；文档引用的权重/日志/编译产物只在本机存在时
> 会被归为 `local-only` 跳过，仓库跟踪文件缺失仍会报错。

## 训练怎么做的

| 阶段 | 内容 | 数据量 | 脚本 |
|---|---|---|---|
| 预训练（v2） | 中文维基 + firefly，seq 2048 | 1,496M tokens | `scripts/v2_build.py`、`scripts/v2_train.py` |
| Plan A SFT（v2） | 27B 教师行为数据 ×6 + 身份 ×25 + 过滤后真实闲聊，再低 LR 补训 1000 步 | 94,478 段 / 22.5M tokens | `scripts/build_planA_corpus.py`、`scripts/v2_train.py` |
| 渐进长文（v3） | 同一 token 流按 4k→8k→16k→32k 切窗，数据量随长度递减 | 72M tokens | `scripts/v3_build_stages.py`、`scripts/v3_train.py` |
| 长上下文 SFT | Plan A 语料重打包成 8192 窗口，避免短序列把窗口压回去 | 22.65M tokens（17.86M 有监督） | `scripts/v3_pack_sft.py`、`scripts/v3_polish.py` |
| 合成检索 SFT | 长文埋事实、只对答案算 loss（关键一步：只喂长文学不会检索） | 12.5M tokens | `scripts/v3_build_retrieval.py`、`scripts/v3_retrieval_sft.py` |
| v3.5 多轮修复 | 2,600 条多轮对话 × 高占比混训 + 检索补强，修"从第 3 轮起复读" | 28.0M tokens | `scripts/v3_5_build_multiturn.py`、`scripts/chat_multi.py` |
| v3.6 日常补丁 | 592 条日常对话 + 系统化小数字运算（×6），单条 SFT 后检索回补 | 38.0M tokens + 补丁轮 | `scripts/v3_6_build_daily_patch.py`、`scripts/v3_6_sft_patch.py` |
| v3.7 KV-QAT | 训练时注入 q2 block8 KV 噪声；补长文召回与股票拒答 | 约 10M tokens | `scripts/v3_7_build_qat_data.py`、`scripts/v3_7_kv_qat.py` |
| v3.8 上下文升级 | 检索过采样 + soup + **末层微调**修对话 | 约 15M | `scripts/soup_models.py`、`scripts/v3_8_build_fix_data.py` |
| v3.9 零代价修行为 | 末层微调（最后 2 层 + norm）用 40× 股票拒答数据修范围评测 | ≤1M tokens | `scripts/v3_6_sft_patch.py --train-last 2` |
| v3.10~v3.11 嵌入式 QAT | q2 KV-QAT → 算术边界补丁 → **Q4 权重/q2 KV 双 QAT** | 约 20M tokens | `scripts/v3_7_kv_qat.py --wqat`、`scripts/v3_11_build_arith_patch.py` |
| v3.12~v3.13 算术/记忆 | 算术走末层微调；合成 5,600 条记忆对话（PC 末层 / 板端双 QAT） | 约 24M tokens | `scripts/v3_11_build_repair.py --pc-fix`、`scripts/v3_13_build_memory.py` |
| v3.14 tool 版 | 算术/时间/随机数做成引擎 tool；训练数据过滤纯算式样本；GGUF 发行取消 | 约 14M tokens（无算术混训） | `scripts/v3_14_build_noarith_mix.py`、`esp32s3-feng-llm/main/feng_calc.cpp` |
| v3.15~v3.19 打磨 | 身份上下文锚点 → 板端串名修复实验（未采用路线已记录）→ 引擎记忆 tool → 440 条定向数据末 2 层 15ep | 约 12M tokens | `scripts/v3_15_build_identity_ctx.py`、`scripts/v3_19_build_mix.py` |
| v3.20 引擎 | 序列数数 tool + 统一采样器；**全项目迁移 C++23**（零堆/无异常/RTTI）并做体积/性能优化 | tool 为 C++ 零堆实现，不占权重 | `esp32s3-feng-llm/main/`、`CHANGELOG.md` v3.20 节 |

完整超参、每阶段 loss/耗时/显存见 [`CHANGELOG.md`](CHANGELOG.md) 与
[`DELIVERY.md`](DELIVERY.md)。

## 仓库结构

```
scripts/            数据构建 / 训练 / 评测 / 导出脚本（路径解析见 scripts/paths.py）
esp32s3-feng-llm/   ESP32 固件 + 可移植 C++23 推理引擎 + PC 端一致性检查
student/ v2/ v3/    三代模型的训练记录（summary.json / train_log.jsonl / config.json / 分词器）
eval/               评测结果 JSON（范围、针检索、记忆、标准基准）
logs/               构建 / 训练 / 烧录 / 板上测试日志（board_baseline_lut.txt 是板上精度基线）
tools/check_md.py   文档自检（代码围栏、路径、过时数字）
tools/check_docs.py 文档事实校验（模型规格 / 评测数字与实际产物对齐）
```

## 配置

脚本里**没有硬编码盘符**：项目根目录由脚本位置自动推导，外部依赖按
**环境变量 > `scripts/local_paths.json` > 默认值** 的顺序解析：

| 环境变量 | 用途 |
|---|---|
| `FENG_ROOT` | 项目根目录（默认：脚本上一级目录） |
| `FENG_PY` | Python 解释器（默认：当前解释器） |
| `FENG_DATA_DIR` | 原始语料目录（**只有重建 v1/v2 语料时才需要**） |
| `FENG_LLAMA_DIR` | llama.cpp 仓库目录（GGUF 转换 / benchmark，可选） |
| `FENG_GXX` | PC 端 g++ 路径（默认 PATH 里的 `g++`，见 `build_pc_chat.ps1`） |

复制 `scripts/local_paths.example.json` 为 `scripts/local_paths.json` 填自己的路径（后者已 gitignore）。
自检命令：`python scripts/paths.py`，会逐条打印路径是否可用。

## 已知限制

- **30M 容量上限**：没覆盖到的自由问答会答偏或编造；复杂推理、专业领域、英文任务不可靠
  （标准基准大多贴近随机，见上）。
- **推荐类只保训练分布内的**：书/电影推荐是补丁里逐条写的，换书名或要最新榜单会露馅。
- **板端上下文 2048（q2 KV）**，32k 只在 PC 上可用；板上 32k 受 PSRAM/KV 内存限制不可能。
- **PC 权重与板端权重不能互换**：PC 的 HF 权重没做量化感知训练，导进引擎会明显退化
  （32 题矩阵 22/27 vs 27/27）；板端用双 QAT 权重（`feng-30m-c-engine-model-v3.19-embed.zip`）。
- **留出 30 题是开发集**：19/30 只用于版本对比；未参与调参的独立留出为 18/30（双版本持平）。
- **记忆不是持久化**：`\reset`/断电/上下文写满即忘；记忆 tool 只覆盖可枚举句式。
- **板端时间需要宿主对时**：串口脚本连接时自动发 `\settime`（宿主 NTP），否则时间 tool 会提示未对时。
- **板端速度 2.26 tok/s**（938 ms/token），长回答要等十几秒；标量内核已到极限，
  下一步杠杆是 PIE（128 位 int8 SIMD）。
- 量化抗性对权重回插极敏感：掺 20% 旧权重就掉点（CHANGELOG v3.10）；改板端行为要走
  "补数据 + 权重/KV 双 QAT"链路，不要手动 soup。
- ESP32 固件的模型分区偏移必须与 `esp32s3-feng-llm/partitions.csv` 一致。

## 许可证

代码与权重均为 **Apache-2.0**（见 [`LICENSE`](LICENSE)）。本项目为个人项目，
与任何模型厂商无隶属或背书关系。
