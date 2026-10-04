# feng-30m

**一个 29.43M 参数的中文对话模型：从零训练、原生 32k 上下文，Q4 量化后能塞进 ESP32-S3 离线对话。**

当前版本 **PC = v3.14（tool 版）、板端 = v3.16-embed 权重 + v3.17 引擎（记忆 tool）**：
**算式 / 网络时间(UTC+8) / 随机数 / 多轮记忆 四个 tool 写进 C 引擎**——板端与 PC 都 0.5 秒秒回、100% 正确，
模型不再学算术；PC = 范围 10/10 + 对话 42/42 + 记忆 **24/24** + 单类别 108，
板端 = q2/2048 矩阵 27/27 + 4/4、工具 13/13；身份与常见事实记忆由引擎确定性作答
（12 题连续记忆 **12/12**、「你叫什么名字」永不串名，均 0.5 秒秒回） ｜ 代码与权重均 **Apache-2.0**
｜ 身份自述：**「我是 feng，一个由个人开发者 jiaheng 独立开发训练的 AI」**
> PC 用 **v3.14**（长上下文、记忆、tool）；板端用 **v3.16-embed**（Q4+q2 双量化 + 多轮上下文 + tool + 身份稳定）；
> v3.4 单类别检索历史最高（113/128，v3.13 并列）。**GGUF 发行已取消**（llama.cpp 路径没有这些 tool）。

## 下载与使用

权重（fp32 / ESP32 板端模型）与蒸馏数据集打包在 **[Releases](https://github.com/jiaheng0815/feng-30m/releases)**：
`feng-30m-v3.14-release.zip`（PC 版：HF 权重 + 蒸馏数据集）、
板端版 `feng-30m-v3.16-embed-release.zip`（含可直接烧录的 ESP32 `model.bin`/`tokenizer.bin`）、
`feng-30m-v3.14-engine.zip`（PC C 引擎源码，v3.17 引擎）、
`feng-30m-c-engine-model-v3.16-embed.zip`（**C 引擎预导出模型**，免装 torch 直接跑）、
`feng-30m-v3.16-embed-firmware.zip`（**板端一包到底**：预编译固件 + 模型 + 烧录说明，免装 ESP-IDF）。
**本仓库只放代码与文档，训练数据与权重不入库。**

包内结构：

- `weights/hf/` —— v3.14 完整权重（fp32 safetensors + 分词器 + chat template），transformers 直接加载
- ~~`weights/gguf/`~~ —— **v3.14 起取消发行**：GGUF/llama.cpp 路径没有 tool，请用仓库自带的
  PC C 引擎（`esp32s3-feng-llm/pc/pc_chat.c`）或 Python 脚本（`scripts/runtime_tools.py`）
- `weights/esp32/` —— 板端 `model.bin`（14.93 MB）+ `tokenizer.bin`（413 KB）+ 参考 logits
- `datasets/` —— 蒸馏训练数据（教师输出与提示词）

安装、推理、烧录的完整步骤见 [`USAGE.md`](USAGE.md)。

## 亮点

- **引擎里的 tool（v3.14，本项目现在的算术方案）**：`算式`（多位数/小数/括号/中文数字/百分号，`feng_calc.c`）、
  `网络时间→UTC+8`（宿主 SNTP 对时 + `\settime`，`feng_tools.c`）、
  `随机数/骰子/硬币`（seed = 运行时间×1.54×1000，丢弃第一个，`feng_tools.c`）——
  板端与 PC 都 **0.5 秒秒回**，实测 `4854+4411=9265`、`5.3+4.1=9.4`、`现在几点？→ 2026年10月04日 15:10:23（周日，UTC+8）`。
  `现在的时间戳是多少？` 会返回原始 Unix 秒 + UTC+8 换算；UTC+8 日历与
  「丢第一个取第二个」的随机数语义有 C 单测（53 项全过，`logs/pc_tools_test.txt`），
  板端专项 13/13（`logs/board_v3_15ci4_tools_time_rand.txt`）。
  中文数字（`五十九加一`）、`乘以/除以`、`15%`、`掷骰子`、`抛硬币` 也都走 tool。
  **模型不再学算术**（训练数据用 tool 识别器过滤纯算式样本）；GGUF 因没有 tool 已取消发行。
- **真的能上板**：Q4 块64 量化后 14.93 MB，落在 **flash 前 16 MB 的 mmap 窗口**内（NOR flash 24 位地址上限，**与模块 32 MB 容量无关**）；int8 KV 让板端上下文从 256 提到 **1024**，v3.14 的 **q2 KV（block8）用 9.62 MB 跑 2048 上下文**，板端实测 **1.80 tok/s**
- **嵌入式 32 题满分 + 多轮 + 记忆（v3.14）**：28 个日常短任务 + 4 个 1.5k token 长文召回（**1963 token 时仍 27/27+4/4**），在 q2 KV / 2048 下 = **27/27 + 4/4**；板端默认/情绪/工具三组 **10/10 ｜ 10/10 ｜ 8/8**；跨轮记忆 12 题抽测 **10/12**；固件保留跨轮上下文（`\reset` 清空）
- **长对话可用性（v3.14 附录 4）**：64 轮连续对话板端 **64/64 成功**；修掉 q2 KV 的 112 B 跨步读
  （缓存行利用率 1/32）后，第 24–26 轮从 108/64/128 s 降到 47/26/50 s（**约 2.5×**），
  且改动与修复前**逐行零差异**（逐位一致）；上下文写满 256 的测试版也验证了自动开新对话（16/16 成功）
- **多轮记忆（v3.14）**：PC 记忆评测（陈述事实→追问）从 v3.12 的 5/24 一路做到 **24/24**；板端 12 题 **10/12**。
  记忆是上下文内记忆（靠 2048 token KV），不是持久记忆
- **长上下文可用（v3.14）**：原生 32k 训练 + 合成检索 SFT + 硬负样本拒答训练；v3.14 针检索按 **每长度 32 题** 复测：
  @4k/8k/16k/32k = **28/28/25/27（单类别 108/128）**，**多类别 107/128**；
  文中没有答案时 **61/64（单类别）与 62/64（多类别）会说明"没有提到"**（v1~v3 是 0/64 全编造）。
  > 单类别历史最高是 v3.4（113/128）；v3.9→v3.12 把 16k 从 23/32 修到 26/32。
- **算术边界可用（v3.13）**：281 题加/减/乘网格（含 0 操作数、结果 0/负）从 v3.9 的 **170 → 274/281**；
  板端对应权重 C 引擎 q2 算术子集 **21/21**、板端算术 10/10。
- **末层微调零代价修行为（v3.9）**：只训练最后 2 层 + norm（4.02M/29.43M 参数）修股票拒答，
  **范围 8/10 → 10/10，检索/对话/拒答一个点都没掉**——全参修复此前要吃掉 3–5 个检索点。
- **基础聊天可用（v3.6 起）**：42 题广谱日常探针 **42/42**（v3.8 保持），
  情绪回应 8/8，多轮 7 轮不同回答比例 **1.00**（v3.0~v3.4 只有 0.57 且第 3 轮起复读）；会拒答炸弹/诈骗请求
- **同一套引擎**：C11 推理核心 PC 与板端共用，与 PyTorch(Q4) **逐位一致**（max|diff| = 0.0000），改内核有基线可回归
- **过程全公开**：每一代模型、每一次改动、每一次实测数字（含失败尝试）都写在文档里，数字都能对上脚本与日志

## 版本速览

| | v3.10 | v3.11 | v3.12 | v3.13 | **v3.14（当前）** |
|---|---|---|---|---|---|
| 累计训练量 | v3.9 + 约 6M KV-QAT | v3.10 + 约 14M 双 QAT（权重+KV） | v3.9 + 约 12M 末层算术微调 | v3.12/v3.11 + 约 12M 记忆混训 | v3.13/v3.9 底座 + **约 14M 无算术混训**（tool 接管算术） |
| 身份自述 | 同（12/12） | 同（12/12） | 同（12/12） | 同（12/12） | **同（12/12）** |
| 范围内评测 | 10/10 | 10/10 | 10/10 | 10/10 | **10/10** |
| 日常对话探针（42 题） | 42/42 | 42/42 | 42/42 | 42/42 | **42/42** |
| 针检索·单类别（4k/8k/16k/32k） | 28/30/28/21 | 29/30/27/20 | 28/29/26/27 | 28/29/29/27 | **28/28/25/27** |
| 针检索·多类别 | 94 | 90 | 108 | 108 | **107** |
| 多轮记忆（PC 24 题 / 板端 12 题抽测） | — | — | 5/24 | 21/24 ｜ 板端 6/6 抽查 | **24/24 ｜ 板端 10/12** |
| 算式 / 时间 / 随机数 | 模型硬算 | 模型硬算 | 模型硬算 | 模型硬算 | **引擎 tool：0.5s 全对**（含多位数/小数） |
| GGUF Q4_K_M | 23.7 MB | 23.7 MB | 23.7 MB | 23.7 MB | **取消发行**（llama.cpp 没有 tool） |
| 嵌入式 32 题矩阵（q2 KV） | 27/27+4/4 | 27/27+4/4 | — | 27/27+4/4 | **27/27+4/4** |
| 板端算术子集 21 题（q2，C 引擎） | 12/21 | 21/21 | — | 21/21 | **21/21（数学题由 tool 回答）** |
| ESP32-S3 实机 | 10/10 + 10/10 | 30/30 + 记忆 6/6 抽查 | — | 30/30 + 记忆 6/6 抽查 | ✅ **工具 8/8 + 30/30 + 记忆 10/12，1.80 tok/s** |

> 范围内评测为 `scripts/eval_planA_scope.py` 的同口径复测（结果 JSON 在 `eval/`）：
> **v3.7 = 10/10、v3.9 = 10/10、v3.10 = 10/10、v3.11 = 10/10、v3.12 = 10/10**、v3.8 = 8/10、v3.6 = 8/10、v3.5 = 7/10、v3.4 = 9/10。
> 42 题日常探针同口径：**v3.4 = 27/42、v3.5 = 29/42、v3.6~v3.12 = 42/42**
> （`eval/chat_probe_v3_4.json`、`chat_probe_v3_5_current.json`、`chat_probe_v3_9_sf2.json`、`chat_probe_v3_10p3.json`）。
> v3.9 针检索：单类别 29/29/23/27 = 108，多类别 28/25/32/23 = 108（`eval/longctx32*_v3_9sf2.json`）。
> **v3.14（PC）**：单类别 28/28/25/27 = 108、多类别 107、记忆 **24/24**、范围 10/10、探针 42/42、
> 身份 12/12（`eval/longctx32*_v3_14pc2.json`、`eval/memory_v3_14pc2.json`）。
> **v3.14（板端）**：C 引擎 q2 矩阵 **27/27+4/4**、算术子集 **21/21**（数学题由 tool 回答）、
> 板端 工具 8/8 + 默认 10/10 + 情绪 10/10、记忆 12 题 **10/12**
> （`logs/board_v3_14b6_tools.txt`、`logs/board_v3_14b6_memory12.txt`）。
> 历史最好：单类别 113（v3.4 / v3.13 并列）；单类别 113 与多类别 108 的细节见 CHANGELOG v3.12/v3.13。

横向对比与全部实测见 [`COMPARISON.md`](COMPARISON.md)，逐版本演进（含失败记录）见 [`CHANGELOG.md`](CHANGELOG.md)。

## 模型规格（v3.8）

| 项目 | 值 |
|---|---|
| 架构 | Qwen3 结构（RMSNorm / QK-norm / RoPE / SwiGLU / tied embedding） |
| 层数 / 宽度 | 11 层 / hidden 448 / FFN 896 |
| 注意力 | 7 头 MHA（7 Q 头 = 7 KV 头），head_dim 64 |
| 词表 | 16384（自训 BPE），特殊符 `<|im_start|>` `<|im_end|>` `<|endoftext|>` `<|pad|>` |
| 参数量 | 29.43M（tied embedding） |
| 上下文 | 32768（`rope_theta=1e6`，无 RoPE 插值） |
| 量化 | Q4 块64（4.25 bpw，板端）/ Q4_K_M / Q8_0 / f16（llama.cpp） |

## 快速开始

**推荐：C 引擎 `pc_chat`（与板端同款引擎，自带算式/时间/随机数/记忆 tool）**——
下载 `feng-30m-v3.14-engine.zip`（引擎）与 `feng-30m-c-engine-model-v3.16-embed.zip`（已导出的模型），
解压到相邻目录后：

```powershell
.\build_pc_chat.ps1                      # MSYS2 gcc，编译 q2/2048 引擎
.\pc_chat_q2b8.exe ..\feng-30m-c-engine-model
# you> 4854+4411         -> [tool] 4854 加 4411 等于 9265。
# you> 现在的时间戳是多少？ -> [tool] 时间戳：1791134671 —— 2026年10月05日 01:24:31（周一，UTC+8）。
# you> 我叫小明，请记住    -> 模型正常回应
# you> 我叫什么名字？      -> [tool] 你叫小明。（0.5 秒秒回）
```

> 引擎的 `model.bin` 是 **Q4 权重 + q2 KV 双 QAT** 格式，必须用板端 v3.16-embed 权重导出
> （预导出包已做好）。**PC 的 v3.14 HF 权重没做量化感知训练，导出给引擎会明显退化**
> （同套 32 题矩阵实测 22/27 vs 27/27，`logs/pc_kv_suite32_v3_14pc2_q2b8.txt`）。
> GGUF/llama.cpp 自 v3.14 起不再发行（没有 tool，算术/时间会退化成模型硬算）。

HF 权重 + transformers：

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

ESP32-S3-WROOM-2-N32R16V（32 MB Octal flash + 16 MB Octal PSRAM）的编译、烧录与串口协议见
[`USAGE.md`](USAGE.md) 第 4 节和 [`esp32s3-feng-llm/README.md`](esp32s3-feng-llm/README.md)。

## 训练怎么做的（v3 链条）

| 阶段 | 内容 | 数据量 | 脚本 |
|---|---|---|---|
| 预训练（v2 完成） | 中文维基 + firefly，seq 2048 | 1,496M tokens | `scripts/v2_build.py`、`scripts/v2_train.py` |
| Plan A SFT（v2 完成） | 27B 教师行为数据 ×6 + 身份 ×25 + 过滤后真实闲聊，再低 LR 补训 1000 步 | 94,478 段 / 22.5M tokens | `scripts/build_planA_corpus.py`、`scripts/v2_train.py` |
| 渐进长文 | 同一 token 流按 4k→8k→16k→32k 切窗，数据量随长度递减 | 72M tokens | `scripts/v3_build_stages.py`、`scripts/v3_train.py` |
| 长上下文 SFT | Plan A 语料重打包成 8192 窗口，避免短序列把窗口压回去 | 22.65M tokens（17.86M 有监督） | `scripts/v3_pack_sft.py`、`scripts/v3_polish.py` |
| 合成检索 SFT | 长文埋事实、只对答案算 loss（关键一步：只喂长文学不会检索） | 12.5M tokens | `scripts/v3_build_retrieval.py`、`scripts/v3_retrieval_sft.py` |
| v3.5 多轮修复 | 2,600 条多轮对话 × 高占比混训 + 检索补强，修"从第 3 轮起复读" | 28.0M tokens（v3_5b/c/d summary 合计） | `scripts/v3_5_build_multiturn.py`、`scripts/chat_multi.py` |
| v3.6 日常补丁 | 592 条日常对话 + 系统化小数字运算（×6），单条对话 SFT 后再检索回补 | 38.0M tokens（v3_6a/b/i/j/r summary 合计）+ 单条 SFT 补丁轮 | `scripts/v3_6_build_daily_patch.py`、`scripts/v3_6_build_drill.py`、`scripts/v3_6_sft_patch.py`、`scripts/chat_probe.py` |
| v3.7 KV-QAT | 训练时注入 q2 block8 KV 噪声；数据加大常识/乘法/书影推荐，并补取件码长文召回与股票拒答 | 约 10M tokens（短样本 + 4k/8k/16k/32k 检索窗口） | `scripts/v3_7_build_qat_data.py`、`scripts/v3_7_build_needle_qat.py`、`scripts/v3_7_kv_qat.py` |
| v3.8 上下文升级 | 从 v3.6r 出发：单类别过采样 + 重复事实的检索 SFT → 与 v3.4 soup → **末层微调**修对话 | 约 15M 检索 + soup/末层修复 | `scripts/v3_1_build_retrieval.py`、`scripts/soup_models.py`、`scripts/v3_8_build_fix_data.py`、`scripts/v3_6_sft_patch.py --train-last` |
| v3.9 零代价修行为 | 末层微调（最后 2 层 + norm）用 40× 股票拒答数据修范围评测 | ≤1M tokens（802 条 × 5 epochs） | `scripts/v3_6_sft_patch.py --train-last 2`、`v3_9/fix_stock.jsonl` |
| v3.10 嵌入式 QAT | v3.9 底座上做 q2 block8 KV-QAT（通用 → 针专项 soup → 9,410 条日常低 lr 回补） | 约 6M tokens（5,516 条 QAT + 4k/8k 回放 + 专项/回补） | `scripts/v3_7_kv_qat.py`、`scripts/soup_models.py`、`v3_10/chatfix_all.jsonl` |
| v3.11 算术边界 + 双 QAT | 补齐 0 操作数/结果≤0 的算术网格（边界加权）+ 27 题验收锚点精修；训练时同时模拟 **Q4 权重（block64/fp16 scale）与 q2 KV** | 约 14M tokens（4,946 算术补丁 + 1,876 修复 + 锚点，2~4 epochs） | `scripts/eval_arith.py`、`scripts/v3_11_build_arith_patch.py`、`scripts/v3_11_build_repair.py`、`scripts/v3_7_kv_qat.py --wqat` |
| v3.12 PC 算术修复 | 同一套算术数据走「末层微调」（最后 2 层 + norm）两轮：第一轮补网格，第二轮定点修漏题 + `×1` 乘法族 + 加法对照 | 约 12M tokens（两轮 patch 轮） | `scripts/v3_6_sft_patch.py --train-last 2`、`scripts/v3_11_build_repair.py --pc-fix` |
| v3.13 记忆版 | 固件多轮上下文（KV 跨轮累积 + `\reset`）；合成 5,600 条记忆对话，PC 走末层微调、板端走 Q4+q2 双 QAT，混训含 27 题/算术/日常硬锚点 | 约 12M tokens（记忆混训两套权重） | `scripts/v3_13_build_memory.py`、`scripts/eval_memory.py`、`scripts/v3_6_sft_patch.py --train-last 2`、`scripts/v3_7_kv_qat.py --wqat` |
| v3.14 tool 版 | 算术/时间/随机数做成 C 引擎 tool（`feng_calc.c` / `feng_tools.c`），所有运行时接入；训练数据用 tool 识别器过滤纯算式样本（模型不再学算术）；GGUF 发行取消 | 约 14M tokens（无算术混训）；tool 为纯 C，不占权重 | `scripts/v3_14_build_noarith_mix.py`、`esp32s3-feng-llm/main/feng_calc.c`、`feng_tools.c`、`scripts/runtime_tools.py`、`esp32s3-feng-llm/pc/pc_chat.c` |

完整超参、每阶段 loss/耗时/显存见 [`DELIVERY.md`](DELIVERY.md) 与 [`CHANGELOG.md`](CHANGELOG.md)。

## 仓库结构

```
scripts/            数据构建 / 训练 / 评测 / 导出脚本（57 个 .py，路径解析见 scripts/paths.py）
esp32s3-feng-llm/   ESP32 固件 + 可移植 C11 推理引擎 + PC 端一致性检查
student/ v2/ v3/    三代模型的训练记录（summary.json / train_log.jsonl / config.json / 分词器）
eval/               评测结果 JSON（范围内 18 题、针检索、各阶段）
logs/               构建 / 训练 / 烧录 / 板上测试日志（board_baseline_lut.txt 是板上精度基线）
tools/check_md.py   文档自检（代码围栏、路径、过时数字）
tools/check_docs.py 文档事实校验（模型规格 / GGUF 体积与模板 / 评测数字与 eval 结果对齐）
```

## 配置（跑脚本前看一眼）

脚本里**没有硬编码盘符**：项目根目录由脚本位置自动推导，外部依赖按
**环境变量 > `scripts/local_paths.json` > 默认值** 的顺序解析：

| 环境变量 | 用途 |
|---|---|
| `FENG_ROOT` | 项目根目录（默认：脚本上一级目录） |
| `FENG_DATA_DIR` | 原始语料目录，v1–v3 的语料脚本共用（默认 `../feng-ai-qwen35/data`） |
| `FENG_LLAMA_DIR` | llama.cpp 仓库目录（GGUF 转换 / 量化 / benchmark） |
| `FENG_PY` | Python 解释器（默认：当前解释器） |
| `FENG_TEACHER_GGUF` | v1 教师 GGUF（**仅 v1 蒸馏脚本需要**；v2/v3 用 HTTP 的 27B 教师） |

不想每次都设环境变量，就复制 `scripts/local_paths.example.json` 为 `scripts/local_paths.json` 填自己的路径
（后者已 gitignore）。自检命令：`python scripts/paths.py`，会逐条打印路径是否可用。

**只跑 v3（推理 / 导出 GGUF / 刷板）：只需要 `FENG_LLAMA_DIR` 和 Python**；其余键只在重建 v1/v2 语料时才用得到。

## 已知限制

- **30M 容量上限**：v3.6 覆盖了常见寒暄/情绪/常识/小数字运算/翻译/推荐等日常问法（42 题探针全过），
  但没覆盖到的自由问答仍可能答偏或编造；复杂推理、长链条计算、专业领域不可靠。
- **推荐类只保训练分布内的**：书/电影推荐是补丁里逐条写的，换别的书名、要最新榜单会露馅。
- 板端上下文 1024（int8 KV 占 9.93 MB PSRAM）或 2048（q2 KV 占 9.62 MB）；**32k 只在 PC 上可用**，板上 32k 受 KV 内存限制不可能。
- **PC 用 v3.14、板端用 v3.14-embed**：PC 版是长上下文 + 记忆 + tool 综合最好；
  板端权重为 Q4 权重 + q2 KV 的量化前向优化（矩阵满分 + 记忆 10/12），PC 端 32k 弱。
- 两个权重不能互换：PC 版没做 KV-QAT（板端 q2 只有 21/27），板端版的 PC 长上下文不如 PC 版。
- **算式/时间/随机数只在带 tool 的运行时里**（板端固件、PC C 引擎 `pc_chat`、Python 脚本）；
  **GGUF/llama.cpp 没有 tool，v3.14 起不再发行 GGUF**。
- **板端时间需要宿主对时**：串口脚本连接时会自动发 `\settime`（宿主走 NTP）；不跑脚本时
  需要手动发一次，否则时间 tool 回答"还没对上网络时间"。
- **记忆是上下文内记忆**：靠 2048 token 的 KV，`\reset`、断电重启或上下文写满后会忘；
  记忆评测 PC 24/24、板端 10/12（颜色/食物会串），属于 30M + q2 KV 下的干扰。
- 板端固件默认**跨轮保留上下文**（v3.13 起）；跑"独立探针"式验收时
  `scripts/esp32_multi.py` 默认每题前发 `\reset`，需要连续对话时加 `--no-reset`。
- 量化抗性对权重回插极敏感：掺 20% v3.9 进 v3.10 就掉到 25/27（CHANGELOG v3.10）；
  想改板端行为请走"补数据 + 权重/KV 双 QAT"链路，不要手动 soup。
- v3.8 是 PC 端上下文版：**16k 仍是弱项**（23/32，v3.4 是 27）；两道股票拒答与 v3.6 一样没修。
- v3.9 已把股票拒答修到 **10/10**；16k 弱项与 32k"文中没有"拒答（61/64）经多轮专项
  （评测同款 filler / 正负配比 / 末层微调）均未突破，记录为平台（CHANGELOG v3.9 附录）。
- 板端生成 ~1.9 tok/s，长回答要等十几秒；标量内核已到极限，下一步是 PIE（128 位 int8 SIMD）。
- ESP32 固件的模型分区偏移必须与 `esp32s3-feng-llm/partitions.csv` 一致（`flash.ps1` 已按当前布局写好）。

## 许可证

代码与权重均为 **Apache-2.0**（见 [`LICENSE`](LICENSE)）。本项目为个人项目，与任何模型厂商无隶属或背书关系。
