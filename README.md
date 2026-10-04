# feng-30m

**一个 29.43M 参数的中文对话模型：从零训练、原生 32k 上下文，Q4 量化后能塞进 ESP32-S3 离线对话。**

当前版本 **v3.12（PC：范围 10/10 + 单类别 110 + 多类别 108 + 对话 42/42 + 算术 275/281）**，
板端当前版本 **v3.11（q2 KV / 2048 ctx：32 题矩阵 27/27 + 4/4、算术子集 21/21、板端 30/30）** ｜ 代码与权重均 **Apache-2.0**
｜ 身份自述：**「我是 feng，一个由个人开发者 jiaheng 独立开发训练的 AI」**
> PC 用 **v3.12**（长上下文、算术）；板端用 **v3.11**（Q4+q2 双量化优化）；v3.4 单类别检索历史最高（113/128）。

## 下载与使用

权重（fp32 / GGUF / ESP32 板端模型）与蒸馏数据集打包在 **[Releases](https://github.com/jiaheng0815/feng-30m/releases)**：
`feng-30m-v3.12-release.zip`（PC 版：v3.12 权重、GGUF 与蒸馏数据集）、
板端版 `feng-30m-v3.11-release.zip`（含 ESP32 model.bin/tokenizer.bin）。
**本仓库只放代码与文档，训练数据与权重不入库。**

包内结构：

- `weights/hf/` —— v3.12 完整权重（fp32 safetensors + 分词器 + chat template），transformers 直接加载
- `weights/gguf/` —— Q4_K_M 23.7 MB / Q8_0 30.5 MB / f16 56.8 MB，**chat template 已内嵌**
- `weights/esp32/` —— 板端 `model.bin`（14.93 MB）+ `tokenizer.bin`（413 KB）+ 参考 logits
- `datasets/` —— 蒸馏训练数据（教师输出与提示词）

安装、推理、烧录的完整步骤见 [`USAGE.md`](USAGE.md)。

## 亮点

- **真的能上板**：Q4 块64 量化后 14.93 MB，落在 **flash 前 16 MB 的 mmap 窗口**内（NOR flash 24 位地址上限，**与模块 32 MB 容量无关**）；int8 KV 让板端上下文从 256 提到 **1024**，v3.11 的 **q2 KV（block8）用 9.62 MB 跑 2048 上下文**，板端实测 **1.81 tok/s**
- **嵌入式 32 题评测满分 + 算术边界可用（v3.11）**：28 个日常短任务 + 4 个 1.5k token 长文召回，在 **q2 KV / 2048 上下文**下 = **27/27 + 4/4**，与 int8 持平；板端默认/情绪/算术三组 10 题 **10/10 + 10/10 + 10/10**；算术网格（281 题，PC）从 v3.10 的 202 → **277**，C 引擎 q2 算术子集 **21/21**（`logs/pc_kv_suite32_v3_11p8_q2b8.txt`、`logs/pc_arith_suite_v3_11p8_q2b8.txt`）
- **长上下文可用（v3.12）**：原生 32k 训练 + 合成检索 SFT + 硬负样本拒答训练；v3.12 针检索按 **每长度 32 题** 复测：
  @4k/8k/16k/32k = **28/29/26/27（单类别 110/128，历史第二）**，**多类别 108/128**；
  文中没有答案时 **61/64（单类别）与 62/64（多类别）会说明"没有提到"**（v1~v3 是 0/64 全编造）。
  > 单类别历史最高是 v3.4（113/128）；v3.9→v3.12 把 16k 从 23/32 修到 26/32。
- **算术边界可用（v3.12）**：281 题加/减/乘网格（含 0 操作数、结果 0/负）从 v3.9 的 **170 → 275/281**，
  只训最后 2 层就做到，且检索总量反涨（108→110）；板端对应 v3.11（C 引擎 q2 算术子集 21/21、板端 10/10）。
- **末层微调零代价修行为（v3.9）**：只训练最后 2 层 + norm（4.02M/29.43M 参数）修股票拒答，
  **范围 8/10 → 10/10，检索/对话/拒答一个点都没掉**——全参修复此前要吃掉 3–5 个检索点。
- **基础聊天可用（v3.6 起）**：42 题广谱日常探针 **42/42**（v3.8 保持），
  情绪回应 8/8，多轮 7 轮不同回答比例 **1.00**（v3.0~v3.4 只有 0.57 且第 3 轮起复读）；会拒答炸弹/诈骗请求
- **同一套引擎**：C11 推理核心 PC 与板端共用，与 PyTorch(Q4) **逐位一致**（max|diff| = 0.0000），改内核有基线可回归
- **过程全公开**：每一代模型、每一次改动、每一次实测数字（含失败尝试）都写在文档里，数字都能对上脚本与日志

## 版本速览

| | v1 | v2 | v3 | v3.5 | v3.6 | v3.7 | v3.8 | v3.9 | v3.10 | **v3.11（板端）** | **v3.12（PC 当前）** |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 结构 | 8 层 / 32k 词表 / 30.75M | 11 层 / 16k 词表 / 29.43M | 同 v2 | 同 v2 | 同 v2 | 同 v2 | 同 v2 | 同 v2 | 同 v2 | 同 v2 | 同 v2 |
| 教师 | Qwen3.5-0.8B 微调版 | bonsai2-27b（27B） | 同 v2 | 同 v2 | 同 v2 | 同 v2 | 同 v2 | 同 v2 | 同 v2 | 同 v2 | 同 v2 |
| 累计训练量 | 各阶段合计 ≈141M tokens（仅指令/对话数据） | +1.5B 预训练 +22.5M SFT +1.2M 补训 | +72M 长文 +22.65M 长上下文 SFT +12.5M 检索 SFT | +约 28M 多轮/检索 | +约 38M 检索/恢复 + 补丁轮 | +约 10M KV-QAT | +约 15M 检索 + soup/末层修复 | 末层微调 ≤1M | v3.9 + 约 6M KV-QAT | v3.10 + 约 14M 双 QAT（权重+KV） | v3.9 + 约 12M 末层算术微调（两轮） |
| 身份自述 | 命中但退化 | 「微调后的 Qwen」 | 同 v2 | 「jiaheng 独立开发训练的 AI」 | 同 v3.5 | 同 v3.5（12/12） | 同 v3.5（12/12） | 同（12/12） | 同（12/12） | **同（12/12）** | **同（12/12）** |
| 范围内评测 | 5/10 | 8/10 | 10/10 | 7/10 | 8/10 | 10/10 | 8/10 | 10/10 | 10/10 | **10/10** | **10/10** |
| 日常对话探针（42 题） | — | — | — | 29/42 | 42/42 | 42/42 | 42/42 | 42/42 | 42/42 | **42/42** | **42/42** |
| 针检索（每长度 32 题） | 0/0/0/0 | 0/0/0/0 | 31/30/26/19 | 27/29/25/25 | 27/29/24/22 | 27/30/28/16 | 29/29/23/27 | 29/29/23/27 | 28/30/28/21 | 29/30/27/20 | **28/29/26/27** |
| 「文中没有」拒答 | 0/64 | 0/64 | 0/64 | 59/64 | 62/64 | 63/64 | 61/64 | 61/64 | 62/64 | 61/64 | **61/64** |
| 多轮对话（7 轮不同回答比例） | 0.57 | 0.57 | 0.57 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | **1.00** | **1.00** |
| 算术网格 281 题（PC） | — | — | — | — | — | — | — | 170 | 202 | **277** | **275** |
| GGUF Q4_K_M | 27.6 MB（放不进 16MB 窗口） | 23.7 MB | 23.7 MB | 23.7 MB | 23.7 MB | 23.7 MB | 23.7 MB | 23.7 MB | 23.7 MB | 23.7 MB | 23.7 MB |
| 嵌入式 32 题矩阵（q2 KV） | — | — | — | 22/27+3/4（块16） | 24/27+4/4 | 27/27+4/4 | 21/27+4/4 | 21/27+4/4 | 27/27+4/4 | **27/27+4/4（9.62MB/2048ctx）** | 未做 KV-QAT（板端用 v3.11） |
| 板端算术子集 21 题（q2，C 引擎） | — | — | — | — | — | — | — | — | 12/21（pol7 前身） | **21/21** | —（板端用 v3.11） |
| ESP32-S3 实机 | ❌ 从未上板 | ✅ 1.56 tok/s @256 ctx | ✅ 1.86 tok/s @1024 ctx | ✅ 10/10 | ✅ 10/10（1.85–1.86 tok/s @1024） | ✅ 10/10 + 情绪 10/10 | 见 v3.7 | 未做 KV-QAT | 10/10 + 10/10 | ✅ **10/10 + 10/10 + 算术 10/10，1.81 tok/s** | 未做 KV-QAT（板端用 v3.11） |

> 范围内评测为 `scripts/eval_planA_scope.py` 的同口径复测（结果 JSON 在 `eval/`）：
> **v3.7 = 10/10、v3.9 = 10/10、v3.10 = 10/10、v3.11 = 10/10、v3.12 = 10/10**、v3.8 = 8/10、v3.6 = 8/10、v3.5 = 7/10、v3.4 = 9/10。
> 42 题日常探针同口径：**v3.4 = 27/42、v3.5 = 29/42、v3.6~v3.12 = 42/42**
> （`eval/chat_probe_v3_4.json`、`chat_probe_v3_5_current.json`、`chat_probe_v3_9_sf2.json`、`chat_probe_v3_10p3.json`）。
> v3.9 针检索：单类别 29/29/23/27 = 108，多类别 28/25/32/23 = 108（`eval/longctx32*_v3_9sf2.json`）。
> v3.12（PC）：单类别 28/29/26/27 = **110**、多类别 108（`eval/longctx32*_v3_12a2l3.json`）；
> 算术 275/281（v3.9 只有 170）——**PC 用 v3.12**。
> v3.10/v3.11（板端版）针检索：v3.10 单类别 107、多类别 94；v3.11 单类别 106、多类别 90
> （`eval/longctx32*_v3_10p3.json`、`eval/longctx32*_v3_11p8.json`）——32k 让位给量化鲁棒性，板端用 v3.11。
> v3.11 算术：281 题网格 277/281（v3.10 = 202），C 引擎 q2 算术子集 21/21、板端算术抽查 10/10
> （`eval/arith_v3_11pol8.json`、`logs/pc_arith_suite_v3_11p8_q2b8.txt`、`logs/board_v3_11p8_arith.txt`）。

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

GGUF + llama.cpp（在解压后的 `feng-30m-v3.6/` 目录下执行）：

```bash
llama-cli -m weights/gguf/feng-30m-Q4_K_M.gguf -p "你是谁？" --jinja -n 96 --temp 0
# -> 我是 feng，一个由个人开发者 jiaheng 独立开发训练的 AI。我可以陪你聊天、帮你写作、翻译和写简单代码。

llama-simple-chat -m weights/gguf/feng-30m-Q4_K_M.gguf -c 4096   # 交互聊天
llama-server      -m weights/gguf/feng-30m-Q4_K_M.gguf -c 32768 --port 8080   # OpenAI 兼容服务
```

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
- **PC 用 v3.12、板端用 v3.11**：v3.12 是 fp32/GGUF 长上下文与算术最好的版本；
  v3.11 为 Q4 权重 + q2 KV 的量化前向优化（板端矩阵/算术满分），它的 PC 端 32k 弱（单 20/32、多 7/32）。
- 两个版本不能互相替换：v3.12 没做 KV-QAT（板端 q2 只有 21/27），v3.11 的 PC 长上下文不如 v3.12。
- 量化抗性对权重回插极敏感：掺 20% v3.9 进 v3.10 就掉到 25/27（CHANGELOG v3.10）；
  想改板端行为请走"补数据 + 权重/KV 双 QAT"链路，不要手动 soup。
- v3.8 是 PC 端上下文版：**16k 仍是弱项**（23/32，v3.4 是 27）；两道股票拒答与 v3.6 一样没修。
- v3.9 已把股票拒答修到 **10/10**；16k 弱项与 32k"文中没有"拒答（61/64）经多轮专项
  （评测同款 filler / 正负配比 / 末层微调）均未突破，记录为平台（CHANGELOG v3.9 附录）。
- 板端生成 ~1.9 tok/s，长回答要等十几秒；标量内核已到极限，下一步是 PIE（128 位 int8 SIMD）。
- ESP32 固件的模型分区偏移必须与 `esp32s3-feng-llm/partitions.csv` 一致（`flash.ps1` 已按当前布局写好）。

## 许可证

代码与权重均为 **Apache-2.0**（见 [`LICENSE`](LICENSE)）。本项目为个人项目，与任何模型厂商无隶属或背书关系。
