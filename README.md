# feng-30m

**一个 29.43M 参数的中文对话模型：从零训练、原生 32k 上下文，Q4 量化后能塞进 ESP32-S3 离线对话。**

当前版本 **v3.6** ｜ 代码与权重均 **Apache-2.0** ｜ 身份自述：**「我是 feng，一个由个人开发者 jiaheng 独立开发训练的 AI」**

## 下载与使用

权重（fp32 / GGUF / ESP32 板端模型）与蒸馏数据集打包在 **[Releases](https://github.com/jiaheng0815/feng-30m/releases)**：
`feng-30m-v3.6-release.zip`（227 MB，解压后 259 MB；含 v3.6 权重、GGUF、板端模型与日常对话补丁数据集）。
**本仓库只放代码与文档，训练数据与权重不入库。**

包内结构：

- `weights/hf/` —— v3.6 完整权重（fp32 safetensors + 分词器 + chat template），transformers 直接加载
- `weights/gguf/` —— Q4_K_M 23.7 MB / Q8_0 30.5 MB / f16 56.8 MB，**chat template 已内嵌**
- `weights/esp32/` —— 板端 `model.bin`（14.93 MB）+ `tokenizer.bin`（413 KB）+ 参考 logits
- `datasets/` —— 蒸馏数据集（教师输出与提示词）+ 多轮对话（`v3_5_multiturn.jsonl`）+ 日常补丁与运算数据（`v3_6_*.jsonl`）

安装、推理、烧录的完整步骤见 [`USAGE.md`](USAGE.md)。

## 亮点

- **真的能上板**：Q4 块64 量化后 14.93 MB，落在 **flash 前 16 MB 的 mmap 窗口**内（NOR flash 24 位地址上限，**与模块 32 MB 容量无关**）；int8 KV 让板端上下文从 256 提到 **1024**，实测 **1.86 tok/s**
- **长上下文可用**：原生 32k 训练 + 合成检索 SFT + 硬负样本拒答训练；v3.6 针检索按 **每长度 32 题** 复测：
  @4k/8k/16k/32k = **27/29/24/22（合计 102/128，79.7%）**，多类别 99/128；
  文中没有答案时 **60/64（单类别）与 62/64（多类别）会说明"没有提到"**（v1~v3 是 0/64 全编造）。
  > 检索数字最高的历史版本是 v3.4（单类别 **113/128**）；v3.6 主动让出 2~5 题噪声级差距，
  > 换来下面这条"真的能闲聊"。
- **基础聊天可用（v3.6）**：42 题广谱日常探针 **42/42**（寒暄/告别/能力/情绪/写作/常识/小数字运算/翻译/安全拒答），
  情绪回应 8/8，多轮 7 轮不同回答比例 **1.00**（v3.0~v3.4 只有 0.57 且第 3 轮起复读）；会拒答炸弹/诈骗请求
- **同一套引擎**：C11 推理核心 PC 与板端共用，与 PyTorch(Q4) **逐位一致**（max|diff| = 0.0000），改内核有基线可回归
- **过程全公开**：每一代模型、每一次改动、每一次实测数字（含失败尝试）都写在文档里，数字都能对上脚本与日志

## 版本速览

| | v1 | v2 | v3 | v3.5 | **v3.6（当前）** |
|---|---|---|---|---|---|
| 结构 | 8 层 / 32k 词表 / 30.75M | 11 层 / 16k 词表 / 29.43M | 同 v2 | 同 v2 | 同 v2 |
| 教师 | Qwen3.5-0.8B 微调版 | bonsai2-27b（27B） | 同 v2 | 同 v2 | 同 v2 |
| 累计训练量 | ~110M tokens（仅指令数据） | +1.5B 预训练 +22.5M SFT +1.2M 补训 | +72M 长文 +22.7M 长上下文 SFT +12.5M 检索 SFT | +约 90M 多轮/检索 | +约 40M 日常补丁/恢复 |
| 身份自述 | 命中但退化 | 「微调后的 Qwen」 | 同 v2 | 「jiaheng 独立开发训练的 AI」 | **同 v3.5（12/12）** |
| 范围内评测 | 5/10 | 8/10 | 10/10 | 9/10 | 9/10 |
| 日常对话探针（42 题） | — | — | — | 29/42 | **42/42** |
| 针检索（每长度 32 题） | 0/0/0/0 | 0/0/0/0 | 31/30/26/19 | 27/29/25/25 | **27/29/24/22** |
| 「文中没有」拒答 | 0/64 | 0/64 | 0/64 | 59/64（92%） | **60/64（94%）** |
| 多轮对话（7 轮不同回答比例） | 0.57 | 0.57 | 0.57 | 1.00 | **1.00** |
| GGUF Q4_K_M | 27.6 MB（放不进 16MB 窗口） | 23.7 MB | 23.7 MB | 23.7 MB | 23.7 MB |
| ESP32-S3 实机 | ❌ 从未上板 | ✅ 1.56 tok/s @256 ctx | ✅ 1.86 tok/s @1024 ctx | ✅ 1.84 tok/s @1024 ctx | ✅ **10 轮 10/10（含情绪多轮）** |

横向对比与全部实测见 [`COMPARISON.md`](COMPARISON.md)，逐版本演进（含失败记录）见 [`CHANGELOG.md`](CHANGELOG.md)。

## 模型规格（v3.6）

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
| 长上下文 SFT | Plan A 语料重打包成 8192 窗口，避免短序列把窗口压回去 | 22.7M tokens（17.9M 有监督） | `scripts/v3_pack_sft.py`、`scripts/v3_polish.py` |
| 合成检索 SFT | 长文埋事实、只对答案算 loss（关键一步：只喂长文学不会检索） | 12.5M tokens | `scripts/v3_build_retrieval.py`、`scripts/v3_retrieval_sft.py` |
| v3.5 多轮修复 | 2,600 条多轮对话 × 高占比混训 + 检索补强，修"从第 3 轮起复读" | 约 90M tokens | `scripts/v3_5_build_multiturn.py`、`scripts/chat_multi.py` |
| v3.6 日常补丁 | 592 条日常对话 + 系统化小数字运算（×6），单条对话 SFT 后再检索回补 | 约 40M tokens | `scripts/v3_6_build_daily_patch.py`、`scripts/v3_6_build_drill.py`、`scripts/v3_6_sft_patch.py`、`scripts/chat_probe.py` |

完整超参、每阶段 loss/耗时/显存见 [`DELIVERY.md`](DELIVERY.md) 与 [`CHANGELOG.md`](CHANGELOG.md)。

## 仓库结构

```
scripts/            数据构建 / 训练 / 评测 / 导出脚本（44 个，路径解析见 scripts/paths.py）
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
- 板端上下文 1024（int8 KV 占 9.93 MB PSRAM）；**32k 只在 PC 上可用**，板上 32k 受 KV 内存限制不可能。
- 板端生成 ~1.9 tok/s，长回答要等十几秒；标量内核已到极限，下一步是 PIE（128 位 int8 SIMD）。
- ESP32 固件的模型分区偏移必须与 `esp32s3-feng-llm/partitions.csv` 一致（`flash.ps1` 已按当前布局写好）。

## 许可证

代码与权重均为 **Apache-2.0**（见 [`LICENSE`](LICENSE)）。本项目为个人项目，与任何模型厂商无隶属或背书关系。
