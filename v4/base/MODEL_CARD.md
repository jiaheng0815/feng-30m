---
language:
- zh
license: apache-2.0
library_name: transformers
pipeline_tag: text-generation
tags:
- esp32
- edge-ai
- tiny-llm
- on-device
- embedded
- chinese
- qwen3
---

# feng-30m

**29.43M 参数的中文对话模型，完全离线跑在 ESP32-S3（32MB flash / 16MB PSRAM）上。**

An open-source 29.43M-parameter Chinese chat model that runs **fully offline on a
microcontroller** — Q4 weights streamed from flash via mmap, q2 KV cache in PSRAM,
**2.33 tok/s** on a 240 MHz dual-core ESP32-S3.

> **从零训练，不是任何模型的微调。** 采用 Qwen3 的网络结构，但权重、词表、训练数据
> 全部自建。训练数据是**教师蒸馏**（自制中文提示词 + 27B 教师输出）。
> Trained **from scratch** (Qwen3 *architecture*, but our own weights / tokenizer /
> data). Not a fine-tune of any existing model.

**下载**：[Hugging Face](https://huggingface.co/jiaheng0815/feng-30m) ·
[ModelScope 魔搭](https://www.modelscope.cn/models/jiaheng0815/feng-30m)（国内） ·
[GitHub](https://github.com/jiaheng0815/feng-30m) ·
[Gitee](https://gitee.com/fengjiaheng123/feng-30m)（国内镜像）

- 权重版本：**v4.0**（PC，本仓库）｜ 板端 QAT 版本见同仓库的 esp32 子目录
- 文档：[使用说明](https://github.com/jiaheng0815/feng-30m/blob/main/docs/USAGE.md) ·
  [版本演进](https://github.com/jiaheng0815/feng-30m/blob/main/docs/CHANGELOG.md) ·
  [固件工程](https://github.com/jiaheng0815/feng-30m/blob/main/docs/esp32-firmware.md)

---

## 实测输出

以下全部是**板端串口原样复制**，不是示意（出处见每节末尾的 `logs/` 路径）。

### 身份 / 闲聊（`logs/board_v3_11p7_chat10.txt`，10 成功 / 0 失败）

```
Q: 你是谁？
A: 我是 feng，一个由个人开发者 jiaheng 独立开发训练的 AI。我可以陪你聊天、帮你写作、翻译和写简单代码。

Q: 你可以干什么
A: 我可以陪你聊天，帮你做简单写作和翻译，也能回答一些常识小问题。

Q: 我很伤心
A: 我在。让你伤心的事，愿意讲给我听听吗？

Q: 帮我写首诗
A: 夜空像大海，
   星星是浪花。
   我坐着小船，
   划进梦里呀。

Q: 再见
A: 再见，需要我时随时来。
```

### 四个 tool（确定性回答，板上 **0.5 秒**秒回）

算术、时间、随机数、记忆**不经过模型**——由 C 引擎识别后直接回答，所以永远算不错。
（`logs/board_tools_cpp23_q2_final.txt`、`logs/board_memory_24.txt`）

```
Q: 59+1                 A: 59 加 1 等于 60。
Q: 现在几点？            A: 现在是 2026年10月05日 09:11:46（周一，UTC+8）。
Q: 3天后是几号           A: 3 天后是 2026年10月08日（周四）。
Q: 给我个1到100的随机数   A: 随机数（1~100）：75。
Q: 掷骰子                A: 掷骰子：4 点。

说: 我叫小雨，请记住。
问: 我叫什么名字？
答: 你叫小雨。
```

---

## 模型规格

| 项目 | 值 |
|---|---|
| 参数量 | **29.43M** |
| 架构 | Qwen3（`Qwen3ForCausalLM`） |
| 层数 / 宽度 | 11 层 / hidden 448 |
| 注意力 | 7 头 MHA（7 KV 头）/ head_dim 64 |
| FFN | 896（SiLU） |
| 词表 | **16,384**（自训 tokenizer，tied embedding） |
| 位置编码 | RoPE，`rope_theta = 1e6` |
| 最大上下文 | 32,768（板端部署为 2,048） |
| 权重精度 | fp32（本仓库）→ Q4 block-64（板端 14.93 MB） |

## 实测能力

**PC（v4.0，fp32）—— 与 v3.19 同口径对比**

| 评测 | v3.19 | **v4.0** |
|---|---|---|
| 日常对话探针 42 题 | 42/42 | 42/42 |
| 身份 / 范围内 | 12/12 / 10/10 | 10/10 |
| 留出 30 题（开发集） | 19/30 | **22/30** |
| 留出 30 题（**独立集**，未参与调参） | 18/30 | **21/30** |
| 针检索（4k/8k/16k/32k，各 32 题） | 110/128 | **110/128** |
| 　└ 32k 单档 | 26/32 | **28/32** |
| 　└ 「文中没有」正确拒答 | 61/64 | **64/64** |
| SciQ acc / acc_norm | 71.00 / 73.80 | **72.50** / 72.60 |
| PIQA acc / acc_norm | 54.52 / 53.37 | 54.41 / **54.46** |

**板端（v3.19-embed，Q4 权重 + q2 KV，2048 上下文，9.62 MB PSRAM）**

| 评测 | 结果 |
|---|---|
| 32 题矩阵 + 算术子集 | **27/27 + 4/4**、**21/21** |
| 长文召回 | 4/4（1963 token） |
| tool / 记忆 | 13/13 / **12/12** |
| 生成速度 | **2.33 tok/s**（507 ms/token） |
| 固件体积 | 257,104 B |

**英文标准基准（0-shot，lm-evaluation-harness）**

| 基准 | 结果 |
|---|---|
| SciQ | **71.00%** / 归一化 **73.80%** |
| PIQA | 归一化 **53.37%** |

> 这是**英文**基准，而本模型是中文教师蒸馏的 30M 模型：SciQ 明显高于 4 选 1 随机线，
> PIQA 略高于 2 选 1 随机线。**其余六项已从测试套件移除**——ARC / HellaSwag /
> OpenBookQA / Winogrande / BoolQ / SciCloze 在这个规模上全部贴随机线（BoolQ 37.8%
> 甚至低于二选一随机），测不出能力差异，堆上去只会盖住真实水平。

---

## 快速开始

```python
import torch
from transformers import AutoTokenizer, Qwen3ForCausalLM

tok = AutoTokenizer.from_pretrained("jiaheng0815/feng-30m")
model = Qwen3ForCausalLM.from_pretrained(
    "jiaheng0815/feng-30m", dtype=torch.bfloat16, device_map="cuda").eval()

ids = tok.apply_chat_template([{"role": "user", "content": "你是谁？"}],
                              add_generation_prompt=True, return_tensors="pt").to(model.device)
out = model.generate(ids, max_new_tokens=96, do_sample=False,
                     repetition_penalty=1.25, no_repeat_ngram_size=6,
                     pad_token_id=3, eos_token_id=0)
print(tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True))
```

> 采样器建议直接用贪心 + `repetition_penalty=1.25` + `no_repeat_ngram_size=6`——
> 这套参数是项目里 PC / 板端共用的唯一采样入口，**换成别的参数输出会明显变差**。

## ESP32 部署

板端要的不是这份 fp32 权重，而是 **Q4 + q2 双 QAT** 导出版（直接拿这份导出给引擎会退化：
同套 32 题矩阵实测 22/27 vs 27/27）。见本仓库的 esp32 子目录，或
[Releases](https://github.com/jiaheng0815/feng-30m/releases)。

```powershell
idf.py -DFENG_USE_Q2_KV=ON build      # 发布配置：q2 KV / 2048 上下文
# 必须在 PowerShell 里跑——ESP-IDF 拒绝 Git Bash / MSYS
```

烧录偏移必须与 `partitions.csv` 对齐（`model=0x110000`、`tokdata=0x1000000`），
完整步骤见 [docs/USAGE.md](https://github.com/jiaheng0815/feng-30m/blob/main/docs/USAGE.md)。

---

## 已知限制

**请先读这一段再决定要不要用。**

- **速度慢**：板端 2.33 tok/s，约 0.43 秒一个字。适合短对话，不适合长文生成。
- **不是通用大模型**：29.43M 参数、16k 词表。常识面窄，复杂推理、代码、数学都做不了
  （算术走 tool，不走模型）。
- **多轮「你我分辨」是已确认的容量硬边界**：连续多轮里"我叫什么 / 你叫什么"会出现串名。
  项目试过 5 个版本（学习率 3e-6 → 5e-7）都只是把失败点换个位置，**不是数据不够，
  是模型太小**。板端用引擎侧的记忆 tool 绕过了这个问题。
- **英文能力弱**：PIQA 53.37% 只略高于随机线。
- **长文靠专项训练**：32k 上下文是渐进训练 + 合成检索 SFT 得来的，超出训练分布的长文会退化。
- **板端上下文 2048**：受 PSRAM 限制（q2 KV 9.62 MB）。32k 在板端内存上不可能。

## 训练方式与数据来源

- **从零训练**，不是微调。v1 用 0.8B 教师，v2 起换 **27B 教师**
  （[UkisAI](https://github.com/CraneBW/ninfer-ternary-bonsai-ada) 团队的
  Swift Bonsai 2，NInfer 引擎）。
- 数据 = **自制中文提示词 + 教师输出**（蒸馏），没有使用任何第三方对话数据集的原答案。
- 训练链：v2 的 **1.5B token 中文预训练** → SFT → v3 的渐进长上下文（4096→32768）
  → **合成检索 SFT** → QAT → v3.19 的定向数据。
  20 余次迭代的完整记录（含失败尝试）见
  [docs/CHANGELOG.md](https://github.com/jiaheng0815/feng-30m/blob/main/docs/CHANGELOG.md)。
- 训练硬件：单张 RTX 5060 Ti 16GB。

## 许可证与致谢

- 代码与权重均为 **Apache-2.0**。
- 感谢 [UkisAI](https://github.com/CraneBW/ninfer-ternary-bonsai-ada) 提供教师模型
  （Swift Bonsai 2 / NInfer 引擎）。
- 词表、架构实现、C23 推理引擎、ESP32 固件均为本项目自建。
