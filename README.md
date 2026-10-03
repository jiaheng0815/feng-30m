# feng-30m v1：从 feng-0.8b（bf16 微调版）蒸馏的 30M 对话模型

> **本文档记录的是 v1（2026-10-01）**，后续两个版本请见
> [`CHANGELOG.md`](CHANGELOG.md)（v1→v3 更新日志）、[`COMPARISON.md`](COMPARISON.md)（三代对比）、
> [`DELIVERY.md`](DELIVERY.md)（当前交付物与 ESP32 部署）。
> v2/v3 已把词表换成 16k、层数加到 11 层，并换成 27B 教师；v1 的产物仍保留在 `student/` 下。

> **下载与使用**：权重（fp32 / GGUF / ESP32 板端模型）与蒸馏数据集打包在
> [Releases](https://github.com/jiaheng0815/feng-30m/releases)，本仓库只放代码与文档；
> 安装、推理、烧录步骤见 [`USAGE.md`](USAGE.md)。代码与权重均为 Apache-2.0。

目标：用 **feng 0.8B（Qwen3.5-0.8B 全参数微调、个人开发者 jiaheng 微调版，bf16）** 作教师，
**从头训练**一个 30M 参数、**原生 32k 上下文**、可正常对话的小模型。

## 流水线

| 阶段 | 内容 | 脚本 |
|---|---|---|
| 1. 数据 | 采集 7 个优质问题集（ShareGPT-zh 38k / Evol-Instruct-zh / Alpaca-zh / Dolly-15k / UltraChat-200k / Orca-Math-200k / 身份题） | `scripts/build_prompts.py` |
| 2. 教师蒸馏 | feng-0.8b(bfloat16 GGUF) 经 llama-server 持续批处理生成回答（每槽 2048 ctx，`<|im_end|>` 截断） | `scripts/teacher_generate.py` |
| 3. 辅助 SFT | 公开数据集里现成的高质量回答（10.3 万条对话，3900 万 token） | `scripts/extract_aux.py` |
| 4. 分词器 | 32k BPE（ByteLevel），特殊符 `<|im_start|>/<|im_end|>/<|endoftext|>/<|pad|>` | `scripts/prepare_corpus.py` |
| 5. 学生模型 | 从零初始化：Qwen3 架构，**8 层 / hidden 448 / 7 头（7 个 KV 头，MHA）/ head_dim 64 / FFN 896 / tied embedding = 30.75M 参数**（实测值，以 `student/feng-30m-chat/config.json` 为准；设计稿里的"9 层 / MQA"未采用） | `scripts/student_config.py` |
| 6. 预训练+SFT | **原生 32k**（rope_theta 1e6，无插值）：8k 指令阶段 → 32k 长文阶段（维基 + 对话拼接 + 大海捞针） | `scripts/train_student.py`、`scripts/train_longctx.py` |
| 7. 评测/导出 | 身份+通用+32k 检索；GGUF f16/Q8_0/Q4_K_M | `scripts/eval_student.py`、`scripts/export_student_gguf.py` |

## 关键设计

- **原生 32k**：`max_position_embeddings=32768`、`rope_theta=1e6`，**不使用 YaRN/RoPE 插值**；
  32k 阶段直接用 32768 token 的序列训练（含长文档与检索任务）。
- **显存安全**：lm_head+交叉熵按 512 token 分块并 `checkpoint`，32k×32k 词表的 logits 不会整体实例化。
- **身份保留**：上一版 feng 身份对话（约 570 条）在语料中**过采样 15×**，确保 30M 学生仍自称 feng/jiaheng。
- **教师生成配置教训**：llama.cpp 的 `--parallel N` 会把 `-c` 均分给 N 个槽位，
  必须用 `-c = 每槽上下文 × N`，否则长提示被静默截断（本项目第一轮 2 万条数据因此作废重跑）。

## 进度 / 结果

### 交付物

| 目录 | 说明 |
|---|---|
| `student/feng-30m-chat/` | **对话版（推荐）**：8k 指令训练 + 非打包对话微调；`student/feng-30m-chat/gguf/feng-30m-Q4_K_M.gguf` **27.6 MB** |
| `student/feng-30m-32k/` | **原生 32k 版**：在 32768 token 上下文上直接训练（无 RoPE 插值），`student/feng-30m-32k/gguf/feng-30m-Q8_0.gguf` 32.5 MB |
| `student/tokenizer/` | 自训 32k BPE 分词器 |
| `data/teacher_distill.jsonl` | 教师（feng-0.8b bf16）蒸馏的 12,000 条回答（1.61M tokens） |
| `data/aux_sft.jsonl` | 10.27 万条公开高质量对话（39M tokens） |

### 训练结果

| 阶段 | 数据 | 步数 / tokens | loss | 耗时 |
|---|---|---|---|---|
| A 8k 指令（打包） | 38M tokens | 289 / 37.9M | 10.5 → 5.38（val 5.21） | 24 min |
| A2 8k 续训 | 同上 2 轮 | 564 / 73.9M | → 3.82（val 3.86） | 47 min |
| C 原生 32k | 12M tokens（对话 85%） | 120 / 7.9M | 3.85 | 9 min |
| 对话微调（非打包，推荐版） | 12.3 万段对话 | 2900 / ~40M | ~5.0 | ~20 min |

### 能力实测（贪婪解码 + 重复惩罚 1.15）

| 项目 | 结果 |
|---|---|
| 身份（feng / 个人开发者 jiaheng / Qwen 微调） | ✅ 稳定命中（"我是 feng，由个人开发者 jiaheng 微调后的 Qwen"） |
| 问候/寒暄 | ✅ 正常（"你好！今天我能为您做些什么？"） |
| 常识/算术/翻译 | ⚠️ 弱（30M 容量 + 训练量限制），多数回答不准确 |
| 32k 大海捞针检索 | ❌ 0/3（模型能处理 32768 token 输入，但检索能力未学会） |
| GGUF 体积 | Q4_K_M **27.6 MB** / Q8_0 32.5 MB / f16 60 MB |
| llama.cpp 支持 | ✅ `llama-server` 直接可用（chat template 已内嵌） |

> **重要澄清**：上文"原生 32k"指**训练与可接受的输入长度**（无 RoPE 插值，直接在 32768 token 上训练），
> **不等于具备长文检索能力**——同协议针检索在 4k/8k/16k/32k 上实测 **0/3**（对照：v3 是 3/3、3/3、2/3、2/3）。
> 长上下文能力要靠长文阶段 + **合成检索数据 SFT** 才能获得，详见 `CHANGELOG.md`。
> 另：v1 的 Q4_K_M 为 27.6 MB，**放不进 ESP32-S3 只能映射前 16 MB flash 的窗口**，因此 v1 从未上板。

### 结论与建议

30M 模型在 ~110M tokens（含 37M 指令数据）训练后可以做到：**身份正确、能进行简单寒暄与短问答**；
但受参数量限制，常识、算术、翻译等仍不可靠。若需要明显更强的能力，建议：
1. 继续训更多 tokens（当前只相当于 Chinchilla 最优量的 ~1/5）；
2. 或把学生放大到 60-100M（同样流程，改 `scripts/student_config.py` 的层数/宽度即可）。
