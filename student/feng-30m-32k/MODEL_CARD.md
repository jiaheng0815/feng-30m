# feng-30m v1（原生 32k 输入版）

由 feng-0.8b（Qwen3.5-0.8B bf16 微调版）蒸馏、**从头训练**的 **30.75M 参数**模型。
Qwen3 架构：**8 层 / hidden 448 / 7 头（7 个 KV 头，MHA）/ head_dim 64 / FFN 896 /
tied embedding，32k 词表 BPE**，`max_position_embeddings=32768`、`rope_theta=1e6`，
**不使用 YaRN/RoPE 插值**（在 32768 token 上直接训练）。

> 权重下载：见 [Releases](https://github.com/jiaheng0815/feng-30m/releases)。当前 Release 发布 **PC v3.14 + 板端 v3.16-embed**（v3.17 引擎）；
> v1 作为历史对照保留在文档与 `student/` 的训练记录里，未随包发布。

- 训练：8k 指令阶段 → 32k 长文阶段（维基 + 对话拼接 + 大海捞针）；
  各阶段 tokens_seen：stageA 37.9M、stageB 17.7M、stageC 7.9M（`student/*/summary.json`）
- 对话格式：`<|im_start|>user\n…<|im_end|>\n<|im_start|>assistant\n…<|im_end|>`
- 建议采样：`temperature=0.7, top_p=0.9, repetition_penalty=1.15`

## 实测能力（贪婪）

| 项目 | 结果 |
|---|---|
| 接收 32768 token 输入 | ✅ 可以（不报错、能生成） |
| **针检索** | ❌ **0/3 @4k、0/3 @8k、0/3 @16k、0/3 @32k**（输出是复读式乱码） |
| 身份 | ✅ 能命中，但会退化重复 |
| 常识 / 算术 / 翻译 | ⚠️ 基本不可靠 |
| GGUF | Q8_0 32.5 MB / f16 60 MB |

> **结论**：v1 的"原生 32k"是**长度**能力，不是**检索**能力。实测证明只喂长文本学不会长上下文，
> 必须配合长文阶段 + 合成检索数据 SFT——v3 用同样的 30M 预算做到了 4k–32k 检索 3/3、3/3、2/3、2/3。
