# feng-30m v1（对话版）

由 feng-0.8b（Qwen3.5-0.8B 全参数微调，个人开发者 jiaheng 微调版，bf16）蒸馏、**从头训练**的
**30.75M 参数**对话模型。Qwen3 架构：**8 层 / hidden 448 / 7 头（7 个 KV 头，MHA）/ head_dim 64 /
FFN 896 / tied embedding，32k 词表 BPE**。

> 权重下载：见 [Releases](https://github.com/jiaheng0815/feng-30m/releases)。**当前 Release 发布 v3.6 权重**；
> v1 作为历史对照保留在文档与 `student/` 的训练记录里，未随包发布。

- 训练：8k 指令阶段 37.9M + 续训 73.9M tokens（`student/stageA`、`stageA2` 的 summary.json），
  之后做 32k 长文阶段（17.7M + 7.9M），最后做对话微调 chat2（3.4M tokens / 1400 步）
- 上下文：本目录（对话版）为 **8192**；32k 输入版见 `../feng-30m-32k/`
- 对话格式：`<|im_start|>user\n…<|im_end|>\n<|im_start|>assistant\n…<|im_end|>`
- 身份：feng（个人开发者 jiaheng 微调的 Qwen）
- 建议采样：`temperature=0.7, top_p=0.9, repetition_penalty=1.15`

## 实测能力（贪婪 + 重复惩罚 1.15）

| 项目 | 结果 |
|---|---|
| 身份 | ✅ 能命中，但**会退化重复**（"我是 fwen…"，且问什么都答身份） |
| 寒暄 | ✅ 正常 |
| 常识 / 算术 / 翻译 | ⚠️ 基本不可靠 |
| 针检索（4k/8k/16k/32k，同协议） | ❌ **0/3 ×4** |
| 范围评测（18 题） | 5/10 |
| GGUF | Q4_K_M 27.6 MB / Q8_0 32.5 MB / f16 60 MB |
| llama.cpp CPU 8 线程 / GPU（tg64，2026-10-03 复测） | 1,125 ±76 / 2,532 ±142 tok/s |
| ESP32-S3 | ❌ 27.6 MB 超出 16 MB mmap 窗口，未部署 |

> 后续版本：v2（16k 词表 + 11 层 + 1.5B 预训练 + 27B 教师，8/10）、
> v3（渐进长上下文 + 合成检索 SFT，**10/10**，检索 3/3、3/3、2/3、2/3，已上 ESP32-S3 1.86 tok/s）、
> v3.6（日常对话补丁：42 题探针 42/42、检索 27/29/24/22、板端 1.85–1.86 tok/s，**当前发布版**）。
> 详见项目根目录 `CHANGELOG.md` / `COMPARISON.md`。
