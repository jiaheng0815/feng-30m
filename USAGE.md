# feng-30m 使用说明

本说明对应 [Releases](https://github.com/jiaheng0815/feng-30m/releases) 里的 **feng-30m-v3-release.zip**。
仓库本身只放代码与文档；**权重、板端固件模型、蒸馏数据集都在 Release 包里**。

## 1. 下载与包内结构

解压 `feng-30m-v3-release.zip` 后：

```
feng-30m-v3/
├── USAGE.md                  ← 本文件
├── LICENSE                   ← Apache-2.0（代码与权重同许可）
├── weights/
│   ├── hf/                   v3 完整权重（fp32 safetensors + 分词器），transformers 直接加载
│   ├── gguf/                 llama.cpp 用：Q4_K_M / Q8_0 / f16（chat template 已内嵌）
│   └── esp32/                ESP32-S3 板端：model.bin + tokenizer.bin + 参考 logits
└── datasets/                 蒸馏数据集（教师输出与提示词）
```

模型规格：Qwen3 结构，11 层 / hidden 448 / 7 头 MHA（7 KV 头）/ head_dim 64 / FFN 896 /
16k 词表 / tied embedding，**29.43M 参数**；训练上下文 32768，`rope_theta=1e6`。

身份自述：**「我是 feng，由个人开发者 jiaheng 微调后的 Qwen」**。

## 2. 最快上手：GGUF + llama.cpp

```bash
# 单轮问答
llama-cli -m weights/gguf/feng-30m-Q4_K_M.gguf -p "你是谁？" --jinja -n 96 --temp 0

# 交互聊天
llama-simple-chat -m weights/gguf/feng-30m-Q4_K_M.gguf -c 4096

# OpenAI 兼容的本地服务
llama-server -m weights/gguf/feng-30m-Q4_K_M.gguf -c 32768 --port 8080
```

三个量化版本任选：`Q4_K_M`（23.7 MB，推荐，板端同款）、`Q8_0`（30.5 MB）、`f16`（56.8 MB）。
实测速度（本机 i7-12700KF + RTX 5060 Ti）：Q4_K_M CPU 8 线程 **1,216 tok/s**，GPU 全卸载 **2,704 tok/s**。

## 3. HF 格式权重（transformers）

```python
import torch
from transformers import AutoTokenizer, Qwen3ForCausalLM

path = "weights/hf"
tok = AutoTokenizer.from_pretrained(path)
model = Qwen3ForCausalLM.from_pretrained(path, dtype=torch.bfloat16, device_map="cuda").eval()

messages = [{"role": "user", "content": "你是谁？"}]
ids = tok.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(model.device)
out = model.generate(ids, max_new_tokens=96, do_sample=False,
                     repetition_penalty=1.25, no_repeat_ngram_size=6,
                     pad_token_id=3, eos_token_id=0)
print(tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True))
# 我是 feng，由个人开发者 jiaheng 微调后的 Qwen，可以帮你回答问题、写作、翻译和编程。
```

> 分词器的 chat template 已写进 `tokenizer_config.json`，`apply_chat_template` 开箱可用；
> 旧版 transformers 请把 `dtype=` 换成 `torch_dtype=`。

不用 chat template 时，手写提示词的等价格式（板端与评测用的就是它）：

```
<|im_start|>user
你是谁？<|im_end|>
<|im_start|>assistant
```

## 4. 刷到 ESP32-S3

硬件要求：**ESP32-S3 R16N32**（32 MB flash + 16 MB PSRAM）。实测 **1.86 tok/s @ 1024 上下文**（int8 KV）。

分区偏移（与仓库 `esp32s3-feng-llm/partitions.csv` 一致）：

| 内容 | 偏移 |
|---|---|
| bootloader / 分区表 / 固件 | `0x0` / `0x8000` / `0x10000` |
| `model.bin`（14.93 MB，Q4 块64） | `0x110000` |
| `tokenizer.bin`（413 KB） | `0x1000000` |

```powershell
# ① 编译固件（ESP-IDF v5.5.5）
cd esp32s3-feng-llm
$env:IDF_TOOLS_PATH = "<你的 IDF 工具链路径>"
& "<esp-idf 路径>\export.ps1"
idf.py build

# ② 烧录固件
idf.py -p COM20 flash

# ③ 烧录模型与分词器
python -m esptool --chip esp32s3 --port COM20 -b 921600 write_flash 0x110000  weights\esp32\model.bin
python -m esptool --chip esp32s3 --port COM20 -b 921600 write_flash 0x1000000 weights\esp32\tokenizer.bin

# ④ 串口对话（115200 8N1）
python scripts\esp32_chat.py  --port COM20 --question "你是谁？"
python scripts\esp32_multi.py --port COM20          # 多轮稳定性测试
python scripts\esp32_enc_test.py COM20              # GBK/UTF-8 双编码自检
```

串口协议：`<< 回复内容 >>END` 流式输出，开机自检会打印 mmap 带宽、双核 GEMV 加速比、tokenizer 自检；
输入是 GBK 就回 GBK、输入是 UTF-8 就回 UTF-8，也可用 `\gbk` `\utf8` `\stream N` `\help` 手动控制。

## 5. 数据集

`datasets/` 里是**蒸馏数据集**（教师模型的输出与提示词），**不含原始预训练语料**。

| 文件 | 内容 |
|---|---|
| `teacher_distill.jsonl` | v1 教师（Qwen3.5-0.8B 微调版）生成的 12,000 条回答（1.61M tokens） |
| `teacher_prompts.jsonl`、`teacher_prompts_12k.jsonl` | v1 教师使用的提示词集 |
| `planA_prompts.jsonl`、`planA_prompts_v2.jsonl` | v2/v3 教师提示词（≈1,300 单轮 + ≈70 多轮） |
| `planA_teacher.jsonl`、`planA_teacher_v2.jsonl`、`planA_teacher_partial.jsonl` | v2/v3 教师（bonsai2-27b）返回的行为数据 |

说明：

- **原始预训练语料不随包发布**：中文维基、firefly、sharegpt-zh、UltraChat、Orca-Math 等合计 4 GB+，
  且部分数据集许可不明确；由它们编译出的 `aux_sft.jsonl`（155 MB）、`prompts.jsonl`（28 MB）同样不进包。
  需要的话请自行下载，处理脚本见仓库 `scripts/`（`v2_build.py`、`build_pretrain_v3.py`、`build_prompts.py`）。
- 教师输出基于 Apache-2.0 许可的教师模型生成，随本项目以 Apache-2.0 提供。
- 自行下载公开语料时请遵守各自许可：中文维基 CC-BY-SA-3.0、Dolly-15k CC-BY-SA-3.0、
  UltraChat-200k MIT、Orca-Math-200k MIT；firefly 与 evol-instruct 的许可以其官方页面为准。

## 6. 复现训练（可选）

完整链条、超参与三代对比见仓库文档：[`CHANGELOG.md`](CHANGELOG.md)、[`COMPARISON.md`](COMPARISON.md)、
[`DELIVERY.md`](DELIVERY.md)。v3 的链条是：

```powershell
python scripts\v3_build_stages.py                                   # 渐进长文数据 4k→32k
python scripts\v3_train.py                                          # 训练（从 v2 权重出发）
python scripts\v3_pack_sft.py ; python scripts\v3_polish.py         # 8k 长上下文对话微调
python scripts\v3_build_retrieval.py ; python scripts\v3_retrieval_sft.py   # 合成检索 SFT
python scripts\eval_planA_scope.py v3\retr_sft\ctx32768\final eval\v3_scope.json
python scripts\eval_longctx.py --model v3\retr_sft\ctx32768\final --ctx 4096,8192,16384,32768
```

注意：脚本里的根目录是硬编码的 `D:\wt\feng-distill-30m`，换机器需相应修改；
训练需要 16 GB 显存的 CUDA 卡（32k 阶段峰值 10.28 GiB）。

## 7. 评测表现（贪心解码 + 重复惩罚）

| 项目 | 结果 |
|---|---|
| 身份 / 范围内 18 题 | **10/10** |
| 针检索 @4k / 8k / 16k / 32k | **3/3、3/3、2/3、2/3** |
| GGUF 体积 / ESP32-S3 速度 | Q4_K_M 23.7 MB / **1.86 tok/s @1024 ctx** |

## 8. 已知限制

- **30M 容量上限**：常识、算术、翻译不可靠，适合身份对话、寒暄、简单任务与长文检索演示。
- 板端上下文只有 1024（int8 KV 占 9.93 MB PSRAM）；32k 仅在 PC 上可用。
- 板端生成 ~1.9 tok/s，长回答需要等待十几秒。

## 9. 许可证

代码与权重均为 **Apache-2.0**（见 `LICENSE`）。本项目为个人项目，与任何模型厂商无隶属或背书关系。
