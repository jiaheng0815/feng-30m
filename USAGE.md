# feng-30m 使用说明

本说明对应 [Releases](https://github.com/jiaheng0815/feng-30m/releases) 里的 **feng-30m-v3.9-release.zip**。
仓库本身只放代码与文档；**权重、板端固件模型、蒸馏数据集都在 Release 包里**。

## 1. 下载与包内结构

解压 `feng-30m-v3.9-release.zip` 后：

```
feng-30m-v3.9/
├── USAGE.md                  ← 本文件
├── LICENSE                   ← Apache-2.0（代码与权重同许可）
├── weights/
│   ├── hf/                   v3.9 完整权重（fp32 safetensors + 分词器），transformers 直接加载
│   ├── gguf/                 llama.cpp 用：Q4_K_M / Q8_0 / f16（chat template 已内嵌）
│   └── esp32/                ESP32-S3 板端：model.bin + tokenizer.bin + 参考 logits
└── datasets/                 蒸馏训练数据（教师输出与提示词）
```

模型规格：Qwen3 结构，11 层 / hidden 448 / 7 头 MHA（7 KV 头）/ head_dim 64 / FFN 896 /
16k 词表 / tied embedding，**29.43M 参数**；训练上下文 32768，`rope_theta=1e6`。

身份自述：**「我是 feng，一个由个人开发者 jiaheng 独立开发训练的 AI」**（v3.2 起）。

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
实测速度（本机 i7-12700KF + RTX 5060 Ti，llama-bench tg64、3 次平均）：
Q4_K_M CPU 8 线程 **约 1.2k tok/s（1,175 ±93）**，GPU 全卸载 **约 2.6k tok/s（2,638 ±137）**；
逐次波动约 ±8%，pp32 波动更大（±30% 以上）不作为指标（日志见 `logs/bench_v3_6_q4km_*.log`）。

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
# 我是 feng，一个由个人开发者 jiaheng 独立开发训练的 AI。我可以陪你聊天、帮你写作、翻译和写简单代码。
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

硬件要求：**ESP32-S3-WROOM-2-N32R16V**（32 MB Octal SPI flash + 16 MB Octal SPI PSRAM，1.8 V）。
**不能用 WROOM-1 等 Quad/3.3 V 模块替代**——固件按 `ESPTOOLPY_OCT_FLASH` 构建，会烧写或启动失败；
flash 必须 **32 MB**：只有**前 16 MB 能被 mmap 直读**（NOR flash 24 位地址上限，**不是模块容量**），
模型就放这里；tokenizer 分区在 16 MB 之后，用 `esp_partition_read` 读。
PSRAM 必须 **16 MB**（8 MB 版本放不下 1024 ctx 的 KV）。
实测 **1.85–1.86 tok/s @ 1024 上下文**（int8 KV；v3.6 板端两次实测为 1.86 / 1.85 tok/s，
见 `logs/board_v3_6_speed.txt`；v3.4 逐轮 10.2–28.9 s，未单独记录 tok/s）。

v3.7 额外提供 **q2 KV（block8）固件**：把上下文从 1024 提到 **2048**（KV 9.62 MB），
实测 10/10 + 情绪多轮 10/10（`logs/board_v3_7_multi.txt`、`logs/board_v3_7_chat.txt`）。
编译命令：

```powershell
idf.py -DFENG_USE_Q2_KV=ON build          # 默认（不带该参数）仍是 int8 / 1024 ctx
```

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
| `teacher_distill.jsonl` | v1 教师（Qwen3.5-0.8B 微调版）生成的 12,000 条回答（文件内 `tokens` 字段合计 1.61M） |
| `teacher_prompts.jsonl`、`teacher_prompts_12k.jsonl` | v1 教师使用的提示词集 |
| `planA_prompts.jsonl`、`planA_prompts_v2.jsonl` | v2/v3 教师提示词，共 1,408 条请求（376 + 1,032；后者含 20 组多轮对话） |
| `planA_teacher.jsonl`、`planA_teacher_v2.jsonl`、`planA_teacher_partial.jsonl` | v2/v3 教师（bonsai2-27b）返回的行为数据 |

说明：

- **原始预训练语料不随包发布**：中文维基、firefly、sharegpt-zh、UltraChat、Orca-Math 等合计 4 GB+，
  且部分数据集许可不明确；由它们编译出的 `aux_sft.jsonl`（155 MB）、`prompts.jsonl`（28 MB）同样不进包。
  需要的话请自行下载，处理脚本见仓库 `scripts/`（`v2_build.py`、`build_pretrain_v3.py`、`build_prompts.py`）。
- **本地脚本生成的补丁数据也不进包**：v3.5 的多轮对话、v3.6 的日常补丁/运算/推荐修复数据由
  `scripts/v3_5_build_multiturn.py`、`scripts/v3_6_build_daily_patch.py`、`scripts/v3_6_build_drill.py`
  按固定随机种子生成，可完全复现，因此不随 Release 发布。
- 教师输出基于 Apache-2.0 许可的教师模型生成，随本项目以 Apache-2.0 提供。
- 自行下载公开语料时请遵守各自许可：中文维基 CC-BY-SA-3.0、Dolly-15k CC-BY-SA-3.0、
  UltraChat-200k MIT、Orca-Math-200k MIT；firefly 与 evol-instruct 的许可以其官方页面为准。

## 6. 复现训练（可选）

完整链条、超参与版本对比见仓库文档：[`CHANGELOG.md`](CHANGELOG.md)、[`COMPARISON.md`](COMPARISON.md)、
[`DELIVERY.md`](DELIVERY.md)。v3 的链条是：

```powershell
python scripts\v3_build_stages.py                                   # 渐进长文数据 4k→32k
python scripts\v3_train.py                                          # 训练（从 v2 权重出发）
python scripts\v3_pack_sft.py ; python scripts\v3_polish.py         # 8k 长上下文对话微调
python scripts\v3_build_retrieval.py ; python scripts\v3_retrieval_sft.py   # 合成检索 SFT
python scripts\eval_planA_scope.py v3\retr_sft\ctx32768\final eval\v3_scope.json
python scripts\eval_longctx.py --model v3\retr_sft\ctx32768\final --ctx 4096,8192,16384,32768
```

v3.5 / v3.6 的后续链条（在当前发布版之上继续训练时）：

```powershell
python scripts\v3_5_build_multiturn.py --out v3_5d\mt_convs.jsonl --n 2600     # 多轮对话数据
python scripts\v3_6_build_daily_patch.py --out v3_6a\daily_patch.jsonl          # 592 条日常补丁
python scripts\v3_6_build_drill.py --out v3_6e\drill.jsonl                      # 运算/细节打磨
python scripts\v3_6_sft_patch.py --init <起点> --patch v3_6e\drill.jsonl `
  --mt v3_5d\mt_convs.jsonl --mt-n 400 --identity-n 150 --out <输出> --epochs 8 --lr 1e-4
python scripts\eval_longctx_many.py --models "<输出>" --n 32 --neg-n 16          # 检索回归
```

注意：脚本不再写死路径——根目录按脚本位置推导，外部工具（llama.cpp、教师模型、原始数据）
用环境变量 `FENG_LLAMA_DIR` / `FENG_TEACHER_GGUF` / `FENG_DATA_DIR` 或 `scripts/local_paths.json` 指定，
自检命令 `python scripts/paths.py`。**只复现 v3 的话只需要 `FENG_LLAMA_DIR`**——教师模型与原始语料
只在重建 v1/v2 语料时才需要。训练需要 16 GB 显存的 CUDA 卡（32k 阶段峰值 10.28 GiB）。

## 7. 评测表现（v3.9，贪心解码；脚本与结果 JSON 都在仓库里）

| 项目 | 结果 |
|---|---|
| 身份（12 题，自称 jiaheng 独立开发训练） | **12/12** |
| 日常对话探针（42 题，0 模板泄漏 / 0 复读） | **42/42**（`eval/chat_probe_v3_9_sf2.json`） |
| 情绪回应（8 题，与 v3.5 同口径） | **8/8**（v3.5 为 7/8，v3.4 为 5/8） |
| 多轮对话（7 轮不同回答比例） | **1.00**（v3.0~v3.4 为 0.57） |
| 范围内 18 题 | **10/10**（v3.7 = 10/10、v3.8 = 8/10，同口径） |
| 针检索 @4k / 8k / 16k / 32k（每长度 32 题） | **29/29/23/27（单类别 108/128）**；**多类别 108/128（历史最好）** |
| 「文中没有该信息」正确拒答 | **61/64（单类别）、62/64（多类别）** |

嵌入式（v3.7，q2 block8 / 2048 ctx）：32 题矩阵 **27/27 + 4/4**（与 int8 持平），
范围 18 题 10/10，板端 10 轮 10/10 + 情绪多轮 10/10，约 1.8 tok/s（`pc_kv_suite_q2b8.exe`）。

## 8. 已知限制

- **30M 容量上限**：v3.6 覆盖了常见寒暄/情绪/常识/小数字运算/翻译/推荐等日常问法（42 题探针全过），
  但没覆盖到的自由问答仍可能答偏或编造；复杂推理与专业领域不可靠。
- 板端 int8 KV 是 1024 上下文；q2 KV（v3.7）是 2048。32k 仅在 PC 上可用。
- 板端生成约 1.8–1.9 tok/s（约 540 ms/token，不含 prefill），长回答需要等待十几秒。
- PC 版 v3.9：16k 仍是弱项（23/32，v3.4 是 27）；单类别总量 108 低于 v3.4 的 113（多类别 108 历史最好）；
  **嵌入式请用 v3.7**（v3.9 未做 KV-QAT，q2 下 32 题矩阵 21/27 + 4/4）。
- v3.9 已把股票拒答修好（范围 10/10），16k 仍是唯一弱项（23/32）；**板端请用 v3.7**。

## 9. 许可证

代码与权重均为 **Apache-2.0**（见 `LICENSE`）。本项目为个人项目，与任何模型厂商无隶属或背书关系。
