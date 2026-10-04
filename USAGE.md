# feng-30m 使用说明

本说明对应 [Releases](https://github.com/jiaheng0815/feng-30m/releases)：
**PC 用 `feng-30m-v3.19-release.zip`（HF 权重 + 蒸馏数据集），
板端用 `feng-30m-v3.19-embed-release.zip`（q2 KV / 2048 ctx + 双 QAT + 多轮上下文 + tool + 身份稳定）**。
在 PC 上跑 C 引擎（体验 tool）再取 `feng-30m-v3.19-engine.zip`（引擎源码）+
`feng-30m-c-engine-model-v3.19-embed.zip`（**已导出的模型，免装 torch**）。
仓库本身只放代码与文档；**权重、板端固件模型、蒸馏数据集都在 Release 包里**。

## 1. 下载与包内结构

解压 PC 包 `feng-30m-v3.19-release.zip` 后：

```
feng-30m-v3.19/
├── USAGE.md                  ← 本文件
├── LICENSE                   ← Apache-2.0（代码与权重同许可）
├── weights/
│   ├── hf/                   v3.19 完整权重（fp32 safetensors + 分词器），transformers 直接加载
│   └── （v3.14 起不再提供 GGUF：llama.cpp 没有 tool，算术/时间/随机数会退化成模型硬算）
└── datasets/                 蒸馏训练数据（教师输出与提示词）
```

> 板端的 `model.bin` / `tokenizer.bin` 不在 PC 包里，请下载 **v3.19-embed** 的 Release
> （它的 `weights/esp32/` 就是可以直接烧录的板端模型）。
> PC 上跑 C 引擎用同一份双 QAT 权重：已导出好的见 `feng-30m-c-engine-model-v3.19-embed.zip`。

模型规格：Qwen3 结构，11 层 / hidden 448 / 7 头 MHA（7 KV 头）/ head_dim 64 / FFN 896 /
16k 词表 / tied embedding，**29.43M 参数**；训练上下文 32768，`rope_theta=1e6`。

身份自述：**「我是 feng，一个由个人开发者 jiaheng 独立开发训练的 AI」**（v3.2 起）。

## 2. 最快上手：C 引擎 `pc_chat`（自带 tool，取代 llama.cpp）

```powershell
# 1) 取模型：直接下载 Release 的 feng-30m-c-engine-model-v3.19-embed.zip
#    想自己导出：用【板端 v3.19-embed 权重包】的 weights/hf（做过 Q4+q2 双 QAT）
python esp32s3-feng-llm\tools\export_model.py --model <v3.19-embed包>\weights\hf --out model_export
# 2) 编译（MSYS2 gcc，q2 KV；不带 -D 则 int8/1024 ctx）
gcc -O2 -DFENG_KV_Q2=1 -DFENG_KV_Q2_BLOCK=8 -o pc_chat.exe `
  esp32s3-feng-llm\pc\pc_chat.c esp32s3-feng-llm\main\feng_model.c `
  esp32s3-feng-llm\main\feng_llm.c esp32s3-feng-llm\main\feng_quant.c `
  esp32s3-feng-llm\main\feng_smp.c esp32s3-feng-llm\main\feng_tokenizer.c `
  esp32s3-feng-llm\main\feng_calc.c esp32s3-feng-llm\main\feng_tools.c -Iesp32s3-feng-llm\main -lm
# 3) 聊天（算式/时间/随机数 0.5s 秒回，多轮上下文默认开）
.\pc_chat.exe model_export
```

> **别用 PC 的 v3.14 HF 权重导出给 C 引擎**：C 引擎的 `model.bin` 是 Q4 block-64 + q2 KV，
> 只有做过双 QAT 的板端权重扛得住。实测同一套 32 题矩阵：PC 权重 **22/27**（翻译/情绪/推荐崩），
> 板端 QAT 权重 **27/27 + 召回 4/4**（`logs/pc_kv_suite32_v3_14pc2_q2b8.txt` /
> `logs/pc_kv_suite32_v3_16p3_recheck.txt`）。PC 的 HF 权重请走 transformers（第 3 节）或 Python 脚本。

也可以直接用 Python 脚本（同一套 tool，`scripts/runtime_tools.py`）：

```powershell
python scripts\chat_student.py --model v3_14\pc2 --prompt "4854+4411"
python scripts\chat_student.py --model v3_14\pc2 --prompt "现在几点？"
```

> **GGUF 已取消发行**：llama.cpp 路径没有 tool，v3.14 起不再随 Release 提供 GGUF。
> 想复现历史速度数字，仓库里仍保留 `scripts/export_student_gguf.py`，但不作为发行物。
> tool 支持：多位数/小数/括号/中文数字（`五十九加一`）/`乘以、除以`/百分号（`一百*15%`）、
> `100的15%`、`12的平方`、`根号16`、`现在几点`、`3天后是几号`、`随机数`、`掷骰子`、`抛硬币`。

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

v3.11 提供 **q2 KV（block8）固件**：把上下文从 1024 提到 **2048**（KV 9.62 MB），
PC 端 C 引擎 32 题矩阵 **27/27 + 4/4**（与 int8 持平）、算术子集 **21/21**（q2 + Q4 同口径），
板端默认/情绪/算术三组 10 题 **10/10 + 10/10 + 10/10**、实测 **1.81 tok/s**
（`logs/pc_kv_suite32_v3_11p8_q2b8.txt`、`logs/pc_arith_suite_v3_11p8_q2b8.txt`、
`logs/board_v3_11p8_multi.txt`、`logs/board_v3_11p8_arith.txt`）。
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

> **不想装 ESP-IDF？** 直接下载 Release 附件 `feng-30m-v3.19-embed-firmware.zip`
> （预编译 v3.17 固件 + 模型 + 哈希清单，esptool 按上表偏移一次写完即可），跳过下面 ①。

```powershell
# ① 编译固件（ESP-IDF v5.5.5；用预编译包可跳过）
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

## 7. 评测表现（PC = v3.19，贪心解码；脚本与结果 JSON 都在仓库里）

| 项目 | 结果 |
|---|---|
| 身份（12 题，自称 jiaheng 独立开发训练） | **12/12** |
| 日常对话探针（42 题，0 模板泄漏 / 0 复读） | **42/42**（`eval/v3_19pc4_probe42.json`） |
| 情绪回应（8 题，与 v3.5 同口径） | **8/8**（v3.5 为 7/8，v3.4 为 5/8） |
| 多轮对话（7 轮不同回答比例） | **1.00**（v3.0~v3.4 为 0.57） |
| 范围内 18 题 | **10/10**（v3.7 = 10/10、v3.8 = 8/10，同口径） |
| 针检索 @4k / 8k / 16k / 32k（每长度 32 题） | **28/30/26/26（单类别 110/128）**；**多类别 105/128** |
| 「文中没有该信息」正确拒答 | **61/64（单类别）、62/64（多类别）** |
| 留出 30 题（开发集，strict-v4 判定） | **19/30**（v3.14 为 16/30；`eval/v3_19pc4_heldout30.json`） |
| 多轮记忆 24 题（说事实→追问） | **23/24**（v3.14 为 24/24；`eval/v3_19pc4_memory24.json`） |
| 算式 / 网络时间 / 随机数 | **C 引擎 tool：0.5s 全对**（`4854+4411=9265`、`5.3+4.1=9.4`、UTC+8 时间、随机数） |
| 多轮记忆 / 身份（v3.17 引擎） | **12 题连续记忆 12/12**、`你叫什么名字？` 永不串名（0.5s 秒回）；`\mem` 查看 |

嵌入式（v3.19-embed，q2 block8 / 2048 ctx）：32 题矩阵 **27/27 + 4/4**、
算术子集 **21/21**（数学题由 tool 回答）、板端 tool 专项（时间/随机数/算式）**13/13**
（`logs/board_tools_time_rand.txt`）+ 记忆 12 题 **12/12**
（`logs/board_v3_19b6_memory12.txt`）、留出 30 题 **19/30**（v3.16-embed 为 17/30；
`eval/v3_19board6_heldout30.json`）、多轮回归套件（PC，固件同款采样）7/10（`logs/pc_mtsuite_v3_19b6.txt`），
约 **1.80 tok/s**；
**固件保留跨轮上下文**：实测"我叫小明 → 你叫小明"、"喜欢蓝色 → 你最喜欢蓝色"、
"养了一只猫 → 你养了一只猫"全对；`\reset` 可清空，上下文满（2048）自动开新对话
（`logs/pc_kv_suite32_v3_13b_q2b8.txt`、`logs/board_v3_13b_memory.txt`）。
  v3.19-embed 在 v3.16-embed 之上用 440 条定向教师数据做 Q4+q2 双 QAT（4 epoch / lr 6e-6），
  板端 7 轮换名身份序列实测 **7/7**（`logs/esp32_multi.txt`，需 `--no-reset` 保留上下文）；
  矩阵/召回/算术/记忆全部保持（`logs/pc_kv_suite32_v3_19b6_q2b8.txt`）；
  另含全部推理优化：长上下文单次 forward 比 v3.15-embed 快 37%（见 CHANGELOG v3.15-embed 附录）。

## 8. 已知限制

- **30M 容量上限**：v3.6 覆盖了常见寒暄/情绪/常识/小数字运算/翻译/推荐等日常问法（42 题探针全过），
  但没覆盖到的自由问答仍可能答偏或编造；复杂推理与专业领域不可靠。
- 板端 int8 KV 是 1024 上下文；q2 KV（v3.19-embed）是 2048。32k 仅在 PC 上可用。
- 板端生成约 1.9 tok/s（约 520 ms/token，不含 prefill），长回答需要等待十几秒。
- PC 版 v3.19：单类别 110；多类别 105（v3.14 为 107）；32k"文中没有"拒答 61/64 是已知平台。
- 板端权重（v3.19-embed）+ v3.17 引擎：q2 矩阵满分、12 题记忆 **12/12**、7 轮身份序列 **7/7**
  （身份/常见事实由引擎记忆 tool 确定性回答），但 **PC 32k 弱于 PC 版**；
  量化鲁棒性对权重回插极敏感（掺 20% v3.9 就掉到 25/27）——要改板端行为请走
  「补数据 + 权重/KV 双 QAT」链路，不要手动 soup（CHANGELOG v3.10/v3.11）。
- 32k 负样本拒答（61/64）经多轮专项训练未突破，已记录为平台（CHANGELOG v3.9 附录）。
- **记忆 = 引擎 tool + 上下文**：可枚举事实由 v3.17 引擎记住并确定性回答（12 题 **12/12**），
  `\mem` 查看、`\reset`/重启/写满即忘；分布外的自由说法仍会错（30M 容量边界）。
- **板端长文很慢**：单行输入上限 4095 字节，但 prefill 是 O(n²)——≈800 tokens 要 ~291 s
  （注意力本身就要 n² 次 KV 访问）。交互输入建议 ≤ ~150 tokens；长文/批量任务用 PC 版。
- **tool 只在带 tool 的运行时里**：板端固件 / PC C 引擎 `pc_chat` / Python 脚本；
  GGUF、llama.cpp 没有 tool，v3.14 起不再发行 GGUF。
- **板端时间靠宿主对时**：串口脚本会自动发 `\settime <unix秒>`（宿主走 NTP）；
  不跑脚本时要手动发一次，否则时间 tool 会回答"还没对上网络时间"。
- **PC 用 v3.19（HF）；板端与 C 引擎用 v3.19-embed**：两个权重不能互换——PC 的 HF 权重没做
  量化感知训练，导进 C 引擎（Q4+q2）会退化（同套 32 题矩阵实测 **22/27 vs 27/27**，
  `logs/pc_kv_suite32_v3_14pc2_q2b8.txt`）；板端 QAT 权重在 PC 32k 长上下文上不如 PC 版。

## 9. 许可证

代码与权重均为 **Apache-2.0**（见 `LICENSE`）。本项目为个人项目，与任何模型厂商无隶属或背书关系。
