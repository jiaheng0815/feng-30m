# feng-30m 使用说明

本说明对应 [Releases](https://github.com/jiaheng0815/feng-30m/releases)：

| 包 | 用途 | 内容 |
|---|---|---|
| `feng-30m-v3.19-release.zip` | PC（transformers） | HF fp32 权重 + 蒸馏数据集 |
| `feng-30m-v3.19-embed-release.zip` | 板端权重 | Q4+q2 双 QAT 权重 + 可直接烧录的 `model.bin` / `tokenizer.bin` |
| `feng-30m-v3.19-engine.zip` | PC 引擎源码 | C++23 引擎（v3.20，自带四个 tool） |
| `feng-30m-c-engine-model-v3.19-embed.zip` | 引擎预导出模型 | 免装 torch，直接跑 `pc_chat` |
| `feng-30m-v3.19-embed-firmware.zip` | 板端一包到底 | 预编译固件 + 模型 + 烧录说明（免装 ESP-IDF） |

仓库本身只放代码与文档；权重、板端模型与蒸馏数据集都在 Release 包里。
引擎当前版本 **v3.20 / C++23**：板端固件 **304,192 B**，板端 **2.26 tok/s**（938 ms/token）。

## 1. 包内结构

PC 包 `feng-30m-v3.19-release.zip`：

```
feng-30m-v3.19/
├── USAGE.md                  ← 本文件
├── LICENSE                   ← Apache-2.0（代码与权重同许可）
├── weights/
│   ├── hf/                   v3.19 完整权重（fp32 safetensors + 分词器 + chat template）
│   └── （v3.14 起不再提供 GGUF：llama.cpp 路径没有 tool，算术/时间会退化成模型硬算）
└── datasets/                 蒸馏训练数据（教师输出与提示词）
```

板端包 `feng-30m-v3.19-embed-release.zip` 的 `weights/esp32/` 即板端模型
（`model.bin` 14.93 MB + `tokenizer.bin` 413 KB + 参考 logits）。

模型规格：Qwen3 结构，11 层 / hidden 448 / 7 头 MHA（7 KV 头）/ head_dim 64 / FFN 896 /
16k 词表 / tied embedding，**29.43M 参数**；训练上下文 32768（`rope_theta=1e6`）；
板端运行 2048 上下文（q2 block8 KV，9.62 MB PSRAM）。
身份自述：**「我是 feng，一个由个人开发者 jiaheng 独立开发训练的 AI」**。

## 2. 最快上手：C++ 引擎 `pc_chat`（自带四个 tool）

```powershell
# 1) 取模型：直接下载 Release 的 feng-30m-c-engine-model-v3.19-embed.zip
#    想自己导出：用【板端 v3.19-embed 权重包】的 weights/hf（做过 Q4+q2 双 QAT）
python esp32s3-feng-llm\tools\export_model.py --model <v3.19-embed包>\weights\hf --out model_export

# 2) 编译（Windows 默认 MSVC + C++23，脚本自动加载 vcvars64；产出 13 个 exe）
esp32s3-feng-llm\build_pc_chat.ps1
# MinGW g++ 备用路径（等价于 .\build_pc_chat.ps1 -MinGW）：
g++ -std=c++23 -fno-exceptions -fno-rtti -fno-threadsafe-statics -O2 -DFENG_KV_Q2=1 -DFENG_KV_Q2_BLOCK=8 -o pc_chat.exe `
  esp32s3-feng-llm\pc\pc_chat.cpp esp32s3-feng-llm\main\feng_model.cpp `
  esp32s3-feng-llm\main\feng_llm.cpp esp32s3-feng-llm\main\feng_quant.cpp `
  esp32s3-feng-llm\main\feng_smp.cpp esp32s3-feng-llm\main\feng_tokenizer.cpp `
  esp32s3-feng-llm\main\feng_sample.cpp esp32s3-feng-llm\main\feng_calc.cpp `
  esp32s3-feng-llm\main\feng_tools.cpp esp32s3-feng-llm\main\feng_memory.cpp -Iesp32s3-feng-llm\main -lm

# 3) 聊天（tool 0.5s 秒回，多轮上下文默认开）
.\pc_chat.exe model_export
```

更省事：在仓库根跑 `esp32s3-feng-llm\build_pc_chat.ps1`，一次编译全部 PC 运行时与单测。
验证构建与数字回归：`python tools\check_all.py`（文档自检 → 构建 → 四套单测 →
32 题矩阵 27/27+4/4 → 算术 21/21 → `pc_check` MATCH）。

> PC CPU 性能：**多线程 654 tok/s**（单线程 157，q2 KV / 32 token / i7-12700KF；
> 输出行级并行、逐字不变）。默认线程数 = 逻辑核 3/4，可用 `OMP_NUM_THREADS` 覆盖。

> **别用 PC 的 HF 权重导出给引擎**：引擎的 `model.bin` 是 Q4 权重 + q2 KV 双 QAT 格式，
> 只有板端 v3.19-embed 权重扛得住。实测同套 32 题矩阵：PC 权重 **22/27**（翻译/情绪/推荐崩），
> 板端 QAT 权重 **27/27 + 召回 4/4**（`logs/pc_kv_suite32_v3_14pc2_q2b8.txt`）。
> PC 的 HF 权重请走 transformers（第 3 节）或 Python 脚本。

也可以直接用 Python 脚本（同一套 tool，`scripts/runtime_tools.py`）：

```powershell
python scripts\chat_student.py --model v3_19\pc4 --prompt "4854+4411"
python scripts\chat_student.py --model v3_19\pc4 --prompt "现在几点？"
```

tool 覆盖：多位数/小数/括号/中文数字（`五十九加一`）/`乘以、除以`/百分号（`一百*15%`）、
`12的平方`、`根号16`、**序列数数**（`把 1 到 5 倒着数一遍`、`从 3 数到 8`）、
`现在几点`、`3天后是几号`、`时间戳`、`随机数`、`掷骰子`、`抛硬币`，
以及记忆句式（`我叫X`、`我最喜欢Y`、`我住在Z`、`我养了W`、`我最喜欢的<键>是<值>`）。

### 2.1 可选：CUDA 加速（RTX 显卡）

```powershell
cd esp32s3-feng-llm
.\build_pc_cuda.ps1        # 需要 VS Build Tools + CUDA Toolkit；产出 pc\*_cuda.exe
.\pc\pc_chat_cuda.exe ..\feng-30m-c-engine-model
.\pc\pc_bench_cuda.exe ..\feng-30m-c-engine-model "你好" 32
```

| 指标（RTX 5060 Ti / q2 KV / 32 token） | CPU（MinGW g++） | CUDA |
|---|---|---|
| 生成 decode | 157 tok/s（6.4 ms/token） | **322 tok/s（3.1 ms/token）** |
| prefill（12 token） | 59 ms | 110 ms |
| 32 题矩阵 | 27/27 + 4/4 | **27/27 + 4/4（输出逐字一致）** |

说明：CUDA 后端只加速 GEMV（权重常驻显存，激活按次拷贝）；prefill 慢是因为每次 GEMV
都有一次 PCIe 往返 + kernel 同步（约 44 次/token），所以它更适合长回答生成。
`FENG_CUDA=0` 环境变量可让同一二进制强制走 CPU 路径对比；板端固件不含 CUDA。

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
# 我是 feng，一个由个人开发者 jiaheng 独立开发训练的 AI。
```

> chat template 已写进 `tokenizer_config.json`，`apply_chat_template` 开箱可用；
> 旧版 transformers 请把 `dtype=` 换成 `torch_dtype=`。

不用 chat template 时，手写提示词的等价格式（板端与评测用的就是它）：

```
<|im_start|>user
你是谁？<|im_end|>
<|im_start|>assistant
```

## 4. 刷到 ESP32-S3

### 硬件要求

| 项目 | 要求 |
|---|---|
| 模块 | **ESP32-S3-WROOM-2-N32R16V**（32 MB Octal flash + 16 MB Octal PSRAM，1.8 V） |
| 为什么必须 WROOM-2 | 固件按 OPI flash / 1.8 V 构建；WROOM-1 等 Quad/3.3 V 模块会烧写或启动失败 |
| flash | 必须 32 MB：**只有前 16 MB 能被 mmap 直读**（NOR flash 24 位地址上限，与模块容量无关），模型放这里；tokenizer 分区在 16 MB 之后，用 `esp_partition_read` 读 |
| PSRAM | 必须 16 MB（8 MB 版放不下 2048 ctx 的 q2 KV，9.62 MB） |
| 串口 | CH343 在 COM20，原生 USB-JTAG 在 COM19；对话 115200 8N1，烧录 921600 |

### 发布固件配置

板端发布配置 = **q2 block8 KV / 2048 ctx**（`idf.py -DFENG_USE_Q2_KV=ON build`）。
不带该参数构建则是 int8 / 1024 ctx。q2 配置实测：

| 指标 | 结果 |
|---|---|
| 32 题矩阵 + 长文召回 | **27/27 + 4/4** |
| 工具 / 记忆 | **13/13** / **12/12** |
| 速度 | **2.26 tok/s**（938 ms/token，ctx 19） |
| 固件体积 | **304,192 B**（C++23 引擎，比 C 版小 2,816 B） |

### 分区偏移（与 `esp32s3-feng-llm/partitions.csv` 一致）

| 内容 | 偏移 |
|---|---|
| bootloader / 分区表 / 固件 | `0x0` / `0x8000` / `0x10000` |
| `model.bin`（14.93 MB，Q4 块64） | `0x110000` |
| `tokenizer.bin`（413 KB） | `0x1000000` |

### 烧录与对话

> **不想装 ESP-IDF？** 直接下载 `feng-30m-v3.19-embed-firmware.zip`
> （预编译固件 + 模型 + 哈希清单，按上表偏移一次写完即可），跳过第 ① 步。

```powershell
# ① 编译固件（ESP-IDF v5.5.5；用预编译包可跳过）
cd esp32s3-feng-llm
$env:IDF_TOOLS_PATH = "<你的 IDF 工具链路径>"     # 本机路径见 scripts/local_paths.json
& "<esp-idf 路径>\export.ps1"
idf.py -DFENG_USE_Q2_KV=ON build                  # 发布配置：q2 / 2048 ctx

# ② 烧录固件
idf.py -p COM20 flash

# ③ 烧录模型与分词器
python -m esptool --chip esp32s3 --port COM20 -b 921600 write_flash 0x110000  weights\esp32\model.bin
python -m esptool --chip esp32s3 --port COM20 -b 921600 write_flash 0x1000000 weights\esp32\tokenizer.bin

# ④ 串口对话与自检
python scripts\esp32_chat.py  --port COM20 --question "你是谁？"
python scripts\esp32_multi.py --port COM20                     # 多轮稳定性（默认每题 \reset）
python scripts\esp32_memory.py --port COM20 --n 12             # 记忆 12 组
python scripts\esp32_tool_test.py --port COM20                 # 时间/随机数/算式 13 项
python scripts\esp32_enc_test.py COM20                         # GBK/UTF-8 双编码自检
```

串口协议：发一行提问 → `<< 回复内容 >>END` 流式输出（`>>END` 行含上下文占用与 tok/s）；
开机自检会打印 mmap 带宽、双核 GEMV 加速比、tokenizer 自检（`你好` → id 5331）。
编码自动跟随终端（GBK 进 GBK 出、UTF-8 进 UTF-8 出），也可用 `\gbk` / `\utf8` / `\stream N` /
`\mem` / `\reset` / `\help` 手动控制。

## 5. 数据集

`datasets/` 里是**蒸馏数据集**（教师模型的输出与提示词），**不含原始预训练语料**。

| 文件 | 内容 |
|---|---|
| `teacher_distill.jsonl` | v1 教师（Qwen3.5-0.8B 微调版）生成的 12,000 条回答（`tokens` 合计 1.61M） |
| `teacher_prompts.jsonl`、`teacher_prompts_12k.jsonl` | v1 教师使用的提示词集 |
| `planA_prompts.jsonl`、`planA_prompts_v2.jsonl` | v2/v3 教师提示词，共 1,408 条请求（含 20 组多轮对话） |
| `planA_teacher.jsonl`、`planA_teacher_v2.jsonl`、`planA_teacher_partial.jsonl` | v2/v3 教师（bonsai2-27b）返回的行为数据 |

说明：

- **原始预训练语料不随包发布**：中文维基、firefly、sharegpt-zh、UltraChat、Orca-Math 等
  合计 4 GB+，且部分数据集许可不明确；由它们编译出的 `aux_sft.jsonl` 等同样不进包。
  需要时按仓库 `scripts/` 里的处理脚本自行下载重建。
- **本地脚本生成的补丁数据也不进包**：v3.5 多轮、v3.6 日常补丁/运算/推荐修复等由
  `scripts/v3_5_build_multiturn.py`、`scripts/v3_6_build_daily_patch.py`、`scripts/v3_6_build_drill.py`
  按固定随机种子生成，可完全复现。
- 教师输出基于 Apache-2.0 许可的教师模型生成，随本项目以 Apache-2.0 提供。
- 自行下载公开语料时请遵守各自许可：中文维基 CC-BY-SA-3.0、Dolly-15k CC-BY-SA-3.0、
  UltraChat-200k MIT、Orca-Math-200k MIT；firefly 与 evol-instruct 以其官方页面为准。

## 6. 复现训练（可选）

完整链条、超参与版本对比见 [`CHANGELOG.md`](CHANGELOG.md)、[`COMPARISON.md`](COMPARISON.md)、
[`DELIVERY.md`](DELIVERY.md)。v3 主线：

```powershell
python scripts\v3_build_stages.py                                   # 渐进长文数据 4k→32k
python scripts\v3_train.py                                          # 训练（从 v2 权重出发）
python scripts\v3_pack_sft.py ; python scripts\v3_polish.py         # 8k 长上下文对话微调
python scripts\v3_build_retrieval.py ; python scripts\v3_retrieval_sft.py   # 合成检索 SFT
python scripts\eval_planA_scope.py archive/v3\retr_sft\ctx32768\final eval\v3_scope.json
python scripts\eval_longctx.py --model archive/v3\retr_sft\ctx32768\final --ctx 4096,8192,16384,32768
```

v3.5 / v3.6 及其后的补丁链条：

```powershell
python scripts\v3_5_build_multiturn.py --out archive/v3_5d\mt_convs.jsonl --n 2600     # 多轮对话数据
python scripts\v3_6_build_daily_patch.py --out archive/v3_6a\daily_patch.jsonl          # 592 条日常补丁
python scripts\v3_6_build_drill.py --out archive/v3_6e\drill.jsonl                      # 运算/细节打磨
python scripts\v3_6_sft_patch.py --init <起点> --patch archive/v3_6e\drill.jsonl `
  --mt archive/v3_5d\mt_convs.jsonl --mt-n 400 --identity-n 150 --out <输出> --epochs 8 --lr 1e-4
python scripts\eval_longctx_many.py --models "<输出>" --n 32 --neg-n 16          # 检索回归
```

脚本不写死路径：根目录按脚本位置推导，外部依赖用环境变量或 `scripts/local_paths.json` 指定，
自检命令 `python scripts/paths.py`。训练需要 16 GB 显存的 CUDA 卡（32k 阶段峰值 10.28 GiB）；
v1 语料脚本已随 v1 归档，重建 v2/v3 数据需要 `FENG_DATA_DIR`（见 [配置](README.md#配置)）。

## 7. 评测表现

### PC（v3.19，贪心解码；脚本与结果 JSON 都在仓库里）

| 项目 | 结果 |
|---|---|
| 身份（12 题） | **12/12**（`eval/v3_19pc4_identity.json`） |
| 日常对话探针（42 题） | **42/42**（`eval/v3_19pc4_probe42.json`） |
| 范围内 18 题 | **10/10**（`eval/v3_19pc4_scope.json`） |
| 针检索（每长度 32 题） | **28/30/26/26 = 110/128（单类别）**；**105/128（多类别）**（`eval/longctx32_v3_19pc4.json`） |
| 「文中没有」拒答 | **61/64（单类别）、62/64（多类别）** |
| 留出 30 题（开发集） | **19/30**（`eval/v3_19pc4_heldout30.json`） |
| 独立留出 30 题（未参与调参） | **18/30**（`eval/heldout2_v3_19pc4.json`） |
| 记忆 24 题 | **23/24**（`eval/v3_19pc4_memory24.json`） |
| 标准英文基准（0-shot） | SciQ 71.00%/73.80%、PIQA 53.37%、ARC-E 26.60%、ARC-C 21.93%、HellaSwag 28.99%、Winogrande 49.33%、OpenBookQA 14.00%/24.00%、BoolQ 37.83%、SciCloze-900 25.56%（`eval/lm_eval_feng_v3_19_pc4.json`） |

### 板端（v3.19-embed + q2 / 2048）

| 项目 | 结果 |
|---|---|
| 32 题矩阵 + 长文召回 | **27/27 + 4/4**（`logs/pc_kv_suite32_v3_19b6_q2b8.txt`） |
| 算术子集（tool） | **21/21**（`logs/pc_arith_suite_v3_19b6_q2b8.txt`） |
| tool 专项 / 记忆 / 身份序列 | **13/13** / **12/12** / **7/7**（`logs/board_tools_cpp23_q2_final.txt`、`logs/board_memory_cpp23_q2_final12.txt`） |
| 速度 | **2.26 tok/s**（938 ms/token @ ctx 19） |
| 多轮回归（PC 模拟固件） | mt-suite 7/10、mem12 12/12、seq 8/10（与 C 版逐值一致） |

## 8. 已知限制

- **30M 容量上限**：没覆盖到的自由问答会答偏或编造；复杂推理与专业领域不可靠。
- **板端上下文 2048（q2 KV）**；32k 仅在 PC 上可用。
- **板端 prefill 是 O(n²)**：≈800 tokens 输入要 ~291 s；交互输入建议 ≤150 tokens，
  长文/批量任务用 PC 版。
- **两个权重不能互换**：PC 的 HF 权重没做量化感知训练，导进引擎会退化（22/27 vs 27/27）；
  板端 QAT 权重在 PC 32k 长上下文上不如 PC 版。
- **记忆不是持久化**：`\reset`/断电/写满即忘；记忆 tool 只覆盖可枚举句式。
- **板端时间靠宿主对时**：串口脚本会自动发 `\settime`；不跑脚本时要手动发一次。
- **tool 只在带 tool 的运行时里**：板端固件 / PC C++ 引擎 / Python 脚本；GGUF 自 v3.14 起不再发行。
- 量化抗性对权重回插极敏感（掺 20% 旧权重就掉点）：改板端行为要走
  「补数据 + 权重/KV 双 QAT」链路（CHANGELOG v3.10/v3.11）。
- 32k 负样本拒答（61/64）经多轮专项未突破，已记录为平台（CHANGELOG v3.9 附录）。

## 9. 常见问题（FAQ）

**串口乱码？** 固件自动跟随终端编码；仍乱码时先跑 `python scripts\esp32_enc_test.py COM20`，
再用 `\gbk` / `\utf8` 锁定。开机横幅是纯 ASCII，任何编码下都不该乱码。

**烧录失败 / 端口占用？** 关闭 monitor/其他串口程序；CH343 在 COM20、原生 USB-JTAG 在 COM19；
烧录用 921600，对话用 115200。`esptool` 报 port busy 时换 USB 口或重新插拔。

**时间 tool 说"还没对上网络时间"？** 属预期：板子没有 RTC。跑一次
`python scripts\esp32_chat.py --port COM20 --question "你好"`（脚本会自动对时），
或手动发 `\settime <unix秒>`。

**算术为什么不由模型算？** 30M 模型算不可靠，v3.14 起算式/时间/随机数交给引擎 tool
（0.5 s、100% 正确）；模型只负责对话。GGUF 路径没有 tool，所以不再发行。

**为什么换了板端权重矩阵分数掉？** 引擎要 Q4+q2 双 QAT 权重（v3.19-embed）。
PC 的 HF 权重用在引擎上会 22/27；两个权重不能互换。

**记忆为什么没了？** 记忆 tool 存在 RAM，`\reset`、断电、上下文写满都会清空；
`\mem` 可以随时查看当前记住的事实。

**能上 32k 上下文吗？** 板端不能：2048 ctx 的 q2 KV 已占 9.62 MB PSRAM，
32k 需要约 150 MB，超出硬件。32k 只在 PC 上可用。

**固件多大？可以塞更多功能吗？** 当前 304,192 B，app 分区 1 MB（还剩 70%）。
新增非热点模块用 `-Os`、热点保持 `-O2`；引擎约定见 [AGENTS.md](AGENTS.md) 的 C++23 条款。

## 10. 许可证

代码与权重均为 **Apache-2.0**（见 `LICENSE`）。本项目为个人项目，
与任何模型厂商无隶属或背书关系。
