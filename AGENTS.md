# AGENTS.md — feng-30m 项目工作指南

本文件供在此仓库工作的 AI 编码代理（Codex 等）阅读。**本项目的文档、注释、提交说明与回复统一用中文。**

## 1. 项目是什么

用大模型当教师，**从零训练一个 ~30M 参数的中文对话模型**，量化到 Q4 后在 **ESP32-S3（32MB flash / 16MB PSRAM）上离线跑**。

- 学生架构（v2/v3 现行）：Qwen3 结构，**11 层 / hidden 448 / 7 头 MHA（7 KV 头）/ head_dim 64 / FFN 896 / 16k 词表 / tied embedding**，29.43M 参数，`rope_theta=1e6`，v3 原生支持 32768 上下文。
- 身份自述（v3.2 起，必须保持）：**「我是 feng，一个由个人开发者 jiaheng 独立开发训练的 AI」**；
  模型不得自称 Qwen/ChatGPT，也不得说自己是"微调"（训练数据来自教师蒸馏，模型本身从零训练）。
- **教师谱系**：**v1** 用 **Qwen 3.5 0.8B 微调版**（feng-0.8b，bf16）生成的数据训练；**v2 / v3** 用 **bonsai2-27b（27B）**训练。
- 版本演进：v1（8 层 / 32k 词表，只能对话，无法上板）→ v2（16k 词表 + 1.5B token 预训练 + 27B 教师 SFT，首次上板）
  → v3（渐进长上下文 + 合成检索 SFT）→ v3.5（修多轮复读）→ v3.6（日常对话大补丁）
  → v3.7（KV-QAT + q2 KV block8，板端 2048 上下文）→ v3.8（上下文专项升级）
  → v3.9 → v3.10 = v3.9 底座 + q2 KV-QAT → v3.11 = 算术边界修复 + Q4 权重/q2 KV 双 QAT
  → v3.12 = PC 算术修复 → v3.13 = 记忆版 → v3.14 = tool 版（算术/时间/随机数交给 C 引擎，
  模型不再学算术；PC 当前发布）→ v3.15-embed（身份漂移修复）→ **v3.16-embed = 板端权重当前发布
  （身份串名修复，现为上一版）** → v3.17 = 引擎/固件（记忆 tool：多轮记忆与身份问答确定性回答，权重未变）
  → **v3.19 = 定向教师数据修"模板串台"（PC `v3_19/pc4`、板端 `v3_19/board6`；留出 30 题 16/17 → 19/19）**。
- v3.19 现状（**当前发布**）：
  - PC `v3_19/pc4`：从 `v3_14/pc2` 出发，用 440 条定向教师数据（常识/列举/情绪/寒暄/推理/身份六类，
    `scripts/v3_19_build_target_prompts.py`）+ 多轮/身份回放做**末 2 层 15 epoch** 微调；
    留出 30 题 **19/30**（strict-v4 判定；v3.14 = 16/30）、单类别检索 **110**（v3.14 = 108）、
    多类别 105（-2）、记忆 23/24（-1）、范围/身份/探针保持（`eval/v3_19pc4_*.json`）。
  - 板端 `v3_19/board6`：在 `v3_16/board_p3` 上做 **Q4+q2 双 QAT（4 epoch / lr 6e-6）**；
    C 引擎矩阵 **27/27+4/4、召回 4/4、算术 21/21、pc_check MATCH**，HF 留出 **19/30**（v3.16 = 17）、
    记忆 24/24；实机 tool 13/13、记忆 12/12、7 轮换名身份 7/7（`logs/pc_kv_suite32_v3_19b6_q2b8.txt`、
    `logs/board_v3_19b6_memory12.txt`、`logs/esp32_multi.txt`）。
  - **评测口径**：留出题判定用 strict-v6（子串假阳性、复读误判、"推荐运动"误命中、
    问题关键词复读四类都拦，关键词大小写不敏感；`chat_probe_heldout.py --rescore`，
    重打分必须传对 case 表，否则未知题按原判保留）；
  - **两套留出题**：第一套 30 题已当开发集（v3.19=19/30）；第二套 `chat_probe_heldout2.py`
    未参与训练/调参（三版权重均 18/30），发版文案必须以第二套为准说明泛化水平；
    **关键词判定会漏假阳性**（答案含"南极/米"等词即判 OK，即使方向答反或退化成词）——
    头条数字必须人工过一遍 `rows[].a`，推荐解码对照存 `eval/heldout2_*_rec.json`；
    留出 30 题已被多轮迭代用作开发集，只作版本对比，不当无偏泛化分数。
  - 别再用 v3.16 时代的"往身份数据加精确链"套路（p4–p7 已证明只是重排失败点）。
- v3.16-embed（上一版板端权重，`v3_16/board_p3/`）：在 v3.15/board_ctxid4 上做
  **上下文双向问名**补丁（报名字后问身份/问名字、同类事实取新、记忆保护），lr 1.5e-6 × 1 epoch；
  板端 12 题记忆 10/12（**v3.17 引擎侧记忆 tool 上线后 12/12**）、6 轮报名字→问身份序列从 3/6 修到 5/6、tool 13/13、
  PC 32 题矩阵 27/27+4/4；新增多轮回归套件 `pc/pc_mt_suite.c`（残余见 CHANGELOG v3.16-embed 节）。
  **注意**：再往上加"精确链"数据（p4–p6）能把 8 轮序列修到 8/8，但会丢长文召回
  （90% 深度）或在板端出现 "我user" 伪影——试过且未采用，别再重复这条路线
  （`CHANGELOG.md` v3.16-embed 附录）。
  （**v3.17 起改用引擎侧记忆 tool 解决**：身份/常见事实追问由 C 引擎确定性回答，
  板端 12 题记忆 12/12、8 轮身份序列 8/8——不要再拿训练数据去磨这类可枚举问答。）
  **p7（lr 5e-7 最小干预）已确认这是容量硬边界**：它保住 27/27+4/4 并把 8 轮序列修到 10/10，
  却把失败换到别的次序（「你好呀→我叫丽丽→你叫什么名字」答"我叫丽丽"）、mem12 掉 1 分。
  p3–p7 五个版本都只是重排失败点——**多轮"你我/事实"分辨不要再靠加数据解决**，
  要提升就得换更大的模型或改架构（见 `CHANGELOG.md` v3.16-embed 附录）。
- v3.15-embed（历史，`v3_15/board_ctxid4/`）：在 v3.14/board6 上做
  「闲聊前缀 + 身份问答」锚点（`scripts/v3_15_build_identity_ctx.py`）+ 召回 ×8、lr 3e-6 × 1 epoch、
  继续双 QAT；修掉多轮里「你叫什么名字 → 你叫小模型/小王子」的漂移
  （7 组前缀 6 组完全正确），矩阵 **27/27+4/4**、算术 21/21、记忆 10/12、工具 8/8 全部保持。
- v3.14（历史 PC 发布）：
  - **tool**（`main/feng_calc.c`、`main/feng_tools.c`、`main/feng_memory.c`）：算式（多位数/小数/括号）、
    **序列数数**（"把 1 到 5 倒着数一遍"→`5、4、3、2、1。`、"从 3 数到 8"；只在祈使句触发，
    "我从1数到100也数不完"这类陈述仍交给模型）、
    UTC+8 时间（宿主 `\settime` 对时 + esp_timer 走时）、随机数（运行时间×1.54×1000，丢第一个取第二个）；
    时间 tool 还含**时钟推算**（`现在7点，再过3小时是几点？`→10 点，跨天说"明天/昨天"）；
    **记忆 tool（v3.17）**：从 `我叫X / 最喜欢Y / 住在Z / 养了W` 里抽事实，追问确定性作答
    （12 题记忆 12/12、身份永不串名），另有通用键值槽 `我最喜欢的<键>是<值>` /
    `我的<键>是<值>`（书/生日/家乡…）；`\mem` 查看、`\reset` 清空；
    列出/遗忘（`你还记得什么？`、`忘掉我的颜色`、`把记住的都忘掉`，遗忘留"墓碑"、
    重学解除）；只覆盖可枚举句式（C 单测 69 项）。**工具顺序：记忆要在时间之前**（"我的生日是几号？"）；
    板端 0.5s 秒回；PC C 引擎 `pc/pc_chat.c` 与 Python `scripts/runtime_tools.py` 同口径；
  - **训练数据不再含纯算式**（`scripts/v3_14_build_noarith_mix.py` 用 tool 识别器过滤）；
  - PC `v3_14/pc2/`：记忆 24/24、范围 10/10、探针 42/42、身份 12/12、单类别 108、多类别 107；
  - 板端 `v3_14/board6/`：q2 矩阵 27/27+4/4、工具 8/8、默认/情绪 10/10、记忆 12 题 10/12、1.80 tok/s；
  - **GGUF 发行取消**：llama.cpp 路径没有 tool，Release 只发 hf + esp32。
- v3.13（历史）：
  - PC `v3_13/mem_pc3/`：单类别 28/29/29/27 = **113/128（并列历史最高）**、多类别 108，
    算术 274/281、**记忆 21/24（v3.12 只有 5/24）**、范围 10/10、探针 42/42（0 未命中）、
    身份 12/12、多轮 1.00；
    - 板端 `v3_15/board_ctxid4/`：q2/2048 矩阵 **27/27 + 4/4**、算术子集 **21/21**、
    板端 默认/情绪/工具 **10/10 ｜ 10/10 ｜ 8/8**、跨轮记忆 10/12、1.80 tok/s；
  - **固件（main.c）默认多轮上下文**：KV 跨轮累积、`\reset` 清空、写满自动开新对话；
    `scripts/esp32_multi.py` 默认每题前 `\reset`（独立探针口径），`--no-reset` 测连续对话。
- v3.12（PC 上一版，`v3_12/arith2l3/`）：算术 275/281、单类别 110、探针 42/42、记忆 5/24。
- v3.11（板端上一版，`v3_11/pol8/`）：q2 矩阵 27/27+4/4、算术子集 21/21、板端 30/30、1.81 tok/s；
  关键技巧是 **Q4 权重/q2 KV 双 QAT**（`v3_7_kv_qat.py --wqat`），量化抗性对权重回插极敏感
  （掺 20% v3.9 权重就掉到 25/27），PC 32k 弱。
- **PC 用 v3.19（`v3_19/pc4`）+ v3.17 引擎；板端用 v3.19-embed 权重（`v3_19/board6`）+ v3.17 固件/引擎**
  （见 `CHANGELOG.md` 的 v3.19/v3.17 节）。
- 硬件：RTX 5060 Ti 16GB（训练）+ i7-12700KF；ESP32-S3-**WROOM-2-N32R16V** 开发板（32MB Octal flash + 16MB Octal PSRAM，1.8V）。

## 2. 运行环境与路径解析（代码里已无硬编码盘符）

所有 Python 脚本统一走 `scripts/paths.py` 解析路径，优先级
**环境变量 > `scripts/local_paths.json`（不入库）> 从脚本位置推导**；自检命令 `python scripts/paths.py`。

| 用途 | 解析方式（本机实际值见 `scripts/local_paths.json`） |
|---|---|
| 仓库根目录 ROOT | 自动按脚本位置推导，可用 `FENG_ROOT` 覆盖 |
| 原始语料（v1–v3 语料脚本共用）/ v1 教师 GGUF（仅 v1 蒸馏需要） | `FENG_DATA_DIR`、`FENG_TEACHER_GGUF` |
| Python 解释器（torch 2.13.0+cu132，CUDA 可用） | 默认当前解释器 `sys.executable`，可用 `FENG_PY` 覆盖 |
| llama.cpp（GGUF 转换 / 量化 / benchmark） | `FENG_LLAMA_DIR` |
| ESP-IDF / esptool / gcc | `flash.ps1 -EspIdfPath -EspToolPy` 或环境变量 `IDF_PATH`/`ESPTOOL_PY`/`FENG_GCC` |
| 串口 | **COM20 = CH343，COM19 = 芯片原生 USB-JTAG**，115200 对话 / 921600 烧录 |

机器相关的实际路径一律写在 `scripts/local_paths.json`（已 gitignore，**不要提交**；新机器复制
`scripts/local_paths.example.json` 填写）。文档和代码里都不出现盘符。

本目录是 git 仓库，远端 `origin = https://github.com/jiaheng0815/feng-30m`（公开仓库）。发布约定：
**主仓库只放代码与文档**——数据集（`data/`、`v2/data/`）与权重/二进制（`*.safetensors`、`*.gguf`、`*.npy`、`*.bin` 等）
 都由 `.gitignore` 排除，随 Release 发布（PC `feng-30m-v3.19-release.zip`、板端 `feng-30m-v3.19-embed-release.zip`）；
**v3.14 起不再发行 GGUF**（llama.cpp 没有 tool，见 CHANGELOG v3.14）。
代码与权重均为 **Apache-2.0**（`LICENSE`）。
开源数据集只含**教师蒸馏数据**（提示词与教师输出）；本地脚本生成的多轮/补丁/运算数据不入 Release 包。
模型权重、训练产物一旦覆盖无法回滚，删除或覆盖已有模型目录前必须先向用户确认。

改动发布物时记得同步：`USAGE.md`（下载/推理/烧录说明）、Release 包内 `weights/`、`datasets/` 的清单，
以及 GGUF 的 chat template（用 `$FENG_LLAMA_DIR\gguf-py\gguf\scripts\gguf_new_metadata.py --chat-template-file` 写入）。

## 3. 目录地图

| 路径 | 内容 |
|---|---|
| `student/` | **v1** 产物（8 层 / 32k 词表）：`student/final/`、`student/feng-30m-chat/`、`student/feng-30m-32k/`、各训练阶段 |
| `v2/` | **v2** 产物：`v2/stage_planA3b/final/`（SFT 对照版）、`v2/gguf_planA3b/`、16k 分词器 `v2/tokenizer/`、预训练数据 |
| `v3/`、`v3_5*/`、`v3_6*/` | v3 及其后续各轮训练记录（历史版本，含 v3.5/v3.6 的补丁链与失败尝试） |
| `v3_7/`…`v3_9/` | v3.7（旧板端）、v3.8、v3.9（PC 上一版，`v3_9/release/`）的训练与评测产物 |
| `v3_10/` | v3.10 板端中间版（`v3_10/qat_pol3/`），另有 cand1/m8k 等未采用实验 |
| `v3_11/` | v3.11 板端权重（算术边界 + Q4 权重/q2 KV 双 QAT） |
| `v3_12/` | v3.12 PC 权重（末层算术微调：275/281 + 单类别 110） |
| `v3_13/` | v3.13 记忆版权重（PC `v3_13/mem_pc3/` + 板端 `v3_13/mem_board/`） |
| `v3_14/` | 历史 PC 发布 `v3_14/pc2/` + 板端 v3.14 版 `v3_14/board6/`；数据不入库 |
| `v3_15/` | v3.15-embed（历史，`v3_15/board_ctxid4/` 上下文身份锚点版）；数据 `identity_ctx*.jsonl` 不入库 |
| `v3_16/` | 上一版板端权重 `v3_16/board_p3/`（v3.16-embed）+ p1/p2/p4–p7 实验（代价见 CHANGELOG）；补丁数据不入库 |
| `v3_19/` | **当前发布权重**：PC `v3_19/pc4/`、板端 `v3_19/board6/` + board1–5 实验；定向数据与训练日志见 `data/`、`logs/`（不入库） |
| `data/` | v1 的提示词集、教师蒸馏数据、公开语料（sharegpt/firefly/dolly/evol 等） |
| `scripts/` | 全部数据构建 / 训练 / 评测 / 导出脚本（77 个 .py，含 `scripts/paths.py` 路径解析；另有 1 个教师启动脚本） |
| `eval/` | 评测结果 JSON（`planA*_scope.json`、`v3_scope.json`、`longctx_*.json` 等） |
| `logs/` | 所有构建 / 训练 / 烧录 / 板上测试日志；`board_baseline_lut.txt` 是板上精度基线 |
| `esp32s3-feng-llm/` | ESP32 固件工程 + 可移植 C 推理引擎 + PC 端一致性检查工具 |
| `tools/check_md.py`、`tools/check_docs.py` | 文档自检：前者查围栏/路径/过时表述，后者把**全部 10 个 md 的关键数字与实际产物对齐**（参数量、GGUF 体积与 chat template、v3.6 探针/检索分数、范围评测、检索 loss） |

## 4. 工作约定

- **训练产物目录结构**：`<版本>/<阶段>/final/`（HF 权重 + tokenizer + `config.json`），阶段汇总写 `summary.json`（steps / tokens / loss / 峰值显存 / 耗时），逐步日志写 `train_log.jsonl`；中途 checkpoint 放 `<阶段>/rolling/stepN/`。
- **日志统一写 `logs/`**，评测结果统一写 `eval/`，不要散落在根目录。
- 训练脚本对已完成的阶段是**跳过式**的（检测到 `<阶段>/final/` 已存在就不重跑）；要重跑先用 `--out` 指向新目录，或显式删除旧目录（需用户确认）。
- 模型卡走 `<模型目录>/MODEL_CARD.md`；`tools/check_md.py` 会校验 README/DELIVERY/COMPARISON/CHANGELOG/esp32 README/MODEL_CARD 里的反引号路径与过时数字。
- Python 脚本开头都 `sys.stdout.reconfigure(encoding="utf-8")`，新增脚本请保持一致；命令行输出用中文没问题（torch 的日志仍是英文）。

## 5. 常用命令

数据与训练（Python 一律用上表的 venv 解释器）：

```powershell
# v3 渐进长文数据 → 训练 → 8k 对话微调 → 检索 SFT（严格按此顺序）
python scripts\v3_build_stages.py
python scripts\v3_train.py                      # --stages ctx4096 可只跑某阶段；--steps-cap N 冒烟
python scripts\v3_pack_sft.py
python scripts\v3_polish.py
python scripts\v3_build_retrieval.py
python scripts\v3_retrieval_sft.py

# v2 主线（预训练 / SFT / 长文三条分支）
python scripts\v2_train.py --stage pretrain --out v2\stage_pre --lr 3e-3
python scripts\v2_train.py --stage sft --model v2\stage_pre\final --out v2\stage_sft

# v3.11 板端版（算术边界 + Q4 权重/q2 KV 双 QAT；完整命令见 CHANGELOG v3.11 节）
python scripts\v3_7_kv_qat.py --init v3_9\stockfix2 --data v3_7\qat_data.jsonl `
  --mt v3_5d\mt_convs.jsonl --mt-n 280 --identity-n 150 --out v3_10\qat_a `
  --epochs 3 --lr 3e-5 --retr v3_8\retr --retr-n "4096:200:2,8192:60:1" --retr-lr 1e-5
# 算术边界 + 双 QAT（发布版 pol8）
python scripts\v3_11_build_arith_patch.py --out v3_11\arith_patch2.jsonl
python scripts\v3_7_kv_qat.py --init v3_11\pol7 --data v3_11\arith_repair.jsonl `
  --identity-n 100 --out v3_11\pol8 --epochs 2 --lr 8e-6 --batch 24 --max-len 1024 --wqat

# v3.12 PC 版（同一套算术数据走末层微调，完整命令见 CHANGELOG v3.12 节）
python scripts\v3_6_sft_patch.py --init v3_9\stockfix2 --patch v3_11\arith_repair.jsonl `
  --mt v3_5d\mt_convs.jsonl --mt-n 150 --identity-n 80 --out v3_12\arith2l `
  --epochs 3 --lr 3e-5 --train-last 2
python scripts\v3_6_sft_patch.py --init v3_12\arith2l --patch v3_12\pcfix2.jsonl `
  --mt v3_5d\mt_convs.jsonl --mt-n 100 --identity-n 80 --out v3_12\arith2l3 `
  --epochs 4 --lr 1.5e-5 --train-last 2

# v3.13 记忆版（PC 末层微调 + 板端双 QAT；混训数据见 CHANGELOG v3.13 节）
python scripts\v3_13_build_memory.py --out v3_13\memory.jsonl
python scripts\v3_6_sft_patch.py --init v3_12\arith2l3 --patch v3_13\mem_mix2.jsonl `
  --mt v3_5d\mt_convs.jsonl --mt-n 300 --identity-n 80 --out v3_13\mem_pc3 `
  --epochs 2 --lr 1.2e-5 --train-last 2
python scripts\v3_7_kv_qat.py --init v3_11\pol8 --data v3_13\mem_mix2.jsonl `
  --identity-n 80 --out v3_13\mem_board --epochs 2 --lr 8e-6 --batch 24 --max-len 1024 --wqat

# v3.14 tool 版（无算术混训；tool 在 C 引擎里）
python scripts\v3_14_build_noarith_mix.py --out v3_14\noarith_mix.jsonl
python scripts\v3_6_sft_patch.py --init v3_9\stockfix2 --patch v3_14\noarith_mix2.jsonl `
  --mt v3_5d\mt_convs.jsonl --mt-n 300 --identity-n 80 --out v3_14\pc2 `
  --epochs 2 --lr 1.2e-5 --train-last 2
python scripts\v3_7_kv_qat.py --init v3_11\pol8 --data v3_14\noarith_mix2.jsonl `
  --identity-n 80 --out v3_14\board --epochs 3 --lr 1e-5 --batch 24 --max-len 1024 --wqat
python scripts\v3_7_kv_qat.py --init v3_14\board --data v3_14\board_memfix.jsonl `
  --identity-n 80 --out v3_14\board6 --epochs 2 --lr 4e-6 --batch 24 --max-len 2048 --wqat

# PC C 引擎运行时（自带四个 tool；编译时务必带上 feng_calc.c + feng_tools.c + feng_memory.c）
gcc -O2 -DFENG_KV_Q2=1 -DFENG_KV_Q2_BLOCK=8 -o pc_chat_q2b8.exe pc_chat.c `
  ../main/feng_model.c ../main/feng_llm.c ../main/feng_quant.c ../main/feng_smp.c `
  ../main/feng_tokenizer.c ../main/feng_calc.c ../main/feng_tools.c ../main/feng_memory.c -I../main -lm
# 更省事：仓库根跑 esp32s3-feng-llm\build_pc_chat.ps1（编 8 个产物 + 跑四套单测）
```

评测与导出：

```powershell
python scripts\eval_planA_scope.py v3_9\stockfix2 eval\v3_9_scope_sf2.json
python scripts\eval_longctx_many.py --models "v3_9sf2=v3_9/stockfix2" --n 32 --neg-n 16
python scripts\chat_probe.py --model v3_10\qat_pol3 --out eval\chat_probe_v3_10p3.json
python scripts\chat_multi.py --model v3_10\qat_pol3    # 多轮坍缩检查
python scripts\chat_student.py --model v3_10\qat_pol3 --prompt "你是谁？"
python scripts\export_student_gguf.py --model v3_10\qat_pol3 --out-dir v3_10\gguf
```

ESP32 固件（在 `esp32s3-feng-llm\` 下）：

```powershell
# 0) 导出板端模型（从 HF 权重生成 model.bin / tokenizer.bin / ref_logits.bin）
$py = "python"        # 换成装了 torch + transformers 的解释器
& $py tools\export_model.py --model <仓库根>\v3_10\qat_pol3 --out model_export_v3_10p3

# 1) PC 端一致性自检（改内核后必跑）
& "<MSYS2>\ucrt64\bin\gcc.exe" -O2 -o pc\pc_check.exe pc_check.c ..\main\feng_model.c `
    ..\main\feng_llm.c ..\main\feng_quant.c ..\main\feng_smp.c ..\main\feng_tokenizer.c -I..\main -lm
.\pc\pc_check.exe ..\model_export_v3_10p3 ..\logs\c_logits_v3_10p3.bin

# 1b) 带 tool 的 PC 运行时 / 板端代理套件（必须带 feng_calc.c + feng_tools.c + feng_memory.c）
& "<MSYS2>\ucrt64\bin\gcc.exe" -O2 -DFENG_KV_Q2=1 -DFENG_KV_Q2_BLOCK=8 -o pc\pc_kv_suite_q2b8.exe `
    pc_kv_suite.c ..\main\feng_model.c ..\main\feng_llm.c ..\main\feng_quant.c `
    ..\main\feng_smp.c ..\main\feng_tokenizer.c ..\main\feng_calc.c ..\main\feng_tools.c `
    ..\main\feng_memory.c -I..\main -lm
.\pc\pc_kv_suite_q2b8.exe ..\model_export_v3_14b6 ..\pc\prompt_long.txt 5200    # 27/27 + 4/4

# 2) 编译固件
$env:IDF_TOOLS_PATH = "<IDF 工具链目录>"      # 本机路径见 scripts/local_paths.json
& "<esp-idf>\export.ps1"
idf.py build

# 3) 烧录（固件；换模型只需后两条；偏移以 partitions.csv 为准）
$esp = "python"                              # 换成带 esptool 的解释器
& $esp -m esptool --chip esp32s3 --port COM20 -b 921600 write_flash `
    0x0 build\bootloader\bootloader.bin 0x8000 build\partition_table\partition-table.bin 0x10000 build\feng_30m.bin
& $esp -m esptool --chip esp32s3 --port COM20 -b 921600 write_flash 0x110000  model_export_v3_10p3\model.bin
& $esp -m esptool --chip esp32s3 --port COM20 -b 921600 write_flash 0x1000000 model_export_v3_10p3\tokenizer.bin

# 或者直接用一键脚本（路径走参数/环境变量，偏移已对齐 partitions.csv）
.\flash.ps1 -Port COM20 -EspIdfPath "<esp-idf>" -ModelDir .\model_export_v3_10p3

# 4) 串口对话 / 稳定性 / 编码自检
python scripts\esp32_chat.py --port COM20 --question "你是谁？"
python scripts\esp32_multi.py --port COM20
python scripts\esp32_enc_test.py COM20
```

文档自检：`python tools\check_md.py` + `python tools\check_docs.py`（改文档后两个都要跑）。
**一键验收**：`python tools\check_all.py`——按顺序跑 文档自检 → PC 引擎构建 + tool 单测 →
多轮回归套件（`pc_mt_suite`）→ 32 题矩阵 + 算术子集（`pc_kv_suite`）→ fp32 参考 logits
（`pc_check`）；模型导出目录缺失时自动跳过套件。提交前建议至少跑一次。

**CI（.github/workflows/ci.yml）**：干净 clone 上跑 Python 语法 + 四套 C 单测 + 四套 C/Python 交叉验证
（算式 29 / 记忆 36 / 随机数 56 / 时间 52，共 173 条）+ 全引擎编译 + 文档自检；
对应脚本 `tools/check_{tool,mem,rand,time}_parity.py`，本地可单独运行（需先按 build 脚本编出对应驱动）。

## 6. 硬性约束与踩过的坑（改代码前先看）

1. **教师生成与训练严格串行**，绝不同时占卡；27B 教师用 `--no-cuda-graph`、`reasoning_effort:none`、并发 8。
2. **llama.cpp `--parallel N` 会把 `-c` 均分给 N 个槽**：总上下文必须写成「每槽上下文 × N」，否则长提示被静默截断（v1 曾因此作废 2 万条教师数据）。
3. **显存安全**：lm_head + 交叉熵必须走 `scripts/student_utils.py::chunked_lm_loss`（512 token 分块 + checkpoint），32k 阶段峰值 10.28 GiB，16GB 卡上不要并发跑其他任务。
4. **注意力后端**：训练用 `torch.nn.attention.sdpa_kernel` 的 EFFICIENT/FLASH 后端；该 torch 构建的 SDPA 不支持 GQA，所以学生用 MHA（7=7），改 GQA 需同步改 C 引擎（`feng_llm.c` 里要加 KV 头广播）。
5. **身份不能掉**：训练语料里身份样本过采样（v1 是 15×），并过滤提及其他 AI 身份（ChatGPT/通义…）的样本；每次出模型都要用 `eval_planA_scope.py` 验证身份题。
6. **flash 前 16MB 的 mmap 窗口是硬边界**（NOR flash 24 位地址上限，**不是模块容量**——模块是 32MB）：`model.bin`（15,659,904 B = 0xEEF380）必须结束在 0x1000000 之前；现行分区为 `model 0x110000/0xEF0000`、`tokdata 0x1000000/0x80000`（tokdata 用 `esp_partition_read` 读，可放窗口外）。烧录偏移必须与 `esp32s3-feng-llm/partitions.csv` 保持一致：`flash.ps1` 已按此修正为 `model=0x110000` / `tokdata=0x1000000`，改动分区表时要同步改脚本。
7. **板端内存账**：权重只能 flash mmap 流式读，不能预载进 SRAM（每层 Q4 ≈0.95MB，内部 SRAM 只剩 ~271KB）；
   KV 默认 int8（`FENG_KV_INT8=1`，`MAX_CTX=1024`，9.93MB PSRAM），可选 q2 block8
   （`idf.py -DFENG_USE_Q2_KV=ON build`，`MAX_CTX=2048`，9.62MB，v3.19-embed 的 32 题 PC 矩阵 27/27 + 召回 4/4
   + 算术子集 21/21；固件默认多轮上下文、`\reset` 清空，见 `CHANGELOG.md` 的 v3.13 节）；板上 32k 上下文在 KV 内存上不可能，
   长文只能走滑窗/attention sink/线性注意力。
8. **速度现状**：标量路径已到 S3 单发射天花板（每步 ~4-5 周期；短上下文 ~1.9 tok/s ≈ 500 ms/token）。默认开启三项小幅数值优化（`FENG_GEMV_MADD` 纯 madd 链、`FENG_FAST_EXP` 快速 exp、`FENG_Q2_VFOLD` V 段折叠）：三者累计 logits 差 **4.8e-6**（Q4 量化误差 2.94）、32 题矩阵输出与优化前**逐字相同**。**长上下文成本仍在注意力本体**（q2/2048 单次 forward **2.22 s** = K 0.84 + softmax 0.05 + V 0.83 + 权重等 0.51，本次会话累计 **-37%**）。q2 注意力另有四处位精确优化（字节 LUT + `[layer][head][t]` 顺序布局 + 内联 fp16→fp32 + 2-token 展开）；prefill 用 `feng_forward_ex(..., want_logits=0)` 跳过中间 token 的 lm head（每个省 ~115 ms，最后一个 token 必须算）。**A8 整数 GEMV 已实测：S3 上比 FPU 慢 36%（`mull` 慢），只在 PC 上快 20%，默认关闭**。改 GEMV/注意力/布局后必须用 PC 32 题矩阵与上一版对比（位精确改动要求**逐字节**，数值改动要求 27/27+4/4 且给出差异量级）。**PIE 路线已实测结案**：裸吞吐 0.63–1.38 周期/MAC 有空间，但 S3 没有字节移位指令、4-bit 权重必须靠 LUT 展开（≥1 次标量 load+store / 权重），正确的整块内核只做到 **1.05×**——不要再写 PIE 内核（`CHANGELOG.md` v3.16-embed 附录）；不要再做没有实测收益的内层微调。
8.5 **采样器统一**：`main/feng_sample.c` 的 `feng_sample_greedy`（1.15 重复惩罚 + no-repeat 3-gram）是
   固件 / `pc_chat` / `pc_kv_suite` / `pc_mt_suite` 的唯一采样入口；改采样器后必须重跑 PC 32 题矩阵
   并与上一版输出对比（v3.20 那次为**逐字节 0 差异**），再上板。
9. **量化格式耦合**：Q4 block-64（4.25 bpw）；改 `QK` 必须同步改 C 侧 `QK`，且 `tools/export_model.py` 会生成 `ref_ids.json` / `ref_logits.bin` 供一致性校验。

## 7. 改动的验收清单

- **改了训练脚本**：用 `--steps-cap 1`（或 `--steps`）跑冒烟，确认能落盘 `<阶段>/final/` 与 `summary.json`。
- **改了 C 推理内核 / KV 量化**：必须重跑 `pc_check`（logits MATCH）→ `pc\verify_c_vs_torch.py`（fp32 路径要求 `max|diff| = 0.0000`）→ 板上 `scripts\esp32_multi.py`，并与 `logs\board_baseline_lut.txt` **逐字对比**回复。
- **改了文档**：跑 `python tools\check_md.py`（围栏/路径/过时数字）**和** `python tools\check_docs.py`
  （模型规格 / GGUF 体积与 chat template / v3.9~v3.12 探针、检索、算术分数 / 范围评测 / 检索 loss 与真实产物对齐）。
- **写了数字**：数字必须能追到 `eval/*.json`、`summary.json` 或 `logs/` 里的实测；没有出处的一律删掉或标注"预期/估算"，
  不要写没有日志支撑的精确值。
- **换了模型版本**：重跑 `eval_planA_scope.py` + `eval_longctx.py`，数字同步进 `CHANGELOG.md` / `README.md` / `USAGE.md`
-  （PC 与板端两套口径必须分别标清：PC=v3.19，板端=v3.19-embed + v3.17 引擎；算式的验收口径是
  **tool 回答**，不是模型算——pc_kv_suite 会把算式任务路由到 feng_calc）。

## 8. 排障速查

| 现象 | 处置 |
|---|---|
| ESP32 推理时 MMU fault / 数据读错 | 检查 `model.bin` 是否越界写进 tokdata；核对 `partitions.csv` 与 PDF 烧录偏移 |
| 板上回复乱码 | 用 `scripts\esp32_enc_test.py` 验证 GBK/UTF-8 自动跟随；`\gbk` / `\utf8` 可手动锁定 |
| 训练 OOM | 降 `--batch` / `--accum`，确认 `chunked_lm_loss` 的 chunk 没被调大，32k 阶段单独跑 |
| 模型答什么都是身份句 | v1 的老毛病，语料配比失衡；参照 v2 的 Plan A 语料 + 低 LR 补训（1000 步 / lr 1.5e-4） |
| 长文阶段跑完检索仍全错 | 长文预训练学不会检索，必须补合成检索 SFT（只对答案算 loss），见 `scripts/v3_build_retrieval.py` |
| esptool 报 port busy | 板子可能不在 USB 列表（换线/供电），或串口被 monitor 占用 |
