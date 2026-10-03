# AGENTS.md — feng-30m 项目工作指南

本文件供在此仓库工作的 AI 编码代理（Codex 等）阅读。**本项目的文档、注释、提交说明与回复统一用中文。**

## 1. 项目是什么

用大模型当教师，**从零训练一个 ~30M 参数的中文对话模型**，量化到 Q4 后在 **ESP32-S3（32MB flash / 16MB PSRAM）上离线跑**。

- 学生架构（v2/v3 现行）：Qwen3 结构，**11 层 / hidden 448 / 7 头 MHA（7 KV 头）/ head_dim 64 / FFN 896 / 16k 词表 / tied embedding**，29.43M 参数，`rope_theta=1e6`，v3 原生支持 32768 上下文。
- 身份自述（必须保持）：**「我是 feng，由个人开发者 jiaheng 微调后的 Qwen」**。
- **教师谱系**：**v1** 用 **Qwen 3.5 0.8B 微调版**（feng-0.8b，bf16）生成的数据训练；**v2 / v3** 用 **bonsai2-27b（27B）**训练。
- 三代演进：v1（8 层 / 32k 词表，只能对话，无法上板）→ v2（16k 词表 + 1.5B token 预训练 + 27B 教师 SFT，首次上板）→ **v3（当前部署版：v2 + 渐进长上下文 + 合成检索 SFT）**。
- v3 现状：范围评测 10/10，针检索 4k/8k/16k/32k = 3/3、3/3、2/3、2/3，板上 **1.86 tok/s @1024 ctx**（int8 KV）。
- 硬件：RTX 5060 Ti 16GB（训练）+ i7-12700KF；ESP32-S3 R16N32 开发板。

## 2. 运行环境与路径解析（代码里已无硬编码盘符）

所有 Python 脚本统一走 `scripts/paths.py` 解析路径，优先级
**环境变量 > `scripts/local_paths.json`（不入库）> 从脚本位置推导**；自检命令 `python scripts/paths.py`。

| 用途 | 解析方式（本机实际值见 `scripts/local_paths.json`） |
|---|---|
| 仓库根目录 ROOT | 自动按脚本位置推导，可用 `FENG_ROOT` 覆盖 |
| 原始数据 / v1 教师 GGUF（同级项目 feng-ai-qwen35） | `FENG_DATA_DIR`、`FENG_TEACHER_GGUF` |
| Python 解释器（本机 torch 2.13.0+cu132，CUDA 可用） | 默认当前解释器 `sys.executable`，可用 `FENG_PY` 覆盖 |
| llama.cpp（GGUF 转换 / 量化 / benchmark） | `FENG_LLAMA_DIR`（本机 `D:\llama.cpp`） |
| ESP-IDF / esptool / gcc | `flash.ps1 -EspIdfPath -EspToolPy` 或环境变量 `IDF_PATH`/`ESPTOOL_PY`；本机 `F:\esp\v5.5.5\esp-idf`、`F:\Espressif`、`F:\msys2\ucrt64\bin\gcc.exe` |
| 串口 | **COM20 = CH343，COM19 = 芯片原生 USB-JTAG**，115200 对话 / 921600 烧录 |

新机器上先复制 `scripts/local_paths.example.json` 为 `scripts/local_paths.json` 填本机路径
（该文件已 gitignore，不会提交）。

本目录是 git 仓库，远端 `origin = https://github.com/jiaheng0815/feng-30m`（公开仓库）。发布约定：
**主仓库只放代码与文档**——数据集（`data/`、`v2/data/`）与权重/二进制（`*.safetensors`、`*.gguf`、`*.npy`、`*.bin` 等）
都由 `.gitignore` 排除，随 Release 的 `feng-30m-v3-release.zip` 发布；代码与权重均为 **Apache-2.0**（`LICENSE`）。
模型权重、训练产物一旦覆盖无法回滚，删除或覆盖已有模型目录前必须先向用户确认。

改动发布物时记得同步：`USAGE.md`（下载/推理/烧录说明）、Release 包内 `weights/`、`datasets/` 的清单，
以及 GGUF 的 chat template（用 `$FENG_LLAMA_DIR\gguf-py\gguf\scripts\gguf_new_metadata.py --chat-template-file` 写入）。

## 3. 目录地图

| 路径 | 内容 |
|---|---|
| `student/` | **v1** 产物（8 层 / 32k 词表）：`student/final/`、`student/feng-30m-chat/`、`student/feng-30m-32k/`、各训练阶段 |
| `v2/` | **v2** 产物：`v2/stage_planA3b/final/`（SFT 对照版）、`v2/gguf_planA3b/`、16k 分词器 `v2/tokenizer/`、预训练数据 |
| `v3/` | **v3 当前版**：`v3/ctx4096/final/` → `v3/ctx32768/final/`（渐进长文）、`v3/polish_ctx8192/final/`（8k 对话微调）、`v3/retr_sft/ctx*/final/`（检索 SFT，最终权重 `v3/retr_sft/ctx32768/final`）、`v3/gguf/` |
| `data/` | v1 的提示词集、教师蒸馏数据、公开语料（sharegpt/firefly/dolly/evol 等） |
| `scripts/` | 全部数据构建 / 训练 / 评测 / 导出脚本（44 个 .py，含 `scripts/paths.py` 路径解析；另有 1 个教师启动脚本） |
| `eval/` | 评测结果 JSON（`planA*_scope.json`、`v3_scope.json`、`longctx_*.json` 等） |
| `logs/` | 所有构建 / 训练 / 烧录 / 板上测试日志；`board_baseline_lut.txt` 是板上精度基线 |
| `esp32s3-feng-llm/` | ESP32 固件工程 + 可移植 C 推理引擎 + PC 端一致性检查工具 |
| `tools/check_md.py` | Markdown 文档自检（围栏配对、路径存在、过时数字），改完文档必须跑 |

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
```

评测与导出：

```powershell
python scripts\eval_planA_scope.py v3\retr_sft\ctx32768\final eval\v3_scope.json
python scripts\eval_longctx.py --model v3\retr_sft\ctx32768\final --ctx 4096,8192,16384,32768
python scripts\chat_student.py --model v3\retr_sft\ctx32768\final --prompt "你是谁？"
python scripts\export_student_gguf.py --model v3\retr_sft\ctx32768\final --out-dir v3\gguf
```

ESP32 固件（在 `esp32s3-feng-llm\` 下）：

```powershell
# 0) 导出板端模型（从 HF 权重生成 model.bin / tokenizer.bin / ref_logits.bin）
$py = "python"        # 换成装了 torch + transformers 的解释器
& $py tools\export_model.py --model <仓库根>\v3\retr_sft\ctx32768\final --out model_export_v3

# 1) PC 端一致性自检（改内核后必跑）
& "<MSYS2>\ucrt64\bin\gcc.exe" -O2 -o pc\pc_check.exe pc_check.c ..\main\feng_model.c `
    ..\main\feng_llm.c ..\main\feng_quant.c ..\main\feng_smp.c ..\main\feng_tokenizer.c -I..\main -lm
.\pc\pc_check.exe ..\model_export_v3 ..\logs\c_logits_v3.bin

# 2) 编译固件
$env:IDF_TOOLS_PATH = "<IDF 工具链目录>"      # 本机 F:\Espressif
& "<esp-idf>\export.ps1"                     # 本机 F:\esp\v5.5.5\esp-idf
idf.py build

# 3) 烧录（固件；换模型只需后两条；偏移以 partitions.csv 为准）
$esp = "python"                              # 换成带 esptool 的解释器
& $esp -m esptool --chip esp32s3 --port COM20 -b 921600 write_flash `
    0x0 build\bootloader\bootloader.bin 0x8000 build\partition_table\partition-table.bin 0x10000 build\feng_30m.bin
& $esp -m esptool --chip esp32s3 --port COM20 -b 921600 write_flash 0x110000  model_export_v3\model.bin
& $esp -m esptool --chip esp32s3 --port COM20 -b 921600 write_flash 0x1000000 model_export_v3\tokenizer.bin

# 或者直接用一键脚本（路径走参数/环境变量，偏移已对齐 partitions.csv）
.\flash.ps1 -Port COM20 -EspIdfPath "<esp-idf>" -ModelDir .\model_export_v3

# 4) 串口对话 / 稳定性 / 编码自检
python scripts\esp32_chat.py --port COM20 --question "你是谁？"
python scripts\esp32_multi.py --port COM20
python scripts\esp32_enc_test.py COM20
```

文档自检：`python tools\check_md.py`。

## 6. 硬性约束与踩过的坑（改代码前先看）

1. **教师生成与训练严格串行**，绝不同时占卡；27B 教师用 `--no-cuda-graph`、`reasoning_effort:none`、并发 8。
2. **llama.cpp `--parallel N` 会把 `-c` 均分给 N 个槽**：总上下文必须写成「每槽上下文 × N」，否则长提示被静默截断（v1 曾因此作废 2 万条教师数据）。
3. **显存安全**：lm_head + 交叉熵必须走 `scripts/student_utils.py::chunked_lm_loss`（512 token 分块 + checkpoint），32k 阶段峰值 10.28 GiB，16GB 卡上不要并发跑其他任务。
4. **注意力后端**：训练用 `torch.nn.attention.sdpa_kernel` 的 EFFICIENT/FLASH 后端；该 torch 构建的 SDPA 不支持 GQA，所以学生用 MHA（7=7），改 GQA 需同步改 C 引擎（`feng_llm.c` 里要加 KV 头广播）。
5. **身份不能掉**：训练语料里身份样本过采样（v1 是 15×），并过滤提及其他 AI 身份（ChatGPT/通义…）的样本；每次出模型都要用 `eval_planA_scope.py` 验证身份题。
6. **ESP32 的 16MB mmap 窗口是硬边界**：`model.bin`（15,659,904 B = 0xEEF380）必须结束在 0x1000000 之前；现行分区为 `model 0x110000/0xEF0000`、`tokdata 0x1000000/0x80000`（tokdata 用 `esp_partition_read` 读，可放窗口外）。`flash.ps1` 里的 `0x310000 / 0x1A10000` 是**已作废的旧偏移**，别照着用。
7. **板端内存账**：权重只能 flash mmap 流式读，不能预载进 SRAM（每层 Q4 ≈0.95MB，内部 SRAM 只剩 ~271KB）；KV 用 int8（`FENG_KV_INT8=1`，`main/main.c` 的 `MAX_CTX=1024`，9.93MB PSRAM）；板上 32k 上下文在 KV 内存上不可能，长文只能走滑窗/attention sink/线性注意力。
8. **速度现状**：标量 Q4 内核已到极限（4.1 周期/权重，1.86 tok/s≈537ms/token），下一个杠杆是 PIE（`ee.vmulas.s8.accx` 128 位 int8 SIMD，预期 2–3x）；不要再做内层展开之类的标量微调（已证明会变慢）。
9. **量化格式耦合**：Q4 block-64（4.25 bpw）；改 `QK` 必须同步改 C 侧 `QK`，且 `tools/export_model.py` 会生成 `ref_ids.json` / `ref_logits.bin` 供一致性校验。

## 7. 改动的验收清单

- **改了训练脚本**：用 `--steps-cap 1`（或 `--steps`）跑冒烟，确认能落盘 `<阶段>/final/` 与 `summary.json`。
- **改了 C 推理内核 / KV 量化**：必须重跑 `pc_check`（logits MATCH）→ `pc\verify_c_vs_torch.py`（fp32 路径要求 `max|diff| = 0.0000`）→ 板上 `scripts\esp32_multi.py`，并与 `logs\board_baseline_lut.txt` **逐字对比**回复。
- **改了文档**：跑 `python tools\check_md.py`（三查：代码围栏配对、反引号路径存在、无过时数字）。
- **换了模型版本**：重跑 `eval_planA_scope.py` + `eval_longctx.py`，数字同步进 `CHANGELOG.md` / `COMPARISON.md` / `DELIVERY.md`（三者口径必须一致）。

## 8. 排障速查

| 现象 | 处置 |
|---|---|
| ESP32 推理时 MMU fault / 数据读错 | 检查 `model.bin` 是否越界写进 tokdata；核对 `partitions.csv` 与 PDF 烧录偏移 |
| 板上回复乱码 | 用 `scripts\esp32_enc_test.py` 验证 GBK/UTF-8 自动跟随；`\gbk` / `\utf8` 可手动锁定 |
| 训练 OOM | 降 `--batch` / `--accum`，确认 `chunked_lm_loss` 的 chunk 没被调大，32k 阶段单独跑 |
| 模型答什么都是身份句 | v1 的老毛病，语料配比失衡；参照 v2 的 Plan A 语料 + 低 LR 补训（1000 步 / lr 1.5e-4） |
| 长文阶段跑完检索仍全错 | 长文预训练学不会检索，必须补合成检索 SFT（只对答案算 loss），见 `scripts/v3_build_retrieval.py` |
| esptool 报 port busy | 板子可能不在 USB 列表（换线/供电），或串口被 monitor 占用 |
