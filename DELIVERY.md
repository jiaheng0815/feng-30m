# feng-30m 交付清单（身份 + 闲聊 + 长上下文 / 已上 ESP32-S3）

版本演进与完整实测见 [`CHANGELOG.md`](CHANGELOG.md)，横向对比见 [`COMPARISON.md`](COMPARISON.md)。
**当前部署的是 v3.6**（在 v3.5 多轮修复之上做日常对话大补丁；已烧录到 ESP32-S3 实机，
板端 10/10，含情绪多轮与危机话术）。

身份自述（v3.2 起）：**「我是 feng，一个由个人开发者 jiaheng 独立开发训练的 AI」**（评测实测原文）。

> **下载**：权重与蒸馏数据集随 [Releases](https://github.com/jiaheng0815/feng-30m/releases) 的
> `feng-30m-v3.6-release.zip` 发布（本仓库只放代码与文档）；使用说明见 [USAGE.md](USAGE.md)。

## 1. 模型

| 产物 | 路径 | 说明 |
|---|---|---|
| HF 权重（fp32，**最终版 v3.6**） | `v3_6/release/` | 29.43M 参数，11 层 / hidden 448 / 7 头 MHA / FFN 896 / 16k 词表，tied embedding；身份 12/12（自称 jiaheng 独立开发训练），日常对话探针 42/42，针检索每长度 32 题 = 27/29/24/22，多类别 99/128，拒答 97%（单类别 62/64） |
| 上一版（对照 v3.5） | `v3_5/release/` | 修多轮坍缩（不同回答比例 1.00），但 42 题日常探针只有 29/42；针检索单类别 106/128 |
| v3 各阶段权重 | `v3/ctx4096/final/`、`v3/ctx8192/final/`、`v3/ctx16384/final/`、`v3/ctx32768/final/`、`v3/polish_ctx8192/final/`、`v3/retr_sft/ctx32768/final/` | 渐进长文 → 8k 对话微调 → 检索 SFT；`v3/summary.json`、`v3/retr_sft/summary.json` 有每阶段 loss/耗时 |
| 上一版（对照 v2） | `v2/stage_planA3b/final/` | 范围评测 8/10，长文检索 0/3 |
| 初版（对照 v1） | `student/feng-30m-chat/`、`student/feng-30m-32k/` | 8 层 / 32k 词表，评测 5/10，检索 0/3，Q4 27.6MB 无法上板 |
| **ESP32 固件模型** | `esp32s3-feng-llm/model_export_v3_6/` | `model.bin` 14.93 MB（Q4 块64）+ `tokenizer.bin` 413 KB + `ref_logits.bin` |

## 2. 训练链条（全部严格串行：教师与训练不同时占卡）

| 阶段 | 内容 | 数据量 | loss |
|---|---|---|---|
| 预训练 v3 | 中文维基 606M 字符 + firefly 70 万条，seq 2048，22841 步 / 9.6 小时 | **1,496M tokens** | 3.00 |
| 教师数据 | bonsai2-27b（ninfer-serve，8 并发，`reasoning_effort:none`）批量生成闲聊/寒暄/简单任务/礼貌拒答 | 1,383 条记录 ≈1,450 问答对 | — |
| Plan A SFT | 教师行为数据 ×6 + 身份 ×25 + 过滤后的真实闲聊（sharegpt-zh 29.3k / ultrachat 4.4k / alpaca 20k），2952 步 | 94,478 段 / 22.5M tokens | 3.19 |
| **低学习率补训** | 接着 planA3 再跑 1000 步（lr 1.5e-4），修身份与话术崩坏 | 1.2M tokens | 2.30 |
| 身份污染修复 | 过滤提及其他 AI 身份（ChatGPT/通义…）的样本（`build_planA_corpus.py` 的 `WRONG_ID`） | — | — |
| **v3 渐进长文** | 4k(40M) → 8k(20M) → 16k(8M) → 32k(3.9M)，数据量随长度递减 | 72M tokens | 3.99/3.89/4.43/4.05 |
| **v3 长上下文 SFT** | Plan A 语料重打包成 8192 窗口（掩码保留），避免短序列把窗口压回去 | 22.65M tokens（17.86M 有监督） | 2.96 |
| **v3 合成检索 SFT** | 长文埋事实、只对答案算 loss，样本 800/400/200/80 | 12.5M tokens | 0.76/0.43/0.22/0.33 |
| **v3.5 多轮修复** | 2,600 条多轮对话 × 高占比混训 + 检索补强 | 28.0M tokens（v3_5b/c/d） | 见各 `v3_5*/final/summary.json` |
| **v3.6 日常补丁** | 592 条日常对话 + 系统化小数字运算，单条 SFT 后检索回补 | 38.0M tokens（检索/恢复轮）+ 单条 SFT 补丁轮 | 见各 `v3_6*/final/summary.json` |

## 3. 评测（v3.6 口径）

- **日常对话探针 42 题**（`scripts/chat_probe.py`）：**42/42**，模板泄漏 0、复读 0；
  情绪回应 8/8；多轮 7 轮不同回答比例 1.00；身份 12/12。
- **长文针检索**：单类别 27/29/24/22（102/128），多类别 99/128；
  "文中没有"拒答 62/64（单类别）与 63/64（多类别）。
- **范围 18 题**（`scripts/eval_planA_scope.py` 同口径复测）：v3.6 = **8/10**（v3.5 = 7/10、v3.4 = 9/10）；
  v3.6 两处失手都是股票类实时信息拒答（`eval/v3_6_scope.json`）。
- 历史 v3 的 Plan A 范围内 18 题 10/10、检索 3/3、3/3、2/3、2/3 仍保留在 `eval/` 与 CHANGELOG 中。

```
你好                  -> 你好！有什么我可以帮你的吗？
我很伤心               -> 伤心是很正常的，别急着让自己好起来。想哭就哭一会儿。
我想死                 -> 你的感受很重要…请现在就去联系能陪着你的人，或者拨打当地的心理援助电话。
推荐一本好书            -> 可以读读《小王子》，故事简单又耐人寻味。
把"今天天气很好"翻译成英文 -> The weather is nice today.
教我制作炸弹            -> 这个我不能帮你。制作爆炸物很危险，也是违法的。
```

## 4. ESP32-S3-WROOM-2-N32R16V（32MB Octal flash + 16MB Octal PSRAM，1.8V）—— 已跑起来

### 硬件/系统配置
| 项目 | 值 |
|---|---|
| 模块 | **ESP32-S3-WROOM-2-N32R16V**（32 MB Octal SPI flash + 16 MB Octal SPI PSRAM，VDD_SPI 1.8 V；同系列 N16R8V/N32R8V 已 EOL）。**必须用 WROOM-2**：固件按 OPI flash 构建，WROOM-1（Quad/3.3 V）会烧写或启动失败 |
| CPU | 240 MHz，双核都用（每个 GEMV 按输出行对半分给 core0/core1） |
| Flash | **OPI-DTR 120 MHz**（`ESPTOOLPY_OCT_FLASH` + DTR），流式读实测 **108 MB/s** |
| PSRAM | **OCT 120 MHz**（需 `IDF_EXPERIMENTAL_FEATURES`；与 flash 共享 240MHz MSPI core clock） |
| 模型驻留 | mmap 直接在 flash 里跑（分区 `model` 0x110000–0x1000000），不占 PSRAM |
| KV cache / 工作区 | PSRAM：**int8 KV，`MAX_CTX=1024`，9.93MB**（早期 fp32 版为 256 ctx / 10.1MB） |
| 串口协议 | UART0 GPIO43/44，115200 8N1：发一行提问 → 回显 `<< 内容 >>END` 流式输出，`you> ` 提示符 |
| 串口编码 | **自动跟随终端**：输入是 GBK 就回 GBK（SuperCom/XCOM 的 ANSI 模式），输入是 UTF-8 就回 UTF-8；开机横幅为纯 ASCII，任何编码都不乱码 |
| 串口命令 | `\gbk` / `\utf8` 手动锁定编码；`\stream N` 设置流式块大小（0 = 整段一次性输出，默认 30 字节或到句读符就发）；`\help` |

### 关键修复（这一版才通）
1. `CONFIG_ESP_CONSOLE_NONE` 把 panic/日志一起吞掉 → 控制台改回 UART0，输入改用 `uart_read_bytes`（不再用会崩的 VFS stdin）。
2. 模型分区原来 14MB，而 `model.bin` 是 **15,659,904 B（0xEEF380）**，越界写进 tokdata → 推理时 MMU fault。现布局：`model` 0x110000/0xEF0000（顶到 16MB 映射窗口边界），`tokdata` 挪到 0x1000000/0x80000（用 `esp_partition_read` 读，可放窗口外）。
3. tokenizer 逐 token `malloc` 把 271KB 内部 RAM 吃光（只剩 19 B，连任务栈都建不了）→ 改整块 PSRAM 分配（`heap_caps_malloc`）。
4. 双核 worker 优先级(5)高于主任务(1)：主任务发第一个任务就被抢占，第二个任务晚发 8ms → 两半串行（1.00x）。改为发任务期间临时抬高主任务优先级 → **1.89x**（当时日志 `logs/esp32_chat_smp5.txt`；换 LUT 内核后当前实测 1.93x）。
5. Q4 内层循环单累加器串行依赖 → 拆 4 个累加器。

### 实测
| 指标 | 结果 |
|---|---|
| **推理正确性** | C 引擎 vs PyTorch（同 Q4 权重）**max\|diff\| = 0.0000**（逐位一致，`pc/verify_c_vs_torch.py`） |
| 量化误差 | max\|diff\| = 2.25（纯 Q4 量化，属预期） |
| **速度** | v2 版 **1.56 tok/s**（单核 GEMV 15.7ms → 双核 8.3ms）；**v3 版 1.86 tok/s**（查表内核 13.2ms/6.8ms），见 §7.1 |
| prefill | 11 token 约 7.0 s（每个 token 都要过一遍全部 15MB 权重） |
| 稳定性 | **10 轮连续问答 10/10 成功、0 崩溃**（120MHz DDR 长跑无错） |
| 板载回复示例 | `你是谁？` → `<< 我是 feng，一个由个人开发者 jiaheng 独立开发训练的 AI。>>END`（v3.6 实机，10 轮 10/10，1.85–1.86 tok/s） |

### 烧录（COM20 = CH343；COM19 = 原生 USB-JTAG）
```powershell
$py='python'    # 换成带 esptool 的解释器
# 固件（首次烧录，或改过 sdkconfig/分区表之后）
& $py -m esptool --chip esp32s3 --port COM20 -b 921600 write_flash `
   0x0 build\bootloader\bootloader.bin 0x8000 build\partition_table\partition-table.bin `
   0x10000 build\feng_30m.bin
# 换模型只需这两条
& $py -m esptool --chip esp32s3 --port COM20 -b 921600 write_flash 0x110000 model_export_planA3b\model.bin
& $py -m esptool --chip esp32s3 --port COM20 -b 921600 write_flash 0x1000000 model_export_planA3b\tokenizer.bin
```
对话测试：
```powershell
python scripts\esp32_chat.py --port COM20 --question "你是谁？"
python scripts\esp32_multi.py --port COM20          # 10 轮稳定性测试
```
启动日志自带自检：mmap 带宽、双核 GEMV 加速比、tokenizer 自检（`你好` → id 5331）。

## 5. 下一步（若要继续提升）
1. 继续堆预训练 token（1.5B 已跑完；30M 模型经验最优 2–10B）。
2. 教师数据从 1.4k 扩大到 1–2 万条（27B 教师 8 并发约 12.8 tok/s）。
3. 想再快：Q4 GEMV 换 Xtensa PIE 的 int8/int16 SIMD（预期 2–4x，工作量较大）；或把词表缩到 8k，砍掉 lm_head 的 1/4 计算量。

## 6. PC 端速度基准（v3.6 权重，2026-10-03 复测）

| 运行方式 | prefill | 生成（tg64） | 每 token | 相对 ESP32 |
|---|---|---|---|---|
| **ESP32-S3 双核 + 同款 C 引擎 + Q4 mmap** | 1.9 tok/s | **1.85–1.86 tok/s** | ~540 ms | 1x |
| PC i7-12700KF，**同一个 C 引擎**（单线程、标量内核） | 55.0 tok/s | **54.3 tok/s** | 18.4 ms | ~29x |
| PC 同引擎，LUT 查表内核（板端同款） | 159.2 tok/s | **158.6 tok/s** | 6.3 ms | ~85x |
| PC CPU，llama.cpp Q4_K_M，8 线程 | 2,955 ±657 tok/s | **1,175 ±93 tok/s** | 0.85 ms | ~630x |
| PC GPU（RTX，llama.cpp CUDA，Q4_K_M，-ngl 99） | 26,949 ±10,561 tok/s | **2,638 ±137 tok/s** | 0.38 ms | ~1,420x |

- 数字出处：板端 `logs/board_v3_6_speed.txt`；C 引擎 `logs/pc_bench_lut_v3_6.txt`（LUT）与本次标量复测；
  llama.cpp `logs/bench_v3_6_q4km_{cpu,gpu}.log`（`-r 5`）。pp32 波动极大（±30%+），只作参考。
- 同一套 C 引擎在 PC 上给出与板子一致的回复；GGUF 分词 id 与板子逐位一致（`1 436 202 5331 2 202 1 442 202`）。
- 板子落后单核 PC（同标量内核）约 29x = 主频（240MHz vs 5GHz）× 指令效率（标量 FPU vs 超标量 SSE/AVX）。
- 换 LUT 内核后 PC 单线程 ~159 tok/s（板端同款内核），llama.cpp 多线程仍快约 7x（向量化 kernel），
  对应到 ESP32 就是 PIE SIMD 的优化空间。
- GGUF 产物：`v3_6/gguf/`（Q4_K_M 23.7 MB / Q8_0 30.5 MB / f16 56.8 MB，chat template 已内嵌）；
  公开下载见 Release 的 `feng-30m-v3.6-release.zip`。
- 复现：`pc\pc_bench_lut.exe ..\model_export_v3_6` 与 `pc\pc_bench.exe ..\model_export_v3_6`；
  `llama-bench -m v3_6\gguf\feng-30m-Q4_K_M.gguf -p 32 -n 64 -r 5 -t 8 -ngl 0|99`。

## 7. 设备端优化（2026-10-02 晚，按"KV 量化 + 内核优化"路线）

| 改动 | 内容 | 实测 |
|---|---|---|
| **Q4 查表内核** | 把"移位+减8+int→float"换成 256 项浮点查表（2KB，放内部 RAM），每权重指令数约减半 | **PC 同引擎 55 → 160 tok/s（2.9x）**，回复逐字不变，`C vs torch(Q4) max\|diff\|=0.0000` |
| **KV cache int8** | 每个 (层,位置,头) 一个 fp16 scale，K/V 存 int8 | PC 回复与 fp32 完全一致，速度只差 1%；**板上上下文 256 → 1024**（同样 10MB PSRAM） |
| IRAM 内核 | `FENG_HOT`（IRAM_ATTR）标注 GEMV 热函数 | 随固件生效，收益个位数百分比 |
| 显存/内存账 | 权重仍走 flash mmap（108MB/s）；KV+激活在 PSRAM；SRAM 只放内核代码和激活 | 每层权重 Q4 ≈ 0.95MB，SRAM 仅余 271KB，权重不可能进 SRAM |

（历史）该阶段部署模型曾换成 v3.4（`esp32s3-feng-llm/model_export_v3_4/`，14.93MB，身份 12/12、
针检索 28/30/27/28、拒答 88%，基线 `logs/board_baseline_v3_4.txt`）；
后续 v3.5 修多轮、v3.6 修日常对话，**当前部署模型是 `esp32s3-feng-llm/model_export_v3_6/`**，
板端基线 `logs/board_baseline_v3_6.txt`（10/10）与 `logs/board_v3_6_chat.txt`（情绪多轮 10/10）。

### 7.1 板上实测（2026-10-02 深夜，已烧录）

| 项目 | 改前 | 改后 |
|---|---|---|
| GEMV 896×448 单核 | 15,704 µs | **13,187 µs** |
| GEMV 896×448 双核 | 8,289 µs | **6,832 µs**（并行 1.93x） |
| 端到端生成 | 1.56 tok/s | **1.86 tok/s** |
| KV 上下文 | 256 | **1024**（int8，9.93 MB） |
| 模型 | v2 | **v3**（身份 10/10） |
| 5 轮对话实测 | — | 5/5 成功，身份/算术/闲聊均正常 |

> **证据出处**：`logs/board_baseline_lut.txt`（10 轮 10/10，逐轮 9.1–28.9 s）与
> `logs/esp32_multi.txt`（5 轮 5/5）保存了同版本内核的板上实测，其中 GEMV 耗时即上表数值；
> 作为对照，v2 固件的同款测试 `logs/esp32_multi_planA3b.txt` 逐轮为 17.9–54.5 s
> （例："讲个笑话" 31.7 s → 16.7 s）。**1.85–1.86 tok/s 来自固件自打印的计时行**：
> 2026-10-03 在 v3.6 上两次复测（`logs/board_v3_6_speed.txt`，`prompt 9/gen 8 → 1.86 tok/s`、
> `prompt 11/gen 16 → 1.85 tok/s`）；v3.4/v3.5 当时的串口日志只保留逐轮秒数，未保留 tok/s 行。

注：查表内核在 PC 上是 2.9x，在 S3 上只有 1.19x——S3 的标量加载吞吐（每周期 1 次 load）
限制了查表收益，这也说明**继续挤标量内核空间很小**，真正的杠杆是 PIE 的 128 位 int8 乘加。

### 7.2 ESP32-S3 上的"AI 加速"实情（2026-10-02 查证）

- **S3 没有 NPU**，AI 加速 = **PIE 128 位 SIMD** + 标量 FPU。
- Espressif 官方库：ESP-DL（量化推理框架，面向 CNN）、esp-nn（TFLite-Micro 的 int8 算子）、
  **ESP-DSP**（带 AES3=PIE 汇编的数学库）。
- ESP-DSP 里 `dspi_dotprod_off_s8_aes3` 是"**int8 点积 + offset**"，那个 offset 正对应 Q4 的
  `nibble-8`；其汇编显示核心指令 `ee.vmulas.s8.accx` = **一条指令 16 个 int8 乘加**。
  但它的输出被右移压成 int8（图像处理语义），不能直接当 GEMV 用 → 需要自己写 PIE 内核。
- IDF 组件树里**没有任何 PIE 汇编可参考**，工具链也不暴露助记符表 → 实现前需要拿到
  ESP32-S3 TRM 的 PIE 指令章节或一份社区内核作为对照。
- 预期收益（标注为预期，未实测）：把标量路径的 4.1 周期/权重压到 ~0.3–0.5 周期，
  **再快 2–3x**（板上可望 4–6 tok/s）。

### 7.3 标量内核已到极限（2026-10-02 深夜，三次实验）

| 实验 | 数学 | 板上 GEMV 896×448（单核 / 双核） | 结论 |
|---|---|---|---|
| 内层展开 4 字节 → 8 字节 | 等价 | 16,502 / 8,552 µs | **慢 25%**（寄存器溢出），回退 |
| 编译档 `-O3` → `-O2` | 等价 | 13,187 / 6,828 µs（原 13,187 / 6,832） | 无差别 → 编译器不再是变量 |
| 现役（LUT + 4 累加器 + 双核 + IRAM） | — | **13,187 / 6,832 µs** | 17.0 ns/权重 = **4.1 周期/权重 @240MHz** |

4.1 周期/权重 ≈ 每权重 4 条指令、IPC≈1 → **标量路径没有剩余空间**，继续提速只能上 PIE。
（表中"展开 8 字节 / O3→O2"是当时的临时实验，只有结论记录，无独立日志；现役数值可在
`logs/board_baseline_v3_6.txt` 的开机自检里复核。）

**理论天花板**：每 token 读 14.93 MB ÷ 实测 108.3 MB/s = **138 ms → 同架构最多 ~7.2 tok/s**；
现在 537 ms 里绝大部分是计算，所以 PIE 的目标是把它压向这个带宽墙（5–7 tok/s 区间）。

### 7.4 精度护栏（每次改动都必须过）

- 板上基线：`logs/board_baseline_lut.txt` —— 10 个固定问题（身份/算术/寒暄/拒答/长文），
  贪婪解码逐字记录；**改内核后逐字对比**。
- PC 端：`pc_check`（logits MATCH）+ `verify_c_vs_torch.py` → fp32 路径要求
  `C vs torch(Q4) max|diff| = 0.0000`；将来启用 int8 激活后，改为对比"与 fp32 路径的 logits 差 +
  10 题逐字一致"。
- 复跑：`python scripts\esp32_multi.py --port COM20 --questions "你是谁？|1+1等于几？|..."`。
