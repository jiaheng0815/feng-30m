# feng-30m on ESP32-S3-WROOM-2-N32R16V（32 MB Octal flash + 16 MB Octal PSRAM）

把 **feng-30m**（Qwen3 架构：11 层 / hidden 448 / 7 头 MHA / FFN 896 / 16k 词表 /
tied embedding，29.43M 参数）量化成 Q4 后**在 ESP32-S3 上离线对话**。

当前部署的是 **v3.16-embed 嵌入式权重**（`../v3_16/board_p3/`：v3.15/board_ctxid4 +
上下文双向问名补丁，继续做 **Q4 权重 + q2 KV 双 QAT**：block8、板端 2048 上下文；
嵌入式 32 题矩阵 27/27 + 长文召回 4/4；修掉多轮里的身份漂移与「报名字后的串名」，
见 `../CHANGELOG.md` 的 v3.16-embed 节）。

**固件自带四个 tool**（模型不再学算术，见 `main/feng_calc.c`、`feng_tools.c`、`feng_memory.c`）：

- **算式**：`4854+4411`、`5.3+4.1`、`(3+4)*2` → 0.5 秒秒回，SoC 运算器直接算；
- **时间**：`现在几点？` → UTC+8 日历。板端放不下 WiFi 协议栈（app+model 占满 16MB mmap 窗口），
  由宿主连接时发 `\settime <unix秒>`（宿主走 NTP）对时，固件用 `esp_timer` 走时；
  另支持**时钟推算**：`现在7点，再过3小时是几点？` → "再过 3 小时是 10 点。"
  （跨天自动说"明天/昨天"，不写起点按当前时间算）；
- **随机数**：`给我个1到100的随机数` → seed = 运行时间×1.54×1000，丢弃第一个随机数取第二个。
- **记忆**（v3.17）：`我叫小明，请记住` / `我最喜欢的颜色是蓝色` / `我住在成都` / `我养了一只乌龟`
  → 事实存进引擎（不拦截，模型照常回应）；追问 `我叫什么名字？` / `我最喜欢什么颜色？`
  / `我住在哪里？` / `我养了什么？` / `你叫什么名字？` **由引擎 0.5 秒确定性作答**
  （12 题连续记忆 **12/12**，身份永不串名；`\mem` 查看、`\mem clear`/`\reset` 清空）。
  另有**通用键值槽**：`我最喜欢的书是《小王子》` → `我最喜欢什么书？`、
  `我的生日是3月5日` → `我的生日是几号？`（生日/职业/家乡…任意短键，满 8 条轮换最旧）。
  还会**列出**（`你还记得什么？`）与**遗忘**（`忘掉我的颜色`、`把记住的都忘掉`；
  遗忘后该键追问统一答"我不记得了"，重新学习即恢复）。
  只覆盖可枚举句式（单测 69 项），其他说法仍走模型自己的多轮能力。

**固件从 v3.13 起默认保留跨轮上下文**：每轮把 `<|im_end|>` 补进 KV，下一轮只 prefill
新增的用户片段；实测 `我叫小明，请记住 → 你叫小明`、`喜欢蓝色 → 你最喜欢蓝色`、
`养了一只猫 → 你养了一只猫` 全对。`\reset` 清空上下文，写满 2048 自动开新对话
（`../logs/board_v3_13b_memory.txt`）。
**当前固件 + 权重的 64 轮连续对话压测**（8 问循环 ×8，2026-10-05）：
**64/64 成功**、平均 12.5 s/轮、最深上下文 33.4 s，其中 24 轮由 tool **0.5 s 秒回**
（`../logs/board_v3_17_stress64.txt`）。

**输入长度**：单行最多 4095 字节，超出会提示并丢弃多余部分；正文按 token 预算截断时
保留 UTF-8 边界与模板收尾。注意板端 prefill 成本 = **每 token 权重 ~0.38 s + 注意力
0.85 ms×n²/2**：491 tokens 的长文实测 **291 s**（分两轮：正文一行、提问一行，
提问能答对文中取件码；217 tokens 的同类测试答"文中没有提到"，板上长文检索不稳定），
固件会在 >80 tokens 时打印预估时间（实测 289 s 预估 vs 290.6 s 实际），
**交互输入建议 ≤ ~150 tokens（≈400 字节中文）**，长文请在 PC 上跑。
日志：`../logs/board_longprompt_2turn_491tok.txt`、`../logs/board_longprompt_2turn.txt`。

**实机实测（2026-10-04，v3.16-embed + q2block8 固件）**：KV `q2/block8, ctx 2048, 9.62 MB`，
PSRAM 余 5006 KB；工具（时间/随机数/算式）13/13
（`../logs/board_v3_16p3_tools.txt`）、跨轮记忆 12 题 **10/12**
（`../logs/board_v3_16p3_memory12.txt`）、速度 **2.18 tok/s**（短回答实测，
`../logs/board_v3_16p3_tail_trailer.txt`；优化前 1.80-1.85）——
回复尾行会打印 `>>END (ctx N/2048, X.XX tok/s)`，和 PC 版同一口径；
长上下文另有专项：q2/2048 单次 forward **2224 ms**（K 835 / softmax 48 / V 832 / 权重等 509 ms），
比优化前 3534 ms 快 **37%** 且 32 题矩阵输出逐字一致（`../CHANGELOG.md` 的 v3.15-embed 附录、
`../logs/board_bench_attn_vfold.txt`、`../logs/board_bench_attn_prof.txt`）；
prefill 还会跳过中间 token 的 lm head（每个省 ~115 ms），同一组 4 轮连续提问从
13.0/12.5/11.7/12.1 s 降到 **10.5/9.8/9.1/9.4 s**，回答逐字相同
（`../logs/board_v3_15ci4_memory_after_vfold.txt`）；
历史版本：v3.6 在 int8 KV @1024 ctx 下 1.85–1.86 tok/s（`../logs/board_v3_6_speed.txt`），GEMV 双核加速 1.93x；
板端回复实测：`你是谁？` → `我是 feng，一个由个人开发者 jiaheng 独立开发训练的 AI。我可以陪你聊天、帮你写作、翻译和写简单代码。`，
`你是Qwen吗？` → `不是。我是 feng，由个人开发者 jiaheng 独立开发训练的 AI。`
基线记录见 `../logs/board_baseline_v3_6.txt` 与 `../logs/board_v3_6_chat.txt`
（v3.5/v3.4 的历史基线见 `board_baseline_v3_5.txt`、`board_baseline_v3_4.txt`）。

板端要的两个文件（`model.bin` + `tokenizer.bin`）有两个来源：**① 直接下载**
[Releases](https://github.com/jiaheng0815/feng-30m/releases) 里的 `feng-30m-v3.16-embed-release.zip`，
取包内 `weights/esp32/`；**② 按下面第 1 节自己从 HF 权重导出**。只想跑起来就选 ①，跳过第 1 节。

> 下文命令里的 `<...>` 都是占位符，换成你本机的路径；Python 脚本会自动解析项目内路径
> （见 `scripts/paths.py`），工具链位置可写在 `scripts/local_paths.example.json` 的副本里。

## 0. 硬件要求（复现必读）

实测板：**ESP32-S3-WROOM-2-N32R16V**——32 MB **Octal SPI** flash + 16 MB **Octal SPI** PSRAM，
VDD_SPI 1.8 V（同系列 N16R8V / N32R8V 已 EOL）。

| 约束 | 原因 |
|---|---|
| 必须是 **WROOM-2**（Octal / 1.8 V） | `sdkconfig` 开了 `ESPTOOLPY_OCT_FLASH=y` + `FLASHMODE_OPI`；换成 WROOM-1（Quad / 3.3 V）会烧写或启动失败 |
| Flash 必须 **32 MB** | 只有**前 16 MB 能被 mmap**（NOR flash 24 位地址上限），模型必须落在这里；16 MB 以上的部分用 `esp_partition_read` 读，`tokdata` 就是放那里的。16 MB 模块放不下整个布局 |
| PSRAM 必须 **16 MB** | KV（1024 ctx，int8）9.93 MB + 工作区/分词 ≈1 MB，8 MB 版本不够 |
| 120 MHz PSRAM 需要 `CONFIG_IDF_EXPERIMENTAL_FEATURES=y` | 已在 `sdkconfig` 里开启，换 IDF 版本时别把它关掉 |
| **GPIO33–37 不可用** | WROOM-2 的 Octal flash 占用这几个引脚（模块也未引出） |

板子自己的开机自检可复核以上配置（`logs/board_baseline_lut.txt`）：ROM 打印
`Octal Flash Mode Enabled`、`SPI Flash Size : 32MB`，PSRAM 打印 `VCC 0x00 (1.8V)`、
`Found 16MB PSRAM device`、`Speed: 120MHz`，mmap 流式读 108.3 MB/s。

> **别把两个「16 MB」搞混**：**mmap 窗口 = 16 MB**（NOR flash 24 位地址上限）是地址限制；
> **模块 flash = 32 MB** 是容量。模型放前 16 MB 内做 mmap 直读，tokenizer 放在 16 MB 之后用
> `esp_partition_read` 读——所以板子必须是 32 MB flash，而不是 16 MB。

```
┌─ ESP32-S3-WROOM-2-N32R16V（32MB Octal flash + 16MB Octal PSRAM，1.8V） ──────────┐
│  flash: 0x110000   model.bin      14.93 MB, Q4 block-64, mmap 直读（不占 RAM）   │
│         0x1000000  tokenizer.bin  413 KB, 启动时读入 PSRAM                       │
│         0x0010000  app            288 KB（factory 分区 1 MB）                    │
│  PSRAM: KV cache int8 1024 ctx = 9.93 MB + 工作区/分词 ≈ 1.0 MB                  │
│  时钟 : CPU 240MHz ×2   flash OPI-DTR 120MHz   PSRAM OCT 120MHz                  │
│  串口 : UART0 115200 8N1（`you> ` 提示符，`<< 内容 >>END` 流式回复）             │
└──────────────────────────────────────────────────────────────────────────────────┘
```

实测（v3.6 + 本工程内核）：**1.85–1.86 tok/s**，GEMV 896×448 单核 13,187 µs / 双核 6,832 µs（1.93x），
flash mmap 流式读 **108.3 MB/s**，C 引擎与 PyTorch(Q4) **逐位一致**。

> 现行发布固件 = **q2 KV / 2048 ctx**（本工程默认 int8 / 1024，`idf.py -DFENG_USE_Q2_KV=ON build` 切 q2），
> app 当前 306,624 B；一键刷机包见 Release 附件 `feng-30m-v3.16-embed-firmware.zip`。
瓶颈是"每个 token 都要把 15 MB 权重从 flash 读一遍 + 逐权重标量计算"，实测证明是**算力瓶颈**
（flash 提频 80→120MHz 只快 5%，双核则 +89%）。

## 1. 拿模型（PC 上执行，二选一）

推荐直接下载 Release 附件 **`feng-30m-c-engine-model-v3.16-embed.zip`**（已导出，板端/PC C 引擎通用）。
想自己导出（当前板端权重 = v3.16-embed，Q4+q2 双 QAT）：

```powershell
$py = "python"                    # 换成装了 torch + transformers 的解释器
& $py tools\export_model.py --model <v3.16-embed包>\weights\hf --out model_export_v3_16p3
# -> model_export_v3_16p3/model.bin      14.93 MB（Q4 块64 + fp16 norms/scales）
#    model_export_v3_16p3/tokenizer.bin  413 KB
#    ref_logits.bin / ref_ids.json / export_info.json
```

> 注意：PC 的 v3.14 HF 权重没做量化感知训练，导出给 C 引擎会退化（同套 32 题矩阵 22/27 vs 27/27，
> `../logs/pc_kv_suite32_v3_14pc2_q2b8.txt`）。

## 2. PC 端一致性自检（强烈建议先跑）

同一套 C 核心在 PC 上编译运行，与 PyTorch 参考逐值比对：

```powershell
cd esp32s3-feng-llm
$src = @('pc_check.c','..\main\feng_model.c','..\main\feng_llm.c','..\main\feng_quant.c',
         '..\main\feng_smp.c','..\main\feng_tokenizer.c','-I..\main','-lm')
& "<MSYS2>\ucrt64\bin\gcc.exe" -O2 -o pc\pc_check.exe @src
.\pc\pc_check.exe ..\model_export_v3_16p3 ..\logs\c_logits_v3_16p3.bin
```

> 上面两条命令里的 gcc 路径是作者机器的 MSYS2 路径，换成你本机的即可；
> `model_export_v3_16p3` 若没自己导出，把它指向 Release 包（v3.16-embed 或预导出模型包）里的 `model.bin` 所在目录。

期望输出（v3.16-embed 实测）：

```
logits check: n=16384 max|diff|=2.7896 mean|diff|=0.48480  argmax c=5331 ref=5331 MATCH
```

这里的 2.27 是**纯 Q4 量化误差**（对 fp32 参考）。要验证实现本身是否等价，再跑：

```powershell
$env:CUDA_VISIBLE_DEVICES=''
& "<带 torch 的 python>" pc\verify_c_vs_torch.py `
   --export ..\model_export_v3_16p3 --model <v3.16-embed包>\weights\hf `
   --c-logits ..\logs\c_logits_v3_16p3.bin
# [C vs torch(Q4)] max|diff| = 0.0000   ← 实现逐位一致
```

## 3. 编译固件（ESP-IDF v5.5.5）

```powershell
$env:IDF_TOOLS_PATH = "<IDF 工具链目录>"
& "<esp-idf 目录>\export.ps1"
cd <仓库>\esp32s3-feng-llm
idf.py build          # 目标/分区表已在 sdkconfig 和 partitions.csv 里配好
```

> 目标芯片（ESP32-S3）、分区表与 flash/PSRAM 配置都已在 `sdkconfig.defaults`、`partitions.csv`、
> `main/CMakeLists.txt` 里写死；路径同样按你本机的 ESP-IDF 安装位置修改。

## 4. 烧录（COM20 = CH343；COM19 = 芯片原生 USB-JTAG，两个都能烧）

```powershell
$py = "python"        # 换成带 esptool 的解释器（ESP-IDF 自带的那个也行）
# ① 固件（首次烧录、或改过 sdkconfig/分区表之后）
& $py -m esptool --chip esp32s3 --port COM20 -b 921600 write_flash `
    0x0 build\bootloader\bootloader.bin 0x8000 build\partition_table\partition-table.bin `
    0x10000 build\feng_30m.bin
# ② 换模型只需这两条
& $py -m esptool --chip esp32s3 --port COM20 -b 921600 write_flash 0x110000  model_export_v3_16p3\model.bin
& $py -m esptool --chip esp32s3 --port COM20 -b 921600 write_flash 0x1000000 model_export_v3_16p3\tokenizer.bin
```

> 第 ② 步的两个文件也可以直接用 Release 包里的 `weights/esp32/model.bin` 与 `weights/esp32/tokenizer.bin`；
> 偏移量必须与 `partitions.csv` 一致（`model` 0x110000 / `tokdata` 0x1000000），否则会出现 MMU fault 或读不到模型。
> **不想装 ESP-IDF**：Release 附件 `feng-30m-v3.16-embed-firmware.zip`（约 13 MB）已含
> bootloader + 分区表 + app + 模型 + 哈希与烧录命令，esptool 一次烧完（本机同款板卡实测）。
> 也可以直接用一键脚本：`.\flash.ps1 -Port COM20 -EspIdfPath <esp-idf> -ModelDir ..\model_export_v3_16p3`。

> 提示：模型写 15 MB 约需 3.5 分钟（921600 波特率，压缩后约 13 MB）。
> 板子不在 USB 列表里时 esptool 会报 "port is busy or doesn't exist"——先检查线/供电。

## 5. 串口对话协议

| 项目 | 值 |
|---|---|
| 波特率 | **115200** 8N1（UART0 = GPIO43/44） |
| 输入 | 一行一句（回车发送），串口会回显你输入的字符 |
| 输出 | 流式 `<< 内容 >>END`；默认按句读符或 30 字节成块发送，减少终端时间戳刷屏 |
| 编码 | **自动跟随终端**：发 GBK 就回 GBK（SuperCom/XCOM 的 ANSI 模式），发 UTF-8 就回 UTF-8 |
| 命令 | `\gbk` `\utf8`（手动锁编码）、`\stream N`（流式块大小，0 = 整段一次性输出）、`\help` |

开机自检会打印：mmap 带宽、双核 GEMV 加速比、KV 配置、tokenizer 自检（`你好` → id 5331）。
PC 侧测试脚本：`python scripts\esp32_chat.py --port COM20 --question "你是谁？"`、
`python scripts\esp32_multi.py --port COM20`（多轮稳定性）、`python scripts\esp32_enc_test.py COM20`（双编码自检）。

## 6. 关键参数与调优

| 项 | 值 | 说明 |
|---|---|---|
| 上下文 | **1024 token（int8 KV，默认）** | `main/main.c` 的 `MAX_CTX`；int8 KV = 9.93 MB。改回 fp32 则只够 256 ctx |
| KV 量化 | int8 + 每 (层,位置,头) 一个 fp16 scale；可选 **q2 block8（2048 ctx）** | 开关在 `main/CMakeLists.txt`：`FENG_KV_INT8=1`（默认）或 `idf.py -DFENG_USE_Q2_KV=ON build`（2bit、块 8，KV 9.62 MB / 2048 ctx） |
| 生成长度 | 96 token | `MAX_NEW` |
| 采样 | 贪心 + 重复惩罚 1.15 | `sample_next()` |
| 量化 | Q4 block-64（4.25 bpw） | `tools/export_model.py`；改 `QK` 需同步改 C 的 `QK` |
| 内核 | Q4 查表（256 项浮点 LUT）+ 4 累加器 + 双核分半 + IRAM | 见 `feng_quant.c` / `feng_smp.c` |
| 速度 | **1.85–1.86 tok/s**（v3.6 实机，`../logs/board_v3_6_speed.txt`） | PIE 路线已实测结案（整块内核只有 1.05×，见 `../CHANGELOG.md` v3.16-embed 附录）；标量 FPU 是 Q4 GEMV 的最优解 |

## 7. 目录

```
esp32s3-feng-llm/
├── main/            推理核心（可移植 C11）+ ESP-IDF 应用
│   ├── feng.h            模型容器 / 推理 API / KV 模式开关
│   ├── feng_model.c      model.bin 解析、张量定位
│   ├── feng_quant.c      Q4 查表/fp16 GEMV
│   ├── feng_llm.c        RMSNorm / QK-norm / RoPE / MHA / SwiGLU / tied head / int8 KV
│   ├── feng_smp.c        双核 GEMV（PC 上退化为单核直通）
│   ├── gbk.c / gbk_table.c  串口编码转换（UTF-8 ↔ GBK，表由词表生成）
│   ├── feng_tokenizer.c  设备端 BPE（二分查找合并表）
│   └── main.c            分区 mmap + PSRAM 分配 + 串口对话
├── tools/export_model.py     HF → model.bin / tokenizer.bin / 参考 logits
├── tools/gen_gbk_table.py    生成 GBK 转换表
├── pc/pc_check.c             PC 端一致性验证（对比 ref_logits / PyTorch）
├── pc/pc_bench.c             PC 端速度基准（同引擎单线程）
├── pc/gbk_selftest.c         编码转换自检
└── partitions.csv / sdkconfig.defaults / CMakeLists.txt
```

> 本引擎按 **MHA（7 个 Q 头 = 7 个 KV 头）** 实现；若改成 GQA/MQA（如 7 头 / 1 KV 头），
> 需要在 `feng_llm.c` 里加 KV 头广播。（v1 也是 MHA，网上"v1 是 MQA"的说法不成立。）

## 8. 已验证结果（v3.16-embed，2026-10-04 实机）

```
$ .\pc\pc_check.exe ..\model_export_v3_7f ..\logs\c_logits_v3_7.bin
model: 11 layers hidden 448 heads 7 x 64 ffn 896 vocab 16384 rope 1000000
tokenizer: vocab 16384 merges 16124 im_start=1 im_end=2 eot=0
encode("你好") -> 9 tokens: 1 436 202 5331 2 202 1 442 202   ← 与 PyTorch 分词完全一致
logits check: n=16384 max|diff|=2.8167 mean=0.48101 argmax c=5331 ref=5331 MATCH
greedy continuation: 你好，今天想聊点什么？

$ python pc\verify_c_vs_torch.py ... 
[C vs torch(Q4)]           max|diff| = 0.0000   ← 实现逐位一致
[torch(Q4) vs torch(fp32)] max|diff| = 2.8167   ← 纯 Q4 量化误差（预期）
```

板子启动自检（节选）：

```
I (961)  feng: mmap stream read: 15296 KB in 144 ms -> 108.3 MB/s
I (1827) feng: KV cache: q2/block8, ctx 2048, 9.62 MB
I (1828) feng: PSRAM free after setup: 5006 KB
I (2228) feng: gemv 896x448: 1-core 13186 us | 2-core 6837 us | speedup 1.93x
```

对话实测：v3.16-embed + q2block8 固件工具（时间/随机数/算式）13/13
（`../logs/board_v3_16p3_tools.txt`）+ 跨轮记忆 12 题 **12/12**（v3.17 引擎记忆 tool，
`../logs/board_v3_16p3_memory12_engmem.txt`）+ 报名字后身份修复
（`../logs/board_v3_16p3_nameleak.txt`、`../logs/board_v3_16p3_identity_ctx.txt`）；
历史记录：v3.11 q2 27/27+4/4、v3.10 q2 27/27+4/4、v3.7 q2 27/27+4/4、
v3.6 int8 1.85–1.86 tok/s、
v3.4 10/10、v3 5 轮 5/5、v2 固件 10/10。
