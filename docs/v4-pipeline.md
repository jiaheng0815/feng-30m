# feng-30m 4.0 训练链路

数据（`scripts/v4_teacher_gen.py` 的产出）跑完之后，按本文从头走到尾。
起点是 **`v4/base/`**（v3.19 pc4 的 HF 权重，含自带 tokenizer）。

> ⚠️ **每一步都要看数，不要盲跑。** 下面标了每步该出现什么数字；
> 对不上就停下来查，别往下走——4.0 的全部价值就在这份新数据上，
> 训练配置错了会把它浪费掉。

---

## 0. 前置检查与显卡交接

```powershell
# 一条命令做完：确认数据跑完 → 停生成器与教师服务 → 等显存真的回落 → 放行判定
pwsh tools\switch_to_train.ps1
```

**为什么要用脚本而不是手敲 `taskkill`**：Windows 是 WDDM 显存模型，
「进程退出」和「显存归还」之间有几秒到几十秒的异步延迟。杀完立刻开训练会撞上
还没释放的显存——轻则 OOM，重则 PyTorch 自己降配、跑出跟预期不一样的结果。
脚本会轮询到显存回落为止，**确认干净了才放行**。

两个判据：
- **数据跑完没**：`journal + rejects` 加起来达到提示词池规模才允许切；没跑完时
  除非加 `-Force`（提前收工用）否则直接拒绝，不会误杀还在写的任务。
- **显存回落没**：降到 2500 MiB 以下算干净。注意**不是 0**——Windows 桌面
  常驻 1~2 GB（dwm / 浏览器 / 微信 / 壁纸引擎都在用 GPU），这是正常的。
  v4 的 SFT 只要 2~4 GiB（1024 长度 / batch 24×2 / 30M 模型），余量四倍以上。

`-TimeoutSec 180` 可以延长等待；返回非 0 就是"别开训练"，脚本会列出还在占卡的进程。

---

## 1. 合并 SFT 数据

```powershell
python scripts\v4_build_sft.py --out data\v4_sft.jsonl `
    --val-out data\v4_val.jsonl --val-ratio 0.02 `
    --extra data\v4_daily.jsonl --identity-repeat 4 --extra-repeat 8
```

`data\v4_daily.jsonl` 是 `scripts\v3_6_build_daily_patch.py` 的产出（592 条手写
日常对话），**先用 `calc_tool.extract` 滤掉纯算式**（v3.14 起算术交给 tool、不进训练）：

```python
from calc_tool import extract          # 项目自己的算式识别器，别自己写正则
keep = [r for r in rows if extract(user_turn(r)) is None]   # 592 → 440
```

**为什么必须过采样（2026-10-17 实测，血的教训）**

教师数据看着有 3.4 万条，但**目标类别在里面的密度极低**——47,874 个回答里：

| 探针考的题 | 训练集条数 |
|---|---|
| 纯寒暄 | 59 |
| 「对不起」 | 16 |
| 翻译成英文 | 75 |
| 「我很伤心」 | 2 |
| 「我想死」 | **1** |

第一次 `--epochs 1 --lr 2e-5` 就是这么翻车的：42 题探针从 42/42 掉到 **5 题未命中**。
表面像学习率过高，**根因是这些行为在数据里几乎没有**。把身份 ×4、补丁 ×8 之后
定向数据占比 2.9% → 14.6%，才修回来。

**期望**：单轮 + 多轮 + 身份合计 **3.8 万条左右**；剔除项里**只应看到
「答案提到别家模型」和少量质检项**，不应出现大批量的"重复"或"长度越界"。
过采样会打印 `过采样 identity: 547 条 x4 -> 补 1641 条` 之类的行。

> **验证集是在过采样之前切的**——复制出来的条目落进 val 会让 val_loss 虚低。

**这一步自带二次质检**（身份泄漏、复读、截断、代码块、格式化列表），
生成时已经过一遍，这里是最后一道。注意 `--extra` 走的是**放宽版**：手写补丁允许
"中文提问 + 非中文回答"（翻译样本），教师数据不开这个口子。

---

## 2. 重建长文检索数据（**别省这一步**）

```powershell
python scripts\v3_build_retrieval.py --out data\retr_v4
```

**为什么必须**：v4.0 要在 4 万条**短对话**上做全参微调，而 v3.19 的长文取件码能力
来自 v3 时代的检索 SFT。不做这一步，长文召回大概率会退化——这是
`AGENTS.md` 里记过的教训（"长文预训练学不会检索，必须补合成检索 SFT"）。

**期望**：4 个窗口档（4096/8192/16384/32768）各出 `.npy`，合计约 12.5M token，
并且**每个档位都带负样本**（见下）。

> ⚠️ **负样本的"难度"必须和评估口径一致**（2026-10-17 血泪）。
> `eval_longctx_many.py` 的负样本会在正文里**撒 1~3 条别的类别的事实当诱饵**，
> 只缺被问的那一类；如果训练端造的是"纯填充文本、一条事实都没有"，
> 模型学的是"空文档里找不到"，考的是"四条信息里缺哪一条"——**根本不是同一个能力**。
> 实测：12.5% 无诱饵负样本 → 拒答只有 4/64；改成带诱饵、18~49% → **64/64**。
> 比例按长度递增（越长越容易编）。

**这一步的数据有问题的话，第 3 步 SFT 出来的模型会"永远硬抽"**——
问它文中没有的东西，它编一个给你。

---

## 3. PC 端 SFT（全参）

```powershell
python scripts\v4_sft.py --init v4/base --data data\v4_sft.jsonl `
    --val data\v4_val.jsonl --out v4\sft_pc2 `
    --epochs 3 --lr 1e-5 --batch 24 --accum 2 --max-len 1024 --save-every 150
# 实际产物：v4\sft_pc2\final（2,392 步 / 47.5 min / val_loss 3.9072 / 峰值 12.81 GiB）
```

**为什么用 `v4_sft.py` 而不是 `v3_6_sft_patch.py`**：后者每步独立随机抽样本
（`rng.sample`），4 万条跑一轮只看得到约 63%，其余永远轮不到。`v4_sft.py` 是
按 epoch 完整洗牌 + 长度分桶。

**超参依据（2026-10-17 用两次实测改的，别改回去）**：
- `--epochs 3`：原文档写 1，理由是"4 万条算完整 SFT 了，多轮会过拟合"。
  **实测反过来**——1 epoch 的监督量只有 **3.24M token**（v3 那轮是 22.65M，**只训了 1/7**），
  val_loss 3.96 且行为大面积丢失。3 epoch 后 val_loss 3.9072，留出第二套从 15/30
  回到 19/30（旧 strict-v6 口径；strict-v7 下是 15→16/30）。
- `--lr 1e-5`：原文档写 2e-5。全参 + 4 万条下 2e-5 会把 v3.19 磨出来的行为冲掉；
  1e-5 稳住。
- `--accum 2`：等效 batch 48；
- `--save-every 150`：中途存 checkpoint。**没有它就没法回答"从第几步开始坏的"**——
  第一次失败时想回溯都没得看。峰值显存 12.81 GiB（batch 24 / max-len 1024）。
- 不加 `--train-last`：**全参**。v3.21 的教训是"日聊/记忆增益只在全参强训练下出现"
  （代价是长文拒答会被伤，所以第 2 步的检索回放不能省）。

**冒烟先跑**：`--steps-cap 3` 确认能落盘再正式开。

**🔍 必看的数字**：
| 指标 | 该是多少 | 不对说明什么 |
|---|---|---|
| 开头几步的 loss | **3~6** | 若在 **9.7**（=ln16384）附近 → 标签错位了，见 `v4_sft.py` 的 `make_labels` 注释 |
| loss 趋势 | 稳步下降 | 不降 → LR 太小或数据有问题 |
| val_loss | 与 train loss 同量级、略高 | 远高于 train → 过拟合 |
| peak_gib | 远小于 16 | 接近 16 就要降 batch/max-len |

---

## 3b. 检索回放（**不能省**）

纯 SFT 会把 32k 长文打回 **7/32**（`v4/sft_pc2`），必须从 SFT 产物再做一遍检索回放：

```powershell
python scripts\v3_retrieval_sft.py --init v4\sft_pc2\final --data-root data\retr_v4 `
    --out v4\retr_sft4 --lr-scale 0.3 --epochs 2
# 产物：v4\retr_sft4\ctx32768\final（四个窗口档依次续训）
```

**为什么 `--lr-scale 0.3`**：脚本默认 5e-5，对刚做完全参 SFT 的模型太猛，会把对话行为冲掉。
**验收（2026-10-18 改为 token 精确口径，预留 32 个生成位）**：v4.0 最终权重
**103/128**（32k **20/32**）、拒答 **64/64**；v3.19 基线为 106/128（32k 24/32）。
**检索命中没有超过 v3.19**，换来的只有拒答（62→64/64）——发版时不要写反。
没到就回去查负样本分布，不要靠加 epoch 硬拉（见 CHANGELOG §4d）。

---

## 3c. 名字补丁（`我叫X，请记住`）

4.8 万轮训练数据里没有「请记住」这个指令句式，原始 4.0 会答非所问。补 160 条定向数据，
**末 2 层 / 5 epoch / lr 5e-6**：

```powershell
python scripts\v3_6_sft_patch.py --init v4\retr_sft4\ctx32768\final `
    --patch data\v4_name_patch.jsonl --out v4\patch_name_e5_lr5e-6 `
    --epochs 5 --lr 5e-6 --train-last 2
```

产物 `v4/patch_name_e5_lr5e-6/`（HF 直接放在阶段目录，**没有 `final` 子目录**），
即 v4.0 的**最终 PC 权重**。补丁强度不能再加：10 epoch / lr 1e-5 会把留出第一套打掉 2 题。

---

## 4. 板端权重（Q4 + q2 KV 双 QAT）

> ⚠️ **v4.0 的 QAT 没有过关卡 A，板端权重仍用 v3.19-embed。**
> 这一节保留完整流程供下一次尝试；实际发布**不要**把 `v4/qat_*` 的导出当板端权重。
> 三版结果：24/27、23/27、`qat_20261017D` 27/27（D 版未做板端 tool/记忆/实机验收）。
> 详见 `CHANGELOG.md` §4f。

板端必须重新做 QAT——**PC 权重直接导出给引擎会退化**（`AGENTS.md`：同套 32 题矩阵
22/27 vs 27/27）。

```powershell
python scripts\v3_7_kv_qat.py --init v4\patch_name_e5_lr5e-6 --data data\v4_sft.jsonl `
    --mt data\v4_sft.jsonl --mt-n 300 --identity-n 80 --out v4\qat_board `
    --epochs 3 --lr 6e-6 --batch 24 --max-len 1024 --wqat `
    --retr data\retr_v4 --retr-n "4096:200:2,8192:60:1" --retr-lr 1e-5
```

`--wqat` 是关键（对权重做 Q4 block64 的 STE），少了它量化误差会直接吃掉几个点。

---

## 5. 导出与验证

```powershell
# 导出 QAT 权重（v4.0 的几版都只用于验收，**没有发布**）
python board\tools\export_model.py --model v4\qat_20261017D --out board\model_export_v4_qat_rejected
# 发布用的板端权重（v3.19-embed）已经在这里，不要覆盖：
#   board\model_export_v3_19b6\
```

> `board/model_export_v4*` 是失败样本，已 gitignore；**不要**把它们打进 Release，
> 也不要用它们覆盖 `model_export_v3_19b6/`。

### 关卡 A：PC 一致性（先过这个再上板）

```powershell
cd board
.\pc\pc_kv_suite_q2b8.exe model_export_v4_qat_rejected ..\pc\prompt_long.txt 5200
```

**验收线（2026-10-17 重定）**：**≥25/27 且长文召回 4/4，且不得出现类别性退化**。
（这是"能否上板"的线；v4.0 三版分别为 24/27、23/27、27/27，前两版被这道线拦下，
D 版虽过线但没做关卡 C/D，所以最终仍未替换板端权重。）

> 原文档写的是「必须 27/27」。那 27/27 是 **v3.19 在定向数据上刷出来的**，其中
> 「推荐-书→小王子」「推荐-电影→千与千寻」本质是背下来的答案。拿它当新模型的硬门槛，
> 等于要求 4.0 复刻 3.19 的口癖。**v4.0 实测 25/27 + 4/4**，差的 2 题是：
> - `推荐-书` → "《小豆芽》或者《小刚开始》"（**编造书名**）
> - `推荐-电影` → "《龙猫》和《龙梦》"（龙猫真的，龙梦编造）
>
> 这 2 题**不要**靠加训练数据糊过去（那是拿测试集当训练集，见 `v3x-lesson.md`）。
> 要在数据侧过滤「编造具体专名」的样本，留给 4.1。
>
> 另有 2 题按同义词放宽了期望（`伤|难过`、`再见|回头见|拜拜`），理由写在
> `pc/pc_kv_suite.cpp` 的注释里。

**多轮（关卡 C）注意口径**：`pc_mt_suite` 的 `mem12` **不经过任何 tool**，测的是裸模型；
而固件 `main.c` 和 `pc_chat` 都会先过 `feng_memory`。所以：
- **板端产品路径**：`[tool] 你叫小雨。` 确定性作答，与模型无关（已实测 ✓）
- `pc_mt_suite mem12` 4/12 是**裸模型**的多轮记忆，v3.19 的 12/12 来自 v3.13 那轮专门的记忆训练

别把这两个数混为一谈。

### 关卡 B：留出题（**两套都要跑**）

```powershell
python scripts\chat_probe_heldout.py  --model v4\sft_pc\final --out eval/heldout_v4_pc.json
python scripts\chat_probe_heldout2.py --model v4\sft_pc\final --out eval/heldout2_v4_pc.json
python scripts\chat_probe.py          --model v4\sft_pc\final --out eval/chat_probe_v4_pc.json
python scripts\eval_planA_scope.py    v4\sft_pc\final eval\v4_scope.json
```

**对照基线（v3.19，strict-v7）**：留出第一套 17/30、第二套 15/30、探针 42/42、
身份 12/12、单类别 110。

**v4.0 实测（2026-10-17，最终权重 `v4/patch_name_e5_lr5e-6`，fp32）**：

| 指标 | v3.19 | 第一次（1ep/2e-5） | **v4.0（3ep/1e-5/过采样）** |
|---|---|---|---|
| 留出第一套（strict-v7） | 17/30 | 20/30（旧口径） | **18/30** |
| 留出第二套（strict-v7，v4 用于选型） | 15/30 | 15/30（旧口径） | **16/30** |
| 42 题探针 | 42/42 | 5 题未命中 | 40/42（1 条同义误判 + 1 条 2+3 算成 6，见 CHANGELOG §4c） |
| 范围 / 身份 | 10/10 | 9/10 | **10/10** |

探针剩下那 2 题**不算失败**：「我很伤心」答的是同义词（判定词太死）、
「2+3等于几」由 `feng_calc` 接管（模型答错用户也看不到）。

> **判定口径用 strict-v7**（`chat_probe_heldout.py --rescore`，重打分必须传对 case 表）。
> 旧表里的 22/30 + 19/30 是 strict-v6 的自动分；改成 strict-v7 后是 18/30 + 16/30，
> 基线 v3.19 也从 19/30 + 18/30 降到 17/30 + 15/30——**仍然领先基线各 1 题，但优势比旧口径小**。
> **第二套在 v4 期间被用于选型**，不再是独立测试集；发版文案要写明"两套都只能做版本对比"。
> 关键词判定仍可能漏假阳性——**头条数字必须人工过一遍 `rows[].a`**。

### 关卡 C：多轮

```powershell
cd board
.\pc\pc_mt_suite.exe model_export_v4_qat_rejected
```

**对照基线（v3.19-embed，套件保留跨轮 nhist 后的新口径）**：mt **8/10** +
mem12 **12/12** + seq **10/10**（`tools/check_all.py` 的门禁值）。
v4 的 QAT 候选在旧口径下是 mt 7/10、seq 6/10、mem12 4/12，且 mem12 是**裸模型**——
产品路径会先过 `feng_memory` tool，不受影响。
多轮"你我分辨"是**容量硬边界**（`AGENTS.md` 已验证 p3–p7 五个版本都只是重排失败点），
**不要因为这项没涨就重训**——那是白费。

### 关卡 D：板端实机

```powershell
.\board\flash.ps1 -Port COM20 -EspIdfPath <esp-idf> -ModelDir board\model_export_v4_qat_rejected
python scripts\esp32_tool_test.py --port COM20      # 期望 13/13
python scripts\esp32_memory.py --port COM20 --n 12  # 期望 12/12
python scripts\esp32_multi.py --port COM20
```

**速度对照**：v3.19-embed 是 2.33 tok/s（固件统计口径，短回复最佳值）。C23 引擎没动数值路径，
新权重同架构，**速度应该一样**；掉超过 5% 说明 QAT 把权重分布改坏了。

---

## 已知风险与对策

| 风险 | 症状 | 对策 |
|---|---|---|
| **长文能力退化** | 32 题矩阵的长文召回掉出 4/4 | 第 2 步的检索回放不能省；仍在掉就调 `--retr-lr` |
| **身份漂移** | 自称别的模型 | 数据侧的答案闸门已拦，若仍漂移查 `data/v4_identity.jsonl` 是否被污染 |
| **拒答能力被伤** | 该拒的不拒 | v3.21 记过"全参强训练首先伤长文拒答"；降 LR 或多放拒答样本 |
| **多轮没涨** | mt-suite 停在 7/10 | **预期之内**，容量硬边界，别重训 |
| **标签错位** | loss ≈ 9.7 | 见第 3 步的表 |

---

## 与 v3 的口径差异（写发版文案时注意）

- **PC 与板端两套权重**：PC 用 `v4/patch_name_e5_lr5e-6/`（HF 直接放在阶段目录）；
  板端发布权重仍是 `board/model_export_v3_19b6/`（v3.19-embed），4.0 的 QAT 全部未采用。
- **`final` 路径规则**：`v4_sft.py` 与 `v3_retrieval_sft.py` 的 HF 权重在 `<阶段>/final` 目录里；
  `v3_6_sft_patch.py`、`v3_7_kv_qat.py` 的产物直接写在 `--out`（没有 `final` 子目录）。
  训练脚本的 `--out` 是阶段目录，但所有**输入**（评测、下一阶段的 `--init`）必须指向真正含
  `model.safetensors` 的那一层。2026-10-17 就是照旧文档抄了 `--model v4\sft_pc`，
  四个评测全部在加载 tokenizer 时挂掉、GPU 一步没跑。
- **算式的验收口径是 tool 回答**，不是模型算——`pc_kv_suite` 会把算式路由到 `feng_calc`。
- **英文标准基准只剩 SciQ 与 PIQA**（其余已删除，见 `CHANGELOG.md` v3.20 评测节）。
