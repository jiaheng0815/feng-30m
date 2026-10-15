# feng-30m 4.0 训练链路

数据（`scripts/v4_teacher_gen.py` 的产出）跑完之后，按本文从头走到尾。
起点是 **`v4/base/`**（v3.19 pc4 的 HF 权重，含自带 tokenizer）。

> ⚠️ **每一步都要看数，不要盲跑。** 下面标了每步该出现什么数字；
> 对不上就停下来查，别往下走——4.0 的全部价值就在这份新数据上，
> 训练配置错了会把它浪费掉。

---

## 0. 前置检查

```powershell
# 生成是否跑完（合格条数应接近提示词池规模）
(Get-Content data\v4_teacher.jsonl | Measure-Object -Line).Lines

# 教师服务可以停了——训练要独占显卡（AGENTS.md 硬约束）
taskkill /IM ninfer-serve.exe /F
nvidia-smi --query-gpu=memory.used --format=csv,noheader    # 应回落到 ~1 GB
```

---

## 1. 合并 SFT 数据

```powershell
python scripts\v4_build_sft.py --out data\v4_sft.jsonl `
    --val-out data\v4_val.jsonl --val-ratio 0.02
```

**期望**：单轮 + 多轮 + 身份（~562 条）合计几万条；剔除项里**只应看到
「答案提到别家模型」和少量质检项**，不应出现大批量的"重复"或"长度越界"。

**这一步自带二次质检**（身份泄漏、复读、截断、代码块、格式化列表），
生成时已经过一遍，这里是最后一道。

---

## 2. 重建长文检索数据（**别省这一步**）

```powershell
python scripts\v3_build_retrieval.py --out data\retr_v4
```

**为什么必须**：v4.0 要在 4 万条**短对话**上做全参微调，而 v3.19 的长文取件码能力
来自 v3 时代的检索 SFT。不做这一步，长文召回大概率会退化——这是
`AGENTS.md` 里记过的教训（"长文预训练学不会检索，必须补合成检索 SFT"）。

**期望**：4 个窗口档（4096/8192/16384/32768）各出 `.npy`，合计约 12.5M token。

---

## 3. PC 端 SFT（全参）

```powershell
python scripts\v4_sft.py --init v4/base --data data\v4_sft.jsonl `
    --val data\v4_val.jsonl --out v4\sft_pc `
    --epochs 1 --lr 2e-5 --batch 24 --accum 2 --max-len 1024
```

**为什么用 `v4_sft.py` 而不是 `v3_6_sft_patch.py`**：后者每步独立随机抽样本
（`rng.sample`），4 万条跑一轮只看得到约 63%，其余永远轮不到。`v4_sft.py` 是
按 epoch 完整洗牌 + 长度分桶。

**超参依据**：
- `--epochs 1`：4 万条已经不是"补丁"而是完整 SFT，多轮会过拟合；
- `--lr 2e-5`：v3.6/v3.7 那条链用的是 1e-4~3e-5（补丁量级），样本量上去后要降；
- `--accum 2`：等效 batch 48；
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

## 4. 板端权重（Q4 + q2 KV 双 QAT）

板端必须重新做 QAT——**PC 权重直接导出给引擎会退化**（`AGENTS.md`：同套 32 题矩阵
22/27 vs 27/27）。

```powershell
python scripts\v3_7_kv_qat.py --init v4\sft_pc --data data\v4_sft.jsonl `
    --mt data\v4_sft.jsonl --mt-n 300 --identity-n 80 --out v4\qat_board `
    --epochs 3 --lr 6e-6 --batch 24 --max-len 1024 --wqat `
    --retr data\retr_v4 --retr-n "4096:200:2,8192:60:1" --retr-lr 1e-5
```

`--wqat` 是关键（对权重做 Q4 block64 的 STE），少了它量化误差会直接吃掉几个点。

---

## 5. 导出与验证

```powershell
# 导出板端模型
python esp32s3-feng-llm\tools\export_model.py --model v4\qat_board --out esp32s3-feng-llm\model_export_v4
```

### 关卡 A：PC 一致性（先过这个再上板）

```powershell
cd esp32s3-feng-llm
.\pc\pc_kv_suite_q2b8.exe model_export_v4 ..\pc\prompt_long.txt 5200
```

**必须 27/27 + 4/4**。掉分说明 QAT 没做好或导出有问题。

### 关卡 B：留出题（**两套都要跑**）

```powershell
python scripts\chat_probe_heldout.py  --model v4\sft_pc --out eval/heldout_v4_pc.json
python scripts\chat_probe_heldout2.py --model v4\sft_pc --out eval/heldout2_v4_pc.json
python scripts\chat_probe.py          --model v4\sft_pc --out eval/chat_probe_v4_pc.json
python scripts\eval_planA_scope.py    v4\sft_pc eval\v4_scope.json
```

**对照基线（v3.19）**：留出第一套 19/30、第二套 18/30、探针 42/42、身份 12/12、单类别 110。

> **判定口径用 strict-v6**（`chat_probe_heldout.py --rescore`，重打分必须传对 case 表）。
> 关键词判定会漏假阳性——**头条数字必须人工过一遍 `rows[].a`**。
> 第二套留出题未参与训练/调参，**发版文案以它为准**。

### 关卡 C：多轮

```powershell
cd esp32s3-feng-llm
.\pc\pc_mt_suite.exe model_export_v4
```

**对照基线**：mt 7/10 + mem12 12/12 + seq 8/10。
多轮"你我分辨"是**容量硬边界**（`AGENTS.md` 已验证 p3–p7 五个版本都只是重排失败点），
**不要因为这项没涨就重训**——那是白费。

### 关卡 D：板端实机

```powershell
python esp32s3-feng-llm\flash.ps1 -Port COM20 -EspIdfPath <esp-idf> -ModelDir esp32s3-feng-llm\model_export_v4
python scripts\esp32_tool_test.py --port COM20      # 期望 13/13
python scripts\esp32_memory.py --port COM20 --n 12  # 期望 12/12
python scripts\esp32_multi.py --port COM20
```

**速度对照**：v3.19-embed 是 2.33 tok/s（507 ms/token decode）。C23 引擎没动数值路径，
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

- **PC 与板端两套权重**：PC 用第 3 步的 `v4/sft_pc`，板端用第 4 步的 `v4/qat_board`。
- **算式的验收口径是 tool 回答**，不是模型算——`pc_kv_suite` 会把算式路由到 `feng_calc`。
- **英文标准基准只剩 SciQ 与 PIQA**（其余已删除，见 `CHANGELOG.md` v3.20 评测节）。
