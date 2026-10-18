# 板端权重升级实验（分支 `board-qat`，候选）

目标：**硬件不变、模型大小不变**，把板端权重从 `v3.19-embed` 换成 4.0 的
高质量权重，靠 QAT 过关卡 A，而不是靠放大模型。

## 为什么选 4.0-20261018B

| 权重 | PC PPL（v4_val） | 留出第一套 | 长文 128 |
|---|---:|---:|---:|
| v3.19-embed（当前板端） | 107.49 | 17/30 | 106/128 |
| **4.0-20261018B** | **97.45** | 19/30 | **108/128** |
| 4.0 final | 51.20 | 18/30 | 103/128 |

B 是 PC 长文最好的版本；它的 Q4 导出只差 **26/27 + 3/4**：

- 翻译-再见：Q4 答 "Good morning."（期望 "Goodbye."）
- 召回 25%：Q4 答 4820（期望 483920）

所以这一轮不是重训模型，而是：

1. 给 QAT 数据加 30 条常见中英翻译（含"再见"及同义说法，避免只修一个句子）；
2. 生成 **2048 上下文**的检索回放（板端 MAX_CTX 就是 2048，不用 4096/8192）；
3. 从 B 做 Q4 权重 + q2 KV 双 QAT，`--train-last 2`、1 epoch、lr 4e-6；
4. 重新导出 Q4，过 32 题矩阵 + 长文召回，再上板跑 tool/记忆/多轮。

## 命令

```powershell
# 2048 检索回放
python scripts\v3_build_retrieval.py --out data\retr_v4_2048 --specs 2048:800 --negative-fracs 0.20

# data\v4_sft_qat2.jsonl = data\v4_sft_qat.jsonl + 30 条翻译补丁
python scripts\v3_7_kv_qat.py --init v4\4.0-20261018B\ctx32768\final `
  --data data\v4_sft_qat2.jsonl --mt data\v4_daily.jsonl --mt-n 300 --identity-n 150 `
  --out v4\qat_20261018E --epochs 1 --lr 4e-6 --batch 8 --max-len 1024 --wqat `
  --train-last 2 --retr data\retr_v4_2048 --retr-n "2048:800:2" --retr-lr 2e-6

# 导出到新目录（不覆盖现有模型）
python board\tools\export_model.py --model v4\qat_20261018E\final --out board\model_export_4.0B_qatE
.\pc\pc_kv_suite_q2b8.exe board\model_export_4.0B_qatE pc\prompt_long.txt 5200
```

## 过关标准

- Q4 32 题矩阵 **27/27 + 4/4**；
- PC PPL 不高于 v3.19 的 107.49，留出第一套不低于 17/30；
- 长文 128 不低于 B 的 108/128（板端只用 2048，但 PC 长文不能塌）；
- 板端 tool 13/13、记忆 24/24、多轮 10/10；
- 只有全部满足才替换板端权重；否则保留 v3.19-embed。

> 结果与结论在跑完后补充到本文件；模型目录只新增、不覆盖。

## 结果（2026-10-18）

QAT 路线没有直接成功：

| 实验 | 设置 | Q4 网关 | PPL |
|---|---|---|---:|
| E | B + 30 条翻译 + 2048 检索，末 2 层，1000 步 | 26/27 + 3/4（失败项不变） | 150.0 |
| F | 翻译×100 + 3000 条通用，末 4 层，3 epoch | 26/27 + 3/4（失败项不变） | — |
| G | 同上但全参 QAT | **27/27 + 3/4**（翻译修好，25% 检索仍错） | **209.3** |

结论：QAT 能修翻译，但要么修不动（末层），要么把通用 PPL 打坏（全参）。
继续加 QAT 步数是在过拟合这 30 条补丁，不是提升模型。

权重 soup（B 与 v3.19 线性插值）能过网关，但质量倒退：

| soup | Q4 网关 | PPL | 留出 200 |
|---|---|---|---:|
| α=0.25 | 27/27 + 4/4 | 124.5 | 81 |
| α=0.50 | 27/27 + 4/4 | 133.3 | — |
| α=0.75 | 27/27 + 4/4 | 135.7 | — |

### 最终候选：hybrid embedding（H）

只把 B 的 **embedding/lm_head** 换成 v3.19（其余 attention/FFN 保留 B）：

```
hybrid = B（除 tied embedding/lm_head）+ v3.19-embed（embedding/lm_head）
```

| 指标 | v3.19-embed（当前） | B | **hybrid H** |
|---|---:|---:|---:|
| PPL（v4_val） | 107.49 | 97.45 | **102.99** |
| 留出 200 题 | 83 | 84 | **85** |
| 留出第一套 / 第二套 | 17 / 15 | 19 / 15 | **19 / 15** |
| 长文 128 | 106 | 108 | **108**（32k 24/32） |
| Q4 32 题 + 召回 | 27/27 + 4/4 | 26/27 + 3/4 | **27/27 + 4/4** |
| 模型大小 | 14.93 MB | 14.93 MB | **14.93 MB** |

板端实测：tool **13/13**、记忆 **24/24**、多轮 **10/10**、身份句正确。
这是“同硬件、同大小”下目前唯一全面不差于 v3.19 的候选。

复现：

```powershell
python scripts\make_hybrid_embedding.py `
  --base v4\4.0-20261018B\ctx32768\final `
  --embed-source v4\base_v319b6_deq `
  --out v4\hybrid_B_emb319
python board\tools\export_model.py --model v4\hybrid_B_emb319 --out board\model_export_hybrid_emb319
```

> 候选 H 还没有替换发布权重；release 口径仍是板端 v3.19-embed。
> 若确认发布，需要把 H 重命名成正式版本并同步 README/AGENTS/CHANGELOG/DELIVERY。
