# 三值量化实验（分支 `ternary`，不发布）

> 目标：参考 **Ternary Bonsai 2** 的三值配方，把 feng-30m 的 Linear 权重压到
> {-d, 0, +d}，看能不能在 30M 规模上保住质量，再决定是否做板端 2-bit 内核。
> **目前结论：PTQ 直接崩；SFT-only QAT 也远未恢复到 dense 水平。**
> 不要把这条分支的权重当成发布候选。

## 1. Bonsai 2 的配方（公开资料）

PrismML 的 Ternary Bonsai 2 27B（2026-09-17）：

- 权重 **{-1, 0, +1}**，每 **128** 个权重一个 **FP16 group scale**；
  `w = s_g · t`，`t ∈ {-1,0,+1}`；PTQ1_0 / PQ2_0 约 **1.75 / 2.13 bpw**；
- 量化前做 **blockwise Walsh–Hadamard 旋转**（块 1024，配一条固定 ±1 sign 向量），
  只作用于输入维 K；权重离线折叠成 `W' = W R^T`，运行时对激活做
  `x' = FWHT(x ⊙ s)/√B`，再走三值矩阵乘；
- **selective precision**：少量参数（线性注意力 recurrent state + norm）保持高精度；
- 27B 模型从 ~54GB 压到 **5.9GB**，综合基准保留 **98.2%**。

## 2. 我们怎么搬到 30M

我们的 hidden=448、FFN=896，不能整除 1024/128，所以：

| 项目 | Bonsai 2 | 本实验 |
|---|---|---|
| 三值码 | {-1,0,+1} | 同 |
| group scale | 128 | **64**（448/896 都能整除） |
| Hadamard 块 | 1024 | **64**（448 内最大 2 的幂） |
| sign 向量 | 每个 K 宽度一条 | 同（448 / 896 各一条） |
| selective precision | recurrent + norm | **embedding + lm_head + norm 保持 fp16/Q4** |
| 训练 | 端到端低比特 | **STE QAT**（`ternary_qat.py`） |

实现：

- `scripts/ternary_ptq.py`：不训练，直接把 Linear 权重旋转 + 三值化，量 PPL；
- `scripts/ternary_qat.py`：把 `nn.Linear` 换成 `TernaryLinear`，forward 里做
  `x' = FWHT(x ⊙ s)/√B`、`W' = ternarize(FWHT(W ⊙ s)/√B)`，反传走 STE；
  `lm_head/embedding/norm` 不动；产物仍是普通 Qwen3 state_dict（float），
  导出板端时再按 block/group 重新量化。

## 3. 结果（`data/v4_val.jsonl`，assistant PPL/CE）

先给同口径基线：dense `v4/base` 在 max-len 256 下 **PPL 114.31**（CE 4.739）。

| 实验 | 设置 | 结果 |
|---|---|---:|
| PTQ，全 Linear 三值 | block64/group64，50 条 | **PPL 6222.7（59.4×）** |
| PTQ，保留 o_proj/down_proj | 50 条 | 2978.5（33.3×） |
| PTQ，只三值化 FFN | 50 条 | 2108.8（23.6×） |
| PTQ，group32/block32 | 50 条 | 2560.8（28.6×） |
| QAT，全 Linear 三值 | 1000 步 / batch2×accum2 / max-len256 | **val CE 5.8191 / PPL 337** |
| QAT，只三值化 FFN（保留 attention） | 500 步 | val CE 5.4266 / PPL 228 |
| QAT，group32/block32 | 500 步 | val CE 5.7156 / PPL 303 |
| QAT，全 Linear，lr 5e-5 | 500 步 | val CE 5.4750 / PPL 239 |
| **冻结留出 200（前 50 题）** | 全 Linear QAT 1000 步 | **4/50 OK**（dense v4/base 17/50） |

估算体积（三值 2.25 bpw + 保留参数 Q4 4.25 bpw）：

- 全 Linear 三值（attention+FFN，22.08M）+ embed/lm_head Q4：**~9.3 MiB**；
- 只三值化 FFN（13.25M）+ 其余 Q4：**~11.6 MiB**；
- 当前 Q4 板端 `model.bin`：**14.93 MiB**。

## 4. 板端为什么还没接

即使质量过关，板端也不是 Bonsai 那种“三值 = 5× 吞吐”：

- S3 没有三值/整数矩阵指令；2-bit 码仍要靠 LUT + 标量 madd，**每权重计算量不降**，
  只省 flash 读字节；
- 需要新的 `model.bin` dtype（2-bit 码 + group scale + sign 向量）、导出脚本、
  FWHT 激活变换和三端 GEMV 调度；
- 短上下文是权重流式读 + 标量计算混合，粗估只有 **1.1–1.2×**；
  长上下文瓶颈在 attention（已做双核，1.58s @2048），三值化不改变它。

## 5. 目前的判断

- **PTQ 不可行**：30M 没有 27B 的冗余，Hadamard + g64 也救不回来（23–59× PPL）。
- **SFT-only QAT 不够**：1000 步把 val CE 从 ~8.7 拉到 5.82，仍比 dense 的
  4.74（同 max-len 256 口径）高 1.08 CE；PPL 337 vs 114。冻结留出前 50 题只有
  **4/50**（dense 17/50）。换 group32、提高 lr、只三值化 FFN（保留 attention）
  都只能把 PPL 拉到 228–303，仍差 2–2.6×。
- 三值化的损失不是“多跑几步 SFT”能补的：它需要模型在预训练阶段就适应
  {-d,0,+d} 的权重约束。Bonsai 2 能在 27B 上 PTQ 保留 98.2%，靠的是大模型冗余；
  30M 上照搬配方不成立。
- 真要三值化 30M，正确做法是 **从零做三值预训练**（BitNet 路线）或在
  1.5B token 级语料上做长时间 QAT/蒸馏，而不是在 Q4 权重上补 SFT。
  这需要几小时到几天的 GPU 时间和完整语料重建，远超本轮。
- **板端 2-bit 内核暂不实现**：没有质量过关的权重，先做内核没有意义；
  等三值预训练有结果，再按第 4 节做打包/FWHT/GEMV。

## 6. 参考

- PrismML, *Introducing Bonsai 2 27B*：https://prismml.com/news/bonsai-2-27b
- PrismML, *Ternary Bonsai 2 2B/4B/9B*：https://prismml.com/news/bonsai-2-family
- 打包/旋转细节（社区整理）：https://john-rocky.github.io/coreai-model-zoo/knowledge/bonsai-ternary-hadamard.html
- 量化实践分析：https://derob98.github.io/ailmanac/docs/models/ternary-quantization-bonsai-2-27b/
