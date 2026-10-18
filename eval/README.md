# eval/ —— 评测结果

每次出模型的评测数字都落在这里，`docs/` 里引用的分数应该能对上这些 JSON。

## 命名约定

`<评测名>_<模型标签>.json`

| 前缀 | 评的是什么 | 产出脚本 |
|---|---|---|
| `v3_*` | 各版本的探针 / 身份 / 范围总分 | `eval_planA_scope.py`、`chat_probe.py` |
| `chat_probe_*` | 42 题聊天探针（关键词判定） | `chat_probe.py` |
| `heldout*` / `h1_*` | 30 题留出**第一套**（**strict-v7 判定**） | `chat_probe_heldout.py` |
| `heldout2_*` / `h2_*` | 30 题留出**第二套**（同 strict-v7） | `chat_probe_heldout2.py` |
| `heldout200_*` | **冻结留出 200 题**（2026-10-18；只在发版/大改时跑，不用于调参） | `chat_probe_heldout200.py` |
| `ppl_*` | 留出集困惑度（不依赖关键词判分器的稳定指标） | `eval_ppl.py` |
| `longctx32*` | 长文针检索（4k/8k/16k/32k；**2026-10-18 起 token 精确、预留 32 生成位**） | `eval_longctx_many.py` |
| `identity_*` | 身份题专项 | `eval_identity.py` |
| `arith_*` | 算术子集（tool 口径） | `pc_kv_suite`（`FENG_SUITE=arith`） |
| `lm_eval_*` | 英文标准基准（lm-eval，**现只跑 SciQ 与 PIQA**） | `scripts/bench_standard.py` |

## 读这些数字时注意

- **两套留出题**：第一套（`chat_probe_heldout.py` 的 30 题）已被多轮迭代当过开发集，
  只作**版本对比**用；第二套（`chat_probe_heldout2.py`）在 v4.0 期间**也被用于选型**
  （根据它的分数挑 SFT / 回放 / 补丁版本），所以两套现在都只能当版本对比，
  **不再是"完全没碰过"的无偏泛化分数**——对外文案必须写明这一点。
- **判定版本必须写进 JSON**：strict-v7 在关键词之外加逐题规则（回声/否定/数量/明显错答）。
  老结果用 `python scripts\chat_probe_heldout.py --rescore eval\xxx.json` 重打分，
  别把 strict-v4/v6 的旧分数和新分数混在一张表里。
- **关键词判定仍会漏假阳性**：strict-v7 只覆盖已知的 30 题规则，换一套题仍可能把
  "方向答反但含关键词"判成 OK。头条数字必须人工过一遍 `rows[].a`，
  推荐解码对照存 `*_rec.json`。
- **算术题的验收口径是 tool 回答**，不是模型算——`pc_kv_suite` 会把算式路由到
  `feng_calc`，别把它当成模型的算术能力。
- **长文评测新旧文件不能混比**：2026-10-18 之前的 `longctx32_*` 把 ctx 当字符数
  （标称 32k 实际约 26k token），且没有给生成预留位置。现行口径见
  `longctx32_v319.json` / `longctx32_v4final.json`（token 精确，输入 = ctx − 32）。
- **冻结留出 200 题**：`scripts/chat_probe_heldout200.py` 的 200 条已与
  `data/v4_sft.jsonl` 做**精确 prompt 去重（0 重合）**，只用于发版对比。
  **禁止把它加入训练/补丁/提示词池**；一旦用于调参，这组分数就作废。
  2026-10-18 基线：v3.19 **83/200**、v4 final **92/200**（McNemar p=0.16，
  配对差 95% CI −1~+10 个百分点——方向有利，但 200 题还不足以在 95% 水平显著）。
- **PPL 是稳定指标**：`eval_ppl.py` 在 `data/v4_val.jsonl`（v4 SFT 的验证集）上
  对 assistant token 算困惑度。2026-10-18：v3.19 **107.49**、v4 final **51.20**、
  B **97.45**、D **161.29**。PPL 与关键词分数分开看，能发现"分数没掉但语言能力退化"。
