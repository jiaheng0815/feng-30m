# eval/ —— 评测结果

每次出模型的评测数字都落在这里，`docs/` 里引用的分数应该能对上这些 JSON。

## 命名约定

`<评测名>_<模型标签>.json`

| 前缀 | 评的是什么 | 产出脚本 |
|---|---|---|
| `v3_*` | 各版本的探针 / 身份 / 范围总分 | `eval_planA_scope.py`、`chat_probe.py` |
| `chat_probe_*` | 42 题聊天探针（关键词判定） | `chat_probe.py` |
| `heldout*` | 30 题留出集（**strict-v6 判定**） | `chat_probe_heldout.py` / `chat_probe_heldout2.py` |
| `longctx32*` | 32k 长文针检索（单轮 / 多轮） | `eval_longctx_many.py` |
| `identity_*` | 身份题专项 | `eval_identity.py` |
| `arith_*` | 算术子集（tool 口径） | `pc_kv_suite`（`FENG_SUITE=arith`） |
| `lm_eval_*` | 英文标准基准（lm-eval，**现只跑 SciQ 与 PIQA**） | `scripts/bench_standard.py` |

## 读这些数字时注意

- **两套留出题**：第一套（`chat_probe_heldout.py` 的 30 题）已被多轮迭代当过开发集，
  只作**版本对比**用，不当无偏泛化分数；发版文案要以**第二套**（`heldout2_*`，未参与
  训练/调参）为准。
- **关键词判定会漏假阳性**：答案含"南极/米"等词即判 OK，即使方向答反或退化成词。
  头条数字必须人工过一遍 `rows[].a`，推荐解码对照存 `*_rec.json`。
- **算术题的验收口径是 tool 回答**，不是模型算——`pc_kv_suite` 会把算式路由到
  `feng_calc`，别把它当成模型的算术能力。
