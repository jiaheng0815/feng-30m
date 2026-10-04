"""v3.9：生成"评测同款填充文本"的 token 流，给 v3_1_build_retrieval.py --stream 用。

动机：单类别/多类别评测的填充是固定的两句话循环（eval_longctx_many.FILLER），而 v3.4 起
训练检索数据改用真实语料填充——存在填充分布偏移。本脚本把评测同款 filler 重复成 token 流，
让 16k 专项训练看到与评测一致的填充分布。

用法：python scripts/v3_9_build_filler_stream.py --out v3_9/filler_eval.npy --tokens 4000000
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import ROOT  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "v3_9" / "filler_eval.npy"))
    ap.add_argument("--tokens", type=int, default=4_000_000)
    ap.add_argument("--tokenizer", default=str(ROOT / "v2" / "tokenizer" / "tokenizer.json"))
    args = ap.parse_args()

    sys.path.insert(0, str(ROOT / "scripts"))
    try:
        from eval_longctx_many import FILLER as EVAL_FILLER
    except Exception as exc:                       # 兜底：与 eval 保持一致的手抄文本
        print(f"[warn] 无法导入 eval_longctx_many（{exc}），使用内置文本")
        EVAL_FILLER = ("在遥远的山谷里，风穿过松林。村里的老人说，时间像河水一样一去不回。"
                       "孩子们在田野上奔跑，数着天上的云。")

    from tokenizers import Tokenizer
    tk = Tokenizer.from_file(args.tokenizer)
    unit = tk.encode(EVAL_FILLER).ids
    repeat = args.tokens // len(unit) + 1
    ids = (unit * repeat)[: args.tokens]
    arr = np.asarray(ids, dtype=np.uint16)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, arr)
    meta = {"filler": EVAL_FILLER, "unit_tokens": len(unit), "tokens": int(arr.size),
            "out": str(out)}
    (out.with_suffix(".json")).write_text(json.dumps(meta, ensure_ascii=False, indent=2),
                                          encoding="utf-8")
    print(f"filler 流：{arr.size/1e6:.1f}M tokens（单元 {len(unit)} tokens）-> {out}")


if __name__ == "__main__":
    main()
