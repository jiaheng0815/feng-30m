"""把 B 权重的 embedding/lm_head 换成 v3.19 的（板端候选 H）。

背景：4.0-20261018B 的 PPL/长文都优于 v3.19，但它的 tied embedding/lm_head
在 Q4 block64 下不够稳：Q4 导出后「再见」翻译成 "Good morning"、25% 深度检索
返回 4820（B fp32 都正确）。把 embedding/lm_head 换成 Q4 更稳的 v3.19 版本，
其余 10 层 attention/FFN 保留 B 的权重，得到同尺寸候选：

    hybrid = B（除 embedding/lm_head）+ v3.19（embedding/lm_head）

实测 Q4 32 题 27/27 + 4/4（B 是 26/27 + 3/4），PPL 103.0（v3.19 107.5），
留出 200 题 85（v3.19 83），长文 108/128（v3.19 106）。

用法：
    python scripts\\make_hybrid_embedding.py `
        --base v4/4.0-20261018B/ctx32768/final `
        --embed-source v4/base_v319b6_deq `
        --out v4/hybrid_B_emb319
"""
import argparse
import sys
from pathlib import Path

import torch

sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="主体权重（attention/FFN 来自这里）")
    ap.add_argument("--embed-source", required=True, help="embedding/lm_head 来源")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    base = AutoModelForCausalLM.from_pretrained(a.base, dtype=torch.float32)
    src = AutoModelForCausalLM.from_pretrained(a.embed_source, dtype=torch.float32)
    if base.model.embed_tokens.weight.shape != src.model.embed_tokens.weight.shape:
        raise SystemExit("embedding 形状不一致")
    with torch.no_grad():
        base.model.embed_tokens.weight.copy_(src.model.embed_tokens.weight)
        if not base.config.tie_word_embeddings:
            base.lm_head.weight.copy_(src.lm_head.weight)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    base.save_pretrained(out, safe_serialization=True)
    AutoTokenizer.from_pretrained(a.base).save_pretrained(out)
    print(f"已写出 {out}（tied={base.config.tie_word_embeddings}）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
