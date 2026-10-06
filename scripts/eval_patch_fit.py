"""检查补丁数据的"训练集自拟合"：给定补丁 jsonl，逐条生成并和目标比对。

用法：python scripts/eval_patch_fit.py --model archive/v3_6b\\final\\ctx8192\\final \
        --data archive/v3_6a\\daily_patch.jsonl [--limit 120]
输出命中率（归一化后前缀/完全匹配）并逐条打印未命中样例。
"""
import argparse
import json
import random
import re
import sys
from pathlib import Path

import torch

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).resolve().parent))


def norm(s: str) -> str:
    return re.sub(r"[\s，。！？、,.!?;；“”\"'（）()：:%\-]", "", s).lower()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--limit", type=int, default=120)
    ap.add_argument("--max-new", type=int, default=64)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    rows = []
    with open(args.data, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    rng = random.Random(args.seed)
    rng.shuffle(rows)
    rows = rows[: args.limit]

    from transformers import AutoTokenizer, Qwen3ForCausalLM
    tok = AutoTokenizer.from_pretrained(args.model)
    model = Qwen3ForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16).to("cuda").eval()

    exact = prefix = 0
    misses = []
    for rec in rows:
        msgs = rec["messages"]
        q = msgs[0]["content"]
        target = msgs[1]["content"]
        text = f"<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n"
        ids = tok(text, add_special_tokens=False)["input_ids"]
        inp = torch.tensor([ids], dtype=torch.long, device="cuda")
        with torch.no_grad():
            out = model.generate(input_ids=inp, max_new_tokens=args.max_new, do_sample=False,
                                 pad_token_id=3, eos_token_id=0)
        r = tok.decode(out[0][inp.shape[1]:].tolist()).split("<|im_end|>")[0].strip()
        nr, nt = norm(r), norm(target)
        if nr == nt:
            exact += 1
        elif nt and nt[:12] == nr[:12]:
            prefix += 1
        else:
            misses.append((q, target, r))

    n = len(rows)
    print(f"=== 补丁自拟合 {args.model}（{n} 条）===")
    print(f"完全匹配 {exact}/{n} = {exact / n:.0%} | 前缀匹配 {prefix}/{n} | "
          f"合计 {(exact + prefix) / n:.0%}")
    print("\n未命中样例（最多 15 条）：")
    for q, t, r in misses[:15]:
        print(f"  Q: {q}\n    期望: {t}\n    实际: {r[:80]}")


if __name__ == "__main__":
    main()
