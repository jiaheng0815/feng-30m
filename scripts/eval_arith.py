"""算术能力网格评测：加 / 减 / 乘，贪心解码，按最后出现的整数判分。

覆盖 v3.6 drill 没训到的边界：结果为零的减法（a-a）、结果为负的减法（a<b）。

用法：
    python scripts/eval_arith.py --model v3_10\\qat_pol3 [--out eval/arith.json]
"""
import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

import torch

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import ROOT  # noqa: E402


def build_cases():
    cases = []
    for a in range(0, 10):
        for b in range(0, 10):
            cases.append(("加", f"{a}+{b}等于几？", a + b))
    for a in range(0, 10):
        for b in range(0, 10):
            kind = "减(结果>0)" if a > b else ("减(结果=0)" if a == b else "减(结果<0)")
            cases.append((kind, f"{a}-{b}等于几？", a - b))
    for a in range(1, 10):
        for b in range(1, 10):
            cases.append(("乘", f"{a}乘{b}等于几？", a * b))
    return cases


def last_int(text: str):
    """只认第一句里「等于 X」的 X：模型答对后可能继续续写下一轮，不能被带偏。"""
    first = text.strip().replace("负", "-").split("\n")[0]
    m = re.search(r"等于\s*(-?\d+)", first.replace(" ", ""))
    if m:
        return int(m.group(1))
    nums = re.findall(r"-?\d+", first.replace(" ", ""))
    return int(nums[-1]) if nums else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", default="")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--max-new", type=int, default=24)
    args = ap.parse_args()

    from transformers import AutoTokenizer, Qwen3ForCausalLM

    tok = AutoTokenizer.from_pretrained(args.model, padding_side="left")
    if tok.pad_token_id is None:
        tok.pad_token_id = 3
    model = Qwen3ForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, attn_implementation="sdpa").to("cuda").eval()

    cases = build_cases()
    rows = []
    for i in range(0, len(cases), args.batch):
        chunk = cases[i:i + args.batch]
        prompts = [f"<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n" for _, q, _ in chunk]
        enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=args.max_new, do_sample=False,
                                 pad_token_id=tok.pad_token_id, eos_token_id=0)
        for j, (kind, q, expect) in enumerate(chunk):
            reply = tok.decode(out[j][enc["input_ids"].shape[1]:].tolist(),
                               skip_special_tokens=True).split("<|im_end|>")[0].strip()
            got = last_int(reply)
            rows.append({"kind": kind, "q": q, "expect": expect, "got": got,
                         "ok": got == expect, "reply": reply[:80]})

    per = Counter()
    tot = Counter()
    for r in rows:
        tot[r["kind"]] += 1
        per[r["kind"]] += int(r["ok"])
    print(f"=== 算术网格：{args.model}（{len(rows)} 题，贪心）===")
    for kind in ("加", "减(结果>0)", "减(结果=0)", "减(结果<0)", "乘"):
        if tot[kind]:
            print(f"  {kind:12s} {per[kind]}/{tot[kind]}")
    miss = [r for r in rows if not r["ok"]]
    for r in miss[:30]:
        print(f"  MISS {r['q']} -> {r['reply'][:50]!r}（期望 {r['expect']}）")
    print(f"  合计 {sum(per.values())}/{len(rows)}")
    if args.out:
        p = Path(args.out)
        if not p.is_absolute():
            p = ROOT / p
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({"model": args.model, "rows": rows}, ensure_ascii=False, indent=2),
                     encoding="utf-8")
        print(f"  -> {p}")


if __name__ == "__main__":
    main()
