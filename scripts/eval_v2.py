"""Quality scorecard for the v2 30M student (identity / chat / knowledge / 32k retrieval)."""
import argparse
import json
import random
import re
import sys
from pathlib import Path

import numpy as np
import torch

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

CASES = [
    ("identity", "你是谁？", [r"feng"], None),
    ("identity", "你叫什么名字？", [r"feng"], None),
    ("identity", "谁微调了你？", [r"jiaheng"], None),
    ("identity", "你和 Qwen 是什么关系？", [r"qwen|通义"], None),
    ("chat", "你好", None, None),
    ("chat", "在吗？", None, None),
    ("chat", "你能做什么？", None, None),
    ("knowledge", "中国的首都是哪里？", [r"北京"], None),
    ("knowledge", "水的化学式是什么？", [r"H2O|h2o|H₂O"], None),
    ("math", "3.14 乘以 2 等于多少？", [r"6\.28"], None),
    ("math", "一加一等于几？", [r"2|二"], None),
    ("translate", "把“今天天气很好”翻译成英文。", [r"today|weather"], None),
    ("writing", "用一句话介绍你自己。", [r"feng"], None),
    ("instruct", "给我讲一个关于猫的短故事。", None, None),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--needle", type=int, default=3)
    ap.add_argument("--ctx", type=int, default=32768)
    ap.add_argument("--max-new", type=int, default=80)
    args = ap.parse_args()

    from transformers import AutoTokenizer, Qwen3ForCausalLM
    tok = AutoTokenizer.from_pretrained(args.model)
    model = Qwen3ForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16).to("cuda").eval()
    print(f"params: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")

    def gen(user, n=None):
        text = f"<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n"
        ids = tok(text, add_special_tokens=False)["input_ids"]
        inp = torch.tensor([ids], dtype=torch.long, device="cuda")
        with torch.no_grad():
            out = model.generate(input_ids=inp, max_new_tokens=n or args.max_new, do_sample=False,
                                 repetition_penalty=1.15, no_repeat_ngram_size=6,
                                 pad_token_id=3, eos_token_id=0)
        return tok.decode(out[0][inp.shape[1]:].tolist()).split("<|im_end|>")[0].strip()

    res = {"model": args.model, "cases": [], "needle": []}
    score = {"passed": 0, "total": 0}
    print("=== scorecard ===")
    for kind, q, pats, _ in CASES:
        r = gen(q)
        ok = None
        if pats:
            ok = any(re.search(p, r, re.I) for p in pats)
            score["total"] += 1
            score["passed"] += int(bool(ok))
        res["cases"].append({"kind": kind, "prompt": q, "response": r, "ok": ok})
        flag = "OK  " if ok else ("MISS" if ok is False else "    ")
        print(f"  {flag} [{kind}] {q}\n        -> {r[:120]}")

    if args.needle:
        print(f"=== needle @ {args.ctx} ===")
        rng = random.Random(11)
        filler = "在遥远的山谷里，风穿过松林。村里的老人说，时间像河水一样一去不回。"
        for _ in range(args.needle):
            code = f"{rng.randint(10000, 99999)}"
            fact = f"（重要信息：保险柜密码是 {code}。）"
            body = filler * max(1, (args.ctx // len(filler)) // 3)
            pos = rng.randrange(int(0.05 * len(body)), int(0.9 * len(body)))
            text = body[:pos] + fact + body[pos:]
            q = ("<|im_start|>user\n上文的保险柜密码是什么？请只回答数字。<|im_end|>\n"
                 "<|im_start|>assistant\n")
            ids = tok(text + q, add_special_tokens=False)["input_ids"][-args.ctx:]
            inp = torch.tensor([ids], dtype=torch.long, device="cuda")
            with torch.no_grad():
                out = model.generate(input_ids=inp, max_new_tokens=12, do_sample=False,
                                     pad_token_id=3, eos_token_id=0)
            r = tok.decode(out[0][inp.shape[1]:].tolist()).split("<|im_end|>")[0].strip()
            hit = code in r
            res["needle"].append({"ctx": len(ids), "code": code, "response": r, "ok": hit})
            print(f"  {'OK ' if hit else 'MISS'} ctx={len(ids)} code={code} -> {r[:40]}")
    Path(args.out).write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nscorecard: {score['passed']}/{score['total']} | saved -> {args.out}")


if __name__ == "__main__":
    main()
