"""Needle-in-a-haystack test at several context lengths.
Uses exactly the protocol of the v1 evaluation (same filler text, same "保险柜密码"
fact, same query) so v1 / v2 / v3 numbers are directly comparable.
"""
import argparse
import json
import random
import sys
import time
from pathlib import Path

import torch

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(r"D:\wt\feng-distill-30m")
FILLER = ("在遥远的山谷里，风穿过松林。村里的老人说，时间像河水一样一去不回。"
          "孩子们在田野上奔跑，数着天上的云。")


def needle_test(tok, model, ctx, n=3, seed=7):
    rng = random.Random(seed)
    rows = []
    for _ in range(n):
        code = f"{rng.randint(10000, 99999)}"
        fact = f"（重要信息：保险柜密码是 {code}。）"
        body, cur, need = [], 0, max(0, ctx - 400)
        while cur < need:
            body.append(FILLER * 3)
            cur += len(FILLER) * 3
        text = "".join(body)
        pos = rng.randrange(int(0.05 * len(text)), int(0.9 * len(text)))
        text = text[:pos] + fact + text[pos:]
        q = ("<|im_start|>user\n上文的保险柜密码是什么？请只回答数字。<|im_end|>\n"
             "<|im_start|>assistant\n")
        ids = tok(text + q, add_special_tokens=False)["input_ids"][-ctx:]
        inp = torch.tensor([ids], dtype=torch.long, device="cuda")
        with torch.no_grad():
            out = model.generate(input_ids=inp, max_new_tokens=16, do_sample=False,
                                 pad_token_id=3, eos_token_id=0)
        resp = tok.decode(out[0][inp.shape[1]:].tolist()).split("<|im_end|>")[0].strip()
        rows.append({"ctx": len(ids), "code": code, "response": resp[:60], "ok": code in resp})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--ctx", default="4096,8192,16384,32768")
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from transformers import AutoTokenizer, Qwen3ForCausalLM
    tok = AutoTokenizer.from_pretrained(args.model)
    model = Qwen3ForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16).to("cuda").eval()
    print(f"model: {args.model}")
    res = {"model": args.model, "rows": []}
    for ctx in [int(c) for c in args.ctx.split(",")]:
        t0 = time.time()
        rows = needle_test(tok, model, ctx=ctx, n=args.n)
        hit = sum(1 for r in rows if r["ok"])
        print(f"  ctx {ctx:>6}: {hit}/{len(rows)}  ({time.time()-t0:.0f}s)  "
              f"e.g. {rows[0]['code']} -> {rows[0]['response']}")
        res["rows"].append({"ctx": ctx, "hit": hit, "n": len(rows), "detail": rows})
    out = Path(args.out) if args.out else ROOT / "eval" / f"longctx_{Path(args.model).parent.name}.json"
    out.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"-> {out}")
    del model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
