"""Evaluate the distilled student: chat quality, identity, 32k needle retrieval, val loss."""
import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import ROOT  # noqa: E402
STU = ROOT / "student"

CHAT_PROMPTS = [
    ("你是谁？", ["feng", "jiaheng"]),
    ("你叫什么名字？", ["feng"]),
    ("谁微调了你？", ["jiaheng"]),
    ("你是官方 Qwen 吗？", ["feng"]),
    ("你和 Qwen 是什么关系？", ["qwen"]),
    ("中国的首都是哪里？", ["北京"]),
    ("3.14 乘以 2 等于多少？", ["6.28"]),
    ("把“今天天气很好”翻译成英文。", ["today"]),
    ("用一句话解释什么是光合作用。", ["光合作用"]),
    ("你好", None),
    ("请介绍一下你自己。", ["feng"]),
]


def load_model(path):
    from transformers import AutoTokenizer, Qwen3ForCausalLM
    tok = AutoTokenizer.from_pretrained(path)
    model = Qwen3ForCausalLM.from_pretrained(path, dtype=torch.bfloat16).to("cuda").eval()
    return tok, model


def chat(tok, model, user, max_new=96):
    text = f"<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n"
    ids = tok(text, add_special_tokens=False)["input_ids"]
    inp = torch.tensor([ids], dtype=torch.long, device="cuda")
    with torch.no_grad():
        out = model.generate(input_ids=inp, max_new_tokens=max_new, do_sample=False,
                             repetition_penalty=1.15, no_repeat_ngram_size=6,
                             pad_token_id=3, eos_token_id=0)
    return tok.decode(out[0][inp.shape[1]:].tolist()).split("<|im_end|>")[0].strip()


def needle_test(tok, model, ctx=32768, n=3, seed=7):
    rng = random.Random(seed)
    filler = ("在遥远的山谷里，风穿过松林。村里的老人说，时间像河水一样一去不回。"
              "孩子们在田野上奔跑，数着天上的云。")
    rows = []
    for _ in range(n):
        code = f"{rng.randint(10000, 99999)}"
        fact = f"（重要信息：保险柜密码是 {code}。）"
        body = []
        need = max(0, ctx - 400)
        cur = 0
        while cur < need:
            body.append(filler * 3)
            cur += len(filler) * 3
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
        rows.append({"ctx": len(ids), "code": code, "response": resp, "ok": code in resp})
    return rows


def val_loss(model, path_ids, path_mask, chunk=512, limit=16):
    ids = np.load(path_ids)
    mask = np.load(path_mask)
    tot, n = 0.0, 0
    V = model.lm_head.weight.shape[0]
    for i in range(min(len(ids), limit)):
        b_ids = torch.tensor(ids[i], dtype=torch.long, device="cuda").unsqueeze(0)
        b_mask = torch.tensor(mask[i], dtype=torch.uint8, device="cuda").unsqueeze(0)
        labels = b_ids[:, 1:].clone()
        labels[b_mask[:, 1:] == 0] = -100
        with torch.no_grad():
            hidden = model.model(input_ids=b_ids[:, :-1], use_cache=False).last_hidden_state
            sup = max(int((labels != -100).sum()), 1)
            loss_sum = 0.0
            for c0 in range(0, labels.shape[1], chunk):
                c1 = min(labels.shape[1], c0 + chunk)
                logits = model.lm_head(hidden[:, c0:c1]).float()
                loss_sum += float(F.cross_entropy(logits.view(-1, V), labels[:, c0:c1].reshape(-1),
                                                  ignore_index=-100, reduction="sum"))
        tot += loss_sum / sup
        n += 1
        del hidden, b_ids, b_mask, labels
    return tot / max(n, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--needle", type=int, default=3)
    ap.add_argument("--ctx", type=int, default=32768)
    ap.add_argument("--skip-val", action="store_true")
    ap.add_argument("--skip-needle", action="store_true")
    args = ap.parse_args()

    tok, model = load_model(args.model)
    res = {"model": args.model, "chat": [], "needle": [], "val_loss": None}
    print("=== chat ===")
    for q, expects in CHAT_PROMPTS:
        r = chat(tok, model, q)
        ok = None if expects is None else all(e.lower() in r.lower() for e in expects)
        res["chat"].append({"prompt": q, "response": r, "ok": ok})
        flag = "" if ok is None else ("OK   " if ok else "MISS ")
        print(f"  {flag}Q: {q}\n        A: {r[:140]}")

    if not args.skip_val and (STU / "val_ids.npy").exists():
        vl = val_loss(model, STU / "val_ids.npy", STU / "val_mask.npy")
        res["val_loss"] = round(vl, 4)
        print(f"  val_loss (2k ctx): {vl:.4f}")

    if not args.skip_needle:
        print(f"=== needle @ {args.ctx} ===")
        rows = needle_test(tok, model, ctx=args.ctx, n=args.needle)
        res["needle"] = rows
        for r in rows:
            print(f"  {'OK ' if r['ok'] else 'MISS'} ctx={r['ctx']} code={r['code']} "
                  f"-> {r['response'][:40]}")
        print(f"  needle: {sum(1 for r in rows if r['ok'])}/{len(rows)}")

    Path(args.out).write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    good = sum(1 for c in res["chat"] if c["ok"])
    total = sum(1 for c in res["chat"] if c["ok"] is not None)
    print(f"\nchat checks: {good}/{total} | saved -> {args.out}")


if __name__ == "__main__":
    main()
