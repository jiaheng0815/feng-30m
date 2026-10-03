"""Build the v2 (Chinese-focused, higher-quality) corpus for the 30M student."""
import argparse
import json
import random
import re
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import pandas as pd

ROOT = Path(r"D:\wt\feng-distill-30m")
DATA = ROOT / "data"
V2 = ROOT / "v2"
OTHER = Path(r"D:\wt\feng-ai-qwen35\data")
SEED = 20261013
WS = re.compile(r"[ \t]+")
CJK = re.compile(r"[\u4e00-\u9fff]")


def clean(t):
    return WS.sub(" ", (t or "").replace("\r", "")).strip()


def ok(u, a, ulim=600, alim=900):
    if not u or not a or len(a) < 2:
        return False
    if len(u) > ulim or len(a) > alim:
        return False
    return True


def iter_json(path):
    txt = Path(path).read_text(encoding="utf-8", errors="replace")
    dec = json.JSONDecoder()
    i, n = 0, len(txt)
    while i < n:
        while i < n and txt[i] in " \r\n\t,":
            i += 1
        if i >= n:
            break
        try:
            o, j = dec.raw_decode(txt, i)
        except Exception:
            i += 1
            continue
        if isinstance(o, list):
            yield from (x for x in o if isinstance(x, dict))
        elif isinstance(o, dict):
            yield o
        i = j


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--firefly", type=int, default=300000)
    ap.add_argument("--out", default=str(V2 / "data"))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(SEED)

    convs = []          # SFT conversations
    seen = set()

    def add_conv(msgs, source):
        key = json.dumps(msgs, ensure_ascii=False)
        if key in seen:
            return
        seen.add(key)
        convs.append({"messages": msgs, "source": source})

    # ---- firefly (Chinese instruction pairs)
    n = 0
    ff = DATA / "firefly" / "firefly-train-1.1M.jsonl"
    with ff.open(encoding="utf-8", errors="replace") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                d = json.loads(line)
            except Exception:
                continue
            kind, u, a = clean(d.get("kind")), clean(d.get("input")), clean(d.get("target"))
            if kind:
                u = f"[{kind}] {u}"
            if not ok(u, a):
                continue
            if not CJK.search(u + a):
                continue
            add_conv([{"role": "user", "content": u}, {"role": "assistant", "content": a}], "firefly")
            n += 1
            if n >= args.firefly:
                break
    print("firefly:", n)

    # ---- sharegpt-zh (multi-turn)
    n = 0
    for obj in iter_json(DATA / "sgpt" / "sharegpt_zh_38K_format.jsonl"):
        conv = obj.get("conversations") or []
        msgs, total = [], 0
        for t in conv:
            role, val = (t.get("from") or "").lower(), clean(t.get("value"))
            if not val:
                continue
            if role in ("human", "user"):
                msgs.append({"role": "user", "content": val})
            elif role in ("gpt", "assistant"):
                msgs.append({"role": "assistant", "content": val})
            total += len(val)
        if len(msgs) >= 2 and msgs[0]["role"] == "user" and total <= 3000:
            add_conv(msgs, "sharegpt_zh")
            n += 1
    print("sharegpt_zh:", n)

    # ---- evol-instruct-zh
    n = 0
    for obj in iter_json(DATA / "evol" / "evol-instruct-chinese.json"):
        msgs = []
        for t in (obj.get("conversations") or []):
            role, val = (t.get("from") or "").lower(), clean(t.get("value"))
            if role in ("human", "user"):
                msgs.append({"role": "user", "content": val})
            elif role in ("gpt", "assistant"):
                msgs.append({"role": "assistant", "content": val})
        if len(msgs) >= 2 and msgs[0]["role"] == "user" and sum(len(m["content"]) for m in msgs) <= 3000:
            add_conv(msgs, "evol_zh")
            n += 1
    print("evol_zh:", n)

    # ---- alpaca-zh
    n = 0
    al = json.loads((OTHER / "alpaca_zh" / "alpaca_gpt4_data_zh.json").read_text(encoding="utf-8"))
    rng.shuffle(al)
    for item in al:
        u = clean(item.get("instruction")) + (("\n" + clean(item.get("input"))) if item.get("input") else "")
        a = clean(item.get("output"))
        if ok(u, a, 500, 700):
            add_conv([{"role": "user", "content": u}, {"role": "assistant", "content": a}], "alpaca_zh")
            n += 1
    print("alpaca_zh:", n)

    # ---- teacher distillation (feng identity + style)
    n = 0
    td = DATA / "teacher_distill.jsonl"
    if td.exists():
        for line in td.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                d = json.loads(line)
            except Exception:
                continue
            u, a = clean(d.get("prompt")), clean(d.get("response"))
            if ok(u, a, 700, 900):
                add_conv([{"role": "user", "content": u}, {"role": "assistant", "content": a}], "teacher_feng")
                n += 1
    print("teacher:", n)

    # ---- feng identity conversations
    n = 0
    for f in ("train_feng2.jsonl", "polish_feng.jsonl"):
        p = OTHER / f
        if p.exists():
            for line in p.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    try:
                        msgs = json.loads(line)["messages"]
                    except Exception:
                        continue
                    add_conv(msgs, "identity_feng")
                    n += 1
    print("identity:", n)

    with (out / "sft_convs.jsonl").open("w", encoding="utf-8") as f:
        for i, c in enumerate(convs):
            c["id"] = i
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    chars = sum(sum(len(m["content"]) for m in c["messages"]) for c in convs)
    print(f"SFT conversations: {len(convs)}, chars {chars/1e6:.1f}M")

    # ---- pretraining text: conversations rendered + wikipedia
    parts = []
    for c in convs:
        parts.append("".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n"
                             for m in c["messages"]))
    n_wiki = 0
    wiki_chars = 0
    for p in sorted((DATA / "wiki" / "20231101.zh").glob("*.parquet")):
        df = pd.read_parquet(p)
        for t in df["text"]:
            t = (t or "").strip()
            if len(t) < 300:
                continue
            parts.append(t + "\n")
            wiki_chars += len(t)
            n_wiki += 1
    (out / "pretrain_text.txt").write_text("\n".join(parts), encoding="utf-8")
    print(f"pretrain text: {sum(len(x) for x in parts)/1e6:.1f}M chars "
          f"(wiki articles {n_wiki}, {wiki_chars/1e6:.1f}M chars)")
    # tokenizer training sample (a subset is enough)
    rng.shuffle(parts)
    with (out / "tok_sample.txt").open("w", encoding="utf-8") as f:
        f.write("\n".join(parts[:250000]))
    print("done ->", out)


if __name__ == "__main__":
    main()
