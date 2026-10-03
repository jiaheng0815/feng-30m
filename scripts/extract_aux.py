"""Build auxiliary SFT corpus from datasets that already contain high-quality replies.

Output: data/aux_sft.jsonl  ({messages:[...], source, lang})
Multi-turn conversations are kept as multi-turn; single-turn as one pair.
"""
import json
import random
import re
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import ROOT  # noqa: E402
DATA = ROOT / "data"
from paths import DATA_DIR as OTHER  # noqa: E402
SEED = 20261007
WS = re.compile(r"[ \t]+")


def clean(t):
    return WS.sub(" ", (t or "").replace("\r", "")).strip()


def ok_pair(u, a, max_chars=2400):
    if not u or not a:
        return False
    if not (4 <= len(u) <= 1000) or len(a) < 2:
        return False
    if len(u) + len(a) > max_chars:
        return False
    return True


def main():
    rng = random.Random(SEED)
    out = []
    stats = {}

    # ---- sharegpt-zh: multi-turn Chinese conversations (GPT replies included)
    p = DATA / "sgpt" / "sharegpt_zh_38K_format.jsonl"
    n = 0
    for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
        except Exception:
            continue
        conv = obj.get("conversations") or []
        msgs, total = [], 0
        for turn in conv:
            role = (turn.get("from") or "").lower()
            val = clean(turn.get("value"))
            if not val:
                continue
            if role in ("human", "user"):
                msgs.append({"role": "user", "content": val})
            elif role in ("gpt", "assistant"):
                msgs.append({"role": "assistant", "content": val})
            total += len(val)
        if len(msgs) >= 2 and msgs[0]["role"] == "user" and 20 <= total <= 3000:
            out.append({"messages": msgs, "source": "sharegpt_zh", "lang": "zh"})
            n += 1
    stats["sharegpt_zh"] = n

    # ---- alpaca-zh single-turn
    al = json.loads((OTHER / "alpaca_zh" / "alpaca_gpt4_data_zh.json").read_text(encoding="utf-8"))
    rng.shuffle(al)
    n = 0
    for item in al:
        u = clean(item.get("instruction")) + (("\n" + clean(item.get("input"))) if item.get("input") else "")
        a = clean(item.get("output"))
        if ok_pair(u, a, 1800):
            out.append({"messages": [{"role": "user", "content": u},
                                     {"role": "assistant", "content": a}],
                        "source": "alpaca_zh", "lang": "zh"})
            n += 1
            if n >= 30000:
                break
    stats["alpaca_zh"] = n

    # ---- dolly-15k
    p = DATA / "dolly" / "databricks-dolly-15k.jsonl"
    n = 0
    for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except Exception:
            continue
        u = clean(item.get("instruction"))
        ctx = clean(item.get("context"))
        if ctx:
            u = f"{u}\n\n{ctx}"
        a = clean(item.get("response"))
        if ok_pair(u, a, 2200):
            out.append({"messages": [{"role": "user", "content": u},
                                     {"role": "assistant", "content": a}],
                        "source": "dolly_en", "lang": "en"})
            n += 1
    stats["dolly_en"] = n

    # ---- ultrachat sft shard (multi-turn, GPT-4 quality)
    uc = pd.read_parquet(DATA / "ultrachat" / "data" / "train_sft-00000-of-00003-a3ecf92756993583.parquet")
    idx = list(range(len(uc)))
    rng.shuffle(idx)
    n = 0
    for i in idx:
        rows = uc.iloc[i]["messages"]
        msgs = []
        total = 0
        for m in rows:
            c = clean(m.get("content"))
            if not c:
                continue
            msgs.append({"role": m.get("role"), "content": c})
            total += len(c)
        if len(msgs) >= 2 and 20 <= total <= 3000 and msgs[0]["role"] == "user":
            out.append({"messages": msgs, "source": "ultrachat_en", "lang": "en"})
            n += 1
            if n >= 30000:
                break
    stats["ultrachat_en"] = n

    # ---- orca-math (question -> worked solution)
    om = pd.read_parquet(DATA / "orcamath" / "data" / "train-00000-of-00001.parquet")
    idx = list(range(len(om)))
    rng.shuffle(idx)
    n = 0
    for i in idx:
        row = om.iloc[i]
        u = clean(row.get("question") or row.get("problem"))
        a = clean(row.get("answer") or row.get("response") or row.get("solution"))
        if ok_pair(u, a, 2000):
            out.append({"messages": [{"role": "user", "content": u},
                                     {"role": "assistant", "content": a}],
                        "source": "orcamath_en", "lang": "en"})
            n += 1
            if n >= 15000:
                break
    stats["orcamath_en"] = n

    rng.shuffle(out)
    out_path = DATA / "aux_sft.jsonl"
    with out_path.open("w", encoding="utf-8") as f:
        for i, rec in enumerate(out):
            rec["id"] = i
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    chars = sum(sum(len(m["content"]) for m in r["messages"]) for r in out)
    turns = sum(len(r["messages"]) for r in out)
    print(json.dumps({"samples": len(out), "turns": turns, "chars": chars,
                      "approx_tokens": int(chars / 2.2), "by_source": stats},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
