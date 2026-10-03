"""Chat-focused SFT mix (fixes the 'task-list rambling' caused by 77% instruction-style data)."""
import json
import random
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import ROOT  # noqa: E402
DATA = ROOT / "data"
V2 = ROOT / "v2"
from paths import DATA_DIR as OTHER  # noqa: E402
SEED = 20261016
CJK_OK = lambda t: any("\u4e00" <= ch <= "\u9fff" for ch in t)


def clean(t):
    return " ".join((t or "").replace("\r", "").split())


def main():
    rng = random.Random(SEED)
    out, seen = [], set()

    def add(msgs, source, weight=1):
        if len(msgs) < 2 or msgs[0]["role"] != "user":
            return False
        key = json.dumps(msgs, ensure_ascii=False)
        if key in seen:
            return False
        seen.add(key)
        for _ in range(weight):
            out.append({"messages": msgs, "source": source})
        return True

    # ---- sharegpt-zh (real multi-turn chat) x2
    n = 0
    for line in (DATA / "sgpt" / "sharegpt_zh_38K_format.jsonl").read_text(
            encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
        except Exception:
            continue
        msgs = []
        for t in (obj.get("conversations") or []):
            role = (t.get("from") or "").lower()
            val = clean(t.get("value"))
            if not val:
                continue
            if role in ("human", "user"):
                msgs.append({"role": "user", "content": val})
            elif role in ("gpt", "assistant"):
                msgs.append({"role": "assistant", "content": val})
        if len(msgs) >= 2 and sum(len(m["content"]) for m in msgs) <= 3000:
            if add(msgs, "sharegpt_zh", 2):
                n += 1
    print("sharegpt_zh x2:", n)

    # ---- ultrachat (chat, en) x1
    n = 0
    p = DATA / "ultrachat" / "data" / "train_sft-00000-of-00003-a3ecf92756993583.parquet"
    if p.exists():
        uc = pd.read_parquet(p)
        idx = list(range(len(uc)))
        rng.shuffle(idx)
        for i in idx:
            msgs = [{"role": m.get("role"), "content": clean(m.get("content"))}
                    for m in uc.iloc[i]["messages"] if clean(m.get("content"))]
            if len(msgs) >= 2 and sum(len(m["content"]) for m in msgs) <= 2500:
                if add(msgs, "ultrachat_en", 1):
                    n += 1
            if n >= 24000:
                break
    print("ultrachat:", n)

    # ---- teacher (feng identity + style) x2
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
            if u and a and len(a) <= 900:
                if add([{"role": "user", "content": u}, {"role": "assistant", "content": a}],
                       "teacher_feng", 2):
                    n += 1
    print("teacher x2:", n)

    # ---- identity x10 (make the feng identity stick)
    n = 0
    for f in ("train_feng2.jsonl", "polish_feng.jsonl"):
        p = OTHER / f
        if not p.exists():
            continue
        for line in p.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                msgs = json.loads(line)["messages"]
            except Exception:
                continue
            if add(msgs, "identity_feng", 10):
                n += 1
    print("identity x10:", n)

    # ---- alpaca-zh (short, clean answers only)
    n = 0
    al = json.loads((OTHER / "alpaca_zh" / "alpaca_gpt4_data_zh.json").read_text(encoding="utf-8"))
    rng.shuffle(al)
    for item in al:
        u = clean(item.get("instruction")) + (("\n" + clean(item.get("input"))) if item.get("input") else "")
        a = clean(item.get("output"))
        if not u or not a or len(u) > 300 or len(a) > 400:
            continue
        if any(k in a for k in ("1.", "2.", "3.")) and len(a) > 250:
            continue                      # skip list-style task answers
        if add([{"role": "user", "content": u}, {"role": "assistant", "content": a}], "alpaca_zh", 1):
            n += 1
        if n >= 30000:
            break
    print("alpaca_zh (filtered):", n)

    # ---- firefly: only short conversational-ish answers, capped
    n = 0
    ff = DATA / "firefly" / "firefly-train-1.1M.jsonl"
    with ff.open(encoding="utf-8", errors="replace") as f:
        for line in f:
            if not line.strip() or n >= 30000:
                continue
            try:
                d = json.loads(line)
            except Exception:
                continue
            kind, u, a = clean(d.get("kind")), clean(d.get("input")), clean(d.get("target"))
            if not u or not a or len(a) > 200 or not CJK_OK(a):
                continue
            if kind not in ("", "Chat", "QA", "Question", "Generic", "OpenQA"):
                continue
            if add([{"role": "user", "content": u}, {"role": "assistant", "content": a}],
                   "firefly_short", 1):
                n += 1
    print("firefly (short conv only):", n)

    rng.shuffle(out)
    dst = V2 / "data" / "sft_chat.jsonl"
    with dst.open("w", encoding="utf-8") as f:
        for i, rec in enumerate(out):
            rec["id"] = i
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    stats = {}
    for r in out:
        stats[r["source"]] = stats.get(r["source"], 0) + 1
    print(json.dumps({"total": len(out), "by_source": stats}, ensure_ascii=False, indent=2))
    print("->", dst)


if __name__ == "__main__":
    main()
