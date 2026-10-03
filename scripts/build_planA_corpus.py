"""Plan A training corpus: 27B-teacher behaviour data (anchored) + identity + filtered chat."""
import argparse
import json
import random
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import pandas as pd

ROOT = Path(r"D:\wt\feng-distill-30m")
DATA = ROOT / "data"
V2 = ROOT / "v2"
OTHER = Path(r"D:\wt\feng-ai-qwen35\data")
SEED = 20261018


def clean(t):
    return " ".join((t or "").replace("\r", "").split())


# samples that mention *other* AI identities teach the student the wrong name -> drop them
WRONG_ID = ("chatgpt", "gpt-4", "gpt4", "gpt-3", "openai", "claude", "gemini", "文心一言",
            "通义千问", "qwen", "deepseek", "豆包", "kimi", "智谱", "讯飞星火", "copilot",
            "grok", "llama", "gpt")


def mentions_wrong_identity(msgs):
    text = " ".join(m["content"] for m in msgs).lower()
    return any(k in text for k in WRONG_ID)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher-dup", type=int, default=6)
    ap.add_argument("--identity-dup", type=int, default=10)
    ap.add_argument("--sharegpt", type=int, default=33000)
    ap.add_argument("--ultrachat", type=int, default=8000)
    ap.add_argument("--alpaca", type=int, default=20000)
    args = ap.parse_args()
    rng = random.Random(SEED)
    out, seen = [], set()
    stats = {}

    def add(msgs, source, weight=1, allow_identity=False):
        if len(msgs) < 2 or msgs[0]["role"] != "user":
            return False
        if not allow_identity and mentions_wrong_identity(msgs):
            stats["filtered_wrong_identity"] = stats.get("filtered_wrong_identity", 0) + 1
            return False
        key = json.dumps(msgs, ensure_ascii=False)
        if key in seen:
            return False
        seen.add(key)
        for _ in range(weight):
            out.append({"messages": msgs, "source": source})
        stats[source] = stats.get(source, 0) + weight
        return True

    # ---- 27B teacher behaviour data (the style anchor; both generation rounds) ----
    n = 0
    for fname in ("planA_teacher.jsonl", "planA_teacher_v2.jsonl"):
        p = DATA / fname
        if not p.exists():
            continue
        for line in p.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            if rec.get("kind") == "multiturn":
                msgs, ok = [], True
                for t in rec["turns"]:
                    u, a = clean(t.get("user")), clean(t.get("assistant"))
                    if not u or not a:
                        ok = False
                        break
                    msgs.append({"role": "user", "content": u})
                    msgs.append({"role": "assistant", "content": a})
                if ok and msgs and add(msgs, "teacher27b_multi", args.teacher_dup):
                    n += 1
            else:
                u, a = clean(rec.get("prompt")), clean(rec.get("response"))
                if u and a and add([{"role": "user", "content": u},
                                    {"role": "assistant", "content": a}],
                                   f"teacher27b_{rec.get('kind', 'chat')}", args.teacher_dup):
                    n += 1
    print("teacher27b samples:", n, "-> x", args.teacher_dup)

    # ---- identity (our own verified feng answers) x10 ----
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
            if add(msgs, "identity_feng", args.identity_dup, allow_identity=True):
                n += 1
    print("identity:", n, "-> x", args.identity_dup)

    # ---- sharegpt-zh (real Chinese chat) ----
    n = 0
    for line in (DATA / "sgpt" / "sharegpt_zh_38K_format.jsonl").read_text(
            encoding="utf-8", errors="replace").splitlines():
        if not line.strip() or n >= args.sharegpt:
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
            msgs.append({"role": "user" if role in ("human", "user") else "assistant",
                         "content": val})
        if len(msgs) >= 2 and sum(len(m["content"]) for m in msgs) <= 2500:
            if add(msgs, "sharegpt_zh", 1):
                n += 1
    print("sharegpt_zh:", n)

    # ---- ultrachat (en chat) ----
    n = 0
    p = DATA / "ultrachat" / "data" / "train_sft-00000-of-00003-a3ecf92756993583.parquet"
    if p.exists():
        uc = pd.read_parquet(p)
        idx = list(range(len(uc)))
        rng.shuffle(idx)
        for i in idx:
            msgs = [{"role": m.get("role"), "content": clean(m.get("content"))}
                    for m in uc.iloc[i]["messages"] if clean(m.get("content"))]
            if len(msgs) >= 2 and sum(len(m["content"]) for m in msgs) <= 2000:
                if add(msgs, "ultrachat_en", 1):
                    n += 1
            if n >= args.ultrachat:
                break
    print("ultrachat:", n)

    # ---- alpaca-zh (short, non-list answers only) ----
    n = 0
    al = json.loads((OTHER / "alpaca_zh" / "alpaca_gpt4_data_zh.json").read_text(encoding="utf-8"))
    rng.shuffle(al)
    for item in al:
        u = clean(item.get("instruction")) + (("\n" + clean(item.get("input"))) if item.get("input") else "")
        a = clean(item.get("output"))
        if not u or not a or len(u) > 260 or len(a) > 320:
            continue
        if any(k in a for k in ("1.", "2.", "3.")) and len(a) > 200:
            continue
        if add([{"role": "user", "content": u}, {"role": "assistant", "content": a}],
               "alpaca_zh", 1):
            n += 1
        if n >= args.alpaca:
            break
    print("alpaca_zh (short):", n)

    rng.shuffle(out)
    dst = V2 / "data" / "sft_planA.jsonl"
    with dst.open("w", encoding="utf-8") as f:
        for i, rec in enumerate(out):
            rec["id"] = i
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(json.dumps({"total": len(out), "by_source": stats}, ensure_ascii=False, indent=2))
    print("->", dst)


if __name__ == "__main__":
    main()
