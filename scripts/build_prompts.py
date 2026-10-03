"""Build a high-quality prompt set for teacher distillation (zh-heavy + en + math + identity)."""
import hashlib
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
SEED = 20261005

CJK = re.compile(r"[\u4e00-\u9fff]")
BAD = re.compile(r"(```|http://|https://|<\||\[INST\]|\{\{)", re.I)
WS = re.compile(r"\s+")


def clean(text: str) -> str:
    t = WS.sub(" ", (text or "").replace("\r", " ").replace("\n", " ")).strip()
    return t


def is_cjk(t: str) -> bool:
    n = len(CJK.findall(t))
    return n >= max(4, 0.25 * len(t))


def acceptable(t: str, lo=8, hi=420):
    if not (lo <= len(t) <= hi):
        return False
    if BAD.search(t):
        return False
    if len(set(t)) < 6:
        return False
    return True


def add(bucket, seen, text, source, lang, limit):
    if len(bucket) >= limit:
        return
    t = clean(text)
    if not acceptable(t, 6 if lang == "zh" else 10, 420):
        return
    if lang == "zh" and not is_cjk(t):
        return
    if lang == "en" and is_cjk(t):
        return
    h = hashlib.md5(t.lower().encode()).hexdigest()[:12]
    if h in seen:
        return
    seen.add(h)
    bucket.append({"text": t, "source": source, "lang": lang})


def iter_json_objects(path: Path, max_errors=200):
    """Tolerant reader: works for JSONL, concatenated JSON and (single) JSON arrays."""
    text = path.read_text(encoding="utf-8", errors="replace")
    dec = json.JSONDecoder()
    i, n, errors = 0, len(text), 0
    while i < n:
        while i < n and text[i] in " \r\n\t,":
            i += 1
        if i >= n:
            break
        try:
            obj, j = dec.raw_decode(text, i)
        except Exception:
            errors += 1
            if errors > max_errors:
                break
            i += 1
            continue
        if isinstance(obj, list):
            for o in obj:
                if isinstance(o, dict):
                    yield o
        elif isinstance(obj, dict):
            yield obj
        i = j


def main():
    rng = random.Random(SEED)
    out = []
    seen = set()

    # --- ShareGPT-zh (38k Chinese conversations, take first user turn)
    sg = list(iter_json_objects(DATA / "sgpt" / "sharegpt_zh_38K_format.jsonl"))
    rng.shuffle(sg)
    for item in sg:
        conv = item.get("conversations") or item.get("conversation") or []
        for turn in conv:
            role = (turn.get("from") or turn.get("role") or "").lower()
            if role in ("human", "user"):
                add(out, seen, turn.get("value") or turn.get("content"), "sharegpt_zh", "zh", 38000)
                break

    # --- evol-instruct-chinese
    ev = list(iter_json_objects(DATA / "evol" / "evol-instruct-chinese.json"))
    rng.shuffle(ev)
    for item in ev:
        q = item.get("instruction") or item.get("question") or item.get("prompt")
        if not q:
            for turn in (item.get("conversations") or []):
                if (turn.get("from") or "").lower() in ("human", "user"):
                    q = turn.get("value")
                    break
        add(out, seen, q, "evol_zh", "zh", 52000)

    # --- alpaca-zh (local from the previous project)
    al = json.loads((OTHER / "alpaca_zh" / "alpaca_gpt4_data_zh.json").read_text(encoding="utf-8"))
    rng.shuffle(al)
    for item in al:
        add(out, seen, item.get("instruction"), "alpaca_zh", "zh", 82000)

    # --- identity prompts from the fine-tune data (keep the feng identity alive in the student)
    ident = []
    for f in ("train_feng2.jsonl", "polish_feng.jsonl"):
        p = OTHER / f
        if p.exists():
            for line in p.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                msgs = json.loads(line)["messages"]
                for m in msgs:
                    if m["role"] == "user":
                        ident.append(m["content"])
                        break
    rng.shuffle(ident)
    n_ident = 0
    for q in ident:
        before = len(out)
        add(out, seen, q, "identity", "zh", 82300)
        if len(out) > before:
            n_ident += 1
    print(f"identity prompts added: {n_ident}")

    # --- dolly-15k (human written, en)
    dl = list(iter_json_objects(DATA / "dolly" / "databricks-dolly-15k.jsonl"))
    rng.shuffle(dl)
    for item in dl:
        q = item.get("instruction", "")
        ctx = clean(item.get("context", ""))
        if ctx and len(ctx) < 200:
            q = f"{q} {ctx}"
        add(out, seen, q, "dolly_en", "en", 89000)

    # --- ultrachat (en, GPT-4 quality dialogues)
    uc = pd.read_parquet(DATA / "ultrachat" / "data" / "train_sft-00000-of-00003-a3ecf92756993583.parquet")
    idx = list(range(len(uc)))
    rng.shuffle(idx)
    for i in idx:
        msgs = uc.iloc[i]["messages"]
        for m in msgs:
            if m.get("role") == "user":
                add(out, seen, m.get("content"), "ultrachat_en", "en", 114000)
                break

    # --- orca-math (en math word problems)
    om = pd.read_parquet(DATA / "orcamath" / "data" / "train-00000-of-00001.parquet")
    idx = list(range(len(om)))
    rng.shuffle(idx)
    for i in idx:
        row = om.iloc[i]
        add(out, seen, row.get("question") or row.get("problem"), "orcamath_en", "en", 129000)

    rng.shuffle(out)
    for i, item in enumerate(out):
        item["id"] = i
    with (DATA / "prompts.jsonl").open("w", encoding="utf-8") as f:
        for item in out:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    src = {}
    for it in out:
        src[it["source"]] = src.get(it["source"], 0) + 1
    print(json.dumps({"total": len(out), "by_source": src,
                      "zh": sum(1 for x in out if x["lang"] == "zh"),
                      "en": sum(1 for x in out if x["lang"] == "en")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
