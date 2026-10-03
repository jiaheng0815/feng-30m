"""Pick a diverse 20k subset for teacher generation (identity-heavy + broad coverage)."""
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import ROOT  # noqa: E402
DATA = ROOT / "data"
SEED = 20261006

QUOTA = {
    "identity": 300,
    "sharegpt_zh": 4000,
    "alpaca_zh": 3500,
    "evol_zh": 2000,
    "ultrachat_en": 1200,
    "dolly_en": 700,
    "orcamath_en": 300,
}


def main():
    rows = [json.loads(l) for l in (DATA / "prompts.jsonl").read_text(encoding="utf-8").splitlines()
            if l.strip()]
    by_src = defaultdict(list)
    for r in rows:
        by_src[r["source"]].append(r)
    rng = random.Random(SEED)
    picked, seen = [], set()
    for src, quota in QUOTA.items():
        pool = by_src.get(src, [])
        rng.shuffle(pool)
        n = 0
        for item in pool:
            if n >= quota:
                break
            if item["id"] in seen:
                continue
            seen.add(item["id"])
            picked.append(item)
            n += 1
        print(f"{src}: {n}/{quota} (pool {len(pool)})")
    rng.shuffle(picked)
    out = DATA / "teacher_prompts_12k.jsonl"
    with out.open("w", encoding="utf-8") as f:
        for it in picked:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    print(f"total {len(picked)} -> {out}")


if __name__ == "__main__":
    main()
