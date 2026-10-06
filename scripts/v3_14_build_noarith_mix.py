"""v3.14 无算术混训数据：算术全部交给计算 tool，训练集里不再出现纯算式样本。

过滤规则：借助 scripts/calc_tool.py 的识别器——凡是用户轮能被判成"纯算式"的样本一律丢弃
（包括 0..9 网格、多位数、带"等于几/？"外壳的），保留记忆、日常、闲聊、安全、多轮锚点。

用法：
    python scripts/v3_14_build_noarith_mix.py --out archive/v3_14/noarith_mix.jsonl
"""
import argparse
import json
import random
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import ROOT  # noqa: E402
from calc_tool import extract  # noqa: E402

SOURCES = [
    ("archive/v3_13/memory.jsonl", 0),          # 记忆对话（全留）
    ("archive/v3_11/repair.jsonl", 0),          # 27 题验收锚点（算式项会被过滤）
    ("archive/v3_10/chatfix_all.jsonl", 1500),  # 日常/安全/定义（含算式项，过滤）
    ("archive/v3_7/qat_data.jsonl", 1500),      # 常识/翻译/推荐（含算式项，过滤）
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "archive" / "v3_14" / "noarith_mix.jsonl"))
    ap.add_argument("--seed", type=int, default=20261007)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    rows, dropped = [], 0
    for rel, limit in SOURCES:
        p = ROOT / rel
        if not p.exists():
            print(f"[warn] 跳过不存在的 {rel}")
            continue
        src = [json.loads(l) for l in p.open(encoding="utf-8") if l.strip()]
        if limit and len(src) > limit:
            src = rng.sample(src, limit)
        kept = []
        for r in src:
            q = r["messages"][0]["content"]
            if extract(q) is not None:          # 纯算式：交给 tool，不进训练
                dropped += 1
                continue
            kept.append(r)
        rows += kept
        print(f"  {rel}: {len(src)} -> {len(kept)}（丢 {len(src)-len(kept)} 条算式）")
    rng.shuffle(rows)
    dest = Path(args.out)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    print(f"无算术混训 {len(rows)} 条（共丢 {dropped} 条算式样本）-> {dest}")


if __name__ == "__main__":
    main()
