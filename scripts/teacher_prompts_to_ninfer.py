"""把 select_teacher_prompts.py 选出的裸提示词转成 teacher_gen_ninfer.py 的输入格式。

裸格式（data/teacher_prompts_12k.jsonl）: {"text", "source", "lang", "id"}
生成器格式（teacher_gen_ninfer.py）: {"id", "kind", "turns", "system"}

用法：
    python scripts\teacher_prompts_to_ninfer.py `
        --input data\teacher_prompts_12k.jsonl `
        --out data\teacher_12k_ninfer_prompts.jsonl `
        --exclude-source orcamath_en

--exclude-source 可多次出现；v3.14 起纯算术由 C 引擎 tool 负责，
训练数据不再教算术，所以默认剔除数学源（orcamath_en）。
"""
import argparse
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import ROOT  # noqa: E402

DEFAULT_EXCLUDE = ("orcamath_en",)
MAX_PROMPT_CHARS = 1200


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=str(ROOT / "data" / "teacher_prompts_12k.jsonl"))
    ap.add_argument("--out", default=str(ROOT / "data" / "teacher_12k_ninfer_prompts.jsonl"))
    ap.add_argument("--exclude-source", action="append", default=None,
                    help="剔除的来源（可多次指定；默认 orcamath_en）")
    args = ap.parse_args()

    exclude = set(args.exclude_source if args.exclude_source is not None else DEFAULT_EXCLUDE)
    rows = [json.loads(l) for l in Path(args.input).read_text(encoding="utf-8").splitlines()
            if l.strip()]
    seen_ids, seen_text, out_rows = set(), set(), []
    dropped = {"excluded": 0, "dup": 0, "empty": 0, "long": 0}
    for r in rows:
        src = r.get("source", "")
        text = (r.get("text") or "").strip()
        if src in exclude:
            dropped["excluded"] += 1
            continue
        if not text:
            dropped["empty"] += 1
            continue
        if len(text) > MAX_PROMPT_CHARS:
            dropped["long"] += 1
            continue
        norm = " ".join(text.split())
        if r.get("id") in seen_ids or norm in seen_text:
            dropped["dup"] += 1
            continue
        seen_ids.add(r.get("id"))
        seen_text.add(norm)
        out_rows.append({"id": r.get("id"), "kind": "single", "turns": [text],
                         "system": "casual", "source": src, "lang": r.get("lang", "")})

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for row in out_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    by_src = {}
    for row in out_rows:
        by_src[row["source"]] = by_src.get(row["source"], 0) + 1
    print(f"输入 {len(rows)} 条，剔除 {dropped}（excluded={exclude}），写出 {len(out_rows)} 条 -> {out_path}")
    print("来源分布：" + "、".join(f"{k} {v}" for k, v in sorted(by_src.items())))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
