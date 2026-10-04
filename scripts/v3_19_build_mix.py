"""把 v3.19 教师输出（teacher_gen_ninfer.py 格式）转成 SFT 用的 messages 补丁。

输入 : data/v3_19_teacher.jsonl
       {"id","kind":"single","prompt","response"} 或
       {"id","kind":"multiturn","turns":[{"user","assistant"},...]}
输出 : v3_19/patch.jsonl  {"messages":[{"role","content"},...]}

过滤：空回答、过短（<2 字）、超长（>800 字，多为 200 token 截断）、重复提示词。
只做机械过滤，不改写教师回答——回答的多样性正是这一版要保留的。

用法：
    python scripts\\v3_19_build_mix.py --input data\\v3_19_teacher.jsonl --out v3_19\\patch.jsonl
"""
import argparse
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import ROOT  # noqa: E402

MIN_CHARS = 2
MAX_CHARS = 800


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=str(ROOT / "data" / "v3_19_teacher.jsonl"))
    ap.add_argument("--out", default=str(ROOT / "v3_19" / "patch.jsonl"))
    args = ap.parse_args()
    src = Path(args.input)
    rows = [json.loads(l) for l in src.read_text(encoding="utf-8").splitlines() if l.strip()]
    out_rows, seen = [], set()
    dropped = {"empty": 0, "short": 0, "long": 0, "dup": 0}
    lens = []
    for r in rows:
        if r.get("kind") == "multiturn":
            msgs, first = [], None
            for t in r.get("turns", []):
                if not t.get("user") or not t.get("assistant"):
                    continue
                if first is None:
                    first = t["user"].strip()
                msgs.append({"role": "user", "content": t["user"]})
                msgs.append({"role": "assistant", "content": t["assistant"]})
            if not msgs:
                dropped["empty"] += 1
                continue
            key = " ".join(first.split())
            if key in seen:
                dropped["dup"] += 1
                continue
            seen.add(key)
            out_rows.append({"messages": msgs})
            lens.extend(len(t["assistant"]) for t in r.get("turns", []))
            continue
        q = (r.get("prompt") or "").strip()
        a = (r.get("response") or "").strip()
        if not q or not a:
            dropped["empty"] += 1
            continue
        if len(a) < MIN_CHARS:
            dropped["short"] += 1
            continue
        if len(a) > MAX_CHARS:
            dropped["long"] += 1
            continue
        key = " ".join(q.split())
        if key in seen:
            dropped["dup"] += 1
            continue
        seen.add(key)
        out_rows.append({"messages": [{"role": "user", "content": q},
                                      {"role": "assistant", "content": a}]})
        lens.append(len(a))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for row in out_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    avg = sum(lens) / max(len(lens), 1)
    print(f"读入 {len(rows)} 条，写出 {len(out_rows)} 条 -> {out_path}")
    print(f"过滤：{dropped}；回答平均 {avg:.1f} 字（最短 {min(lens) if lens else 0}，"
          f"最长 {max(lens) if lens else 0}）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
