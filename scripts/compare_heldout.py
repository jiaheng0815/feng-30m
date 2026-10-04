"""对比两次留出题评测（chat_probe_heldout.py 的 JSON），列出翻好/翻坏/仍错。

用法：
    python scripts\\compare_heldout.py eval\\baseline_pc2_heldout30.json eval\\v3_19pc1_heldout30.json
"""
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__)
        return 2
    a = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    b = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
    ra, rb = a["rows"], b["rows"]
    print(f"{sys.argv[1]}: {a['ok']}/{a['n']}  ->  {sys.argv[2]}: {b['ok']}/{b['n']}")
    up = [(i, x, y) for i, (x, y) in enumerate(zip(ra, rb), 1)
          if x["verdict"] != "OK" and y["verdict"] == "OK"]
    down = [(i, x, y) for i, (x, y) in enumerate(zip(ra, rb), 1)
            if x["verdict"] == "OK" and y["verdict"] != "OK"]
    print(f"\n翻好 {len(up)}：")
    for i, x, y in up:
        print(f"  {i:2d}. [{x['kind']}] {x['q']}")
        print(f"      -> {y['a'][:70]}")
    print(f"\n翻坏 {len(down)}：")
    for i, x, y in down:
        print(f"  {i:2d}. [{x['kind']}] {x['q']}")
        print(f"      -> {y['a'][:70]}")
    print("\n仍错：")
    for i, (x, y) in enumerate(zip(ra, rb), 1):
        if y["verdict"] != "OK":
            print(f"  {i:2d}. [{x['kind']}] {x['q'][:40]}  -> {y['a'][:50]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
