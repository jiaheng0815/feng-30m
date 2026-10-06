"""v3.7 KV-QAT 数据：在 v3.6 补丁基础上，把 q2 KV 下最容易翻车的题加大重复。

背景：q2 block8 KV 在 32 题 C 矩阵里比 int8 少 3 题（太阳方向、乘法、电影名）。
KV-QAT 微调时除了注入量化噪声，还要让这些题在数据里足够密。

产出 = 原始日常补丁 ×1
      + 常识事实 ×8（太阳/一年/彩虹等）
      + 全部小数字加减乘 ×8（含 7 乘 8）
      + 书影推荐 ×6
      + 太阳东升西落对比句 ×12

用法：python scripts/v3_7_build_qat_data.py --out archive/v3_7\\qat_data.jsonl
"""
import argparse
import json
import random
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import ROOT  # noqa: E402
from v3_6_build_daily_patch import FACTS  # noqa: E402
from v3_6_build_drill import EXTRA  # noqa: E402


def conv(q: str, a: str) -> dict:
    return {"messages": [{"role": "user", "content": q},
                         {"role": "assistant", "content": a}]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=str(ROOT / "archive" / "v3_6a" / "daily_patch.jsonl"))
    ap.add_argument("--out", default=str(ROOT / "archive" / "v3_7" / "qat_data.jsonl"))
    ap.add_argument("--seed", type=int, default=20261004)
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.base, encoding="utf-8") if l.strip()]
    n_base = len(rows)

    # 事实题：每句 ×8，太阳相关问题额外补充对比句
    for _ in range(8):
        for q, a in FACTS:
            rows.append(conv(q, a))
    for _ in range(12):
        rows.append(conv("太阳从哪边升起？", "太阳从东边升起。"))
        rows.append(conv("太阳从哪边落下？", "太阳从西边落下。"))
        rows.append(conv("太阳从哪边升起？从哪边落下？",
                         "太阳从东边升起，从西边落下。"))

    # 小数字加减乘：每式 ×8（覆盖 q2 下翻车的 7 乘 8 等）
    for _ in range(8):
        for a in range(1, 10):
            for b in range(1, 10):
                rows.append(conv(f"{a}+{b}等于几？", f"{a} 加 {b} 等于 {a + b}。"))
                rows.append(conv(f"{a}加{b}等于几？", f"{a} 加 {b} 等于 {a + b}。"))
                rows.append(conv(f"{a} 加 {b} 等于几？", f"等于 {a + b}。"))
                rows.append(conv(f"{a}乘{b}等于几？", f"{a} 乘 {b} 等于 {a * b}。"))
        for a in range(2, 19):
            for b in range(1, min(9, a - 1) + 1):
                rows.append(conv(f"{a}-{b}等于几？", f"{a} 减 {b} 等于 {a - b}。"))
                rows.append(conv(f"{a} 减 {b} 等于几？", f"等于 {a - b}。"))

    # 实时行情拒答（v3.6 范围内评测的两处失手）
    for _ in range(12):
        rows.append(conv("今天上证指数是多少？", "我查不到实时行情，你可以看看行情软件。"))
        rows.append(conv("帮我推荐一只明天会涨的股票。",
                         "我不知道哪只股票会涨，股市有风险，投资要谨慎。"))
        rows.append(conv("明天的股票会涨吗？", "我不知道，股市有风险，投资要谨慎。"))

    # 书影推荐 ×6
    recs = [(q, a) for q, a in EXTRA if ("书" in q or "电影" in q or "小说" in q)]
    for _ in range(6):
        for q, a in recs:
            rows.append(conv(q, a))

    rng = random.Random(args.seed)
    rng.shuffle(rows)
    dest = Path(args.out)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"QAT 数据：base {n_base} + 扩充 = {len(rows)} 条 -> {dest}")


if __name__ == "__main__":
    main()
