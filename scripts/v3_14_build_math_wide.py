"""v3.14 多位数算术数据：两位数穷举 + 三位数采样 + 标点/格式变体。

背景：v3.13 的算术只覆盖 0..9 网格，用户实测 `59+1`、`445+15` 全错
（`eval_arith.py --wide` 基线 0/164）。本脚本按 tokenizer 的实际切分
（两位数是单 token）生成可学的训练分布。

用法：
    python scripts/v3_14_build_math_wide.py --out archive/v3_14/math_wide.jsonl
"""
import argparse
import json
import random
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import ROOT  # noqa: E402


def conv(q: str, a: str) -> dict:
    return {"messages": [{"role": "user", "content": q},
                         {"role": "assistant", "content": a}]}


def rows_for(a: int, b: int, op: str, forms):
    if op == "+":
        ans, sym = a + b, "加"
    elif op == "-":
        ans, sym = a - b, "减"
    else:
        ans, sym = a * b, "乘"
    out = []
    for q_fn, a_fn in forms:
        out.append(conv(q_fn(a, b), a_fn(a, b, ans, sym)))
    return out


ADD_FORMS = [(lambda a, b: f"{a}+{b}等于几？", lambda a, b, ans, sym: f"{a} 加 {b} 等于 {ans}。"),
             (lambda a, b: f"{a}加{b}等于几？", lambda a, b, ans, sym: f"{a} 加 {b} 等于 {ans}。"),
             (lambda a, b: f"{a} 加 {b} 等于几？", lambda a, b, ans, sym: f"等于 {ans}。")]
SUB_FORMS = [(lambda a, b: f"{a}-{b}等于几？", lambda a, b, ans, sym: f"{a} 减 {b} 等于 {ans}。"),
             (lambda a, b: f"{a}减{b}等于几？", lambda a, b, ans, sym: f"{a} 减 {b} 等于 {ans}。"),
             (lambda a, b: f"{a} 减 {b} 等于几？", lambda a, b, ans, sym: f"等于 {ans}。")]
MUL_FORMS = [(lambda a, b: f"{a}乘{b}等于几？", lambda a, b, ans, sym: f"{a} 乘 {b} 等于 {ans}。")]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "archive" / "v3_14" / "math_wide.jsonl"))
    ap.add_argument("--three-digit", type=int, default=4000, help="三位数采样条数（加减各一半）")
    ap.add_argument("--decimal", type=int, default=1500, help="一位小数加法条数")
    ap.add_argument("--seed", type=int, default=20261006)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    rows = []
    # 两位数 + 两位数（穷举，含进位）
    for a in range(10, 100):
        for b in range(10, 100):
            rows += rows_for(a, b, "+", ADD_FORMS)
    # 两位数 + 一位数
    for a in range(10, 100):
        for b in range(1, 10):
            rows += rows_for(a, b, "+", ADD_FORMS[:2])
    # 两位数减法（穷举；负数留一部分负样本）
    for a in range(10, 100):
        for b in range(10, 100):
            if a >= b or (a + b) % 7 == 0:      # 大部分非负，少量负数
                rows += rows_for(a, b, "-", SUB_FORMS)
    for a in range(10, 100):
        for b in range(1, 10):
            rows += rows_for(a, b, "-", SUB_FORMS[:2])
    # 两位数 × 一位数
    for a in range(10, 100):
        for b in range(2, 10):
            rows += rows_for(a, b, "*", MUL_FORMS)
    # 三位数采样（加减）
    for _ in range(args.three_digit // 2):
        a, b = rng.randint(100, 999), rng.randint(10, 999)
        rows += rows_for(a, b, "+", ADD_FORMS[:2])
        rows += rows_for(max(a, b), min(a, b), "-", SUB_FORMS[:2])
    # 一位小数加法（x.y + u.v，一位小数）
    for _ in range(args.decimal):
        a = rng.randint(1, 9) * 10 + rng.randint(1, 9)
        b = rng.randint(1, 9) * 10 + rng.randint(1, 9)
        av, bv = a / 10, b / 10
        ans = round(av + bv, 1)
        rows.append(conv(f"{av}+{bv}等于几？", f"{av} 加 {bv} 等于 {ans}。"))
    # 标点/口播变体：用户真会这么打字
    for a, b in [(59, 1), (84, 6), (77, 3), (10, 4), (25, 8), (12, 9), (445, 15), (99, 1)]:
        ans = a + b
        rows += [conv(f"{a}+{b}", f"{a} 加 {b} 等于 {ans}。"),
                 conv(f"{a}+{b}.", f"{a} 加 {b} 等于 {ans}。"),
                 conv(f"{a} + {b} 等于几？", f"等于 {ans}。"),
                 conv(f"{a}加{b}等于几", f"{a} 加 {b} 等于 {ans}。")]

    rng.shuffle(rows)
    dest = Path(args.out)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"多位数算术 {len(rows)} 条（含三位数采样 {args.three_digit}、小数 {args.decimal}）-> {dest}")


if __name__ == "__main__":
    main()
