"""v3.11 算术边界补丁：补齐 v3.6 drill 没覆盖的三类情况。

现有 drill（scripts/v3_6_build_drill.py）只枚举了：
  - 加法 1..9 × 1..9（没有 0 操作数）
  - 减法 a-b 只取结果 >= 1（没有 a==b 的 0，也没有负数）
所以板端会出现 `7-7=1`、`1+0=0` 这类错误（eval_arith.py 实测）。

本脚本生成完整网格并给边界情况加权：
  - 0 参与加法（0+b / a+0 / 0+0）
  - 减法结果 = 0（a==b）
  - 减法结果 < 0（a<b，答案为 -k）
再混入日常补丁与旧 drill 采样做锚点，防止低 lr 微调把别的能力带跑。

用法：
    python scripts/v3_11_build_arith_patch.py --out archive/v3_11/arith_patch.jsonl
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


def math_rows():
    """完整 0..9 网格：每个问法都给出「a 运算符 b 等于 c。」句式。"""
    rows, boundary = [], []
    for a in range(0, 10):
        for b in range(0, 10):
            ans = a + b
            forms = [(f"{a}+{b}等于几？", f"{a} 加 {b} 等于 {ans}。"),
                     (f"{a}加{b}等于几？", f"{a} 加 {b} 等于 {ans}。"),
                     (f"{a} 加 {b} 等于几？", f"等于 {ans}。")]
            for q, a_ in forms:
                rows.append((q, a_))
                if a == 0 or b == 0:
                    boundary.append((q, a_))
    for a in range(0, 10):
        for b in range(0, 10):
            ans = a - b
            forms = [(f"{a}-{b}等于几？", f"{a} 减 {b} 等于 {ans}。"),
                     (f"{a}减{b}等于几？", f"{a} 减 {b} 等于 {ans}。"),
                     (f"{a} 减 {b} 等于几？", f"等于 {ans}。")]
            for q, a_ in forms:
                rows.append((q, a_))
                if a <= b:                      # 结果 0 或负数：旧 drill 完全没训过
                    boundary.append((q, a_))
    for a in range(1, 10):
        for b in range(1, 10):
            ans = a * b
            rows.append((f"{a}乘{b}等于几？", f"{a} 乘 {b} 等于 {ans}。"))
            rows.append((f"{a} 乘 {b} 等于几？", f"等于 {ans}。"))
    return rows, boundary


def load_jsonl(path: Path, limit: int, rng: random.Random):
    if not path.exists():
        return []
    rows = [json.loads(l) for l in path.open(encoding="utf-8") if l.strip()]
    if limit and len(rows) > limit:
        rows = rng.sample(rows, limit)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "archive" / "v3_11" / "arith_patch.jsonl"))
    ap.add_argument("--repeat", type=int, default=3, help="完整网格重复次数")
    ap.add_argument("--boundary-extra", type=int, default=4, help="边界情况额外重复次数")
    ap.add_argument("--base", default=str(ROOT / "archive" / "v3_6a" / "daily_patch.jsonl"),
                    help="日常补丁锚点")
    ap.add_argument("--drill", default=str(ROOT / "archive" / "v3_6e" / "drill.jsonl"),
                    help="旧 drill 采样锚点（保住已学会的非负减法/乘法）")
    ap.add_argument("--drill-limit", type=int, default=1200)
    ap.add_argument("--keep-emotion-anchors", action="store_true",
                    help="保留锚点里的情绪样本（默认过滤：避免低 lr 微调改动已验收的情绪句式）")
    ap.add_argument("--seed", type=int, default=20261004)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    rows, boundary = math_rows()
    out = []
    for _ in range(args.repeat):
        out += [conv(q, a) for q, a in rows]
    for _ in range(args.boundary_extra):
        out += [conv(q, a) for q, a in boundary]
    anchors = load_jsonl(Path(args.base), 0, rng)
    drill = load_jsonl(Path(args.drill), args.drill_limit, rng)
    if not args.keep_emotion_anchors:
        emo = ("伤心", "痛苦", "想死", "很快乐", "很高兴", "孤独", "压力", "难过")
        before = len(anchors) + len(drill)

        def keep(r):
            q = r["messages"][0]["content"]
            return not any(w in q for w in emo)

        anchors = [r for r in anchors if keep(r)]
        drill = [r for r in drill if keep(r)]
        print(f"过滤情绪锚点 {before - len(anchors) - len(drill)} 条")
    out += anchors + drill
    rng.shuffle(out)

    dest = Path(args.out)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"算术网格 {len(rows)} 条 ×{args.repeat} + 边界 ×{args.boundary_extra}（{len(boundary)} 条）"
          f" + 日常锚点 {len(anchors)} + 旧 drill {len(drill)} = {len(out)} 条 -> {dest}")


if __name__ == "__main__":
    main()
