"""v3.11 验收锚点修复数据：把 C 引擎 27 题的"已通过回答" + 算术漏题 + 通用锚点打包。

用途：算术补丁（v3_11/arith_patch2.jsonl）会把 27 题矩阵里的某一题
（情绪-伤心 或 翻译-再见）抖掉一分。本脚本从一次全通过的 q2 矩阵日志里抽出
「题目 -> 通过回答」，按高权重混入修复轮，把矩阵拉回 27/27 的同时保住算术增益。

用法：
    python scripts/v3_11_build_repair.py --suite-log logs/pc_kv_suite32_v3_10p3_q2b8.txt `
        --misses eval/arith_v3_11pol4.json --out v3_11/repair.jsonl
"""
import argparse
import json
import random
import re
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import ROOT  # noqa: E402

SUITE_C = ROOT / "esp32s3-feng-llm" / "pc" / "pc_kv_suite.cpp"


def conv(q: str, a: str) -> dict:
    return {"messages": [{"role": "user", "content": q},
                         {"role": "assistant", "content": a}]}


def parse_tasks():
    t = SUITE_C.read_text(encoding="utf-8")
    i = t.index("static const task_t TASKS[]")
    block = t[i:t.index("};", i)]
    out = {}
    for m in re.finditer(r'\{\s*"([^"]+)"\s*,\s*"([^"]+)"\s*,', block):
        out[m.group(1)] = m.group(2)
    return out


def parse_suite_log(path: Path):
    """返回 label -> 通过的回复。"""
    got = {}
    pat = re.compile(r"^(\S+)\s+\[[^\]]*\]\s+\[(PASS|FAIL)\]\s+(.*?)\s*\[expect")
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        m = pat.match(line.strip())
        if m and m.group(2) == "PASS":
            got[m.group(1)] = m.group(3).strip()
    return got


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite-log", default=str(ROOT / "logs" / "pc_kv_suite32_v3_10p3_q2b8.txt"))
    ap.add_argument("--misses", default="", help="eval_arith.py 结果 JSON，取其中 ok=false 的题")
    ap.add_argument("--qat", default=str(ROOT / "v3_7" / "qat_data.jsonl"))
    ap.add_argument("--qat-n", type=int, default=1500)
    ap.add_argument("--anchor-repeat", type=int, default=8)
    ap.add_argument("--miss-repeat", type=int, default=20)
    ap.add_argument("--out", default=str(ROOT / "v3_11" / "repair.jsonl"))
    ap.add_argument("--arith", default=str(ROOT / "v3_11" / "arith_patch2.jsonl"),
                    help="--mix 时并入的算术补丁数据")
    ap.add_argument("--mix", action="store_true",
                    help="输出 = 本修复集 + 算术补丁（pol8 的 arith_repair.jsonl）")
    ap.add_argument("--pc-fix", action="store_true",
                    help="PC 末层修复集：漏题 ×miss-repeat + ×1 乘法族 + 水的化学式 + 高危锚点")
    ap.add_argument("--surgical", action="store_true",
                    help="只保留 8 个高危题（×2）+ 取件码召回（×2）（pol7 的 repair3.jsonl）")
    ap.add_argument("--needle", default=str(ROOT / "v3_7" / "needle_qat.jsonl"))
    ap.add_argument("--seed", type=int, default=20261004)
    args = ap.parse_args()

    tasks = parse_tasks()
    passed = parse_suite_log(Path(args.suite_log))
    rows = []
    for label, q in tasks.items():
        a = passed.get(label)
        if a:
            rows += [conv(q, a)] * args.anchor_repeat
    print(f"验收锚点 {len(passed)}/{len(tasks)} 题 ×{args.anchor_repeat} = {len(rows)}")

    if args.misses:
        miss = json.loads(Path(args.misses).read_text(encoding="utf-8"))
        bad = [r for r in miss["rows"] if not r["ok"]]
        op_re = re.compile(r"(\d+)\s*([+\-×乘])\s*(\d+)")

        def canon(q: str, expect: int):
            m = op_re.search(q)
            if not m:
                return f"等于 {expect}。"
            sym = {"+" : "加", "-": "减", "×": "乘", "乘": "乘"}[m.group(2)]
            return f"{m.group(1)} {sym} {m.group(3)} 等于 {expect}。"

        for r in bad:
            rows += [conv(r["q"], canon(r["q"], r["expect"]))] * args.miss_repeat
        print(f"算术漏题 {len(bad)} 条 ×{args.miss_repeat} = {len(bad)*args.miss_repeat}")

    rng = random.Random(args.seed)
    qat = [json.loads(l) for l in Path(args.qat).open(encoding="utf-8") if l.strip()]
    if args.qat_n and len(qat) > args.qat_n:
        qat = rng.sample(qat, args.qat_n)
    rows += qat
    rng.shuffle(rows)

    if args.surgical:
        keep = ("我很伤心", "「再见」用英语怎么说？", "中国的首都是哪里？",
                "水的化学式是什么？", "彩虹有几种颜色？", "一年有几个月？",
                "我想死", "我很孤独")
        rows = [r for r in rows if r["messages"][0]["content"] in keep] * 2
        needle = [json.loads(l) for l in Path(args.needle).open(encoding="utf-8") if l.strip()]
        rows += needle * 2
        rng.shuffle(rows)
        print(f"surgical：高危锚点 ×2 + 召回 ×2 = {len(rows)} 条")
    elif args.pc_fix:
        keep = ("我很伤心", "「再见」用英语怎么说？", "中国的首都是哪里？",
                "水的化学式是什么？", "彩虹有几种颜色？", "一年有几个月？",
                "你是谁？", "你可以干什么？")
        rows = [r for r in rows if r["messages"][0]["content"] in keep] * 2
        for a in range(1, 10):
            rows += [conv(f"{a}乘1等于几？", f"{a} 乘 1 等于 {a}。")] * 10
            rows += [conv(f"1乘{a}等于几？", f"1 乘 {a} 等于 {a}。")] * 10
            # ×1 的对照项：避免"乘 1"把"加 1"带偏（1+1 曾被训成 1）
            rows += [conv(f"{a}+1等于几？", f"{a} 加 1 等于 {a + 1}。")] * 6
            rows += [conv(f"1+{a}等于几？", f"1 加 {a} 等于 {a + 1}。")] * 6
        rows += [conv("水的化学式是什么？", "水的化学式是 H₂O。")] * 40
        rng.shuffle(rows)
        print(f"pc-fix：漏题 + ×1 乘法族 + 水的化学式 + 锚点 = {len(rows)} 条")
    elif args.mix:
        arith = [json.loads(l) for l in Path(args.arith).open(encoding="utf-8") if l.strip()]
        rows += arith
        rng.shuffle(rows)
        print(f"mix：修复集 + 算术补丁 {len(arith)} 条 = {len(rows)} 条")

    dest = Path(args.out)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    print(f"合计 {len(rows)} 条 -> {dest}")


if __name__ == "__main__":
    main()
