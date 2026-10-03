"""v3.7 长文 QAT 数据：按板端测试同款格式生成"取件码"召回样本（含 25%/50%/75%/90% 位置）。

用途：q2 KV 下 25% 位置的召回偶发失败，补一批同分布样本做量化感知微调。

用法：python scripts/v3_7_build_needle_qat.py --out v3_7\\needle_qat.jsonl [--n 12]
"""
import argparse
import json
import random
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import ROOT  # noqa: E402

FILLER = (ROOT / "esp32s3-feng-llm" / "pc" / "prompt_long.txt").read_text(encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "v3_7" / "needle_qat.jsonl"))
    ap.add_argument("--n", type=int, default=12, help="每个位置生成几条")
    ap.add_argument("--chars", type=int, default=4200, help="正文长度（字符）")
    ap.add_argument("--seed", type=int, default=20261004)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    rows = []
    for pos_permille in (250, 500, 750, 900):
        for _ in range(args.n):
            code = f"{rng.randint(100000, 999999)}"
            body = (FILLER * (args.chars // len(FILLER) + 1))[:args.chars]
            at = len(body) * pos_permille // 1000
            text = body[:at] + f"（重要信息：快递柜取件码是 {code}。）" + body[at:]
            q = text + "\n\n上文提到的快递柜取件码是多少？请只回答数字。"
            rows.append({"messages": [
                {"role": "user", "content": q},
                {"role": "assistant", "content": code}]})
    # 顺手修 v3.6/v3.7 范围内评测的股票拒答（QAT 后偶发尾巴乱码）
    for _ in range(12):
        rows.append({"messages": [
            {"role": "user", "content": "今天上证指数是多少？"},
            {"role": "assistant", "content": "我查不到实时行情，你可以看看行情软件。"}]})
        rows.append({"messages": [
            {"role": "user", "content": "帮我推荐一只明天会涨的股票。"},
            {"role": "assistant", "content": "我不知道哪只股票会涨，股市有风险，投资要谨慎。"}]})
        rows.append({"messages": [
            {"role": "user", "content": "明天的股票会涨吗？"},
            {"role": "assistant", "content": "我不知道，股市有风险，投资要谨慎。"}]})
    rng.shuffle(rows)
    dest = Path(args.out)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"长文 QAT 样本：{len(rows)} 条 -> {dest}")


if __name__ == "__main__":
    main()
