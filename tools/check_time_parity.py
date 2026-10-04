"""C 引擎与 Python 的时间 tool 一致性检查（固定 epoch，逐条比对输出）。

Python 侧会比 C 多一个时间来源标注（"，网络时间"/"，系统时间…"），对比前剥掉。

用法：
    gcc -O2 -o pc_time_ask pc/pc_time_ask.c main/feng_tools.c -Imain -lm
    python tools/check_time_parity.py --c-bin pc_time_ask
"""
import argparse
import subprocess
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from runtime_tools import time_answer  # noqa: E402

EPOCHS = [1791095400, 0, 1709222399, 1735675200]
QUESTIONS = [
    "现在几点？", "今天几号", "现在的时间戳是多少？", "今天是星期几",
    "3天后是几号", "明天是几号", "昨天是几号", "前天是几号",
    "现在7点，再过3小时是几点？", "3小时后是几点？", "现在1点，3小时前是几点？",
    "讲个笑话", "1+1",
]


def py_answer(q, epoch):
    out = time_answer(q, epoch=float(epoch), source="system")
    if out is None:
        return None
    return out.replace("，系统时间（未取到网络时间）", "").replace("，网络时间", "")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--c-bin", required=True, help="pc_time_ask 可执行文件路径")
    args = ap.parse_args()
    lines = []
    for epoch in EPOCHS:
        lines.append(f"@{epoch}")
        lines.extend(QUESTIONS)
    payload = ("\n".join(lines) + "\n").encode("utf-8")
    r = subprocess.run([args.c_bin], input=payload, capture_output=True)
    c_lines = r.stdout.decode("utf-8", "replace").splitlines()
    if len(c_lines) != len(lines):
        print(f"C 输出 {len(c_lines)} 行 != 脚本 {len(lines)} 行")
        return 1
    bad = 0
    total = 0
    epoch = None
    for sent, c_out in zip(lines, c_lines):
        if sent.startswith("@"):
            epoch = int(sent[1:])
            continue
        total += 1
        py = py_answer(sent, epoch)
        c_ans = None if c_out == "NONE" else c_out
        if c_ans != py:
            bad += 1
            print(f"[FAIL] epoch={epoch} {sent!r}\n   C     : {c_ans!r}\n   Python: {py!r}")
    print(f"时间一致性：{total - bad}/{total} 条一致"
          + ("" if bad == 0 else f"，{bad} 条不一致"))
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
