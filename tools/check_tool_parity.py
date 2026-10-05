"""C++ 引擎与 Python 工具的口径一致性检查。

把同一批提示词分别喂给：
  - C++：`pc/pc_calc_ask.cpp` 驱动 + `main/feng_calc.cpp`（stdin 一行一题，输出回答或 NONE）
  - Python：`scripts/calc_tool.py`
逐条比较，输出不一致就退出码 1。CI 里编译好 C 驱动并传 --c-bin。

用法：
    g++ -std=c++23 -fno-exceptions -fno-rtti -fno-threadsafe-statics -O2 -o pc_calc_ask pc/pc_calc_ask.cpp main/feng_calc.cpp -Imain -lm
    python tools/check_tool_parity.py --c-bin pc_calc_ask
"""
import argparse
import subprocess
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from calc_tool import calc_answer  # noqa: E402

CASES = [
    # 基础算术
    "59+1", "4854+4411", "5.3+4.1", "10+4.", "(3+4)*2", "36/6", "1/0",
    # 中文数字 / 百分号 / 平方 / 根号
    "五十九加一", "一百零五加二十", "十五乘以四", "100的15%", "一百*15%", "15%+5%", "12的平方", "根号16",
    # 外壳
    "把59+1算一下", "麻烦算一下 445+15 是多少", "59+1等于几？",
    # 序列数数（v3.20）
    "把 1 到 5 倒着数一遍。", "从 3 数到 8。", "把十到十五倒着数。", "把 8 到 3 倒着数",
    "把 1 到 100 倒着数。",
    # 不该被 tool 拦截
    "你好", "我2-3点有空", "今天25度", "加起来", "数到 8", "我从1数到100也数不完",
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--c-bin", required=True, help="pc_calc_ask 可执行文件路径")
    args = ap.parse_args()
    payload = ("\n".join(CASES) + "\n").encode("utf-8")
    r = subprocess.run([args.c_bin], input=payload, capture_output=True)
    c_lines = r.stdout.decode("utf-8", "replace").splitlines()
    if len(c_lines) != len(CASES):
        print(f"C 驱动输出行数 {len(c_lines)} != 用例数 {len(CASES)}（stderr: "
              f"{r.stderr.decode('utf-8', 'replace')[:200]}）")
        return 1
    bad = 0
    for q, c_out in zip(CASES, c_lines):
        c_ans = None if c_out == "NONE" else c_out
        py_ans = calc_answer(q)
        if c_ans != py_ans:
            bad += 1
            print(f"[FAIL] {q!r}\n   C     : {c_ans!r}\n   Python: {py_ans!r}")
    print(f"一致性检查：{len(CASES) - bad}/{len(CASES)} 条一致"
          + ("" if bad == 0 else f"，{bad} 条不一致"))
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
