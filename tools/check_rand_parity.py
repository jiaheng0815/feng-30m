"""C++ 引擎与 Python 的随机数 tool 逐值一致性检查（同 seed 必须得到同一个数）。

用法：
    g++ -std=c++23 -fno-exceptions -fno-rtti -fno-threadsafe-statics -O2 -o pc_rand_ask pc/pc_rand_ask.cpp main/feng_tools.cpp -Imain -lm
    python tools/check_rand_parity.py --c-bin pc_rand_ask
"""
import argparse
import subprocess
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from runtime_tools import rand_range  # noqa: E402

SEEDS = [1, 7, 1540, 88172645463325252, 1791095400123, 3735928559, 2 ** 63 + 5, 0]
RANGES = [(1, 100), (0, 9), (1, 6), (-5, 5), (7, 7), (100, 1), (0, 0)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--c-bin", required=True, help="pc_rand_ask 可执行文件路径")
    args = ap.parse_args()
    cases = [(s, lo, hi) for s in SEEDS for lo, hi in RANGES]
    payload = "\n".join(f"{s} {lo} {hi}" for s, lo, hi in cases) + "\n"
    r = subprocess.run([args.c_bin], input=payload.encode("utf-8"), capture_output=True)
    c_lines = r.stdout.decode("utf-8", "replace").split()
    if len(c_lines) != len(cases):
        print(f"C 输出 {len(c_lines)} 行 != 用例 {len(cases)} 条")
        return 1
    bad = 0
    for (seed, lo, hi), c_out in zip(cases, c_lines):
        py = rand_range(seed, lo, hi)
        if str(py) != c_out:
            bad += 1
            print(f"[FAIL] seed={seed} [{lo},{hi}] C={c_out} Python={py}")
    print(f"随机数一致性：{len(cases) - bad}/{len(cases)} 条一致"
          + ("" if bad == 0 else f"，{bad} 条不一致"))
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
