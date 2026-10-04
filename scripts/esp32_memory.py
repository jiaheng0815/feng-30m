"""板端记忆评测：串口跑 24 组「说事实 → 追问」，按是否复述事实判分。

和 scripts/eval_memory.py 用同一套题（build_cases），这样 PC 与板端口径一致。
默认连续对话（--no-reset 语义），每题问完不清上下文。

用法：
    python scripts/esp32_memory.py --port COM20 [--out logs/board_memory_24.txt]
"""
import argparse
import json
import sys
import time
from pathlib import Path

import serial

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import ROOT  # noqa: E402
from eval_memory import build_cases  # noqa: E402
from runtime_tools import set_board_time  # noqa: E402


def read_until(ser, needle, timeout):
    buf, t0 = "", time.time()
    while time.time() - t0 < timeout:
        data = ser.read(4096)
        if data:
            buf += data.decode("utf-8", "replace")
            if needle in buf:
                break
    return buf


def ask(ser, q, wait):
    ser.reset_input_buffer()
    ser.write((q + "\n").encode("utf-8"))
    ser.flush()
    out = read_until(ser, ">>END", wait)
    i = out.find("<<")
    return out[i + 2:out.find(">>END")].strip() if i >= 0 else ""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default="COM20")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--wait", type=float, default=180.0)
    ap.add_argument("--out", default=str(ROOT / "logs" / "board_memory_24.txt"))
    args = ap.parse_args()

    cases = build_cases(n=args.n)
    ser = serial.Serial(args.port, args.baud, timeout=0.5)
    ser.dtr = False
    ser.rts = True
    time.sleep(0.12)
    ser.rts = False
    boot = read_until(ser, "FENG_READY", 30)
    if "FENG_READY" not in boot:
        print("[!] 启动未完成:", boot[-200:])
    synced, src = set_board_time(ser)
    print(f"[时间] 板端对时 {'成功' if synced else '失败'}"
          f"（来源：{'网络(NTP)' if src == 'ntp' else '系统时钟'}）", flush=True)
    read_until(ser, "you> ", 3)

    ok = 0
    lines = []
    for i, c in enumerate(cases, 1):
        ack = ask(ser, c["statement"], args.wait)
        reply = ask(ser, c["ask"], args.wait)
        good = c["value"] in reply
        ok += int(good)
        line = (f"[{i:2d}] {'OK ' if good else 'MISS'} {c['kind']} {c['value']}\n"
                f"     说: {c['statement']}\n     问: {c['ask']}\n     答: {reply[:80]}")
        print(line, flush=True)
        lines.append(line + "\n")
    ser.close()
    summary = f"\n=== 板端记忆 {ok}/{len(cases)} ==="
    print(summary)
    lines.append(summary)
    p = Path(args.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("".join(lines), encoding="utf-8")
    print(f"-> {p}")


if __name__ == "__main__":
    main()
