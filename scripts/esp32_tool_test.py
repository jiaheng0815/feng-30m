"""板端 tool 专项验收：网络时间(UTC+8) + 随机数/骰子/硬币 + 算式外壳。

板子没有 RTC/WiFi 协议栈，脚本先走 NTP 取网络时间戳，再发 `\\settime` 推进固件
（固件用 esp_timer 走时）；随后逐题检查板端 `<<` 回复：正则匹配 + 随机数范围 +
板端时间与网络时间的一致性（NTP 与板端时刻差）。

用法：
    python scripts/esp32_tool_test.py --port COM20 [--out logs/board_tools_time_rand.txt]
"""
import argparse
import re
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import serial

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import ROOT  # noqa: E402
from runtime_tools import ntp_epoch, set_board_time  # noqa: E402

# (问句, 回复正则, 需要范围校验的整数取值区间)
CASES = [
    ("现在几点？", r"现在是 \d{4}年\d{2}月\d{2}日 \d{2}:\d{2}:\d{2}（周[一二三四五六日]，UTC\+8）。", None),
    ("今天是几号", r"现在是 \d{4}年\d{2}月\d{2}日 \d{2}:\d{2}:\d{2}（周[一二三四五六日]，UTC\+8）。", None),
    ("3天后是几号", r"3 天后是 \d{4}年\d{2}月\d{2}日（周[一二三四五六日]）。", None),
    ("明天是几号", r"1 天后是 \d{4}年\d{2}月\d{2}日（周[一二三四五六日]）。", None),
    ("昨天是几号", r"1 天前是 \d{4}年\d{2}月\d{2}日（周[一二三四五六日]）。", None),
    ("给我个1到100的随机数", r"随机数（1~100）：(\d+)。", (1, 100)),
    ("随机 0-9", r"随机数（0~9）：(\d+)。", (0, 9)),
    ("掷骰子", r"掷骰子：(\d+) 点。", (1, 6)),
    ("抛硬币", r"抛硬币：(正面|反面)。", None),
    ("59+1", r"59 加 1 等于 60。", None),
    ("把59+1算一下", r"59 加 1 等于 60。", None),
]


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
    t0 = time.time()
    out = read_until(ser, ">>END", wait)
    dt = time.time() - t0
    i = out.find("<<")
    return (out[i + 2:out.find(">>END")].strip() if i >= 0 else ""), dt


def parse_board_time(reply):
    m = re.search(r"(\d{4})年(\d{2})月(\d{2})日 (\d{2}):(\d{2}):(\d{2})", reply)
    if not m:
        return None
    y, mo, d, h, mi, s = (int(x) for x in m.groups())
    return datetime(y, mo, d, h, mi, s, tzinfo=timezone(timedelta(hours=8)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default="COM20")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--wait", type=float, default=30.0)
    ap.add_argument("--out", default=str(ROOT / "logs" / "board_tools_time_rand.txt"))
    args = ap.parse_args()

    ser = serial.Serial(args.port, args.baud, timeout=0.5)
    ser.dtr = False
    ser.rts = True
    time.sleep(0.12)
    ser.rts = False
    boot = read_until(ser, "FENG_READY", 30)
    if "FENG_READY" not in boot:
        print("[!] 启动未完成:", boot[-200:])
    synced, src = set_board_time(ser)
    host_epoch, host_src = ntp_epoch()
    print(f"[时间] 板端对时 {'成功' if synced else '失败'}"
          f"（来源：{'网络(NTP)' if src == 'ntp' else '系统时钟'}）", flush=True)
    read_until(ser, "you> ", 3)

    lines = [f"[时间] 板端对时 {'成功' if synced else '失败'}"
             f"（来源：{'网络(NTP)' if src == 'ntp' else '系统时钟'}）\n"]
    ok = total = 0
    first_time_reply = ""

    for q, pattern, rng in CASES:
        total += 1
        reply, dt = ask(ser, q, args.wait)
        m = re.search(pattern, reply)
        good = bool(m)
        note = ""
        if good and rng is not None:
            v = int(m.group(1))
            note = f"值 {v} {'在' if rng[0] <= v <= rng[1] else '不在'} {rng[0]}~{rng[1]}"
            good = rng[0] <= v <= rng[1]
        ok += int(good)
        first_time_reply = reply if q == "现在几点？" else first_time_reply
        line = f"[{'OK ' if good else 'MISS'}] {dt:5.1f}s  Q: {q}\n      A: {reply[:90]}\n"
        if note:
            line = line.rstrip("\n") + f"（{note}）\n"
        print(line, end="", flush=True)
        lines.append(line)

    # 随机数随时间变化：连问 3 次，3 个值不应完全相同（seed 来自运行时间）
    total += 1
    vals = []
    for _ in range(3):
        reply, _ = ask(ser, "给我个1到100的随机数", args.wait)
        m = re.search(r"随机数（1~100）：(\d+)。", reply)
        vals.append(int(m.group(1)) if m else None)
    good = all(v is not None and 1 <= v <= 100 for v in vals) and len(set(vals)) >= 2
    ok += int(good)
    line = (f"[{'OK ' if good else 'MISS'}] 随机数随时间变化: 连续 3 次 -> {vals}\n")
    print(line, end="", flush=True)
    lines.append(line)

    # 板端时间 vs 网络时间（NTP 取样后再比对，允许 300s 容差）
    if first_time_reply:
        total += 1
        board_dt = parse_board_time(first_time_reply)
        diff = (board_dt.timestamp() - host_epoch) if board_dt else None
        good = diff is not None and abs(diff) <= 300
        ok += int(good)
        line = (f"[{'OK ' if good else 'MISS'}] 板端时间与网络时间一致"
                f"（板端 {board_dt.strftime('%Y-%m-%d %H:%M:%S') if board_dt else '?'}"
                f" vs 宿主 NTP，差 {diff:+.0f}s）\n" if diff is not None else
                "[MISS] 板端时间与网络时间一致（未能解析板端时间）\n")
        print(line, end="", flush=True)
        lines.append(line)

    ser.close()
    summary = f"\n=== 板端 tool 专项（时间/随机数/算式）{ok} 成功 / {total - ok} 失败（共 {total} 轮）==="
    print(summary)
    lines.append(summary)
    p = Path(args.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("".join(lines), encoding="utf-8")
    print(f"-> {p}")


if __name__ == "__main__":
    main()
