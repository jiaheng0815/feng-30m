"""Multi-turn stability test over the serial chat protocol.

Resets the board once, then sends every question from --questions (or a built-in
list) and waits for the << ... >>END stream of each reply.  Flags crashes
(panic / abort / Guru) or timeouts, so a 120 MHz flash/PSRAM configuration can be
validated by actually running for a while instead of one single token.

默认每道题前发 \\reset（把题目当独立探针，和历次记录可比）；
加 --no-reset 则保留上下文，用来测真正的多轮对话。
"""
import argparse
import sys
import time
from pathlib import Path

import serial

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import ROOT  # noqa: E402

DEFAULT_QUESTIONS = [
    "你好",
    "你是谁？",
    "你是谁开发的？",
    "讲个笑话",
    "中国的首都是哪里？",
    "1+1等于几？",
    "把'今天天气很好'翻译成英文",
    "什么是人工智能？",
    "推荐一本好书",
    "谢谢，再见",
]


def read_until(ser, needle, timeout):
    buf = ""
    t0 = time.time()
    while time.time() - t0 < timeout:
        data = ser.read(4096)
        if data:
            buf += data.decode("utf-8", "replace")
            if needle in buf:
                break
    return buf, time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default="COM20")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--questions", default=None, help="'|'-separated list")
    ap.add_argument("--reply-wait", type=float, default=300.0)
    ap.add_argument("--out", default=str(ROOT / "logs/esp32_multi.txt"))
    ap.add_argument("--no-reset", action="store_true",
                    help="保留上下文连续对话（默认每题前 \\reset，保持探针相互独立）")
    args = ap.parse_args()

    questions = args.questions.split("|") if args.questions else DEFAULT_QUESTIONS
    ser = serial.Serial(args.port, args.baud, timeout=0.5)
    ser.dtr = False
    ser.rts = True
    time.sleep(0.12)
    ser.rts = False

    log = []
    boot, _ = read_until(ser, "FENG_READY", 30)
    log.append(boot)
    if "FENG_READY" not in boot:
        print("[!] 启动未完成:", boot[-300:])

    ok = fail = 0
    for i, q in enumerate(questions, 1):
        if not args.no_reset:
            ser.reset_input_buffer()
            ser.write(b"\\reset\n")
            ser.flush()
            read_until(ser, "context cleared", 5)
        t0 = time.time()
        ser.reset_input_buffer()
        ser.write((q + "\n").encode("utf-8"))
        ser.flush()
        reply, _ = read_until(ser, ">>END", args.reply_wait)
        dt = time.time() - t0
        crash = any(k in reply for k in ("Guru Meditation", "abort() was called", "CPU halted"))
        idx = reply.find("<<")
        text = reply[idx + 2:reply.find(">>END")].strip() if idx >= 0 else ""
        status = "OK" if (">>END" in reply and not crash) else ("CRASH" if crash else "TIMEOUT")
        if status == "OK":
            ok += 1
        else:
            fail += 1
        line = f"[{i:2d}] {status:7s} {dt:5.1f}s  Q: {q}\n     A: {text[:120]}"
        print(line, flush=True)
        log.append(line + "\n")

    ser.close()
    summary = f"\n=== {ok} 成功 / {fail} 失败 (共 {len(questions)} 轮) ==="
    print(summary)
    log.append(summary)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write("".join(log))


if __name__ == "__main__":
    main()
