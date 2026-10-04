"""Reset the ESP32-S3 through COM20, wait for FENG_READY, send a line, read the streamed reply."""
import argparse
import sys
import time
from pathlib import Path

import serial

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).resolve().parent))
from runtime_tools import set_board_time  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default="COM20")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--question", default="你好")
    ap.add_argument("--boot-wait", type=float, default=25.0)
    ap.add_argument("--reply-wait", type=float, default=240.0)
    ap.add_argument("--no-reset", action="store_true")
    args = ap.parse_args()

    ser = serial.Serial(args.port, args.baud, timeout=0.5)
    if not args.no_reset:
        # ESP32-S3 auto-reset sequence (EN pulse via DTR/RTS)
        ser.dtr = False
        ser.rts = True
        time.sleep(0.12)
        ser.rts = False
        time.sleep(0.05)
        print("[reset] EN pulse sent", flush=True)
    else:
        print("[reset] skipped", flush=True)

    buf = ""
    t0 = time.time()
    print(f"--- boot log ({args.port} @ {args.baud}) ---", flush=True)
    while time.time() - t0 < args.boot_wait:
        data = ser.read(4096)
        if data:
            text = data.decode("utf-8", "replace")
            buf += text
            print(text, end="", flush=True)
            if "you>" in buf or "FENG_READY" in buf:
                time.sleep(0.5)
                break
    synced, src = set_board_time(ser)
    print(f"[时间] 板端对时 {'成功' if synced else '失败'}"
          f"（来源：{'网络(NTP)' if src == 'ntp' else '系统时钟'}）", flush=True)

    print(f"\n>>> 发送: {args.question}", flush=True)
    ser.write((args.question + "\n").encode("utf-8"))
    ser.flush()

    t0 = time.time()
    reply = ""
    print("--- 模型输出（流式）---", flush=True)
    while time.time() - t0 < args.reply_wait:
        data = ser.read(4096)
        if data:
            text = data.decode("utf-8", "replace")
            reply += text
            print(text, end="", flush=True)
            if ">>END" in reply:
                # 板端在 >>END 之后才打印 tok/s 统计行（多轮记账会再花一点时间），
                # 多读 1.5 s 把它带上，方便记录速度。
                t1 = time.time()
                while time.time() - t1 < 1.5:
                    tail = ser.read(4096)
                    if not tail:
                        continue
                    text = tail.decode("utf-8", "replace")
                    reply += text
                    print(text, end="", flush=True)
                break
    ser.close()
    ok = ">>END" in reply
    print(f"\n--- 结果: {'成功收到完整回复' if ok else '超时/未收到完整回复'} ---")
    idx = reply.find("<<")
    if idx >= 0:
        seg = reply[idx + 2:reply.find(">>END")].strip()
        print(f"回复内容: {seg[:200]}")
        print(f"回复字数: {len(seg)}")


if __name__ == "__main__":
    main()
