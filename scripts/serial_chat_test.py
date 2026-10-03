"""Talk to the ESP32-S3 over the serial port: boot log + one question."""
import sys
import time

import serial

sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def main():
    port = sys.argv[1] if len(sys.argv) > 1 else "COM20"
    baud = int(sys.argv[2]) if len(sys.argv) > 2 else 921600
    question = sys.argv[3] if len(sys.argv) > 3 else "你好"
    read_secs = float(sys.argv[4]) if len(sys.argv) > 4 else 40.0

    ser = serial.Serial(port, baud, timeout=1)
    print(f"listening on {port} @ {baud} for {read_secs:.0f}s ...", flush=True)
    t0 = time.time()
    buf = ""
    while time.time() - t0 < read_secs:
        data = ser.read(4096)
        if data:
            text = data.decode("utf-8", "replace")
            buf += text
            print(text, end="", flush=True)
    if "FENG_READY" not in buf and "you>" not in buf:
        print("\n[!] no prompt seen yet — device may still be booting")
    print(f"\n>>> sending: {question}")
    ser.write((question + "\n").encode("utf-8"))
    ser.flush()
    t0 = time.time()
    got_end = False
    while time.time() - t0 < 180 and not got_end:
        data = ser.read(4096)
        if data:
            text = data.decode("utf-8", "replace")
            print(text, end="", flush=True)
            if ">>END" in text:
                got_end = True
    ser.close()
    print("\n[done]")


if __name__ == "__main__":
    main()
