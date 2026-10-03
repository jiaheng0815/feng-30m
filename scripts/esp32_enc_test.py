"""Verify the board mirrors the terminal encoding.
Sends one question as GBK and one as UTF-8 on the same connection and decodes each
reply with the encoding that was used for the question.
"""
import sys
import time

import serial

sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def read_until(ser, needle, timeout):
    buf = b""
    t0 = time.time()
    while time.time() - t0 < timeout:
        data = ser.read(4096)
        if data:
            buf += data
            if needle in buf:
                break
    return buf


def main():
    port = sys.argv[1] if len(sys.argv) > 1 else "COM20"
    ser = serial.Serial(port, 115200, timeout=0.5)
    ser.dtr = False
    ser.rts = True
    time.sleep(0.12)
    ser.rts = False
    boot = read_until(ser, b"FENG_READY", 30)
    print("[boot ascii-ok]", b"FENG_READY" in boot, "| non-ascii bytes:", sum(1 for b in boot if b > 127))

    for name, enc in (("GBK", "gbk"), ("UTF-8", "utf-8")):
        q = "你是谁？"
        ser.reset_input_buffer()
        ser.write((q + "\n").encode(enc))
        ser.flush()
        raw = read_until(ser, b">>END", 200)
        idx = raw.find(b"<<")
        end = raw.find(b">>END")
        text = raw[idx + 2:end].decode(enc, "replace").strip() if idx >= 0 and end > idx else ""
        print(f"[{name:5s} in] {q}  ->  {text[:90]}")

    ser.close()


if __name__ == "__main__":
    main()
