"""运行时 tool（Python 版）：算式 / 网络时间(UTC+8) / 随机数，口径与 C 引擎一致。

- 算式：复用 calc_tool（与 feng_calc.c 同规则）
- 时间：走 SNTP（真·网络时间戳）转 UTC+8；取不到时回退系统时钟并注明
- 随机数：seed = 运行时间(秒) × 1.54 × 1000，丢弃第一个随机数后取第二个

串口脚本用 set_board_time() 把网络时间推给板子（板端放不下 WiFi 协议栈）。
"""
import random
import re
import socket
import struct
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from calc_tool import calc_answer, extract  # noqa: E402,F401

_START = time.perf_counter()
_NTP_HOSTS = ("ntp.aliyun.com", "ntp.tencent.com", "pool.ntp.org")
_TIME_KW = ("几点", "现在时间", "现在的时间", "当前时间", "时间是多少", "什么时间",
            "今天几号", "今天几月", "日期", "星期几", "时间戳")


def ntp_epoch(timeout=1.5):
    """SNTP 查询网络时间戳；失败回退系统时钟。返回 (epoch, 'ntp'|'system')。"""
    for host in _NTP_HOSTS:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
                s.settimeout(timeout)
                pkt = b"\x1b" + 47 * b"\0"
                s.sendto(pkt, (host, 123))
                data, _ = s.recvfrom(48)
            if len(data) >= 48:
                secs = struct.unpack("!I", data[40:44])[0] - 2208988800
                if secs > 1600000000:
                    return float(secs), "ntp"
        except Exception:
            continue
    return time.time(), "system"


def time_answer(user, epoch=None, source=None):
    if not any(k in user for k in _TIME_KW):
        return None
    if epoch is None:
        epoch, source = ntp_epoch()
    t = datetime.fromtimestamp(epoch, tz=timezone(timedelta(hours=8)))
    week = "一二三四五六日"[t.weekday()]
    tag = "网络时间" if source == "ntp" else "系统时间（未取到网络时间）"
    return f"现在是 {t.strftime('%Y年%m月%d日 %H:%M:%S')}（周{week}，UTC+8，{tag}）。"


def random_answer(user):
    if "随机" not in user:
        return None
    nums = [int(x) for x in re.findall(r"\d+", user)]
    lo, hi = (nums[0], nums[1]) if len(nums) >= 2 and nums[0] != nums[1] else (1, 100)
    if hi < lo:
        lo, hi = hi, lo
    seed = int((time.perf_counter() - _START) * 1.54 * 1000)
    rng = random.Random(seed)
    rng.random()                                   # 第 1 个按需求丢弃
    v = rng.randint(lo, hi)                        # 第 2 个随机数
    return f"随机数（{lo}~{hi}）：{v}。"


def tool_answer(user, epoch=None, source=None):
    """按 算式 → 时间 → 随机数 的顺序尝试；都不是则返回 None。"""
    for fn in (lambda: calc_answer(user),
               lambda: time_answer(user, epoch, source),
               lambda: random_answer(user)):
        r = fn()
        if r is not None:
            return r
    return None


def set_board_time(ser, timeout=3.0):
    """给板端发 \\settime（板子重启后调用一次）。返回是否收到确认。"""
    epoch, src = ntp_epoch()
    ser.reset_input_buffer()
    ser.write(f"\\settime {int(epoch)}\n".encode())
    ser.flush()
    t0 = time.time()
    buf = ""
    while time.time() - t0 < timeout:
        data = ser.read(256)
        if data:
            buf += data.decode("utf-8", "replace")
            if "time synced" in buf:
                return True, src
    return False, src


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    e, s = ntp_epoch()
    print(f"ntp_epoch -> {e:.0f} ({s})")
    for q in ["59+1", "4854+4411", "现在几点？", "今天几号", "给我个1到100的随机数",
              "随机 0-9", "你好"]:
        print(f"{q!r:18s} -> {tool_answer(q, e, s)}")
