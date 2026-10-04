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
from calc_tool import calc_answer, cn_to_ascii, extract  # noqa: E402,F401

_START = time.perf_counter()
_NTP_HOSTS = ("ntp.aliyun.com", "ntp.tencent.com", "pool.ntp.org")
_TIME_KW = ("几点", "现在时间", "现在的时间", "当前时间", "时间是多少", "什么时间",
            "今天几号", "今天几月", "几号", "几月", "日期", "星期几", "时间戳",
            "明天", "后天", "昨天", "前天")

# ---- 引擎侧记忆（与 C 版 feng_memory.c 同口径）----
_MEM = {}          # 通用键值槽："颜色"/"书"/"生日"…；空串键 = 无名偏好（"我最喜欢X"）
_MEM_SPECIAL = {}  # 名字/城市/宠物（专用槽，问法固定）
_QUESTION_PREFIX = ("什么", "啥", "哪", "几", "谁", "多少", "吗", "呢", "怎么")


def mem_clear():
    _MEM.clear()


def _mem_take(text, kw):
    i = text.find(kw)
    if i < 0:
        return None
    rest = text[i + len(kw):].lstrip(" \t：:")
    val = re.split(r"[。，、！？；,.!?;\n\r]", rest, maxsplit=1)[0]
    prev = None
    while prev != val:                       # 逐层去掉尾部虚词（与 C 版一致）
        prev = val
        for tail in ("请记住", "一下", "了", "的", "呀", "啊", "哦", "吧", "嘛", "呢"):
            if val.endswith(tail):
                val = val[: -len(tail)]
    val = val.strip()
    if not val or val.startswith(_QUESTION_PREFIX):
        return None
    return val


def mem_learn(user):
    changed = False
    def put(k, v):
        nonlocal changed
        if v is not None and _MEM.get(k) != v:
            _MEM[k] = v
            changed = True
    # 专用槽
    for kw in ("我的名字是", "我叫"):
        v = _mem_take(user, kw)
        if v is not None:
            _MEM_SPECIAL["name"] = v
            changed = True
            break
    for kw in ("住在", "搬到"):
        v = _mem_take(user, kw)
        if v is not None:
            _MEM_SPECIAL["city"] = v
            changed = True
            break
    for kw in ("养了一只", "养的是", "养了"):
        v = _mem_take(user, kw)
        if v is not None:
            _MEM_SPECIAL["pet"] = v
            changed = True
            break
    m_pet = re.search(r"我的宠物(?:是|叫)([^，。！？\s]+)", user)
    if m_pet and not m_pet.group(1).startswith(_QUESTION_PREFIX):
        if _MEM_SPECIAL.get("pet") != m_pet.group(1):
            _MEM_SPECIAL["pet"] = m_pet.group(1)
            changed = True
    # 通用槽 1：最喜欢的<键>是/改成/换成<值>
    m = re.search(r"最喜欢的([^，。！？\s]{1,8})?(是|改成|换成)([^，。！？\s]+)", user)
    if m:
        put(m.group(1) or "", m.group(3))
    else:
        m2 = re.search(r"(?:最喜欢吃|喜欢吃|最喜欢)([^，。！？\s]+)", user)
        if m2 and not m2.group(1).startswith(_QUESTION_PREFIX):
            put("", m2.group(1))
    # 通用槽 2：我的<键>是<值>
    m3 = re.search(r"我的([^，。！？\s]{1,8})(?:是|叫)([^，。！？\s]+)", user)
    if m3 and m3.group(1) != "名字" and not m3.group(2).startswith(_QUESTION_PREFIX):
        put(m3.group(1), m3.group(2))
    return changed


def mem_answer(user):
    if any(k in user for k in ("你叫什么名字", "你叫什么", "你是谁")):
        if "谁训练" in user or "谁开发" in user:
            return "个人开发者 jiaheng 训练了我，我叫 feng。"
        return "我叫 feng，由个人开发者 jiaheng 开发训练。"
    if any(k in user for k in ("我叫什么", "我叫啥", "记得我叫什么",
                               "我的名字是什么", "我的名字是啥")):
        name = _MEM_SPECIAL.get("name")
        return f"你叫{name}。" if name else None
    if "住在哪" in user or "哪个城市" in user:
        city = _MEM_SPECIAL.get("city")
        return f"你住在{city}。" if city else None
    if "养了什么" in user or "养了啥" in user or "我的宠物" in user:
        pet = _MEM_SPECIAL.get("pet")
        return f"你养了{pet}。" if pet else None
    if "我" not in user:
        return None                       # "你最喜欢什么颜色"问的是助手
    key = None
    m = re.search(r"喜欢什么([^，。！？\s]{0,8})", user)
    if m:
        key = m.group(1)
    elif (m := re.search(r"喜欢啥([^，。！？\s]{0,8})", user)) is not None:
        key = m.group(1)
    else:
        m2 = re.search(r"最喜欢的([^，。！？\s]{1,8})(?:是什么|是啥)", user)
        if m2:
            key = m2.group(1)
        elif "我最喜欢什么" in user or "我喜欢什么" in user or "喜欢吃什么" in user:
            key = ""
    if key is not None:
        v = _MEM.get(key)
        return f"你最喜欢{v}。" if v else None
    # 我的<键>是什么 / 是多少 / 是几号 …
    m3 = re.search(r"我的([^，。！？\s]{1,8})(?:是什么|是啥|是多少|是几号|是哪个|是哪里|是几)", user)
    if m3 and m3.group(1) != "名字":
        v = _MEM.get(m3.group(1))
        if v:
            return f"你的{m3.group(1)}是{v}。"
    return None


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
    off = _days_offset(user)
    if off is not None:
        d = datetime.fromtimestamp(epoch + off * 86400, tz=timezone(timedelta(hours=8)))
        wd = "一二三四五六日"[d.weekday()]
        if off > 0:
            return f"{off} 天后是 {d.strftime('%Y年%m月%d日')}（周{wd}）。"
        if off < 0:
            return f"{-off} 天前是 {d.strftime('%Y年%m月%d日')}（周{wd}）。"
        return f"今天是 {d.strftime('%Y年%m月%d日')}（周{wd}）。"
    return f"现在是 {t.strftime('%Y年%m月%d日 %H:%M:%S')}（周{week}，UTC+8，{tag}）。"


def _days_offset(user):
    for kw, d in (("明天", 1), ("后天", 2), ("昨天", -1), ("前天", -2)):
        if kw in user:
            return d
    m = re.search(r"(\d+)\s*天(后|前|之后|之前)", user)
    if m:
        return int(m.group(1)) * (1 if "后" in m.group(2) else -1)
    return None


def random_answer(user):
    coin = "硬币" in user or "正反面" in user
    dice = "骰子" in user or "色子" in user
    if not (coin or dice or "随机" in user):
        return None
    norm = cn_to_ascii(user)
    nums = [int(x) for x in re.findall(r"\d+", norm)]
    default = (1, 6) if dice else (1, 100)
    lo, hi = (nums[0], nums[1]) if len(nums) >= 2 and nums[0] != nums[1] else default
    if hi < lo:
        lo, hi = hi, lo
    seed = int((time.perf_counter() - _START) * 1.54 * 1000)
    rng = random.Random(seed)
    rng.random()                                   # 第 1 个按需求丢弃
    if coin:
        return f"抛硬币：{'正面' if rng.randint(0, 1) else '反面'}。"   # 第 2 个随机数
    v = rng.randint(lo, hi)                        # 第 2 个随机数
    if dice:
        return f"掷骰子：{v} 点。"
    return f"随机数（{lo}~{hi}）：{v}。"


def tool_answer(user, epoch=None, source=None):
    """按 算式 → 记忆 → 时间 → 随机数 的顺序尝试；都不是则返回 None。
    （记忆要排在时间前："我的生日是几号？"不能被时间 tool 的"几号"关键词截走。）
    记忆 tool 会先记录本轮陈述里的可枚举事实（与 C 版固件同口径）。"""
    mem_learn(user)
    for fn in (lambda: calc_answer(user),
               lambda: mem_answer(user),
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
    mem_clear()
    for q in ["59+1", "4854+4411", "现在几点？", "今天几号", "给我个1到100的随机数",
              "随机 0-9", "我叫小雨，请记住。", "你叫什么名字？", "我叫什么名字？",
              "我最喜欢的颜色是蓝色。", "我最喜欢什么颜色？", "我养了一只乌龟。",
              "我养了什么？", "你好"]:
        print(f"{q!r:18s} -> {tool_answer(q, e, s)}")
