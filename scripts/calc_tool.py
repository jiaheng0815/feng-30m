"""计算 tool（Python 版）：和 C 引擎 feng_calc.c 同口径。

算术不该由 30M 模型硬背——所有运行时（板端固件 / PC C 引擎 / Python 脚本）
在把问题交给模型之前，先用本 tool 判断并计算。识别规则与 feng_calc.c 一致：
去掉「计算/帮我算/等于几/？/。」等外壳，只接受由数字与 + - * / × ÷ ( ) 组成的算式。

用法：
    from calc_tool import calc_answer
    reply = calc_answer("4854+4411")      # -> "4854 加 4411 等于 9265。"
    reply = calc_answer("你好")           # -> None（不是算式）
"""
import re

_SUFFIXES = ["是多少钱", "是多少元", "多少钱", "多少元", "是多少呢", "是多少呀",
             "算一下", "算算", "算下", "计算一下",
             "等于几", "等于多少", "是多少", "多少", "=?", "＝?", "?", "？",
             "呢", "呀", "。", ".", "!", "！", "=", "＝"]
_PREFIXES = ["帮我算一下", "帮我计算", "帮我算算", "帮我算", "麻烦算一下", "麻烦算算",
             "计算一下", "计算", "算一下", "算算", "请问一下", "请问", "求解", "求", "把"]
_OP_MAP = {"加上": "+", "加": "+", "减去": "-", "减": "-", "乘以": "*", "乘上": "*", "乘": "*",
           "除以": "/", "除": "/", "×": "*", "÷": "/", "（": "(", "）": ")"}
_VALID = re.compile(r"^[0-9.+\-*/()% ]+$")
_CN_DIGITS = {"零": 0, "〇": 0, "一": 1, "二": 2, "两": 2, "三": 3, "四": 4,
              "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}
_CN_UNITS = {"十": 10, "百": 100, "千": 1000, "万": 10000}


def cn_to_ascii(s: str) -> str:
    """把「五十九 / 一百零五 / 两千三」等中文数字段换成阿拉伯数字（与 feng_calc.c 同口径）。"""
    out, i = [], 0
    while i < len(s):
        if s[i] in _CN_DIGITS or s[i] in _CN_UNITS:
            total = section = cur = 0
            while i < len(s) and (s[i] in _CN_DIGITS or s[i] in _CN_UNITS):
                if s[i] in _CN_DIGITS:
                    cur = _CN_DIGITS[s[i]]
                else:
                    u = _CN_UNITS[s[i]]
                    if u < 10000:
                        section += (cur or 1) * u
                        cur = 0
                    else:
                        total += (section + cur) * 10000
                        section = cur = 0
                i += 1
            out.append(str(total + section + cur))
        else:
            out.append(s[i])
            i += 1
    return "".join(out)


def extract(user: str):
    """返回紧凑算式字符串；不是纯算式则返回 None。"""
    s = user.strip()
    for k, v in _OP_MAP.items():
        s = s.replace(k, v)
    for _ in range(6):
        before = s
        s = s.strip()
        for suf in _SUFFIXES:
            if s.endswith(suf):
                s = s[: -len(suf)]
                break
        s = s.strip()
        for pre in _PREFIXES:
            if s.startswith(pre):
                s = s[len(pre):]
                break
        s = s.strip()
        if s == before:
            break
    s = cn_to_ascii(s)          # 外壳剥完再转中文数字（"帮我算一下"里的"一"不动）
    # 100的15% -> 100*15%；12的平方 -> 12*12；12的立方 -> 12*12*12；根号16 -> (16**(0.5))
    m = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*的\s*(平方|立方)\s*", s)
    if m:
        x = m.group(1)
        s = f"{x}*{x}*{x}" if m.group(2) == "立方" else f"{x}*{x}"
    elif "的" in s and s.count("的") == 1:
        s = s.replace("的", "*")
    m = re.fullmatch(r"\s*根号\s*(\d+(?:\.\d+)?)\s*", s)
    if m:
        s = f"({m.group(1)}**(0.5))"
    if not _VALID.match(s):
        return None
    if not any(c.isdigit() for c in s):
        return None
    if not any(c in "+-*/" for c in s):
        return None
    return s.replace(" ", "")


def _fmt(v: float) -> str:
    if abs(v - round(v)) < 1e-9 and abs(v) < 1e15:
        return str(int(round(v)))
    s = f"{v:.4f}".rstrip("0").rstrip(".")
    return s if s not in ("", "-0") else "0"


def calc_answer(user: str):
    """算式请求返回中文回答；否则返回 None。"""
    seq = seq_answer(user)                       # 序列数数（与 feng_calc.c 同口径）
    if seq is not None:
        return seq
    expr = extract(user)
    if expr is None:
        return None
    try:
        # 15% -> (15/100)；其余交给 eval（字符集已限制）
        eval_expr = re.sub(r"(\d+(?:\.\d+)?)%", r"(\1/100)", expr)
        v = eval(eval_expr, {"__builtins__": {}}, {})   # 已限制字符集，安全
    except ZeroDivisionError:
        return "这个算式里除数是 0，算不出来。"
    except Exception:
        return "这个算式我没看懂。"
    if not isinstance(v, (int, float)):
        return None
    m = re.fullmatch(r"(\d+(?:\.\d+)?)([+\-*/])(\d+(?:\.\d+)?)", expr)
    if m:
        sym = {"+": "加", "-": "减", "*": "乘", "/": "除以"}[m.group(2)]
        return f"{m.group(1)} {sym} {m.group(3)} 等于 {_fmt(float(v))}。"
    return f"结果是 {_fmt(float(v))}。"


def seq_answer(user: str):
    """序列数数："把 1 到 5 倒着数一遍" / "从 3 数到 8"；不匹配返回 None。
    只在明确祈使句上触发，避免"我从1数到100也数不完"这类陈述被截走。"""
    rev = any(k in user for k in ("倒着数", "倒序", "倒过来数"))
    starts = user.startswith("从") or user.startswith("把") or "请从" in user
    has_count = ("数一遍" in user) or ("数一下" in user)
    if rev:
        pass
    elif "数到" in user:
        if not (starts or has_count):
            return None
    elif has_count and starts:
        pass
    else:
        return None
    nums = [int(x) for x in re.findall(r"\d+", cn_to_ascii(user))]
    if len(nums) < 2:
        return None
    a, b = nums[0], nums[1]
    if not rev and a > b:
        return None
    if a > b:
        a, b = b, a
    if b - a > 50:
        return f"范围有点大（{a} 到 {b}），给我 50 个以内的区间吧。"
    rng = range(b, a - 1, -1) if rev else range(a, b + 1)
    return "、".join(str(v) for v in rng) + "。"


if __name__ == "__main__":
    import sys

    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    for q in ["4854+4411", "4854+4411等于几", "59+1", "445+15", "5.3+4.1", "10+4.",
              "(3+4)*2", "1/0", "你好", "我2-3点有空"]:
        print(f"{q!r:22s} -> {calc_answer(q)}")
