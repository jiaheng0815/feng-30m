"""C 引擎与 Python 的记忆 tool 一致性检查（对话式，含学习/追问/换值/列表/遗忘/重学）。

把同一段对话分别喂给：
  - C：`pc/pc_mem_ask.c` 驱动 + `main/feng_memory.c`（stdin 一行一句，输出回答或 NONE）
  - Python：`scripts/runtime_tools.py` 的 mem_learn / mem_answer
逐条比较，不一致退出码 1。

用法：
    gcc -O2 -o pc_mem_ask pc/pc_mem_ask.c main/feng_memory.c -Imain -lm
    python tools/check_mem_parity.py --c-bin pc_mem_ask
"""
import argparse
import subprocess
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from runtime_tools import mem_answer, mem_clear, mem_learn  # noqa: E402

SCRIPT = [
    "\\clear",
    # 姓名：学习 → 追问 → 换值
    "我叫小明，请记住。", "我叫什么名字？", "我的名字是什么？",
    "我叫小雨，请记住。", "我叫什么名字？",
    "你叫什么名字？", "你是谁？",
    # 颜色：学习 → 追问 → 改口
    "我最喜欢的颜色是蓝色。", "我最喜欢什么颜色？",
    "我最喜欢的颜色改成红色了。", "我最喜欢什么颜色？",
    # 城市 / 宠物 / 食物
    "我住在成都。", "我住在哪里？",
    "我搬到武汉了。", "我住在哪里？",
    "我养了一只乌龟。", "我养了什么？",
    "我最喜欢吃饺子。", "我最喜欢吃什么？",
    # 通用键值槽
    "我最喜欢的书是《小王子》。", "我最喜欢什么书？",
    "我的生日是3月5日。", "我的生日是几号？",
    # 普通陈述不拦
    "我今天被表扬了。", "你好。",
    # 列出 / 遗忘 / 墓碑 / 重学
    "你还记得什么？",
    "忘掉我的颜色。", "我最喜欢什么颜色？",
    "我最喜欢的颜色是绿色。", "我最喜欢什么颜色？",
    "别记我的生日了。", "我的生日是几号？",
    "把记住的都忘掉。", "我叫什么名字？", "你还记得什么？",
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--c-bin", required=True, help="pc_mem_ask 可执行文件路径")
    args = ap.parse_args()

    payload = ("\n".join(SCRIPT) + "\n").encode("utf-8")
    r = subprocess.run([args.c_bin], input=payload, capture_output=True)
    c_lines = r.stdout.decode("utf-8", "replace").splitlines()
    if len(c_lines) != len(SCRIPT):
        print(f"C 驱动输出行数 {len(c_lines)} != 脚本行数 {len(SCRIPT)}（stderr: "
              f"{r.stderr.decode('utf-8', 'replace')[:200]}）")
        return 1

    mem_clear()
    bad = 0
    for line, c_out in zip(SCRIPT, c_lines):
        if line == "\\clear":
            mem_clear()
            py_out = "CLEARED"
        elif line == "\\mem":
            continue
        else:
            mem_learn(line)
            py_out = mem_answer(line) or "NONE"
        c_ans = c_out
        if c_ans != py_out:
            bad += 1
            print(f"[FAIL] {line!r}\n   C     : {c_ans!r}\n   Python: {py_out!r}")
    print(f"记忆一致性：{len(SCRIPT) - bad}/{len(SCRIPT)} 条一致"
          + ("" if bad == 0 else f"，{bad} 条不一致"))
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
