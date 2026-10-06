"""v3.15 上下文身份锚点：闲聊几轮之后再问身份，回答必须仍然是 feng。

背景：板端 v3.14（QAT 模型）在「你好 / 讲个笑话 / 推荐一本好书」之后问
「你叫什么名字」会答成「你叫小模型」——把记忆训练里的「我叫X→你叫X」模式
套到了反问上（PC 版同前缀正常，属 QAT + q2 KV 下的身份漂移）。

本脚本生成"N 轮闲聊前缀 + 身份问答"的对话，同时保留一部分
「闲聊前缀 + 我叫X → 你叫X」的记忆样本，防止把记忆能力带坏。

用法：
    python scripts/v3_15_build_identity_ctx.py --out archive/v3_15/identity_ctx.jsonl
"""
import argparse
import json
import random
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import ROOT  # noqa: E402
from calc_tool import extract  # noqa: E402

IDENTITY = [
    ("你叫什么名字？", "我叫 feng，由个人开发者 jiaheng 开发训练。"),
    ("你叫什么名字", "我叫 feng，由个人开发者 jiaheng 开发训练。"),
    ("你叫什么？", "我叫 feng，由个人开发者 jiaheng 开发训练。"),
    ("你是谁？", "我是 feng，一个由个人开发者 jiaheng 独立开发训练的 AI。"),
    ("你是谁", "我是 feng，一个由个人开发者 jiaheng 独立开发训练的 AI。"),
    ("你是什么模型？", "我是 feng，一个由个人开发者 jiaheng 独立开发训练的 AI。大概 3000 万参数，是个很小的模型。"),
    ("你是谁开发的？", "我是 feng，由个人开发者 jiaheng 独立开发训练的 AI。"),
    ("谁训练了你？", "个人开发者 jiaheng 训练了我，我叫 feng。"),
    ("你是通义千问吗？", "不是。我是 feng，和通义千问没有关系，由 jiaheng 开发训练。"),
    ("你是 ChatGPT 吗？", "不是。我是 feng，由个人开发者 jiaheng 开发训练，和 OpenAI 没有关系。"),
]
NAMES = ["小明", "小红", "阿杰", "小芳", "丽丽", "小雨", "晓峰", "婷婷"]


def conv(*msgs):
    return {"messages": [{"role": ("user" if i % 2 == 0 else "assistant"), "content": m}
                         for i, m in enumerate(msgs)]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=str(ROOT / "archive" / "v3_6a" / "daily_patch.jsonl"),
                    help="闲聊前缀来源（用户/助手成对）")
    ap.add_argument("--out", default=str(ROOT / "archive" / "v3_15" / "identity_ctx.jsonl"))
    ap.add_argument("--n", type=int, default=900, help="身份锚点条数")
    ap.add_argument("--mem-n", type=int, default=400, help="记忆保护条数")
    ap.add_argument("--seed", type=int, default=20261008)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    pairs = []
    base = Path(args.base)
    if base.exists():
        for line in base.open(encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            msgs = json.loads(line)["messages"]
            if len(msgs) == 2 and msgs[0]["role"] == "user" and msgs[1]["role"] == "assistant":
                if extract(msgs[0]["content"]) is not None:
                    continue          # 算术交给 tool，不放进前缀（保持"无算术"训练口径）
                pairs.append((msgs[0]["content"], msgs[1]["content"]))
    if not pairs:
        pairs = [("你好", "你好，今天想聊点什么？"), ("谢谢你", "不客气。"), ("再见", "再见，需要我时随时来。")]

    rows = []
    for _ in range(args.n):
        pre = [pairs[i] for i in rng.sample(range(len(pairs)), rng.randint(1, 3))]
        q, a = rng.choice(IDENTITY)
        flat = []
        for u, r in pre:
            flat += [u, r]
        flat += [q, a]
        rows.append(conv(*flat))
    for _ in range(args.mem_n):
        pre = [pairs[i] for i in rng.sample(range(len(pairs)), rng.randint(1, 2))]
        name = rng.choice(NAMES)
        flat = []
        for u, r in pre:
            flat += [u, r]
        flat += [f"我叫{name}，请记住。", f"好的，{name}，我记住了。", "我叫什么名字？", f"你叫{name}。"]
        rows.append(conv(*flat))
    rng.shuffle(rows)
    dest = Path(args.out)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    print(f"上下文身份锚点 {len(rows)} 条（身份 {args.n} + 记忆 {args.mem_n}）-> {dest}")


if __name__ == "__main__":
    main()
