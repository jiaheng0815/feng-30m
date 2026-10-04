"""v3.13 记忆补丁：让模型在多轮里记住用户说过的事实并复述。

背景：板端固件已经保留多轮上下文（KV 累积），但 30M 模型对
「我叫小明」→「我叫什么名字？」这类记忆题答不出来（PC/板端一致）。
本脚本生成合成的多轮记忆对话（姓名/颜色/城市/宠物/食物/运动），
含单事实、多事实与干扰项，用于末层微调。

用法：
    python scripts/v3_13_build_memory.py --out v3_13/memory.jsonl
"""
import argparse
import json
import random
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import ROOT  # noqa: E402

NAMES = ["小明", "小红", "阿杰", "小芳", "老王", "丽丽", "大勇", "小雨", "阿强", "婷婷",
         "晓峰", "佳佳", "小龙", "阿敏", "小雅", "文文", "阿豪", "静静", "浩子", "琳琳"]
COLORS = ["蓝色", "红色", "绿色", "紫色", "橙色", "黑色", "白色", "黄色"]
CITIES = ["北京", "上海", "广州", "深圳", "成都", "杭州", "西安", "武汉", "南京", "重庆"]
PETS = ["猫", "狗", "兔子", "仓鼠", "鹦鹉", "乌龟"]
FOODS = ["火锅", "饺子", "面条", "烧烤", "蛋糕", "米饭"]
SPORTS = ["篮球", "足球", "羽毛球", "跑步", "游泳", "乒乓球"]


def conv(*msgs):
    return {"messages": [{"role": ("user" if i % 2 == 0 else "assistant"), "content": m}
                         for i, m in enumerate(msgs)]}


def fact_pairs(rng):
    """(类别, 陈述问法, 提问问法, 值, 回答模板)。"""
    return [
        ("姓名", ["我叫{v}，请记住。", "我叫{v}。", "记住，我叫{v}。", "我的名字是{v}。"],
         ["我叫什么名字？", "我叫什么？", "你还记得我叫什么吗？", "我的名字是什么？"],
         rng.choice(NAMES), "你叫{v}。"),
        ("颜色", ["我最喜欢的颜色是{v}。", "我喜欢{v}。", "我喜欢的颜色是{v}。"],
         ["我最喜欢什么颜色？", "我喜欢什么颜色？", "你记得我喜欢什么颜色吗？"],
         rng.choice(COLORS), "你最喜欢{v}。"),
        ("城市", ["我住在{v}。", "我来自{v}。", "我家在{v}。"],
         ["我住在哪里？", "我来自哪里？", "你还记得我住在哪吗？"],
         rng.choice(CITIES), "你住在{v}。"),
        ("宠物", ["我养了一只{v}。", "我家有一只{v}。"],
         ["我养了什么？", "我家有什么宠物？", "你记得我养了什么吗？"],
         rng.choice(PETS), "你养了一只{v}。"),
        ("食物", ["我最喜欢吃{v}。", "我喜欢吃{v}。"],
         ["我最喜欢吃什么？", "我喜欢吃什么？"],
         rng.choice(FOODS), "你最喜欢吃{v}。"),
        ("运动", ["我最喜欢的运动是{v}。", "我常玩{v}。"],
         ["我最喜欢什么运动？", "我常玩什么？"],
         rng.choice(SPORTS), "你最喜欢{v}。"),
    ]


def ack(rng, kind, v):
    tpl = {
        "姓名": [f"好的，{v}，我记住了。", f"记住了，你叫{v}。"],
        "颜色": [f"记住了，你喜欢{v}。", f"好的，你最喜欢{v}。"],
        "城市": [f"记住了，你住在{v}。", f"好的，{v}。"],
        "宠物": [f"记住了，你养了一只{v}。"],
        "食物": [f"记住了，你最喜欢吃{v}。"],
        "运动": [f"记住了，你最喜欢{v}。"],
    }[kind]
    return rng.choice(tpl)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "v3_13" / "memory.jsonl"))
    ap.add_argument("--n", type=int, default=2400, help="单事实对话条数")
    ap.add_argument("--multi-n", type=int, default=800, help="多事实（含干扰）条数")
    ap.add_argument("--seed", type=int, default=20261005)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    rows = []
    # 1) 单事实：陈述 → 确认 → 追问 → 复述
    for _ in range(args.n):
        kind, stmts, asks, v, ans = rng.choice(fact_pairs(rng))
        s = rng.choice(stmts).format(v=v)
        q = rng.choice(asks)
        rows.append(conv(s, ack(rng, kind, v), q, ans.format(v=v)))
        # 同一事实换问法再来一轮（同一对话里多问一次）
        rows.append(conv(s, ack(rng, kind, v), q, ans.format(v=v),
                         rng.choice(asks), ans.format(v=v)))
    # 2) 多事实 + 干扰：说 2~3 条，再挑一条问
    for _ in range(args.multi_n):
        k = rng.randint(2, 3)
        facts = rng.sample(fact_pairs(rng), k)
        msgs = []
        for kind, stmts, asks, v, ans in facts:
            msgs += [rng.choice(stmts).format(v=v), ack(rng, kind, v)]
        kind, stmts, asks, v, ans = rng.choice(facts)
        msgs += [rng.choice(asks), ans.format(v=v)]
        rows.append(conv(*msgs))

    rng.shuffle(rows)
    dest = Path(args.out)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"记忆对话 {len(rows)} 条（单事实 {args.n}×2 + 多事实 {args.multi_n}）-> {dest}")


if __name__ == "__main__":
    main()
