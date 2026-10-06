"""v3.16 质量补丁数据：报名字后的身份串名 + 同类事实取旧值 + 类目退化。

板端 v3.15-embed 的三类残余错例（12 题记忆测评 + 交互复现）：
  1) 「我叫小明，请记住」之后问「你叫什么名字？」→ 答「我叫小明，由个人开发者
     jiaheng 开发训练」——把**用户的名字**套到了自己身份上。需要在"刚听完用户
     自报姓名"的上下文里把身份问答和用户记忆分开锚死。
  2) 同类事实说两次后问「我最喜欢什么？」→ 答了更早的那条（recency 失效）。
  3) 「我最喜欢的运动是羽毛球」→ 退化成"看不到实时信息"模板（类目覆盖不足）。

生成六组数据：name_identity / name_memory / recency / cross_category /
generic_memory（保护）/ identity_anchor（保护），全部沿用 v3.15 的身份口径。

用法：
    python scripts/v3_16_build_identity_memory_patch.py --out archive/v3_16/patch.jsonl
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

SELF = "我叫 feng，由个人开发者 jiaheng 开发训练。"
SELF_FULL = "我是 feng，一个由个人开发者 jiaheng 独立开发训练的 AI。"

IDENTITY = [
    ("你叫什么名字？", SELF),
    ("你叫什么名字", SELF),
    ("你叫什么？", SELF),
    ("你的名字是什么？", SELF),
    ("你是谁？", SELF_FULL),
    ("你是谁", SELF_FULL),
    ("你是什么模型？", SELF_FULL + "大概 3000 万参数，是个很小的模型。"),
    ("你是谁开发的？", "我是 feng，由个人开发者 jiaheng 独立开发训练的 AI。"),
    ("谁训练了你？", "个人开发者 jiaheng 训练了我，我叫 feng。"),
    ("你是通义千问吗？", "不是。我是 feng，和通义千问没有关系，由 jiaheng 开发训练。"),
    ("你是 ChatGPT 吗？", "不是。我是 feng，由个人开发者 jiaheng 开发训练，和 OpenAI 没有关系。"),
]
NAME_Q = [
    ("我叫什么名字？", "你叫{name}。"),
    ("我叫什么名字", "你叫{name}。"),
    ("我叫啥？", "你叫{name}。"),
    ("你还记得我叫什么吗？", "你叫{name}。"),
    ("我叫什么来着？", "你叫{name}。"),
]
ANNOUNCE = ["我叫{name}，请记住。", "我是{name}。", "我的名字是{name}，别叫错了。",
            "叫我{name}就行。", "我叫{name}，你可以这么称呼我。"]
ACK = ["好的，{name}，我记住了。", "记住了，{name}。", "好的，{name}。"]

NAMES = ["小明", "小红", "阿杰", "小芳", "丽丽", "小雨", "晓峰", "婷婷", "老王", "阿强",
         "欣欣", "大壮", "小雅", "浩然", "子墨", "婉婉", "磊磊", "小贝"]
CATS = {
    "颜色": ["蓝色", "黄色", "绿色", "紫色", "红色", "橙色", "黑色", "白色"],
    "食物": ["蛋糕", "火锅", "饺子", "面条", "寿司", "烧烤", "冰淇淋", "包子"],
    "运动": ["羽毛球", "游泳", "跑步", "篮球", "足球", "乒乓球", "骑行", "登山"],
    "城市": ["武汉", "成都", "杭州", "西安", "南京", "长沙", "青岛", "厦门"],
    "宠物": ["猫", "狗", "兔子", "乌龟", "仓鼠", "鹦鹉", "金鱼"],
}


def conv(*msgs):
    return {"messages": [{"role": ("user" if i % 2 == 0 else "assistant"), "content": m}
                         for i, m in enumerate(msgs)]}


def fact(kind, value, change=False, rng=random):
    """一类事实的陈述与应答；change=True 表示"改成…了"的更新说法。"""
    if kind == "颜色":
        u = (f"我现在最喜欢的颜色改成{value}了。" if change else f"我最喜欢的颜色是{value}。")
        a = f"好的，你最喜欢{value}。"
    elif kind == "食物":
        u = (f"我最近更喜欢吃{value}了。" if change else f"我最喜欢{value}。")
        a = f"好的，你最喜欢{value}。"
    elif kind == "运动":
        u = (f"我现在最喜欢的运动换成{value}了。" if change else f"我最喜欢的运动是{value}。")
        a = f"好的，你最喜欢{value}。"
    elif kind == "城市":
        u = (rng.choice([f"我搬到{value}了。", f"我现在住在{value}了。",
                         f"我搬到{value}住了。"]) if change else f"我住在{value}。")
        a = f"好的，你住在{value}。"
    else:
        u = (f"我现在养的是{value}。" if change else f"我养了一只{value}。")
        a = f"好的，你养了{value}。"
    return u, a


def q_for(kind):
    if kind == "颜色":
        return ["我最喜欢什么颜色？", "我最喜欢什么颜色", "我喜欢什么颜色来着？"]
    if kind == "食物":
        return ["我最喜欢什么？", "我最喜欢吃什么？", "我喜欢吃什么来着？"]
    if kind == "运动":
        return ["我最喜欢什么运动？", "我喜欢什么运动来着？"]
    if kind == "城市":
        return ["我住在哪里？", "我住在哪？"]
    return ["我养了什么？", "我养了什么宠物？"]


def answer_for(kind, value):
    if kind == "城市":
        return f"你住在{value}。"
    if kind == "宠物":
        return f"你养了{value}。"
    return f"你最喜欢{value}。"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=str(ROOT / "archive" / "v3_6a" / "daily_patch.jsonl"))
    ap.add_argument("--out", default=str(ROOT / "archive" / "v3_16" / "patch.jsonl"))
    ap.add_argument("--name-identity", type=int, default=500)
    ap.add_argument("--name-memory", type=int, default=300)
    ap.add_argument("--both-ways", type=int, default=0, help="同一对话里双向问名（防反向串名）")
    ap.add_argument("--chain", type=int, default=0, help="精确链：先问'我叫什么'再问'你叫什么'")
    ap.add_argument("--recency", type=int, default=450)
    ap.add_argument("--cross", type=int, default=250)
    ap.add_argument("--memory", type=int, default=450)
    ap.add_argument("--anchor", type=int, default=250)
    ap.add_argument("--seed", type=int, default=20261009)
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
                    continue                      # 算术交给 tool，不进前缀
                pairs.append((msgs[0]["content"], msgs[1]["content"]))
    if not pairs:
        pairs = [("你好", "你好，今天想聊点什么？"), ("谢谢你", "不客气。"), ("再见", "再见，需要我时随时来。")]

    def prefix(max_turns):
        if max_turns <= 0 or rng.random() < 0.25:
            return []
        return [pairs[i] for i in rng.sample(range(len(pairs)), rng.randint(1, max_turns))]

    rows = []

    # 1) 报名字之后问身份：答案必须是 feng（不许串成用户的名字）
    for _ in range(args.name_identity):
        pre = prefix(2)
        name = rng.choice(NAMES)
        q, a = rng.choice(IDENTITY[:6])
        flat = [m for p in pre for m in p]
        flat += [rng.choice(ANNOUNCE).format(name=name), rng.choice(ACK).format(name=name), q, a]
        rows.append(conv(*flat))

    # 2) 报名字之后问"我叫什么"：答案必须是用户的名字
    for _ in range(args.name_memory):
        pre = prefix(2)
        name = rng.choice(NAMES)
        q, a = rng.choice(NAME_Q)
        flat = [m for p in pre for m in p]
        flat += [rng.choice(ANNOUNCE).format(name=name), rng.choice(ACK).format(name=name),
                 q, a.format(name=name)]
        rows.append(conv(*flat))

    # 2b) 同一对话里双向问名：先问助手再问用户（或反过来），两个方向都要对
    for _ in range(args.both_ways):
        pre = prefix(2)
        name = rng.choice(NAMES)
        qn, an = rng.choice(NAME_Q)
        qi, ai = rng.choice(IDENTITY[:6])
        flat = [m for p in pre for m in p]
        flat += [rng.choice(ANNOUNCE).format(name=name), rng.choice(ACK).format(name=name)]
        if rng.random() < 0.5:
            flat += [qi, ai, qn, an.format(name=name)]
        else:
            flat += [qn, an.format(name=name), qi, ai]
        if rng.random() < 0.5:                      # 再补一轮，拉长上下文
            qx, ax = rng.choice(IDENTITY[:6])
            flat += [f"我再说一遍，我叫{name}。", f"记住了，{name}。", qx, ax]
        rows.append(conv(*flat))

    # 2c) 精确链：先答"你叫X"，紧接着被问"你叫什么名字"——必须切回 feng
    for _ in range(args.chain):
        pre = prefix(2)
        name = rng.choice(NAMES)
        qn, an = rng.choice(NAME_Q)
        qi, ai = rng.choice(IDENTITY[:6])
        flat = [m for p in pre for m in p]
        flat += [rng.choice(ANNOUNCE).format(name=name), rng.choice(ACK).format(name=name)]
        flat += [qn, an.format(name=name)]
        if rng.random() < 0.4:                      # 中间插一轮闲聊
            flat += [rng.choice(["你好", "谢谢你"]),
                     rng.choice(["你好！有什么我可以帮你的吗？", "不客气。"])]
        flat += [qi, ai]
        if rng.random() < 0.5:                      # 再交替一轮
            flat += [qn, an.format(name=name), qi, ai]
        rows.append(conv(*flat))

    # 3) 同类事实两次 → 答最新的
    for _ in range(args.recency):
        kind = rng.choice(list(CATS))
        v1, v2 = rng.sample(CATS[kind], 2)
        u1, a1 = fact(kind, v1)
        u2, a2 = fact(kind, v2, change=True)
        q = rng.choice(q_for(kind))
        flat = [m for p in prefix(1) for m in p]
        flat += [u1, a1, u2, a2, q, answer_for(kind, v2)]
        rows.append(conv(*flat))

    # 4) 多类目事实交错 → 每个类目各答各的（含运动类目，修退化模板）
    for _ in range(args.cross):
        kinds = rng.sample(list(CATS), rng.randint(2, 4))
        picks = [(k, rng.choice(CATS[k])) for k in kinds]
        flat = [m for p in prefix(1) for m in p]
        for k, v in picks:
            u, a = fact(k, v)
            flat += [u, a]
        for k, v in rng.sample(picks, rng.randint(1, len(picks))):
            flat += [rng.choice(q_for(k)), answer_for(k, v)]
        rows.append(conv(*flat))

    # 5) 通用单事实记忆保护
    for _ in range(args.memory):
        kind = rng.choice(list(CATS))
        v = rng.choice(CATS[kind])
        u, a = fact(kind, v)
        q = rng.choice(q_for(kind))
        flat = [m for p in prefix(2) for m in p]
        flat += [u, a, q, answer_for(kind, v)]
        rows.append(conv(*flat))

    # 6) 身份锚点（闲聊前缀 + 身份问答）
    for _ in range(args.anchor):
        pre = prefix(3)
        q, a = rng.choice(IDENTITY)
        flat = [m for p in pre for m in p] + [q, a]
        rows.append(conv(*flat))

    rng.shuffle(rows)
    dest = Path(args.out)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    print(f"v3.16 补丁 {len(rows)} 条 -> {dest}")
    print(f"  报名字→问身份 {args.name_identity} | 报名字→问名字 {args.name_memory} | "
          f"同类取新 {args.recency} | 多类目 {args.cross} | 记忆保护 {args.memory} | 身份锚点 {args.anchor} | 双向问名 {args.both_ways}" + f" | 精确链 {args.chain}")


if __name__ == "__main__":
    main()
