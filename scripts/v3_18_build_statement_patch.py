"""v3.18 补丁数据：把"普通陈述句"和"要求记住"分开。

留出题评测（`scripts/chat_probe_heldout.py`）发现的最大失败模式：
模型把"我说X → 好的，我记住了"这条记忆训练模板**泛化到所有陈述**——
  "我今天被表扬了" → "记住了，你叫你'小白'…"
  "我和朋友吵架了" → "好的，我记住了。"
  "祝你周末愉快"   → "好的，我记住了。"

本脚本生成三类数据教它区分：
  A 情绪/生活陈述（没说"记住"）→ 自然共情/回应；
  B 礼貌寒暄（麻烦你了/辛苦你了/祝你…）→ 自然回应；
  C **明确要求记住**（请记住/记一下/帮我记住）→ 记忆确认（保留这条行为）；
  D 身份/记忆保护（沿用 v3_16 的锚点与问答，防止把底座带坏）。

用法：
    python scripts/v3_18_build_statement_patch.py --out archive/v3_18/stmt_patch.jsonl
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

POS = ["我今天被表扬了。", "我今天升职了。", "我中奖了。", "我考了满分。", "我找到工作了。",
       "我把论文写完了。", "我学会做菜了。", "我今天跑步五公里。", "我买到票了。",
       "我养的猫生小猫了。", "我收到礼物了。", "我体检结果很好。"]
POS_R = ["那挺好的呀，恭喜你！", "厉害！替你高兴。", "真不错，替你开心。", "恭喜恭喜！",
         "太好了，继续保持。", "这值得庆祝一下。"]
NEG = ["我和朋友吵架了。", "我考试考砸了。", "我丢钱包了。", "我有点紧张。", "我睡不着。",
       "我今天很累。", "我被批评了。", "我感冒了。", "我面试没过。", "我的手机摔坏了。",
       "我养的鱼死了。", "我错过火车了。"]
NEG_R = ["有点难受吧，想聊聊的话我在。", "别太往心里去，慢慢会好的。", "抱抱你，先好好休息。",
         "先深呼吸，事情一件一件来。", "辛苦了，别为难自己。", "嗯，我听着呢，想说说吗？"]
NEU = ["我周末要去旅行。", "我搬家了。", "我换了个新发型。", "我开始学吉他。",
       "我把房间重新布置了。", "我今天吃了火锅。", "我打算早点睡。", "我在看一本小说。"]
NEU_R = ["听起来不错，祝你顺利！", "挺好呀，注意休息。", "了解，玩得开心点。",
         "不错不错，慢慢来。", "嗯，挺好的。", "希望一切顺利！"]
POLITE = ["麻烦你了。", "辛苦你了。", "谢谢你帮我。", "打扰一下。", "让你费心了。"]
POLITE_R = ["不麻烦，应该的。", "不辛苦，乐意帮忙。", "不客气，有需要随时说。",
            "没事，你说。", "别客气，举手之劳。"]
WISH = ["祝你周末愉快。", "祝你早点好起来。", "祝你生日快乐。", "祝一切顺利。"]
WISH_R = ["谢谢，你也是！", "谢谢你，我会的。", "谢谢！也祝你开心。", "借你吉言，谢谢！"]

NAMES = ["小明", "小雨", "阿强", "丽丽", "晓峰", "婷婷", "小雅", "子墨"]
CATS = {"颜色": ["蓝色", "黄色", "绿色", "紫色"], "运动": ["游泳", "羽毛球", "跑步"],
        "城市": ["武汉", "成都", "杭州"], "宠物": ["乌龟", "兔子", "猫"],
        "食物": ["蛋糕", "火锅", "饺子"]}
MEM_INSTR = ["请记住。", "记一下。", "帮我记住。", "请记下来。"]


def conv(*msgs):
    return {"messages": [{"role": ("user" if i % 2 == 0 else "assistant"), "content": m}
                         for i, m in enumerate(msgs)]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=str(ROOT / "archive" / "v3_6a" / "daily_patch.jsonl"))
    ap.add_argument("--out", default=str(ROOT / "archive" / "v3_18" / "stmt_patch.jsonl"))
    ap.add_argument("--a-n", type=int, default=420, help="情绪/生活陈述")
    ap.add_argument("--b-n", type=int, default=140, help="礼貌寒暄")
    ap.add_argument("--c-n", type=int, default=260, help="明确要求记住 -> 确认")
    ap.add_argument("--d-n", type=int, default=240, help="身份/记忆保护")
    ap.add_argument("--seed", type=int, default=20261010)
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
            if len(msgs) == 2 and msgs[0]["role"] == "user":
                if extract(msgs[0]["content"]) is not None:
                    continue                 # 算术交给 tool，不进前缀（保持"无算术"口径）
                pairs.append((msgs[0]["content"], msgs[1]["content"]))
    if not pairs:
        pairs = [("你好", "你好，今天想聊点什么？")]

    def prefix(n=1):
        return [pairs[i] for i in rng.sample(range(len(pairs)), rng.randint(0, n))]

    rows = []
    # A 情绪/生活陈述 -> 自然回应（不带"记住"）
    for _ in range(args.a_n):
        kind = rng.random()
        if kind < 0.4:
            s, r = rng.choice(POS), rng.choice(POS_R)
        elif kind < 0.8:
            s, r = rng.choice(NEG), rng.choice(NEG_R)
        else:
            s, r = rng.choice(NEU), rng.choice(NEU_R)
        flat = [m for p in prefix(1) for m in p] + [s, r]
        rows.append(conv(*flat))
    # B 礼貌寒暄 -> 自然回应
    for _ in range(args.b_n):
        if rng.random() < 0.7:
            s, r = rng.choice(POLITE), rng.choice(POLITE_R)
        else:
            s, r = rng.choice(WISH), rng.choice(WISH_R)
        flat = [m for p in prefix(1) for m in p] + [s, r]
        rows.append(conv(*flat))
    # C 明确要求记住 -> 确认（保留记忆确认行为，但只在这种指令下出现）
    for _ in range(args.c_n):
        name = rng.choice(NAMES)
        kind = rng.randrange(3)
        instr = rng.choice(MEM_INSTR)
        if kind == 0:
            rows.append(conv(f"我叫{name}，{instr}", f"好的，{name}，我记住了。"))
        elif kind == 1:
            cat = rng.choice(list(CATS))
            v = rng.choice(CATS[cat])
            stmt = {"颜色": f"我最喜欢的颜色是{v}", "运动": f"我最喜欢的运动是{v}",
                    "城市": f"我住在{v}", "宠物": f"我养了一只{v}",
                    "食物": f"我最喜欢{v}"}[cat]
            rows.append(conv(f"{stmt}，{instr}", f"好的，我记住了：{stmt}。"))
        else:
            cat = rng.choice(list(CATS))
            v = rng.choice(CATS[cat])
            stmt = {"颜色": f"我最喜欢的颜色是{v}", "运动": f"我最喜欢的运动是{v}",
                    "城市": f"我住在{v}", "宠物": f"我养了一只{v}",
                    "食物": f"我最喜欢{v}"}[cat]
            ask = {"颜色": "我最喜欢什么颜色？", "运动": "我最喜欢什么运动？",
                   "城市": "我住在哪里？", "宠物": "我养了什么？",
                   "食物": "我最喜欢什么？"}[cat]
            ans = {"颜色": f"你最喜欢{v}。", "运动": f"你最喜欢{v}。",
                   "城市": f"你住在{v}。", "宠物": f"你养了{v}。",
                   "食物": f"你最喜欢{v}。"}[cat]
            rows.append(conv(f"{stmt}，{instr}", f"好的，我记住了：{stmt}。", ask, ans))
    # D 身份保护（沿用 v3.16 的口径）
    ID = [("你叫什么名字？", "我叫 feng，由个人开发者 jiaheng 开发训练。"),
          ("你是谁？", "我是 feng，一个由个人开发者 jiaheng 独立开发训练的 AI。"),
          ("谁训练了你？", "个人开发者 jiaheng 训练了我，我叫 feng。"),
          ("你是通义千问吗？", "不是。我是 feng，和通义千问没有关系，由 jiaheng 开发训练。")]
    for _ in range(args.d_n):
        pre = prefix(2)
        q, a = rng.choice(ID)
        flat = [m for p in pre for m in p] + [q, a]
        rows.append(conv(*flat))

    rng.shuffle(rows)
    dest = Path(args.out)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    print(f"v3.18 陈述补丁 {len(rows)} 条 -> {dest}")
    print(f"  A 情绪/生活 {args.a_n} | B 礼貌寒暄 {args.b_n} | C 记住指令 {args.c_n} | D 身份保护 {args.d_n}")


if __name__ == "__main__":
    main()
