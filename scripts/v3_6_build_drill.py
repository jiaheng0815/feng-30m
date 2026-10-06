"""v3.6 数学专项 + 细节打磨数据：在 daily_patch 基础上系统覆盖小数字运算。

探针显示"1+1/2+3/10-4"这类最简单的算术仍会答错（说成 1/6/8），
原因是 45 个随机加法样例不够系统。这里：
  1. 枚举 1~9 的所有加（81）、非负减（约 90）、2~9 的乘法（64），每个 ×6 次；
  2. 追加电影/睡眠/天气/计划等细节打磨样本；
  3. 原始 daily_patch 原样保留，防止其他映射漂移。

用法：python scripts/v3_6_build_drill.py --base archive/v3_6a\\daily_patch.jsonl --out archive/v3_6e\\drill.jsonl
"""
import argparse
import json
import random
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import ROOT  # noqa: E402
from v3_6_build_daily_patch import COURTESY, FAREWELLS, GREETINGS  # noqa: E402

EXTRA = [
    ("推荐一部电影", "可以看看《千与千寻》，画面温暖，故事也耐看。"),
    ("推荐一部电影吧", "《肖申克的救赎》不错，故事完整，看完很受鼓舞。"),
    ("给我推荐一部电影", "《飞屋环游记》很治愈，适合放松的时候看。"),
    ("推荐电影", "《龙猫》很适合放松的时候看，画面很温柔。"),
    ("推荐一部喜剧电影", "《人在囧途》比较轻松，适合打发时间。"),
    ("推荐一本好书", "可以读读《小王子》，故事简单又耐人寻味。"),
    ("推荐一本书", "《夏洛的网》不错，讲友情，读起来很温暖。"),
    ("推荐一本适合孩子看的书", "《窗边的小豆豆》很适合，故事轻松又真诚。"),
    ("推荐一本小说", "《老人与海》不长，讲坚持，读完很有力量。"),
    ("推荐一本历史书", "《上下五千年》适合入门，故事多，读起来不累。"),
    ("我睡不着怎么办", "睡前别看手机，喝点温水，做几次慢呼吸，房间调暗一点。"),
    ("晚上睡不着", "试试固定时间上床，白天多晒晒太阳，睡前不要喝咖啡。"),
    ("我失眠了", "偶尔失眠很常见。别硬逼自己睡，起来安静待一会儿再躺下。"),
    ("帮我写一句晚安", "晚安，好梦。"),
    ("给我讲个笑话吧", "老师问：为什么你的作业和同桌一模一样？学生说：因为我们用的是同一本参考书。"),
    ("讲个冷笑话", "有人问我为什么带伞，我说：因为天气预报说今天有雨。"),
    ("今天适合做什么", "适合出门走走，或者把房间收拾一下，心情会轻快些。"),
    ("周末怎么安排", "可以挑半天做喜欢的事，再留半天好好休息。"),
    ("我想学做饭", "从简单的开始，比如番茄炒蛋，先切好材料再开火。"),
    ("怎么才能早起", "前一晚早点睡，把闹钟放远一点，起床后先拉开窗帘。"),
    ("怎么记单词", "每天少量多次，把新词放进句子里记，隔几天复习一遍。"),
    ("怎么放松心情", "慢慢呼吸几次，去窗边站一会儿，或者听一首喜欢的歌。"),
    ("我很无聊", "可以听歌、散步、整理照片，或者学做一道简单的菜。"),
    ("我很孤独", "孤独的时候找人说话是有用的，我陪你说说话吧。"),
    ("我很孤独", "我陪你聊一会儿，你现在想说什么都可以。"),
    ("给我推荐一首歌", "可以听听《明天会更好》，节奏明快，听着有力量。"),
    ("形容一下春天", "春天是风变软、草变绿、白天一点点变长的季节。"),
    ("用一句话形容秋天", "秋天是风里带凉、树叶变黄的季节。"),
    ("什么是人工智能？", "人工智能是让机器学会完成需要动脑的任务的技术。"),
    ("什么是人工智能？", "简单说，就是让机器像人一样学习、判断和处理问题的技术。"),
    ("我今天心情不太好", "抱抱你，想聊点什么，我都在。"),
    ("我心情不好", "怎么了？愿意说给我听听吗？"),
    ("我今天有点低落", "先别急着让自己好起来，我陪你慢慢缓一缓。"),
    ("我有点郁闷", "郁闷的时候说出来会轻一点，我在听。"),
    ("我今天很烦", "烦的时候先离开让你烦的地方，喝口水，慢慢来。"),
]


def conv(q: str, a: str) -> dict:
    return {"messages": [{"role": "user", "content": q},
                         {"role": "assistant", "content": a}]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=str(ROOT / "archive" / "v3_6a" / "daily_patch.jsonl"))
    ap.add_argument("--out", default=str(ROOT / "archive" / "v3_6e" / "drill.jsonl"))
    ap.add_argument("--repeat", type=int, default=6)
    ap.add_argument("--profile", default="full", choices=["full", "books"],
                    help="books：只输出书籍推荐专项（强重复）+ 原补丁锚点")
    ap.add_argument("--seed", type=int, default=20261006)
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.base, encoding="utf-8") if l.strip()]
    n_base = len(rows)

    if args.profile == "books":
        books = [(q, a) for q, a in EXTRA
                 if ("书" in q or "小说" in q or "电影" in q or "剧" in q
                     or "人工智能" in q)]
        for _ in range(40):
            for q, a in books:
                rows.append(conv(q, a))
                rows.append(conv(q + "？", a))
        greet = [(p, r) for prompts, replies in GREETINGS + COURTESY + FAREWELLS
                 for p in prompts for r in replies]
        for _ in range(20):                    # 问候/客套强锚定，防止"你好"漂成"早上好"
            for q, a in greet:
                rows.append(conv(q, a))
        rng = random.Random(args.seed)
        rng.shuffle(rows)
        dest = Path(args.out)
        dest.parent.mkdir(parents=True, exist_ok=True)
        with dest.open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"书籍专项：原补丁 {n_base} + 书籍 {len(books)}×80 = {len(rows)} 条 -> {dest}")
        return

    math = []
    for a in range(1, 10):
        for b in range(1, 10):
            ans = a + b
            math.append((f"{a}+{b}等于几？", f"{a} 加 {b} 等于 {ans}。"))
            math.append((f"{a}加{b}等于几？", f"{a} 加 {b} 等于 {ans}。"))
            # 带空格问法单独用"等于 X。"作答：此前照抄算子会退化成
            # "1 加 1。 2 等于 3。"（v3_6e/f 踩过），简化目标能压住
            math.append((f"{a} 加 {b} 等于几？", f"等于 {ans}。"))
    for a in range(2, 19):
        for b in range(1, min(9, a - 1) + 1):
            ans = a - b
            math.append((f"{a}-{b}等于几？", f"{a} 减 {b} 等于 {ans}。"))
            math.append((f"{a}减{b}等于几？", f"{a} 减 {b} 等于 {ans}。"))
            math.append((f"{a} 减 {b} 等于几？", f"等于 {ans}。"))
    for a in range(2, 10):
        for b in range(2, 10):
            math.append((f"{a}乘{b}等于几？", f"{a} 乘 {b} 等于 {a * b}。"))
    spaced = [(q, a) for q, a in math if " " in q]

    for _ in range(args.repeat):
        for q, a in math:
            rows.append(conv(q, a))
    for _ in range(args.repeat * 2):                 # 带空格问法 ×3，压住"1 加 1。"截断
        for q, a in spaced:
            rows.append(conv(q, a))
    base_len = len(rows)
    for q, a in EXTRA:
        rows.append(conv(q, a))
        rows.append(conv(q + "？", a))

    rng = random.Random(args.seed)
    rng.shuffle(rows)
    dest = Path(args.out)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"基础 {n_base} 条 + 数学 {len(math)}×{args.repeat} + 打磨 {len(EXTRA)*2} "
          f"= {len(rows)} 条 -> {dest}")


if __name__ == "__main__":
    main()
