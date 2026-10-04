"""v3.19 定向提示词：针对留出题暴露的六类失败模式，给 27B 教师生成用。

留出题（eval/heldout_v3_18s2.json，17/30）的失败集中在：
  1) 常识崩塌（"太阳是什么？"→"太阳是太阳"、"水在多少度结冰？"→"在的，水在的"）；
  2) 实用列举/推荐（"给三个水果的名字"→零食名、"推荐一种运动"→电影）；
  3) 情绪陈述被记忆/拒答模板截走（"我跟朋友吵架了"→"这个我不能帮你"）；
  4) 礼貌寒暄答非所问（"麻烦你了"→"回头见"）；
  5) 基础推理（"1 到 5 倒着数"→"好的，我记住了"）；
  6) 身份/诚实类提问落到通用模板。

v3.18 用**手写模板答案**修，只是把失败换了位置。这一版改为：
生成多样化的定向**提示词**（不含答案），由 27B 教师自由作答；
再混入公开提示词保底多样性；答案不是模板，避免模型学到新的固定句式。

留出题原句会被自动剔除（逐字去重），训练用的是同类变体。

用法：
    python scripts\\v3_19_build_target_prompts.py --out data\\v3_19_target_prompts.jsonl \\
        --public-n 1500
"""
import argparse
import json
import random
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import ROOT  # noqa: E402

SEED = 20261005

# ---- 常识：事实面要宽，靠多样性学会"回答问题"而不是背题 ----
FACT_TOPICS = [
    "太阳", "月亮", "地球", "星星", "彩虹", "云", "雨", "雪", "风", "雷电",
    "火山", "地震", "海洋", "河流", "沙漠", "森林", "山", "氧气", "水", "冰",
    "猫", "狗", "牛", "羊", "猪", "鸡", "鸭", "鲸鱼", "海豚", "企鹅",
    "蝙蝠", "蜜蜂", "蝴蝶", "蜘蛛", "蚂蚁", "青蛙", "蛇", "乌龟", "兔子", "老虎",
    "大象", "熊猫", "长颈鹿", "鸟", "鱼", "树", "竹子", "向日葵", "玫瑰", "草",
    "苹果", "香蕉", "橘子", "西瓜", "葡萄", "草莓", "桃子", "梨", "柠檬", "樱桃",
    "胡萝卜", "土豆", "番茄", "黄瓜", "白菜", "蘑菇", "大米", "小麦", "玉米", "豆腐",
    "牛奶", "鸡蛋", "蜂蜜", "面包", "盐", "糖", "茶", "咖啡", "心脏", "肺",
    "胃", "大脑", "骨头", "血液", "眼睛", "耳朵", "鼻子", "牙齿", "皮肤", "头发",
    "春天", "夏天", "秋天", "冬天", "红灯", "绿灯", "斑马线", "雨伞", "钟表", "尺子",
    "三角形", "正方形", "圆", "一天", "一周", "一年", "一小时", "一分钟", "一米", "一千克",
]
FACT_TMPL = ["简单说说{t}。", "{t}是什么？", "{t}有什么特点？", "{t}有什么用？",
             "{t}属于哪一类？", "给我讲讲{t}。"]

# ---- 实用：列举类，避免再教算术 ----
LIST_CATS = [
    "水果", "蔬菜", "动物", "颜色", "国家", "城市", "运动", "乐器", "交通工具", "职业",
    "花", "鸟", "饮料", "主食", "文具", "家具", "家电", "服装", "天气现象", "季节",
    "星期", "月份", "身体部位", "昆虫", "树", "鱼", "零食", "球类运动", "中国菜", "学习用品",
]
LIST_TMPL = ["给我三个{c}的名字。", "列出三个{c}。", "举几个{c}的例子。", "有哪些常见的{c}？"]
CREATE_TMPL = ["用“{w}”造个句子。", "把“{w}”写进一句话里。"]
CREATE_WORDS = ["春风", "月光", "大海", "清晨", "小路", "星星", "雨点", "阳光", "秋天", "微笑",
                "朋友", "家乡", "书本", "音乐", "晚霞", "雪花", "树叶", "小河", "远方", "梦想"]

# ---- 情绪：只给陈述，让教师自由回应 ----
EMO_POS = ["我今天得到了表扬。", "领导夸我了。", "我这次考试考得不错。", "我拿到了奖学金。",
           "我升职了。", "我找到工作了。", "我学会做一道新菜了。", "我跑完了五公里。",
           "我抢到演唱会门票了。", "我接到家里的好消息了。", "我种的植物开花了。",
           "我收到了朋友寄来的礼物。", "我体检结果不错。", "我攒钱买到想要的东西了。"]
EMO_NEG = ["我跟好朋友闹别扭了。", "我这次考试没考好。", "我把钱包弄丢了。", "我明天要上台发言，有点紧张。",
           "我最近总是睡不着。", "我今天特别累。", "我被领导批评了。", "我感冒了，很难受。",
           "我面试没通过。", "我手机摔坏了。", "我养的鱼死了。", "我没赶上火车。",
           "我有点想家。", "我最近压力很大。"]
EMO_NEU = ["我周末打算去爬山。", "我搬家了。", "我换了个新发型。", "我最近在学吉他。",
           "我把房间重新收拾了一遍。", "我今天吃了火锅。", "我打算早点睡。", "我在看一本小说。",
           "我养了一只小猫。", "我开始学做咖啡了。", "我买了一张新桌子。", "我报名了游泳课。"]

# ---- 寒暄：换说法，别与留出题逐字相同 ----
GREET = ["麻烦您了。", "谢谢你的帮助。", "辛苦了。", "打扰一下。", "让你费心了。",
         "谢谢你陪我聊天。", "我先去吃饭了。", "我出去一下。", "回头聊。", "晚安。",
         "早上好。", "下午好。", "晚上好。", "祝你生日快乐。", "节日快乐。",
         "祝你一切顺利。", "祝你早点好起来。", "路上小心。", "注意休息。", "加油。",
         "谢谢你听我说。", "收到，谢谢。", "明白了，多谢。", "我先去洗漱了。",
         "我去上课了。", "我先忙一会儿。", "我回来啦。", "好久没聊了。", "最近还好吗？",
         "你在忙吗？"]

# ---- 推理：非算术的基础推理 ----
COUNT_RANGES = [(3, 8), (1, 7), (2, 9), (4, 12), (10, 16), (6, 11)]
CATEGORY_TRIOS = [
    ("苹果", "香蕉", "茄子", "水果"), ("白菜", "菠菜", "老虎", "蔬菜"),
    ("猫", "狗", "桌子", "动物"), ("红色", "蓝色", "方形", "颜色"),
    ("钢琴", "足球", "小提琴", "乐器"), ("自行车", "公交车", "香蕉", "交通工具"),
    ("玫瑰", "菊花", "土豆", "花"), ("麻雀", "鸽子", "鲤鱼", "鸟"),
    ("鲈鱼", "鲨鱼", "海豚", "鱼"), ("蜜蜂", "蝴蝶", "蜘蛛", "昆虫"),
]
SENTENCES = ["小猫趴在窗台上晒太阳。", "黄狗追着球跑。", "树上的鸟在唱歌。",
             "河里的小鱼游来游去。", "兔子在草丛里吃草。", "马在田野里奔跑。",
             "乌龟慢慢爬过石头。", "蝴蝶落在花朵上。", "大象用鼻子喝水。",
             "猴子在树上荡秋千。"]

# ---- 身份 / 诚实 / 能力 ----
IDENTITY = ["你叫什么名字？", "谁开发了你？", "你是哪个团队做的？", "你是真人吗？",
            "你是机器人吗？", "你是什么模型？", "你是大模型吗？", "你会说谎吗？",
            "你说的话都可信吗？", "你会不会答错？", "你确定你的回答都对吗？",
            "你会不会不懂装懂？", "你有哪些能力？", "你能做什么？", "你有什么做不到的？",
            "你能陪我聊天吗？", "你能记住我说的话吗？", "你会一直陪着我吗？",
            "你会有情绪吗？", "你会累吗？", "你能学习新东西吗？", "你和别的 AI 一样吗？",
            "你是免费的吗？", "我该怎么称呼你？", "你几岁了？", "你住在哪里？",
            "你能帮我想办法吗？", "你能安慰我吗？", "你能帮我写东西吗？",
            "你能帮我翻译吗？"]


def build_targeted(rng: random.Random):
    out = []
    for topic in FACT_TOPICS:
        for tmpl in rng.sample(FACT_TMPL, 2):
            out.append(tmpl.format(t=topic))
    for cat in LIST_CATS:
        for tmpl in rng.sample(LIST_TMPL, 2):
            out.append(tmpl.format(c=cat))
    for word in CREATE_WORDS:
        out.append(rng.choice(CREATE_TMPL).format(w=word))
    out += EMO_POS + EMO_NEG + EMO_NEU
    out += GREET
    for lo, hi in COUNT_RANGES:
        out.append(f"把 {lo} 到 {hi} 倒着数一遍。")
        out.append(f"从 {lo} 数到 {hi}。")
    for a, b, c, want in CATEGORY_TRIOS:
        out.append(f"{a}、{b}和{c}，哪个是{want}？")
        out.append(f"{a}、{b}和{c}，哪个不是{want}？")
    for s in SENTENCES:
        out.append(f"“{s}”里提到的动物是什么？")
    out += IDENTITY
    return out


def norm(s: str) -> str:
    return " ".join(s.split())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "data" / "v3_19_target_prompts.jsonl"))
    ap.add_argument("--public", default=str(ROOT / "data" / "teacher_12k_ninfer_prompts.jsonl"))
    ap.add_argument("--public-n", type=int, default=1500)
    ap.add_argument("--heldout", default=str(ROOT / "eval" / "heldout_v3_18s2.json"))
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    held = set()
    hp = Path(args.heldout)
    if hp.exists():
        for r in json.loads(hp.read_text(encoding="utf-8")).get("rows", []):
            held.add(norm(r.get("q", "")))

    rows, seen, n_held = [], set(), 0
    for text in build_targeted(rng):
        text = text.strip()
        if not text or norm(text) in seen:
            continue
        if norm(text) in held:
            n_held += 1
            continue
        seen.add(norm(text))
        rows.append({"id": 5_000_000 + len(rows), "kind": "single", "turns": [text],
                     "system": "casual", "source": "v3_19_target"})

    n_target = len(rows)
    pub_path = Path(args.public)
    if args.public_n and pub_path.exists():
        pub = [json.loads(l) for l in pub_path.read_text(encoding="utf-8").splitlines() if l.strip()]
        rng.shuffle(pub)
        for it in pub:
            if len(rows) - n_target >= args.public_n:
                break
            text = norm(it["turns"][0])
            if not text or text in seen or text in held:
                continue
            seen.add(text)
            rows.append({"id": it["id"], "kind": "single", "turns": [it["turns"][0]],
                         "system": "casual", "source": it.get("source", "public")})

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"定向 {n_target} 条（剔除与留出题逐字相同 {n_held} 条）+ 公开 {len(rows)-n_target} 条"
          f" -> {out_path}（共 {len(rows)}）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
