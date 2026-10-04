"""v3.21 定向提示词：针对 v3.19 剩余失败（覆盖缺口）的第二批。

v3.19 留出 19/30（strict-v4）后仍失败的模式与诊断：
  - "太阳是什么？" → 复读"太阳是太阳系…"（v3.19 只随机训了 2/6 个句式，没覆盖"是什么"）
  - "水在多少度结冰？" → 答"8 度"（教师数据里根本没有 0°C 这条事实）
  - "给我推荐一种运动" → 错推荐（只训过"列出三个运动"）
  - "把 1 到 5 倒着数" → 乱答（只训过 6 个区间）
  - "祝你周末愉快" → 话题错位；"三角形有几条边？" → "边是边"
做法：事实主题**全 6 句式覆盖** + 显式知识事实 + 推荐类 + 计数扩展 + 诚实/身份 + 祝福/情绪变体；
自动剔除与留出题逐字相同的原句，并与 v3.19 的定向提示去重。

用法：
    python scripts\\v3_21_build_target_prompts.py --out data\\v3_21_target_prompts.jsonl
"""
import argparse
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import ROOT  # noqa: E402
from v3_19_build_target_prompts import FACT_TOPICS, FACT_TMPL  # noqa: E402

# 显式知识事实（避开留出题原句，用变体句式）
KNOWLEDGE = [
    "水在什么温度会结冰？", "水多少度沸腾？", "冰是水的什么形态？", "水在零下会怎么样？",
    "水的沸点是多少？", "冰融化成什么？", "水蒸气是什么形态的水？",
    "三角形有几个角？", "三角形的内角和是多少度？", "三角形有多少个顶点？",
    "正方形有几条边？", "长方形有几条边？", "五边形有几条边？", "圆形有几条边？",
    "一天有几个小时？", "一周有几天？", "一年有几个月？", "一小时有多少分钟？",
    "一分钟有多少秒？", "一年四季分别是什么？", "一个月大约有多少天？",
    "太阳是一颗什么星？", "太阳会发光吗？", "月亮自己会发光吗？", "月亮围着什么转？",
    "地球是行星吗？", "地球围着什么转？", "星星为什么会发亮？", "白天和黑夜是怎么来的？",
    "猫属于什么动物？", "鲸鱼是鱼吗？", "企鹅会飞吗？", "蝙蝠是鸟吗？",
    "红灯表示什么？", "绿灯表示什么？", "过马路要走哪里？",
]

RECOMMEND_CATS = ["运动", "水果", "书", "电影", "城市", "学科", "乐器", "饮料", "零食",
                  "学习方法", "休闲活动", "早餐", "花", "宠物", "锻炼方式", "旅行目的地"]
RECOMMEND_EXTRA = [
    "推荐一个适合放松的运动。", "推荐一个适合睡前看的书。", "推荐一部适合全家看的电影。",
    "推荐一种好养活的植物。", "推荐一道简单的家常菜。", "推荐一个周末能去的地方。",
    "推荐一首适合学习时听的音乐。", "推荐一个适合新手的爱好。",
]

WISHES = ["生日快乐", "节日快乐", "一切顺利", "早点好起来", "旅途愉快", "考试顺利",
          "工作顺利", "身体健康", "天天开心", "新年快乐", "假期愉快", "早日康复",
          "心想事成", "好运连连", "晚安好梦", "前程似锦"]

COUNT_RANGES = [(2, 7), (1, 6), (3, 10), (2, 11), (5, 14), (1, 8), (4, 9), (6, 13),
                (7, 15), (8, 12)]

HONESTY = ["你会说谎吗？", "你能保证你说的都对吗？", "你是不是在骗我？", "你是真人吗？",
           "你的答案可靠吗？", "你会不会不懂装懂？", "你确定你的回答对吗？",
           "你是机器人还是真人？", "你会骗我吗？", "你说的是真的吗？",
           "你答错了怎么办？", "你能承认自己不会吗？"]

PRAISE = ["我今天被老师夸了。", "领导表扬我了。", "我今天拿了第一名。", "我考试进步了。",
          "我被选为班长了。", "我拿到了奖学金。", "我做的菜被夸好吃。", "我坚持早起一周了。"]
UPSET = ["我和朋友闹矛盾了。", "我跟家人吵架了。", "我被朋友误会了。", "我心情有点低落。",
         "我觉得很委屈。", "我和同学闹别扭了。", "我今天摔了一跤。", "我把作业忘在家里了。"]


def build():
    out = []
    for topic in FACT_TOPICS:
        for tmpl in FACT_TMPL:                     # 全 6 句式，不再随机挑 2 个
            out.append(tmpl.format(t=topic))
    out += KNOWLEDGE
    for cat in RECOMMEND_CATS:
        out.append(f"给我推荐一种{cat}。")
    out += RECOMMEND_EXTRA
    for w in WISHES:
        out.append(f"祝你{w}。")
    for lo, hi in COUNT_RANGES:
        out.append(f"把 {lo} 到 {hi} 倒着数一遍。")
        out.append(f"从 {lo} 数到 {hi}。")
    out += HONESTY + PRAISE + UPSET
    return out


def norm(s):
    return " ".join(s.split())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "data" / "v3_21_target_prompts.jsonl"))
    ap.add_argument("--heldout", default=str(ROOT / "eval" / "heldout_v3_18s2.json"))
    ap.add_argument("--dedup", default=str(ROOT / "data" / "v3_19_target_prompts.jsonl"))
    args = ap.parse_args()

    held = set()
    hp = Path(args.heldout)
    if hp.exists():
        for r in json.loads(hp.read_text(encoding="utf-8")).get("rows", []):
            held.add(norm(r.get("q", "")))
    done = set()
    dp = Path(args.dedup)
    if dp.exists():
        for line in dp.read_text(encoding="utf-8").splitlines():
            if line.strip():
                done.add(norm(json.loads(line)["turns"][0]))

    rows, seen, n_held, n_dup = [], set(), 0, 0
    for text in build():
        text = text.strip()
        key = norm(text)
        if not text or key in seen:
            continue
        if key in held:
            n_held += 1
            continue
        if key in done:
            n_dup += 1
            continue
        seen.add(key)
        rows.append({"id": 6_000_000 + len(rows), "kind": "single", "turns": [text],
                     "system": "casual", "source": "v3_21_target"})

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"生成 {len(rows)} 条（剔除留出原句 {n_held}、v3.19 重复 {n_dup}）-> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
