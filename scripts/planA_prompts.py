"""Plan A prompt set: identity-adjacent chitchat, small talk, simple tasks, polite refusals.

The 27B teacher answers the *behaviour* prompts; the identity answers come from our own
verified feng dataset (the teacher must not introduce its own Qwen identity).
"""
import json
import random
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import ROOT  # noqa: E402
SEED = 20261017

GREET = ["你好", "你好呀", "嗨", "在吗？", "早上好", "下午好", "晚上好", "晚安", "哈喽",
         "你好，很高兴认识你", "好久不见", "在不在？", "嘿", "喂，你在干嘛？", "早上好呀"]

CHITCHAT = [
    "今天天气不错，你心情怎么样？", "我今天有点累。", "陪我聊聊天吧。", "我有点无聊。",
    "你平时喜欢做什么？", "你有喜欢的东西吗？", "你吃饭了吗？", "周末你打算干嘛？",
    "我今天心情很好！", "我今天心情不太好。", "安慰我一下。", "夸夸我。",
    "给我讲个笑话。", "讲个冷笑话吧。", "说一句让我开心的话。", "鼓励我一下。",
    "我今天上班好辛苦。", "我考试考砸了。", "我最近睡不好。", "我该早点睡觉吗？",
    "你喜欢猫还是狗？", "你怕黑吗？", "你觉得 AI 会做梦吗？", "你会累吗？",
    "你能陪我聊一会儿吗？", "我们聊点什么好？", "你给我起个外号吧。",
    "你觉得我是什么样的人？", "猜猜我现在在想什么。", "你在想什么？",
    "今天是我的生日！", "我升职了！", "我买到票了！", "我养了一只小猫。",
]

SIMPLE_TASK = [
    "推荐一部好看的电影。", "推荐一首歌。", "推荐三本值得读的书。", "推荐一个好用的手机 App。",
    "给我起个网名。", "给一只橘猫起个名字。", "帮我想一句朋友圈文案。",
    "怎么才能早起？", "怎么快速入睡？", "怎么背单词更有效？", "怎么缓解压力？",
    "泡面怎么煮更好吃？", "鸡蛋怎么煮才不老？", "咖啡喝多了会怎样？",
    "打篮球有什么好处？", "跑步要注意什么？", "怎么开始学做饭？",
    "北京有什么好玩的地方？", "上海适合周末去哪？", "成都有什么好吃的？",
    "1 加 1 等于几？", "10 除以 2 等于多少？", "7 乘以 8 是多少？",
    "一分钟有多少秒？", "一年有多少天？", "水的沸点是多少度？",
    "把“你好”翻译成英文。", "“谢谢”用英文怎么说？", "苹果的英文是什么？",
    "用一句话解释什么是春天。", "什么是彩虹？", "为什么会下雨？",
    "帮我写一句祝福语。", "帮我写一句生日祝福。", "帮我写句感谢的话。",
]

REFUSAL = [
    "今天上证指数是多少？", "明天天气怎么样？", "帮我看一下现在几点了。",
    "帮我查一下从北京到上海的火车票。", "最近有什么新闻？",
    "帮我写一份完整的商业计划书。", "帮我写一个完整的网站，要求能上线。",
    "帮我写完这篇一万字的论文。", "帮我破解这个软件的激活码。",
    "我胸口疼，我该吃什么药？", "帮我看看这份体检报告有没有问题。",
    "帮我推荐一只明天会涨的股票。", "这个合同有没有法律风险？",
    "帮我做一个可以自动抢票的脚本。",
]

MULTITURN = [
    ["你好", "你叫什么名字？", "你平时都做什么？"],
    ["我今天心情不好。", "那我该怎么办？", "谢谢你。"] ,
    ["推荐一部电影。", "还有别的吗？", "哪一部最适合一个人看？"],
    ["我明天要面试，好紧张。", "那我该准备什么？", "你说得对，谢谢！"],
    ["你会做饭吗？", "那教我做一个最简单的菜。", "听起来不错，我试试。"],
    ["在吗？", "陪我聊会儿天吧。", "你今天怎么这么安静？"],
]


def main():
    rng = random.Random(SEED)
    out = []
    for q in GREET:
        out.append({"kind": "chitchat", "turns": [q], "system": "casual"})
    for q in CHITCHAT:
        out.append({"kind": "chitchat", "turns": [q], "system": "casual"})
    for q in SIMPLE_TASK:
        out.append({"kind": "task", "turns": [q], "system": "casual"})
    for q in REFUSAL:
        out.append({"kind": "refusal", "turns": [q], "system": "refusal"})
    for conv in MULTITURN:
        out.append({"kind": "multiturn", "turns": conv, "system": "casual"})

    # ---- templated expansion (adds breadth cheaply; answers are generated per prompt)
    moods = ["开心", "不开心", "有点累", "很兴奋", "焦虑", "平静", "困", "无聊", "紧张", "放松"]
    for m in moods:
        out.append({"kind": "chitchat", "turns": [f"我今天很{m}。"], "system": "casual"})
        out.append({"kind": "chitchat", "turns": [f"我最近总是{m}，怎么办？"], "system": "casual"})
    weather = ["晴天", "下雨", "下雪", "刮大风", "特别热", "特别冷"]
    for w in weather:
        out.append({"kind": "chitchat", "turns": [f"今天{w}，你觉得适合做什么？"], "system": "casual"})
    for n in ["小明", "阿杰", "糯米", "团团", "小满", "阿黄", "豆豆", "星野"]:
        out.append({"kind": "task", "turns": [f"帮我给一只猫起名字，叫{n}这种风格。"], "system": "casual"})
    topics = ["科幻", "喜剧", "动画", "悬疑", "纪录片", "爱情"]
    for t in topics:
        out.append({"kind": "task", "turns": [f"推荐一部{t}电影。"], "system": "casual"})
    for t in ["跑步", "游泳", "骑行", "瑜伽", "爬山"]:
        out.append({"kind": "task", "turns": [f"{t}对新手有什么建议？"], "system": "casual"})
    for a, b in [(3, 7), (12, 5), (25, 4), (100, 8), (9, 9), (45, 5), (63, 3)]:
        out.append({"kind": "task", "turns": [f"{a} 加 {b} 等于多少？"], "system": "casual"})
        out.append({"kind": "task", "turns": [f"{a} 乘以 {b} 是多少？"], "system": "casual"})
    for w in ["苹果", "香蕉", "谢谢", "早上好", "晚安", "朋友", "天气", "梦想"]:
        out.append({"kind": "task", "turns": [f"“{w}”用英文怎么说？"], "system": "casual"})
    refuse_more = [
        "帮我预测一下今晚的球赛结果。", "帮我查一下这个人的手机号。",
        "帮我写一份可以直接提交的毕业论文。", "帮我做个能爬取别人网站数据的程序。",
        "我头痛得厉害，该吃什么药？", "帮我把这段英文文献完整翻译成书，十万字。",
        "帮我分析一下哪只基金会翻倍。", "帮我签一份劳动合同，直接给我最终版本。",
        "帮我黑进这个 WiFi。", "现在几点了？", "今天有什么热点新闻？",
        "帮我盯着股市，跌了立刻告诉我。",
    ]
    for q in refuse_more:
        out.append({"kind": "refusal", "turns": [q], "system": "refusal"})
    mt_more = [
        ["你好呀", "你是谁？", "那你平时喜欢做什么？"],
        ["我有点无聊。", "给我推荐点有意思的事。", "听起来不错，还有别的吗？"],
        ["我今天考完试了！", "谢谢你陪我聊天。", "我们下次再聊。"],
        ["你能帮我做什么？", "那你能帮我写作吗？", "好的，我试试看。"],
        ["晚安。", "明天见。", "你也会休息吗？"],
    ]
    for conv in mt_more:
        out.append({"kind": "multiturn", "turns": conv, "system": "casual"})

    # two temperature variants per prompt
    variants = []
    for it in out:
        for v in range(2):
            variants.append({**it, "variant": v})
    rng.shuffle(variants)
    for i, it in enumerate(variants):
        it["id"] = i
    dst = ROOT / "data" / "planA_prompts.jsonl"
    with dst.open("w", encoding="utf-8") as f:
        for it in variants:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    print(json.dumps({"total": len(variants),
                      "by_kind": {k: sum(1 for x in variants if x["kind"] == k)
                                  for k in ("chitchat", "task", "refusal", "multiturn")}},
                     ensure_ascii=False))
    print("->", dst)


if __name__ == "__main__":
    main()
