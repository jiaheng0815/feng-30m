"""对话质量评测：抓"模板坍缩"——模型对不同输入回同一句话（v3.4 踩过的坑）。

指标：
  distinct     不同回答数 / 总题数（越高越好）
  top_share    最高频回答的占比（越低越好）
  blurb_leak   非"能力类"问题上回答里出现能力介绍模板的次数（越低越好）
  emotion_ok   情绪类输入是否给出与情绪相关的回应（含情绪词）

用法：python scripts/eval_chat.py <模型目录> [输出.json]
"""
import json
import re
import sys
from collections import Counter
from pathlib import Path

import torch

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import ROOT  # noqa: E402

# (类别, 输入, 期望出现的词（任一）)
CASES = [
    ("情绪", "我很高兴", ("高兴", "开心", "好", "棒", "恭喜")),
    ("情绪", "我很难过", ("难过", "别", "抱", "陪", "安慰", "发生", "愿意听")),
    ("情绪", "我很伤心", ("伤心", "别", "抱", "陪", "安慰", "发生", "愿意听")),
    ("情绪", "我很痛苦", ("痛苦", "难", "别", "抱", "陪", "安慰", "愿意听")),
    ("情绪", "我很快乐", ("快乐", "开心", "好", "棒")),
    ("情绪", "我很孤独", ("孤独", "陪", "抱", "朋友", "聊聊", "愿意听")),
    ("情绪", "我想死", ("别", "不", "陪", "找", "帮", "专业", "热线", "抱")),
    ("情绪", "我今天心情不太好", ("心情", "别", "抱", "陪", "放松", "聊聊")),
    ("寒暄", "你好", ("你好", "嗨", "在", "feng")),
    ("寒暄", "在吗？", ("在", "你好", "feng")),
    ("寒暄", "早上好", ("早", "好")),
    ("寒暄", "晚安", ("晚安", "睡", "好")),
    ("寒暄", "谢谢你", ("不客气", "不用谢", "别客气", "高兴")),
    ("闲聊", "给我讲个笑话", None),
    ("任务", "推荐一部电影", None),
    ("任务", "给我起个网名", None),
    ("任务", "把“今天天气很好”翻译成英文", ("weather", "fine", "nice", "good")),
    ("任务", "1 加 1 等于几？", ("2", "二")),
    ("事实", "中国的首都是哪里？", ("北京",)),
    ("事实", "水的化学式是什么？", ("H2O", "H₂O", "水")),
]
BLURB = ("帮你写作、翻译和写简单代码", "可以陪你聊天", "陪你聊天、帮你写作")
CAPABILITY_Q = ("你能做什么", "你会做什么", "你有什么能力")


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("用法：python scripts/eval_chat.py <模型目录> [输出.json]")
    model_dir = sys.argv[1]
    out_path = sys.argv[2] if len(sys.argv) > 2 else str(ROOT / "eval" / "chat_quality.json")
    from transformers import AutoTokenizer, Qwen3ForCausalLM

    tok = AutoTokenizer.from_pretrained(model_dir)
    model = Qwen3ForCausalLM.from_pretrained(model_dir, dtype=torch.bfloat16).to("cuda").eval()

    def reply(user: str, n: int = 72) -> str:
        text = f"<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n"
        ids = tok(text, add_special_tokens=False)["input_ids"]
        inp = torch.tensor([ids], dtype=torch.long, device="cuda")
        with torch.no_grad():
            out = model.generate(input_ids=inp, max_new_tokens=n, do_sample=False,
                                 repetition_penalty=1.25, no_repeat_ngram_size=6,
                                 pad_token_id=3, eos_token_id=0)
        return tok.decode(out[0][inp.shape[1]:].tolist()).split("<|im_end|>")[0].strip()

    rows = []
    print(f"=== 对话质量评测：{model_dir} ===")
    for kind, q, expect in CASES:
        r = reply(q)
        leak = any(b in r for b in BLURB) and not any(c in q for c in CAPABILITY_Q)
        topic_ok = None if expect is None else any(w in r for w in expect)
        rows.append({"kind": kind, "prompt": q, "response": r,
                     "blurb_leak": leak, "topic_ok": topic_ok})
        flag = "LEAK" if leak else ("OK  " if topic_ok else ("MISS" if topic_ok is False else "    "))
        print(f"  {flag} [{kind}] {q}\n        -> {r[:96]}")

    norms = [re.sub(r"\s+", "", r["response"])[:24] for r in rows]
    cnt = Counter(norms)
    top, top_n = cnt.most_common(1)[0]
    distinct = len(cnt) / len(rows)
    leak_n = sum(1 for r in rows if r["blurb_leak"])
    emo = [r for r in rows if r["kind"] == "情绪"]
    emo_ok = sum(1 for r in emo if r["topic_ok"])
    summary = {"model": model_dir, "n": len(rows), "distinct_ratio": round(distinct, 3),
               "top_response_share": round(top_n / len(rows), 3),
               "top_response": top, "blurb_leak": leak_n,
               "emotion_ok": f"{emo_ok}/{len(emo)}"}
    print(f"\n不同回答比例 {distinct:.2f} | 最高频回答占比 {top_n/len(rows):.2f} "
          f"（\"{top[:28]}…\"）| 能力模板泄漏 {leak_n} 次 | 情绪回应 {emo_ok}/{len(emo)}")
    Path(out_path).write_text(json.dumps({"summary": summary, "rows": rows},
                                         ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"-> {out_path}")


if __name__ == "__main__":
    main()
