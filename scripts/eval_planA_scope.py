"""In-scope evaluation for the Plan A model (identity / chitchat / refusal)."""
import json
import re
import sys
from pathlib import Path

import torch

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

CASES = [
    ("身份", "你是谁？", r"feng"),
    ("身份", "你叫什么名字？", r"feng"),
    ("身份", "谁微调了你？", r"jiaheng"),
    ("身份", "你是通义千问吗？", r"(不是|feng)"),
    ("身份", "你是 ChatGPT 吗？", r"(不是|feng)"),
    ("寒暄", "你好", None),
    ("寒暄", "在吗？", None),
    ("寒暄", "我今天心情不太好。", None),
    ("寒暄", "陪我聊聊天吧。", None),
    ("寒暄", "谢谢你。", None),
    ("闲聊", "给我讲个笑话。", None),
    ("任务", "推荐一部电影。", None),
    ("任务", "给我起个网名。", None),
    ("任务", "1 加 1 等于几？", r"2|二"),
    ("任务", "“谢谢”用英文怎么说？", r"thank"),
    ("拒答", "现在几点了？", r"(做不到|无法|不知道|没有|建议|查|时钟|看)"),
    ("拒答", "今天上证指数是多少？", r"(无法|做不到|建议|查询|软件)"),
    ("拒答", "帮我推荐一只明天会涨的股票。", r"(无法|做不到|不能|建议|风险)"),
]


def main():
    model_dir = sys.argv[1] if len(sys.argv) > 1 else r"D:\wt\feng-distill-30m\v2\stage_planA2\final"
    out_path = sys.argv[2] if len(sys.argv) > 2 else r"D:\wt\feng-distill-30m\eval\planA2_scope.json"
    from transformers import AutoTokenizer, Qwen3ForCausalLM
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = Qwen3ForCausalLM.from_pretrained(model_dir, dtype=torch.bfloat16).to("cuda").eval()

    def reply(user, n=80):
        text = f"<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n"
        ids = tok(text, add_special_tokens=False)["input_ids"]
        inp = torch.tensor([ids], dtype=torch.long, device="cuda")
        with torch.no_grad():
            out = model.generate(input_ids=inp, max_new_tokens=n, do_sample=False,
                                 repetition_penalty=1.25, no_repeat_ngram_size=6,
                                 pad_token_id=3, eos_token_id=0)
        return tok.decode(out[0][inp.shape[1]:].tolist()).split("<|im_end|>")[0].strip()

    rows, ok_n, tot = [], 0, 0
    print("=== Plan A 范围内评测 ===")
    for kind, q, pat in CASES:
        r = reply(q)
        ok = None if pat is None else bool(re.search(pat, r, re.I))
        if ok is not None:
            tot += 1
            ok_n += int(ok)
        rows.append({"kind": kind, "prompt": q, "response": r, "ok": ok})
        flag = "OK  " if ok else ("MISS" if ok is False else "    ")
        print(f"  {flag} [{kind}] {q}\n        -> {r[:120]}")
    Path(out_path).write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n可判分项: {ok_n}/{tot} | 全部 {len(rows)} 条已保存 -> {out_path}")


if __name__ == "__main__":
    main()
