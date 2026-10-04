"""多轮对话测试：按给定话题顺序连续聊，检查模型会不会"复读自己上一轮的话"。

用法：
    python scripts/chat_multi.py --model v3_4\\release --turns "你是谁？;你可以干什么;我很高兴;我很伤心;我很难过"
可选：--rep 1.25（重复惩罚，默认 1.0 = 关闭，模拟 llama-cli 的默认行为）、--temp 0.0
"""
import argparse
import re
import sys
from collections import Counter
from pathlib import Path

import torch

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import ROOT  # noqa: E402
from runtime_tools import tool_answer  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--turns", default="你是谁？;你可以干什么;我很高兴;我很伤心;我很痛苦;我很快乐;我想死")
    ap.add_argument("--rep", type=float, default=1.0, help="重复惩罚（1.0 = 关闭）")
    ap.add_argument("--temp", type=float, default=0.0)
    ap.add_argument("--max-new", type=int, default=72)
    args = ap.parse_args()

    from transformers import AutoTokenizer, Qwen3ForCausalLM
    tok = AutoTokenizer.from_pretrained(args.model)
    if not getattr(tok, "chat_template", None):      # 训练中间产物没带模板，用项目标准模板
        tok.chat_template = (
            r"{%- for message in messages %}"
            r"{{- '<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n' }}"
            r"{%- endfor %}"
            r"{%- if add_generation_prompt %}{{- '<|im_start|>assistant\n' }}{%- endif %}")
    model = Qwen3ForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16).to("cuda").eval()

    turns = [t.strip() for t in args.turns.split(";") if t.strip()]
    msgs = []
    replies = []
    print(f"=== 多轮测试 {args.model}（rep={args.rep}, temp={args.temp}）===")
    for q in turns:
        msgs.append({"role": "user", "content": q})
        tool = tool_answer(q)                       # 算式/时间/随机数走 tool，不进模型
        if tool is not None:
            r = f"[calc] {tool}"
            msgs.append({"role": "assistant", "content": tool})
            replies.append(r)
            print(f"  [user] {q}\n  [feng] {r[:100]}")
            continue
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        ids = tok(text, add_special_tokens=False)["input_ids"]
        inp = torch.tensor([ids], dtype=torch.long, device="cuda")
        kw = dict(max_new_tokens=args.max_new, pad_token_id=3, eos_token_id=0)
        if args.temp > 0:
            kw.update(do_sample=True, temperature=args.temp, top_p=0.9,
                      repetition_penalty=args.rep, no_repeat_ngram_size=6)
        else:
            kw.update(do_sample=False)
            if args.rep != 1.0:
                kw.update(repetition_penalty=args.rep, no_repeat_ngram_size=6)
        with torch.no_grad():
            out = model.generate(input_ids=inp, **kw)
        r = tok.decode(out[0][inp.shape[1]:].tolist()).split("<|im_end|>")[0].strip()
        msgs.append({"role": "assistant", "content": r})
        replies.append(r)
        print(f"  [user] {q}\n  [feng] {r[:100]}")

    norm = [re.sub(r"\s+", "", r)[:24] for r in replies]
    cnt = Counter(norm)
    top, n = cnt.most_common(1)[0]
    print(f"\n不同回答比例 {len(cnt)/len(replies):.2f} | 最高频回答占比 {n/len(replies):.2f} "
          f"（\"{top[:30]}…\"）")


if __name__ == "__main__":
    main()
