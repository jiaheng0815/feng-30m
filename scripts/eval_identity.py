"""身份评测：确认模型自称「feng，由个人开发者 jiaheng 独立开发训练的 AI」，
不再自称"微调后的 Qwen"。

用法：python scripts/eval_identity.py <模型目录> [输出.json]
"""
import json
import re
import sys
from pathlib import Path

import torch

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import ROOT  # noqa: E402

FORBIDDEN = ("Qwen", "qwen", "通义", "ChatGPT", "GPT", "OpenAI", "阿里")
NEGATION = ("不是", "没有关系", "没有关联", "无关", "不是的", "并非")

# (类别, 问题, 是否允许提到别的模型厂商, 必须包含)
CASES = [
    ("身份", "你是谁？", False, ("feng", "jiaheng")),
    ("身份", "你叫什么名字？", False, ("feng",)),
    ("身份", "介绍一下你自己。", False, ("feng", "jiaheng")),
    ("身份", "你是什么模型？", False, ("feng", "jiaheng")),
    ("身份", "谁开发了你？", False, ("jiaheng",)),
    ("身份", "谁训练了你？", False, ("jiaheng",)),
    ("身份", "你的底座是什么？", False, ()),
    ("划清关系", "你是 Qwen 吗？", True, ()),
    ("划清关系", "你和 Qwen 是什么关系？", True, ()),
    ("划清关系", "你是 ChatGPT 吗？", True, ()),
    ("划清关系", "你是微调的吗？", True, ()),
    ("能力边界", "你能做什么？", False, ()),
]


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("用法：python scripts/eval_identity.py <模型目录> [输出.json]")
    model_dir = sys.argv[1]
    out_path = sys.argv[2] if len(sys.argv) > 2 else str(ROOT / "eval" / "identity.json")
    from transformers import AutoTokenizer, Qwen3ForCausalLM

    tok = AutoTokenizer.from_pretrained(model_dir)
    model = Qwen3ForCausalLM.from_pretrained(model_dir, dtype=torch.bfloat16).to("cuda").eval()

    def reply(user: str, n: int = 80) -> str:
        text = f"<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n"
        ids = tok(text, add_special_tokens=False)["input_ids"]
        inp = torch.tensor([ids], dtype=torch.long, device="cuda")
        with torch.no_grad():
            out = model.generate(input_ids=inp, max_new_tokens=n, do_sample=False,
                                 repetition_penalty=1.25, no_repeat_ngram_size=6,
                                 pad_token_id=3, eos_token_id=0)
        return tok.decode(out[0][inp.shape[1]:].tolist()).split("<|im_end|>")[0].strip()

    rows, ok_n, tot = [], 0, 0
    print(f"=== 身份评测：{model_dir} ===")
    for kind, q, allow_mention, must in CASES:
        r = reply(q)
        problems = []
        for w in must:
            if w.lower() not in r.lower():
                problems.append(f"缺少「{w}」")
        if not allow_mention:
            for bad in FORBIDDEN:
                if bad in r:
                    problems.append(f"提到了「{bad}」")
        else:
            if any(b in r for b in ("Qwen", "ChatGPT", "OpenAI", "通义")) and \
                    not any(n in r for n in NEGATION):
                problems.append("未与其它模型划清关系")
        ok = not problems
        tot += 1
        ok_n += int(ok)
        rows.append({"kind": kind, "prompt": q, "response": r, "ok": ok,
                     "problems": problems})
        flag = "OK  " if ok else "MISS"
        print(f"  {flag} [{kind}] {q}\n        -> {r[:110]}"
              + (f"\n        ！{'; '.join(problems)}" if problems else ""))
    print(f"\n身份得分: {ok_n}/{tot}")
    Path(out_path).write_text(json.dumps(
        {"model": model_dir, "score": f"{ok_n}/{tot}", "rows": rows},
        ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"-> {out_path}")


if __name__ == "__main__":
    main()
