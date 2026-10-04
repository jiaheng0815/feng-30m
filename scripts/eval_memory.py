"""多轮记忆评测：说一个事实 → 追问 → 看模型能不能复述。

用法：
    python scripts/eval_memory.py --model v3_13/mem_pc [--out eval/memory_v3_13.json]
"""
import argparse
import json
import random
import sys
from pathlib import Path

import torch

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import ROOT  # noqa: E402
from v3_13_build_memory import CITIES, COLORS, FOODS, NAMES, PETS, SPORTS  # noqa: E402


def build_cases(seed=777, n=24):
    rng = random.Random(seed)
    kinds = [("姓名", NAMES, "我叫{v}，请记住。", "我叫什么名字？"),
             ("颜色", COLORS, "我最喜欢的颜色是{v}。", "我最喜欢什么颜色？"),
             ("城市", CITIES, "我住在{v}。", "我住在哪里？"),
             ("宠物", PETS, "我养了一只{v}。", "我养了什么？"),
             ("食物", FOODS, "我最喜欢{v}。", "我最喜欢什么？"),
             ("运动", SPORTS, "我最喜欢的运动是{v}。", "我最喜欢什么运动？")]
    cases = []
    for i in range(n):
        kind, pool, stmt, ask = kinds[i % len(kinds)]
        v = rng.choice(pool)
        cases.append({"kind": kind, "value": v, "statement": stmt.format(v=v), "ask": ask})
    return cases


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", default="")
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--max-new", type=int, default=32)
    args = ap.parse_args()

    from transformers import AutoTokenizer, Qwen3ForCausalLM

    tok = AutoTokenizer.from_pretrained(args.model)
    if not getattr(tok, "chat_template", None):      # 训练中间产物没带模板，用项目标准模板
        tok.chat_template = (
            r"{%- for message in messages %}"
            r"{{- '<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n' }}"
            r"{%- endfor %}"
            r"{%- if add_generation_prompt %}{{- '<|im_start|>assistant\n' }}{%- endif %}")
    model = Qwen3ForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, attn_implementation="sdpa").to("cuda").eval()

    cases = build_cases(n=args.n)
    rows, ok = [], 0
    for c in cases:
        msgs = [{"role": "user", "content": c["statement"]}]
        # 第一轮：模型确认
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        ids = torch.tensor([tok(text, add_special_tokens=False)["input_ids"]], device="cuda")
        with torch.no_grad():
            out = model.generate(ids, max_new_tokens=16, do_sample=False,
                                 pad_token_id=3, eos_token_id=0)
        ack = tok.decode(out[0][ids.shape[1]:].tolist(), skip_special_tokens=True).split("<|im_end|>")[0].strip()
        msgs.append({"role": "assistant", "content": ack})
        # 第二轮：追问
        msgs.append({"role": "user", "content": c["ask"]})
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        ids = torch.tensor([tok(text, add_special_tokens=False)["input_ids"]], device="cuda")
        with torch.no_grad():
            out = model.generate(ids, max_new_tokens=args.max_new, do_sample=False,
                                 pad_token_id=3, eos_token_id=0)
        reply = tok.decode(out[0][ids.shape[1]:].tolist(), skip_special_tokens=True).split("<|im_end|>")[0].strip()
        good = c["value"] in reply
        ok += int(good)
        rows.append({**c, "ack": ack[:40], "reply": reply[:80], "ok": good})
        print(f"  [{'OK ' if good else 'MISS'}] {c['kind']} {c['value']}: {reply[:50]!r}")
    print(f"=== 记忆评测 {args.model}: {ok}/{len(cases)} ===")
    if args.out:
        p = Path(args.out)
        if not p.is_absolute():
            p = ROOT / p
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({"model": args.model, "score": f"{ok}/{len(cases)}", "rows": rows},
                                ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"  -> {p}")


if __name__ == "__main__":
    main()
