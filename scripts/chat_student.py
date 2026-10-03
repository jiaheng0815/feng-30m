"""Interactive / scripted chat with the distilled student."""
import argparse
import sys
from pathlib import Path

import torch

sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=r"D:\wt\feng-distill-30m\student\stageB\final")
    ap.add_argument("--prompt", default=None, help="single prompt; omit for interactive")
    ap.add_argument("--max-new", type=int, default=128)
    ap.add_argument("--temp", type=float, default=0.0)
    args = ap.parse_args()

    from transformers import AutoTokenizer, Qwen3ForCausalLM
    tok = AutoTokenizer.from_pretrained(args.model)
    model = Qwen3ForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16).to("cuda").eval()

    history = []

    def reply(user):
        history.append(("user", user))
        text = "".join(
            f"<|im_start|>{r}\n{c}<|im_end|>\n" for r, c in history) + "<|im_start|>assistant\n"
        ids = tok(text, add_special_tokens=False)["input_ids"]
        inp = torch.tensor([ids], dtype=torch.long, device="cuda")
        with torch.no_grad():
            out = model.generate(input_ids=inp, max_new_tokens=args.max_new, do_sample=args.temp > 0,
                                 temperature=max(args.temp, 1e-5), top_p=0.9,
                                 repetition_penalty=1.15, no_repeat_ngram_size=6,
                                 pad_token_id=3, eos_token_id=0)
        resp = tok.decode(out[0][inp.shape[1]:].tolist()).split("<|im_end|>")[0].strip()
        history.append(("assistant", resp))
        return resp

    if args.prompt:
        print(f"你: {args.prompt}")
        print(f"feng-30m: {reply(args.prompt)}")
        return
    print("(输入 q 退出)")
    while True:
        try:
            u = input("你: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not u or u.lower() in ("q", "quit", "exit"):
            break
        print(f"feng-30m: {reply(u)}")


if __name__ == "__main__":
    main()
