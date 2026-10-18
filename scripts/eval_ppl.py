"""留出集困惑度（PPL）：不依赖关键词判分器的稳定回归指标。

数据默认用 `data/v4_val.jsonl`（v4 SFT 在过采样前切出的 689 条验证集，
只用于 val_loss，不参与训练）。指标同时给：
  - ppl_asst：只对 assistant token 算 loss（与 SFT 目标一致）
  - ppl_all ：对全部 token 算 loss（观察通用语言建模）
用法：
    python scripts/eval_ppl.py --model v4/base --out eval/ppl_v3_19pc4.json
"""
import argparse
import json
import math
import sys
from pathlib import Path

import torch

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", default=str(ROOT / "data" / "v4_val.jsonl"))
    ap.add_argument("--out", default="")
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    from transformers import AutoTokenizer, Qwen3ForCausalLM
    tok = AutoTokenizer.from_pretrained(args.model)
    model = Qwen3ForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, attn_implementation="sdpa").to(args.device).eval()

    rows = [json.loads(l) for l in Path(args.data).read_text(encoding="utf-8").splitlines()
            if l.strip()]
    if args.limit > 0:
        rows = rows[:args.limit]

    nll_a = nll_all = 0.0
    n_a = n_all = 0
    samples = 0
    with torch.no_grad():
        for r in rows:
            ids, mask = [], []
            for m in r["messages"]:
                t = tok.encode(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n",
                               add_special_tokens=False)
                ids += t
                mask += [1 if m["role"] == "assistant" else 0] * len(t)
            if len(ids) < 2:
                continue
            ids, mask = ids[-args.max_len:], mask[-args.max_len:]
            inp = torch.tensor([ids[:-1]], dtype=torch.long, device=args.device)
            lab = torch.tensor([ids[1:]], dtype=torch.long, device=args.device)
            m = torch.tensor([mask[1:]], dtype=torch.bool, device=args.device)
            logits = model(input_ids=inp).logits.float()
            logp = torch.log_softmax(logits, dim=-1)
            tok_lp = logp.gather(-1, lab.unsqueeze(-1)).squeeze(-1)
            nll_all += float(-tok_lp.sum())
            n_all += int(tok_lp.numel())
            if m.any():
                nll_a += float(-tok_lp[m].sum())
                n_a += int(m.sum())
            samples += 1
            if samples % 100 == 0:
                print(f"  {samples}/{len(rows)}", flush=True)

    ppl_a = math.exp(nll_a / n_a) if n_a else float("nan")
    ppl_all = math.exp(nll_all / n_all) if n_all else float("nan")
    out = {"model": args.model, "data": args.data, "samples": samples,
           "assistant_tokens": n_a, "all_tokens": n_all,
           "nll_assistant": nll_a, "nll_all": nll_all,
           "ppl_assistant": ppl_a, "ppl_all": ppl_all}
    print(json.dumps(out, ensure_ascii=False, indent=1))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=1),
                                  encoding="utf-8")


if __name__ == "__main__":
    main()
