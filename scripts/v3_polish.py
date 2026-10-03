"""Long-context chat polish: masked SFT on 8192-token packed conversations.
Run after the progressive long-context stages so the assistant behaviour (identity,
greetings, refusals) comes back without shrinking the window again.
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).parent))
from student_utils import chunked_lm_loss  # noqa: E402

ROOT = Path(r"D:\wt\feng-distill-30m")
V3 = ROOT / "v3"
ATTN = [SDPBackend.EFFICIENT_ATTENTION, SDPBackend.FLASH_ATTENTION]


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", default=str(V3 / "ctx32768" / "final"))
    ap.add_argument("--out", default=str(V3 / "polish_ctx8192"))
    ap.add_argument("--seq", type=int, default=8192)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=20261002)
    args = ap.parse_args()

    from transformers import AutoTokenizer, Qwen3ForCausalLM
    ids = np.load(V3 / "data" / f"sft{args.seq}_ids.npy", mmap_mode="r")
    mask = np.load(V3 / "data" / f"sft{args.seq}_mask.npy", mmap_mode="r")
    n_all = ids.shape[0]
    steps = max(1, int(n_all // (args.batch * args.accum) * args.epochs))
    log(f"[polish] {n_all} windows x {args.seq} | init {args.init} | "
        f"{args.batch}x{args.accum} | {steps} steps | lr {args.lr:g}")
    tok = AutoTokenizer.from_pretrained(ROOT / "v2" / "tokenizer")
    model = Qwen3ForCausalLM.from_pretrained(args.init, dtype=torch.float32)
    model.config.max_position_embeddings = args.seq
    model = model.to("cuda").train()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), eps=1e-8,
                            weight_decay=0.05, fused=True)
    rng = np.random.default_rng(args.seed)
    t0, tok_seen, peak, loss_last = time.time(), 0, 0.0, None
    for step in range(1, steps + 1):
        if step <= args.warmup:
            cur_lr = args.lr * step / args.warmup
        else:
            prog = (step - args.warmup) / max(1, steps - args.warmup)
            cur_lr = args.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * prog)))
        for g in opt.param_groups:
            g["lr"] = cur_lr
        opt.zero_grad(set_to_none=True)
        loss_sum = 0.0
        for _ in range(args.accum):
            idx = rng.integers(0, n_all, size=args.batch)
            a = np.asarray(ids[idx])[:, :args.seq]
            m = np.asarray(mask[idx])[:, :args.seq]
            t = torch.tensor(a, dtype=torch.long, device="cuda")
            mt = torch.tensor(m, dtype=torch.uint8, device="cuda")
            inp, lab = t[:, :-1], t[:, 1:]
            lab = lab.clone()
            lab[mt[:, 1:] == 0] = -100
            with torch.autocast("cuda", dtype=torch.bfloat16), sdpa_kernel(ATTN):
                loss_all = chunked_lm_loss(model, inp, lab, chunk=512)
            (loss_all / args.accum).backward()
            loss_sum += float(loss_all.detach())
            tok_seen += int(inp.numel())
            del inp, lab, loss_all, t, mt, a, m
        gn = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
        opt.step()
        loss_last = loss_sum / args.accum
        peak = max(peak, torch.cuda.max_memory_allocated() / 2**30)
        if step % 10 == 0 or step == steps:
            dt = time.time() - t0
            log(f"[polish] step {step}/{steps} loss {loss_last:.4f} lr {cur_lr:.2e} "
                f"gnorm {gn:.2f} {tok_seen/dt:,.0f} tok/s {dt/60:.1f}min peak {peak:.2f}GiB")
    dest = Path(args.out) / "final"
    dest.mkdir(parents=True, exist_ok=True)
    model.config.use_cache = True
    model.save_pretrained(dest, safe_serialization=True)
    tok.save_pretrained(dest)
    (Path(args.out) / "summary.json").write_text(json.dumps(
        {"init": args.init, "seq": args.seq, "steps": steps, "tokens": tok_seen,
         "loss": loss_last, "peak_gib": peak, "minutes": (time.time() - t0) / 60,
         "out": str(dest)}, indent=2), encoding="utf-8")
    log(f"polish done -> {dest} (loss {loss_last:.4f}, {(time.time()-t0)/60:.1f} min)")


if __name__ == "__main__":
    main()
