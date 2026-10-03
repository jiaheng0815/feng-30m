"""Chat fine-tuning on UNPACKED conversations (no cross-conversation contamination)."""
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
from student_config import build_config  # noqa: E402
from student_utils import chunked_lm_loss  # noqa: E402

ROOT = Path(r"D:\wt\feng-distill-30m")
STU = ROOT / "student"
ATTN = [SDPBackend.EFFICIENT_ATTENTION]


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(STU / "stageA" / "final"))
    ap.add_argument("--out", default=str(STU / "chat"))
    ap.add_argument("--tag", default="_8k")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--warmup", type=int, default=40)
    ap.add_argument("--steps", type=int, default=0)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--max-len", type=int, default=2048)
    ap.add_argument("--chunk", type=int, default=512)
    ap.add_argument("--seed", type=int, default=20261012)
    ap.add_argument("--log-every", type=int, default=20)
    ap.add_argument("--save-every", type=int, default=300)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    ids = np.load(STU / f"conv_ids{args.tag}.npy")
    mask = np.load(STU / f"conv_mask{args.tag}.npy")
    off = np.load(STU / f"conv_offsets{args.tag}.npy")
    lengths = np.diff(off)
    keep = (lengths >= 8) & (lengths <= args.max_len)
    conv_idx = np.where(keep)[0]
    # length-bucketed order so batches have little padding
    order = conv_idx[np.argsort(lengths[conv_idx])]
    n_conv = len(order)
    batch_tokens = args.batch * 512  # rough
    steps_per_epoch = max(1, int(n_conv / (args.batch * args.accum)))
    total_steps = args.steps or int(steps_per_epoch * args.epochs)
    log(f"conversations {n_conv} (dropped {len(lengths)-n_conv}), "
        f"avg len {lengths[conv_idx].mean():.0f}, steps/epoch {steps_per_epoch}, total {total_steps}")

    from transformers import AutoTokenizer, Qwen3ForCausalLM
    tok = AutoTokenizer.from_pretrained(args.model)
    model = Qwen3ForCausalLM.from_pretrained(args.model, dtype=torch.float32).to("cuda")
    model.config.use_cache = False
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), eps=1e-8,
                            weight_decay=0.05, fused=True)

    def make_batch(indices):
        rows = [ids[off[i]:off[i + 1]] for i in indices]
        mrows = [mask[off[i]:off[i + 1]] for i in indices]
        L = max(len(r) for r in rows)
        b_ids = np.zeros((len(rows), L), dtype=np.int64)
        b_mask = np.zeros((len(rows), L), dtype=np.uint8)
        pad = tok.pad_token_id if tok.pad_token_id is not None else 3
        for k, (r, m) in enumerate(zip(rows, mrows)):
            b_ids[k, :len(r)] = r
            b_mask[k, :len(m)] = m
        return (torch.tensor(b_ids[:, :-1], device="cuda"),
                torch.tensor(b_mask[:, 1:], device="cuda"),
                torch.tensor(b_ids[:, 1:], device="cuda"))

    logf = (out / "train_log.jsonl").open("a", encoding="utf-8")
    t0 = time.time()
    tok_seen = 0
    step = 0
    cursor = 0
    while step < total_steps:
        lr = args.lr * min(1.0, (step + 1) / args.warmup)
        if step > args.warmup:
            prog = (step - args.warmup) / max(1, total_steps - args.warmup)
            lr = args.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * prog)))
        for g in opt.param_groups:
            g["lr"] = lr
        opt.zero_grad(set_to_none=True)
        loss_sum = 0.0
        for _ in range(args.accum):
            if cursor + args.batch > n_conv:
                cursor = 0
                order = order[np.random.default_rng(args.seed + step).permutation(n_conv)]
            idx = order[cursor:cursor + args.batch]
            cursor += args.batch
            inp, m, lab = make_batch(idx)
            labels = lab.clone()
            labels[m == 0] = -100
            with torch.autocast("cuda", dtype=torch.bfloat16), sdpa_kernel(ATTN):
                loss_all = chunked_lm_loss(model, inp, labels, chunk=args.chunk)
            (loss_all / args.accum).backward()
            loss_sum += float(loss_all.detach())
            tok_seen += int(inp.numel())
            del inp, m, lab, labels, loss_all
        gn = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
        opt.step()
        step += 1
        if step % args.log_every == 0 or step == 1:
            el = time.time() - t0
            rec = {"step": step, "loss": round(loss_sum / args.accum, 4), "lr": lr,
                   "grad_norm": round(gn, 2), "tok_per_s": round(tok_seen / el),
                   "elapsed_min": round(el / 60, 1),
                   "peak_gib": round(torch.cuda.max_memory_allocated() / 1024**3, 2)}
            logf.write(json.dumps(rec) + "\n"); logf.flush()
            log(f"step {step}/{total_steps} loss {rec['loss']} lr {lr:.2e} gnorm {gn:.2f} "
                f"{rec['tok_per_s']} tok/s {rec['elapsed_min']}min peak {rec['peak_gib']}GiB")
        if args.save_every and (step % args.save_every == 0 or step == total_steps):
            model.config.use_cache = True
            model.save_pretrained(out / f"step{step}", safe_serialization=True)
            tok.save_pretrained(out / f"step{step}")
            model.config.use_cache = False
            log(f"  saved {out / f'step{step}'}")

    model.config.use_cache = True
    model.save_pretrained(out / "final", safe_serialization=True)
    tok.save_pretrained(out / "final")
    (out / "summary.json").write_text(json.dumps(
        {"steps": step, "tokens": tok_seen, "loss": loss_sum / args.accum,
         "elapsed_min": round((time.time() - t0) / 60, 1)}, ensure_ascii=False, indent=2),
        encoding="utf-8")
    log(f"done: {step} steps, {tok_seen/1e6:.1f}M tokens, "
        f"{(time.time()-t0)/60:.1f} min -> {out/'final'}")


if __name__ == "__main__":
    main()
