"""Stage B: extend the 30M student to 32k context (YaRN + chunked cross-entropy)."""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.utils.checkpoint import checkpoint

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).parent))
from student_config import build_config  # noqa: E402
from student_utils import chunked_lm_loss  # noqa: E402

from paths import ROOT  # noqa: E402
STU = ROOT / "student"


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def chunked_lm_loss(model, input_ids, labels, chunk=512):
    """LM loss without ever materialising [S, vocab] logits (checkpointed per chunk)."""
    hidden = model.model(input_ids=input_ids, use_cache=False).last_hidden_state
    weight = model.lm_head.weight
    V = weight.shape[0]
    total_sup = (labels != -100).sum().clamp(min=1)
    S = input_ids.shape[1]

    def part(h, lab, w, denom):
        logits = h @ w.t()
        return F.cross_entropy(logits.float().view(-1, V), lab.reshape(-1),
                               ignore_index=-100, reduction="sum") / denom

    losses = []
    for c0 in range(0, S, chunk):
        c1 = min(S, c0 + chunk)
        losses.append(checkpoint(part, hidden[:, c0:c1], labels[:, c0:c1], weight, total_sup,
                                 use_reentrant=False))
    return sum(losses)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(STU / "stageA" / "final"))
    ap.add_argument("--out", default=str(STU / "stageB"))
    ap.add_argument("--len", type=int, default=32768)
    ap.add_argument("--factor", type=float, default=0.0,
                    help="0 = native (no RoPE scaling); >0 enables YaRN")
    ap.add_argument("--orig-max", type=int, default=2048)
    ap.add_argument("--steps", type=int, default=240)
    ap.add_argument("--accum", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--chunk", type=int, default=512)
    ap.add_argument("--seed", type=int, default=20261011)
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--save-every", type=int, default=100)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    ids = np.load(STU / "long_ids.npy", mmap_mode="r")
    mask = np.load(STU / "long_mask.npy", mmap_mode="r")
    v_ids, v_mask = np.load(STU / "long_val_ids.npy"), np.load(STU / "long_val_mask.npy")
    log(f"long corpus: {len(ids)} seqs x {ids.shape[1]} tokens "
        f"({len(ids) * ids.shape[1] / 1e6:.1f}M tokens), val {len(v_ids)}")

    from transformers import AutoTokenizer, Qwen3ForCausalLM
    tok = AutoTokenizer.from_pretrained(args.model)
    kernels = [SDPBackend.EFFICIENT_ATTENTION]
    rope = None
    if args.factor and args.factor > 1.0:
        rope = {"type": "yarn", "factor": args.factor,
                "original_max_position_embeddings": args.orig_max}
    cfg = build_config(vocab_size=tok.vocab_size, max_pos=args.len, rope_scaling=rope)
    model = Qwen3ForCausalLM.from_pretrained(args.model, config=cfg, dtype=torch.float32).to("cuda")
    model.config.use_cache = False
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), eps=1e-8,
                            weight_decay=0.05, fused=True)
    log(f"loaded {sum(p.numel() for p in model.parameters())/1e6:.2f}M params, "
        f"{'native' if rope is None else f'yarn x{args.factor}'} ctx {args.len}")

    logf = (out / "train_log.jsonl").open("a", encoding="utf-8")
    t0 = time.time()
    tok_seen = 0
    for step in range(1, args.steps + 1):
        lr = args.lr * min(1.0, step / args.warmup) * (0.1 + 0.9 * 0.5 *
                                                       (1 + math.cos(math.pi * step / args.steps)))
        for g in opt.param_groups:
            g["lr"] = lr
        opt.zero_grad(set_to_none=True)
        loss_sum = 0.0
        for _ in range(args.accum):
            idx = int(rng.integers(0, len(ids)))
            b_ids = torch.tensor(np.asarray(ids[idx]), dtype=torch.long, device="cuda").unsqueeze(0)
            b_mask = torch.tensor(np.asarray(mask[idx]), dtype=torch.uint8, device="cuda").unsqueeze(0)
            labels = b_ids[:, 1:].clone()
            labels[b_mask[:, 1:] == 0] = -100
            with torch.autocast("cuda", dtype=torch.bfloat16), sdpa_kernel(kernels):
                loss = chunked_lm_loss(model, b_ids[:, :-1], labels, chunk=args.chunk) / args.accum
            loss.backward()
            loss_sum += float(loss)
            tok_seen += b_ids.shape[1] - 1
            del b_ids, b_mask, labels, loss
        gn = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
        opt.step()
        if step % args.log_every == 0 or step == 1:
            el = time.time() - t0
            rec = {"step": step, "loss": round(loss_sum, 4), "lr": lr, "grad_norm": round(gn, 3),
                   "tok_per_s": round(tok_seen / el), "elapsed_min": round(el / 60, 1),
                   "peak_gib": round(torch.cuda.max_memory_allocated() / 1024**3, 2)}
            logf.write(json.dumps(rec) + "\n"); logf.flush()
            log(f"step {step}/{args.steps} loss {loss_sum:.4f} lr {lr:.2e} gnorm {gn:.2f} "
                f"{rec['tok_per_s']} tok/s {rec['elapsed_min']}min peak {rec['peak_gib']}GiB")
        if step % args.save_every == 0 or step == args.steps:
            model.config.use_cache = True
            model.save_pretrained(out / f"step{step}", safe_serialization=True)
            tok.save_pretrained(out / f"step{step}")
            model.config.use_cache = False
            log(f"  saved {out / f'step{step}'}")

    model.config.use_cache = True
    model.save_pretrained(out / "final", safe_serialization=True)
    tok.save_pretrained(out / "final")
    summary = {"steps": args.steps, "ctx": args.len, "yarn_factor": args.factor,
               "tokens_seen": tok_seen, "final_loss": loss_sum,
               "elapsed_min": round((time.time() - t0) / 60, 1)}
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                                      encoding="utf-8")
    log(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
