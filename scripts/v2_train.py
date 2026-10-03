"""v2 trainer: stage=pretrain (packed, all-token) | sft (unpacked, masked) | 32k (native long)."""
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

from paths import ROOT  # noqa: E402
V2 = ROOT / "v2"
ATTN = [SDPBackend.EFFICIENT_ATTENTION]


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def build_model(tok, max_pos, rope_scaling=None):
    from transformers import Qwen3Config, Qwen3ForCausalLM
    cfg = Qwen3Config(
        vocab_size=tok.vocab_size, hidden_size=448, num_hidden_layers=11,
        num_attention_heads=7, num_key_value_heads=7, head_dim=64, intermediate_size=896,
        hidden_act="silu", max_position_embeddings=max_pos, rms_norm_eps=1e-6,
        rope_theta=1000000.0, rope_scaling=rope_scaling, attention_bias=False,
        attention_dropout=0.0, tie_word_embeddings=True, initializer_range=0.02,
        bos_token_id=1, eos_token_id=0, pad_token_id=3)
    return Qwen3ForCausalLM(cfg)


def save(model, tok, path):
    model.config.use_cache = True
    Path(path).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(path, safe_serialization=True)
    tok.save_pretrained(path)
    model.config.use_cache = False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["pretrain", "sft", "32k"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default=None, help="init from an existing HF dir (sft/32k stages)")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--steps", type=int, default=0)
    ap.add_argument("--seq", type=int, default=2048)
    ap.add_argument("--tag", default="", help="packed-array suffix (sft stage)")
    ap.add_argument("--data", default=None, help="explicit packed array for the pretrain stage")
    ap.add_argument("--chunk", type=int, default=512)
    ap.add_argument("--seed", type=int, default=20261014)
    ap.add_argument("--log-every", type=int, default=25)
    ap.add_argument("--save-every", type=int, default=400)
    ap.add_argument("--gen-every", type=int, default=0)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(V2 / "tokenizer")

    if args.stage == "pretrain":
        ids = np.load(Path(args.data) if args.data else (V2 / "pretrain_ids.npy"), mmap_mode="r")
        n_all, L = ids.shape
        steps_per_epoch = max(1, n_all // (args.batch * args.accum))
        total = args.steps or int(steps_per_epoch * args.epochs)
        model = build_model(tok, args.seq).to("cuda")
        log(f"[pretrain] seqs {n_all} x {L} | {args.batch}x{args.accum} | "
            f"{steps_per_epoch} steps/epoch | total {total} | "
            f"params {sum(p.numel() for p in model.parameters())/1e6:.2f}M")

        def get_batch():
            idx = rng.integers(0, n_all, size=args.batch)
            arr = np.asarray(ids[idx])[:, :args.seq]
            t = torch.tensor(arr, dtype=torch.long, device="cuda")
            return t[:, :-1], t[:, 1:]

    elif args.stage == "sft":
        cid = np.load(V2 / f"sft_ids{args.tag}.npy")
        cmask = np.load(V2 / f"sft_mask{args.tag}.npy")
        off = np.load(V2 / f"sft_offsets{args.tag}.npy")
        lens = np.diff(off)
        keep = np.where((lens >= 8) & (lens <= args.seq))[0]
        order = keep[np.argsort(lens[keep])]
        n_conv = len(order)
        steps_per_epoch = max(1, n_conv // (args.batch * args.accum))
        total = args.steps or int(steps_per_epoch * args.epochs)
        model = build_model(tok, args.seq)
        if args.model:
            from transformers import Qwen3ForCausalLM
            model = Qwen3ForCausalLM.from_pretrained(args.model, dtype=torch.float32)
        model = model.to("cuda")
        log(f"[sft] conversations {n_conv} | avg len {lens[keep].mean():.0f} | "
            f"steps/epoch {steps_per_epoch} | total {total}")
        cursor = [0]

        def get_batch():
            if cursor[0] + args.batch > n_conv:
                cursor[0] = 0
                perm = np.random.default_rng(args.seed + int(time.time())).permutation(n_conv)
                order[:] = order[perm]
            sel = order[cursor[0]:cursor[0] + args.batch]
            cursor[0] += args.batch
            rows = [cid[off[i]:off[i + 1]] for i in sel]
            rows_m = [cmask[off[i]:off[i + 1]] for i in sel]
            L = max(len(r) for r in rows)
            b = np.zeros((len(rows), L), dtype=np.int64)
            m = np.zeros((len(rows), L), dtype=np.uint8)
            for k, (r, mm) in enumerate(zip(rows, rows_m)):
                b[k, :len(r)] = r
                m[k, :len(mm)] = mm
            t = torch.tensor(b, dtype=torch.long, device="cuda")
            m = torch.tensor(m, dtype=torch.uint8, device="cuda")
            return t[:, :-1], t[:, 1:], m[:, 1:]

    else:  # 32k
        ids = np.load(V2 / "long_ids.npy", mmap_mode="r")
        mask = np.load(V2 / "long_mask.npy", mmap_mode="r")
        n_all, L = ids.shape
        total = args.steps or 150
        from transformers import Qwen3ForCausalLM
        model = Qwen3ForCausalLM.from_pretrained(args.model, dtype=torch.float32).to("cuda")
        log(f"[32k] seqs {n_all} x {L} | total {total} steps | native ctx {L}")

        def get_batch():
            i = int(rng.integers(0, n_all))
            t = torch.tensor(np.asarray(ids[i]), dtype=torch.long, device="cuda").unsqueeze(0)
            m = torch.tensor(np.asarray(mask[i]), dtype=torch.uint8, device="cuda").unsqueeze(0)
            return t[:, :-1], t[:, 1:], m[:, 1:]

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), eps=1e-8,
                            weight_decay=0.05, fused=True)
    logf = (out / "train_log.jsonl").open("a", encoding="utf-8")
    t0 = time.time()
    tok_seen = 0
    last = None
    for step in range(1, total + 1):
        lr = args.lr * min(1.0, step / args.warmup)
        if step > args.warmup:
            prog = (step - args.warmup) / max(1, total - args.warmup)
            lr = args.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * prog)))
        for g in opt.param_groups:
            g["lr"] = lr
        opt.zero_grad(set_to_none=True)
        loss_sum = 0.0
        for _ in range(args.accum):
            b = get_batch()
            if args.stage == "pretrain":
                inp, lab = b
                m = None
            else:
                inp, lab, m = b
                lab = lab.clone()
                lab[m == 0] = -100
            with torch.autocast("cuda", dtype=torch.bfloat16), sdpa_kernel(ATTN):
                loss_all = chunked_lm_loss(model, inp, lab, chunk=args.chunk)
            (loss_all / args.accum).backward()
            loss_sum += float(loss_all.detach())
            tok_seen += int(inp.numel())
            del inp, lab, loss_all
        gn = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
        opt.step()
        last = loss_sum / args.accum
        if step % args.log_every == 0 or step == 1:
            el = time.time() - t0
            rec = {"step": step, "total": total, "loss": round(last, 4), "lr": lr,
                   "grad_norm": round(gn, 2), "tok_per_s": round(tok_seen / el),
                   "elapsed_min": round(el / 60, 1),
                   "peak_gib": round(torch.cuda.max_memory_allocated() / 1024**3, 2)}
            logf.write(json.dumps(rec) + "\n"); logf.flush()
            log(f"step {step}/{total} loss {rec['loss']} lr {lr:.2e} gnorm {gn:.2f} "
                f"{rec['tok_per_s']} tok/s {rec['elapsed_min']}min peak {rec['peak_gib']}GiB")
        if args.save_every and (step % args.save_every == 0 or step == total):
            save(model, tok, out / f"step{step}")
            log(f"  saved {out / f'step{step}'}")
    save(model, tok, out / "final")
    (out / "summary.json").write_text(json.dumps(
        {"stage": args.stage, "steps": total, "tokens": tok_seen, "final_loss": last,
         "elapsed_min": round((time.time() - t0) / 60, 1),
         "params": sum(p.numel() for p in model.parameters())}, ensure_ascii=False, indent=2),
        encoding="utf-8")
    log(f"done {args.stage}: {total} steps, {tok_seen/1e6:.1f}M tokens, "
        f"{(time.time()-t0)/60:.1f} min -> {out / 'final'}")


if __name__ == "__main__":
    main()
