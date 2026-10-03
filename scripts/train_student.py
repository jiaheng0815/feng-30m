"""Train the 30M student from scratch (stage A: 2k context, masked SFT loss)."""
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

ATTN_KERNELS = [SDPBackend.EFFICIENT_ATTENTION]

ROOT = Path(r"D:\wt\feng-distill-30m")
STU = ROOT / "student"


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(STU / "stageA"))
    ap.add_argument("--tag", default="_8k", help="packed corpus suffix")
    ap.add_argument("--chunk", type=int, default=512, help="chunked-CE chunk length")
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--max-steps", type=int, default=0)
    ap.add_argument("--weight-decay", type=float, default=0.1)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=20261009)
    ap.add_argument("--log-every", type=int, default=20)
    ap.add_argument("--eval-every", type=int, default=200)
    ap.add_argument("--save-every", type=int, default=500)
    ap.add_argument("--gen-every", type=int, default=400)
    ap.add_argument("--resume", default=None)
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    ids = np.load(STU / f"train_ids{args.tag}.npy", mmap_mode="r")
    mask = np.load(STU / f"train_mask{args.tag}.npy", mmap_mode="r")
    v_ids = np.load(STU / f"val_ids{args.tag}.npy")
    v_mask = np.load(STU / f"val_mask{args.tag}.npy")
    n_seq, seq_len = ids.shape
    tokens_per_step = args.batch * args.accum * (seq_len - 1)
    steps_per_epoch = max(1, n_seq // (args.batch * args.accum))
    total_steps = int(steps_per_epoch * args.epochs)
    if args.max_steps:
        total_steps = min(total_steps, args.max_steps)
    log(f"corpus {n_seq} seqs x {seq_len} | {tokens_per_step} tok/step | "
        f"{steps_per_epoch} steps/epoch | total {total_steps} steps "
        f"({total_steps * tokens_per_step / 1e6:.0f}M tokens)")

    from transformers import AutoTokenizer, Qwen3ForCausalLM
    tok = AutoTokenizer.from_pretrained(STU / "tokenizer")
    cfg = build_config(vocab_size=tok.vocab_size, max_pos=seq_len - 1)
    model = Qwen3ForCausalLM(cfg).to("cuda")
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.config.use_cache = False
    if args.resume:
        import glob
        sd = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(sd["model"])
        log(f"resumed weights from {args.resume} (step {sd.get('step')})")
    n_params = sum(p.numel() for p in model.parameters())
    log(f"model {n_params/1e6:.2f}M params on cuda")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), eps=1e-8,
                            weight_decay=args.weight_decay, fused=True)

    def lr_at(step):
        if step < args.warmup:
            return args.lr * (step + 1) / args.warmup
        prog = (step - args.warmup) / max(1, total_steps - args.warmup)
        return args.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * prog)))

    logf = (out_dir / "train_log.jsonl").open("a", encoding="utf-8")
    model.train()
    step, t_start, tok_seen = 0, time.time(), 0
    last_loss = None

    @torch.no_grad()
    def evaluate():
        model.eval()
        tot, n = 0.0, 0
        for i in range(0, len(v_ids), args.batch):
            b_ids = torch.tensor(v_ids[i:i + args.batch], dtype=torch.long, device="cuda")
            b_mask = torch.tensor(v_mask[i:i + args.batch], dtype=torch.uint8, device="cuda")
            labels = b_ids[:, 1:].clone()
            labels[b_mask[:, 1:] == 0] = -100
            with torch.autocast("cuda", dtype=torch.bfloat16), sdpa_kernel(ATTN_KERNELS):
                loss = chunked_lm_loss(model, b_ids[:, :-1], labels, chunk=args.chunk)
            tot += float(loss) * len(b_ids)
            n += len(b_ids)
        model.train()
        return tot / max(n, 1)

    @torch.no_grad()
    def sample(prompt="你是谁？", n_new=48):
        model.eval()
        text = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
        ids = tok(text, add_special_tokens=False)["input_ids"]
        inp = torch.tensor([ids], dtype=torch.long, device="cuda")
        out = model.generate(input_ids=inp, max_new_tokens=n_new, do_sample=False,
                             pad_token_id=3, eos_token_id=0)
        resp = tok.decode(out[0][inp.shape[1]:].tolist())
        model.train()
        return resp.split("<|im_end|>")[0].strip()

    while step < total_steps:
        opt.zero_grad(set_to_none=True)
        loss_sum = 0.0
        for _ in range(args.accum):
            idx = rng.integers(0, n_seq, size=args.batch)
            b_ids = torch.tensor(np.asarray(ids[idx]), dtype=torch.long, device="cuda")
            b_mask = torch.tensor(np.asarray(mask[idx]), dtype=torch.uint8, device="cuda")
            labels = b_ids[:, 1:].clone()
            labels[b_mask[:, 1:] == 0] = -100
            with torch.autocast("cuda", dtype=torch.bfloat16), sdpa_kernel(ATTN_KERNELS):
                loss_all = chunked_lm_loss(model, b_ids[:, :-1], labels, chunk=args.chunk)
            loss = loss_all / args.accum
            loss.backward()
            loss_sum += float(loss_all.detach())
            del loss_all, loss, b_ids, b_mask, labels
        gn = float(torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip))
        lr = lr_at(step)
        for g in opt.param_groups:
            g["lr"] = lr
        opt.step()
        step += 1
        tok_seen += tokens_per_step
        last_loss = loss_sum / args.accum
        if step % args.log_every == 0 or step == 1:
            el = time.time() - t_start
            rec = {"step": step, "loss": round(last_loss, 4), "lr": lr, "grad_norm": round(gn, 2),
                   "tok_per_s": round(tok_seen / el), "elapsed_min": round(el / 60, 1),
                   "vram_gib": round(torch.cuda.max_memory_allocated() / 1024**3, 2)}
            logf.write(json.dumps(rec) + "\n"); logf.flush()
            log(f"step {step}/{total_steps} loss {last_loss:.4f} lr {lr:.2e} gnorm {gn:.2f} "
                f"{rec['tok_per_s']} tok/s {rec['elapsed_min']}min vram {rec['vram_gib']}GiB")
        if step % args.eval_every == 0:
            vl = evaluate()
            log(f"  [eval] step {step} val_loss {vl:.4f}")
            logf.write(json.dumps({"step": step, "val_loss": round(vl, 4)}) + "\n"); logf.flush()
        if step % args.gen_every == 0:
            log(f"  [sample] {sample()!r}")
        if step % args.save_every == 0 or step == total_steps:
            ck = out_dir / f"step{step}.pt"
            torch.save({"model": model.state_dict(), "step": step, "cfg": cfg.to_dict()}, ck)
            log(f"  saved {ck}")

    log("saving final model ...")
    model.config.use_cache = True
    model.save_pretrained(out_dir / "final", safe_serialization=True)
    tok.save_pretrained(out_dir / "final")
    summary = {"steps": step, "tokens_seen": tok_seen, "final_loss": last_loss,
               "params": n_params, "elapsed_min": round((time.time() - t_start) / 60, 1),
               "val_loss": evaluate()}
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                                          encoding="utf-8")
    log(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
