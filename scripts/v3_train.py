"""v3: progressive long-context extension, starting from the v2 chat model.

Stage schedule (same data stream, longer windows, fewer tokens -- long stages cost
much more compute per token, which is why every long-context recipe shrinks the
budget as the window grows):
    ctx 4096  : 40.0M tokens
    ctx 8192  : 20.0M tokens
    ctx 16384 :  8.0M tokens
    ctx 32768 :  4.0M tokens
Each stage is one epoch over its own pack, warmup 20 steps then cosine to 10% of the
stage LR, saving a checkpoint at the end.  A short masked chat polish afterwards
(v2_train.py --stage sft) restores the assistant behaviour.
"""
import argparse
import json
import math
import shutil
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
V3 = ROOT / "archive" / "v3"
ATTN = [SDPBackend.EFFICIENT_ATTENTION, SDPBackend.FLASH_ATTENTION]
STAGES = [
    # name, seq, batch, accum, lr
    ("ctx4096", 4096, 4, 4, 8e-5),
    ("ctx8192", 8192, 2, 4, 6e-5),
    ("ctx16384", 16384, 1, 4, 4.5e-5),
    ("ctx32768", 32768, 1, 4, 3e-5),
]


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", default=str(ROOT / "v2" / "stage_planA3b" / "final"))
    ap.add_argument("--out", default=str(V3))
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--seed", type=int, default=20261002)
    ap.add_argument("--stages", default="", help="comma list to run a subset")
    ap.add_argument("--steps-cap", type=int, default=0, help="cap steps per stage (smoke test)")
    ap.add_argument("--data-root", default=str(V3 / "data"))
    ap.add_argument("--save-every", type=int, default=100, help="mid-stage rolling checkpoint")
    args = ap.parse_args()

    from transformers import AutoTokenizer, Qwen3ForCausalLM
    tok = AutoTokenizer.from_pretrained(ROOT / "v2" / "tokenizer")
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    want = set(args.stages.split(",")) if args.stages else None
    init_dir = Path(args.init)
    summary = []

    for name, L, batch, accum, lr in STAGES:
        if want and name not in want:
            continue
        data = np.load(Path(args.data_root) / f"{name}.npy", mmap_mode="r")
        n_all = data.shape[0]
        steps = max(1, n_all // (batch * accum))
        if args.steps_cap:
            steps = min(steps, args.steps_cap)
        stage_dir = out_root / name
        dest = stage_dir / "final"
        if dest.exists() and not args.steps_cap:
            log(f"=== {name}: already finished ({dest}) -> skipped ===")
            init_dir = dest
            continue
        log(f"=== {name}: {n_all} seqs x {L} | init {init_dir} | {batch}x{accum} | "
            f"{steps} steps | lr {lr:g} ===")
        # resume from the newest rolling checkpoint of this stage if there is one
        start_step = 1
        rolling = stage_dir / "rolling"
        ckpts = sorted([p for p in rolling.glob("step*") if p.is_dir()],
                       key=lambda p: int(p.name[4:])) if rolling.exists() else []
        if ckpts:
            last = ckpts[-1]
            start_step = int(last.name[4:]) + 1
            log(f"    resume: found {last.name}, continuing from step {start_step}")
            init_dir = last
        model = Qwen3ForCausalLM.from_pretrained(init_dir, dtype=torch.float32)
        model.config.max_position_embeddings = L
        model = model.to("cuda")
        model.train()
        opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), eps=1e-8,
                                weight_decay=0.05, fused=True)
        rng = np.random.default_rng(args.seed + L)
        torch.manual_seed(args.seed + L)
        t0 = time.time()
        tok_seen = 0
        peak = 0
        loss_last = None
        for step in range(start_step, steps + 1):
            if step <= args.warmup:
                cur_lr = lr * step / args.warmup
            else:
                prog = (step - args.warmup) / max(1, steps - args.warmup)
                cur_lr = lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * prog)))
            for g in opt.param_groups:
                g["lr"] = cur_lr
            opt.zero_grad(set_to_none=True)
            loss_sum = 0.0
            for _ in range(accum):
                idx = rng.integers(0, n_all, size=batch)
                arr = np.asarray(data[idx])[:, :L]
                t = torch.tensor(arr, dtype=torch.long, device="cuda")
                inp, lab = t[:, :-1], t[:, 1:]
                with torch.autocast("cuda", dtype=torch.bfloat16), sdpa_kernel(ATTN):
                    loss_all = chunked_lm_loss(model, inp, lab, chunk=512)
                (loss_all / accum).backward()
                loss_sum += float(loss_all.detach())
                tok_seen += int(inp.numel())
                del inp, lab, loss_all, t, arr
            gn = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
            opt.step()
            loss_last = loss_sum / accum
            peak = max(peak, torch.cuda.max_memory_allocated() / 2**30)
            if step % 5 == 0 or step == steps:
                dt = time.time() - t0
                log(f"[{name}] step {step}/{steps} loss {loss_last:.4f} lr {cur_lr:.2e} "
                    f"gnorm {gn:.2f} {tok_seen/dt:,.0f} tok/s {dt/60:.1f}min peak {peak:.2f}GiB")
            if step % args.save_every == 0 and step < steps:
                snap = rolling / f"step{step}"
                snap.mkdir(parents=True, exist_ok=True)
                model.config.use_cache = True
                model.save_pretrained(snap, safe_serialization=True)
                tok.save_pretrained(snap)
                model.config.use_cache = False
                keep = sorted([p for p in rolling.glob("step*") if p.is_dir()],
                              key=lambda p: int(p.name[4:]))[-2:]
                for p in rolling.glob("step*"):
                    if p.is_dir() and p not in keep and str(p).startswith(str(out_root)):
                        shutil.rmtree(p, ignore_errors=True)
                log(f"    rolling checkpoint -> {snap.name} (keeping last 2)")
        dest.mkdir(parents=True, exist_ok=True)
        model.config.use_cache = True
        model.save_pretrained(dest, safe_serialization=True)
        tok.save_pretrained(dest)
        log(f"saved -> {dest} (loss {loss_last:.4f}, peak {peak:.2f}GiB, "
            f"{(time.time()-t0)/60:.1f} min)")
        summary.append({"stage": name, "seq": L, "steps": steps, "tokens": tok_seen,
                        "loss": loss_last, "peak_gib": peak,
                        "minutes": (time.time() - t0) / 60, "out": str(dest)})
        (out_root / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        del model, opt
        torch.cuda.empty_cache()
        init_dir = dest
    log("v3 progressive stages done -> " + str(init_dir))


if __name__ == "__main__":
    main()
