"""Teach the retrieval skill with the synthesised long-context QA data.
Masked loss (answer tokens only), one epoch per length, decreasing LR as the window
grows.  Run after the 8k chat polish so the assistant behaviour is already restored.
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

from paths import ROOT  # noqa: E402
V3 = ROOT / "v3"
ATTN = [SDPBackend.EFFICIENT_ATTENTION, SDPBackend.FLASH_ATTENTION]
SPECS = [(4096, 2, 4, 5e-5), (8192, 2, 4, 4e-5), (16384, 1, 4, 3e-5), (32768, 1, 4, 2e-5)]


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", default=str(V3 / "polish_ctx8192" / "final"))
    ap.add_argument("--out", default=str(V3 / "retr_sft"))
    ap.add_argument("--data-root", default=str(V3 / "data"),
                    help="检索数据目录（含 retr{L}_ids.npy / retr{L}_mask.npy）")
    ap.add_argument("--lr-scale", type=float, default=1.0,
                    help="学习率缩放（续训已微调过的模型时可调小，如 0.7）")
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--epochs", type=float, default=1.0,
                    help="每个长度跑几个 epoch（补丁轮需要重复多遍才能压住旧行为）")
    ap.add_argument("--lengths", default="",
                    help="只跑这些长度（逗号分隔，如 4096,8192）；默认全部")
    ap.add_argument("--train-last", type=int, default=0,
                    help="只训练最后 N 层 + norm（0=全参）；末层微调对其它长度的扰动更小")
    ap.add_argument("--seed", type=int, default=20261003)
    args = ap.parse_args()
    from transformers import AutoTokenizer, Qwen3ForCausalLM
    tok = AutoTokenizer.from_pretrained(ROOT / "v2" / "tokenizer")
    out_root = Path(args.out)
    data_root = Path(args.data_root)
    out_root.mkdir(parents=True, exist_ok=True)
    init_dir = Path(args.init)
    summary = []
    want = {int(x) for x in args.lengths.split(",") if x.strip()} if args.lengths else None
    for L, batch, accum, lr in SPECS:
        if want and L not in want:
            continue
        ids = np.load(data_root / f"retr{L}_ids.npy", mmap_mode="r")
        mask = np.load(data_root / f"retr{L}_mask.npy", mmap_mode="r")
        lr = lr * args.lr_scale
        n_all = ids.shape[0]
        steps = max(1, int(n_all * args.epochs // (batch * accum)))
        log(f"=== retrieval {L}: {n_all} samples | init {init_dir} | {batch}x{accum} | "
            f"{steps} steps | {args.epochs:g} epoch(s) | lr {lr:g} ===")
        model = Qwen3ForCausalLM.from_pretrained(init_dir, dtype=torch.float32)
        model.config.max_position_embeddings = L
        model = model.to("cuda").train()
        if args.train_last > 0:
            for p in model.parameters():
                p.requires_grad = False
            n_layers = len(model.model.layers)
            for layer in model.model.layers[max(0, n_layers - args.train_last):]:
                for p in layer.parameters():
                    p.requires_grad = True
            for name in ("norm",):
                mod = getattr(model.model, name, None)
                if mod is not None:
                    for p in mod.parameters():
                        p.requires_grad = True
            if L == SPECS[0][0]:
                trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
                total = sum(p.numel() for p in model.parameters())
                log(f"[train-last] 只训练最后 {args.train_last} 层 + norm："
                    f"{trainable/1e6:.2f}M / {total/1e6:.2f}M 参数")
        opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), eps=1e-8,
                                weight_decay=0.05, fused=True)
        rng = np.random.default_rng(args.seed + L)
        t0, tok_seen, peak, loss_last = time.time(), 0, 0.0, None
        for step in range(1, steps + 1):
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
                a = np.asarray(ids[idx])[:, :L]
                m = np.asarray(mask[idx])[:, :L]
                t = torch.tensor(a, dtype=torch.long, device="cuda")
                mt = torch.tensor(m, dtype=torch.uint8, device="cuda")
                inp, lab = t[:, :-1], t[:, 1:]
                lab = lab.clone()
                lab[mt[:, 1:] == 0] = -100
                with torch.autocast("cuda", dtype=torch.bfloat16), sdpa_kernel(ATTN):
                    loss_all = chunked_lm_loss(model, inp, lab, chunk=512)
                (loss_all / accum).backward()
                loss_sum += float(loss_all.detach())
                tok_seen += int(inp.numel())
                del inp, lab, loss_all, t, mt, a, m
            gn = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
            opt.step()
            loss_last = loss_sum / accum
            peak = max(peak, torch.cuda.max_memory_allocated() / 2**30)
            if step % 5 == 0 or step == steps:
                dt = time.time() - t0
                log(f"[retr{L}] step {step}/{steps} loss {loss_last:.4f} lr {cur_lr:.2e} "
                    f"gnorm {gn:.2f} {tok_seen/dt:,.0f} tok/s {dt/60:.1f}min peak {peak:.2f}GiB")
        dest = out_root / f"ctx{L}" / "final"
        dest.mkdir(parents=True, exist_ok=True)
        model.config.use_cache = True
        model.save_pretrained(dest, safe_serialization=True)
        tok.save_pretrained(dest)
        log(f"saved -> {dest} (loss {loss_last:.4f})")
        summary.append({"seq": L, "steps": steps, "tokens": tok_seen, "loss": loss_last,
                        "minutes": (time.time() - t0) / 60, "out": str(dest)})
        (out_root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        del model, opt
        torch.cuda.empty_cache()
        init_dir = dest
    log(f"retrieval SFT done -> {init_dir}")


if __name__ == "__main__":
    main()
