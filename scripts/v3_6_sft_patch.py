"""v3.6 专用补丁 SFT：不用打包窗口，按"单条对话"训练，直接把新映射吃进权重。

背景：用 v3_retrieval_sft.py 的打包混训（窗口里很多对话首尾相接）时，
30M 模型会靠窗口内上下文"蹭"出低 loss，单条提问时却不会答（自拟合只有 2%）。
这里改成每条对话单独一条样本（长度分桶 + padding + 只对 assistant 算 loss），
并混入少量多轮/身份样本防止基本盘漂移。

用法：
    python scripts/v3_6_sft_patch.py --init v3_5d\\final\\ctx32768\\final \
        --patch v3_6a\\daily_patch.jsonl --mt v3_5d\\mt_convs.jsonl \
        --out v3_6d\\final --epochs 8 --lr 1e-4 --mt-n 600 --identity-n 200
"""
import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).resolve().parent))
from v3_2_build_identity_mix import build_identity_pool  # noqa: E402


def log(m: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def load_convs(path: str, limit: int, rng: random.Random) -> list[list[tuple[str, str]]]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rec = json.loads(line)
                rows.append([(m["role"], m["content"]) for m in rec["messages"]
                             if m.get("content")])
    if limit and len(rows) > limit:
        rows = rng.sample(rows, limit)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", required=True)
    ap.add_argument("--patch", required=True)
    ap.add_argument("--mt", default="")
    ap.add_argument("--mt-n", type=int, default=600)
    ap.add_argument("--identity-n", type=int, default=200)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=float, default=8.0)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--batch", type=int, default=24)
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=20261005)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)
    from transformers import AutoTokenizer, Qwen3ForCausalLM
    tok = AutoTokenizer.from_pretrained(args.init)

    convs: list[list[tuple[str, str]]] = load_convs(args.patch, 0, rng)
    n_patch = len(convs)
    if args.mt:
        convs += load_convs(args.mt, args.mt_n, rng)
    if args.identity_n:
        convs += build_identity_pool(rng, args.identity_n)
    log(f"样本：补丁 {n_patch} + 其余 {len(convs) - n_patch} = {len(convs)} 条")

    pad_id = tok.pad_token_id if tok.pad_token_id is not None else 3
    samples = []
    for msgs in convs:
        ids, mask = [], []
        for role, text in msgs:
            t = tok.encode(f"<|im_start|>{role}\n{text}<|im_end|>\n", add_special_tokens=False)
            ids += t
            mask += [1 if role == "assistant" else 0] * len(t)
        if len(ids) > args.max_len:
            ids, mask = ids[:args.max_len], mask[:args.max_len]
        if sum(mask) >= 4:
            samples.append((ids, mask))
    samples.sort(key=lambda s: len(s[0]))          # 长度分桶，减少 padding
    log(f"可用样本 {len(samples)}，平均长度 {np.mean([len(s[0]) for s in samples]):.0f}")

    model = Qwen3ForCausalLM.from_pretrained(args.init, dtype=torch.float32).to("cuda").train()
    model.config.use_cache = False
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), eps=1e-8,
                            weight_decay=0.05, fused=True)

    steps_per_epoch = max(1, len(samples) // args.batch)
    total = max(1, int(steps_per_epoch * args.epochs))
    t0, seen, loss_last, peak = time.time(), 0, None, 0.0
    for step in range(1, total + 1):
        if step <= args.warmup:
            lr = args.lr * step / args.warmup
        else:
            prog = (step - args.warmup) / max(1, total - args.warmup)
            lr = args.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * prog)))
        for g in opt.param_groups:
            g["lr"] = lr
        idx = rng.sample(range(len(samples)), args.batch)
        rows = [samples[i] for i in idx]
        L = max(len(r[0]) for r in rows)
        b_ids = np.full((len(rows), L), pad_id, dtype=np.int64)
        b_lab = np.full((len(rows), L), -100, dtype=np.int64)
        for k, (ids, mask) in enumerate(rows):
            b_ids[k, :len(ids)] = ids
            lab = np.asarray(ids, dtype=np.int64)
            lab[np.asarray(mask) == 0] = -100
            b_lab[k, :len(lab)] = lab
        t_ids = torch.tensor(b_ids, dtype=torch.long, device="cuda")
        t_lab = torch.tensor(b_lab, dtype=torch.long, device="cuda")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(input_ids=t_ids).logits
            loss = torch.nn.functional.cross_entropy(
                out[:, :-1].reshape(-1, out.shape[-1]).float(),
                t_lab[:, 1:].reshape(-1), ignore_index=-100)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
        opt.step()
        loss_last = float(loss.detach())
        seen += int((t_lab[:, 1:] != -100).sum())
        peak = max(peak, torch.cuda.max_memory_allocated() / 2**30)
        if step % 25 == 0 or step == total:
            dt = time.time() - t0
            log(f"step {step}/{total} loss {loss_last:.4f} lr {lr:.2e} gnorm {gn:.2f} "
                f"{seen/dt:,.0f} tok/s {dt/60:.1f}min peak {peak:.2f}GiB")
    dest = Path(args.out)
    dest.mkdir(parents=True, exist_ok=True)
    model.config.use_cache = True
    model.save_pretrained(dest, safe_serialization=True)
    tok.save_pretrained(dest)
    log(f"saved -> {dest}（loss {loss_last:.4f}）")


if __name__ == "__main__":
    main()
