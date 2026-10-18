"""三值 QAT 实验（分支 `ternary`）：Bonsai 2 风格的 ternary + blockwise Hadamard。

和 `ternary_ptq.py` 的区别：这里用 **STE（straight-through estimator）** 训练，
让模型自己适应 {-d,0,+d} 权重的误差；lm_head/embedding/norm 保持 fp16
（selective precision），只把 attention/MLP 的 Linear 变成三值。

实现方式：把 Qwen3 的 `nn.Linear` 换成 `TernaryLinear`（state_dict 键不变），
forward 里做
    x' = FWHT(x ⊙ s) / sqrt(B)
    W' = ternarize(FWHT(W ⊙ s) / sqrt(B), group)
    y  = x' W'^T
反传走 STE：dL/dW = dL/dW'（量化函数视为恒等）。

用法：
    python scripts\\ternary_qat.py --init v4/base --data data\\v4_sft.jsonl `
        --val data\\v4_val.jsonl --out v4/ternary_qat --epochs 1 --lr 1e-5 `
        --batch 8 --accum 2 --max-len 512 --steps-cap 200
"""
import argparse
import json
import math
import random
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).resolve().parent))

from student_utils import chunked_lm_loss  # noqa: E402
from v4_sft import encode, load_convs, make_batches, make_labels  # noqa: E402


def build_hadamard(n: int) -> torch.Tensor:
    h = torch.ones(1, 1)
    while h.shape[0] < n:
        h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
    return h


def block_rotate(x: torch.Tensor, signs: torch.Tensor, H: torch.Tensor, block: int):
    nb = x.shape[-1] // block
    y = x.reshape(*x.shape[:-1], nb, block) * signs.view(nb, block)
    return (y @ H / math.sqrt(block)).reshape(*x.shape)


def ternarize(w: torch.Tensor, group: int):
    out, K = w.shape
    wg = w.reshape(out, K // group, group)
    d = wg.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8)
    q = (wg / d).round().clamp(-1, 1)
    return (q * d).reshape_as(w)


class TernaryLinear(torch.nn.Module):
    """保留原 weight（float）用于导出；forward 里实时旋转+三值化，梯度走 STE。"""

    def __init__(self, base: torch.nn.Linear, signs, H, block: int, group: int):
        super().__init__()
        self.in_features = base.in_features
        self.out_features = base.out_features
        self.weight = base.weight
        self.bias = base.bias
        self.register_buffer("signs", signs, persistent=False)
        self.register_buffer("H", H, persistent=False)
        self.block = block
        self.group = group

    def forward(self, x):
        wr = block_rotate(self.weight, self.signs, self.H, self.block)
        wq = ternarize(wr, self.group)
        w = wr + (wq - wr).detach()          # STE
        xr = block_rotate(x, self.signs, self.H, self.block)
        return F.linear(xr, w, self.bias)


def replace_linears(model, block: int, group: int, seed: int, keep_lm_head: bool, keep: list[str]):
    widths = set()
    targets = []
    for name, mod in model.named_modules():
        if not isinstance(mod, torch.nn.Linear) or isinstance(mod, TernaryLinear):
            continue
        if keep_lm_head and name.endswith("lm_head"):
            continue
        if any(k in name for k in keep):
            print(f"保留 {name}（--keep）")
            continue
        if mod.in_features % block or mod.in_features % group:
            print(f"跳过 {name}（K={mod.in_features} 不能被 block/group 整除）")
            continue
        widths.add(mod.in_features)
        targets.append((name, mod))

    gen = torch.Generator().manual_seed(seed)
    signs = {K: ((torch.rand(K, generator=gen) > 0.5).float() * 2 - 1) for K in widths}
    H = build_hadamard(block)
    for name, mod in targets:
        parts = name.split(".")
        parent = model
        for p in parts[:-1]:
            parent = getattr(parent, p)
        new = TernaryLinear(mod, signs[mod.in_features].to(mod.weight.device),
                            H.to(mod.weight.device), block, group)
        setattr(parent, parts[-1], new)
    tern = sum(m.weight.numel() for m in model.modules() if isinstance(m, TernaryLinear))
    print(f"三值 Linear：{len(targets)} 个 / {tern/1e6:.2f}M 权重；"
          f"K 宽度 {sorted(widths)}；block={block} group={group}", flush=True)
    return tern


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--val", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--accum", type=int, default=2)
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--steps-cap", type=int, default=0)
    ap.add_argument("--save-every", type=int, default=0)
    ap.add_argument("--block", type=int, default=64)
    ap.add_argument("--group", type=int, default=64)
    ap.add_argument("--keep-lm-head", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--keep", default="", help="逗号分隔的名字片段，命中的 Linear 不量化")
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--seed", type=int, default=20261018)
    a = ap.parse_args()

    from transformers import AutoTokenizer, Qwen3ForCausalLM
    rng = random.Random(a.seed)
    torch.manual_seed(a.seed)
    dev = torch.device(a.device)
    if dev.type == "cuda" and not torch.cuda.is_available():
        print("!! CUDA 不可用，回退 CPU", flush=True)
        dev = torch.device("cpu")

    tok = AutoTokenizer.from_pretrained(a.init)
    pad_id = getattr(tok, "pad_token_id", 3) or 3
    model = Qwen3ForCausalLM.from_pretrained(a.init, dtype=torch.float32).to(dev).train()
    model.config.use_cache = False
    keep = [x.strip() for x in a.keep.split(",") if x.strip()]
    replace_linears(model, a.block, a.group, a.seed, a.keep_lm_head, keep)

    samples = []
    for c in load_convs(a.data):
        ids, mask = encode(tok, c, a.max_len)
        if sum(mask) >= 4:
            samples.append((ids, mask))
    print(f"训练样本 {len(samples)} 条", flush=True)
    if not samples:
        return 1
    val_samples = []
    if a.val:
        for c in load_convs(a.val):
            ids, mask = encode(tok, c, a.max_len)
            if sum(mask) >= 4:
                val_samples.append((ids, mask))
        print(f"验证样本 {len(val_samples)} 条", flush=True)

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=a.lr, betas=(0.9, 0.95), eps=1e-8,
                            weight_decay=0.05, fused=(dev.type == "cuda"))
    out = Path(a.out)
    (out / "final").mkdir(parents=True, exist_ok=True)
    logf = (out / "train_log.jsonl").open("w", encoding="utf-8")

    def save(dest: Path):
        dest.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(dest, safe_serialization=True)
        tok.save_pretrained(dest)
        meta = {"block": a.block, "group": a.group, "keep_lm_head": a.keep_lm_head,
                "note": "TernaryLinear 的 state_dict 键与 Qwen3 相同；导出时需按 block/group 重新量化"}
        (dest / "ternary_meta.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")

    @torch.no_grad()
    def eval_val() -> float:
        if not val_samples:
            return float("nan")
        model.eval()
        tot, n = 0.0, 0
        for i in range(0, len(val_samples), a.batch):
            rows = val_samples[i:i + a.batch]
            L = max(len(r[0]) for r in rows)
            bi = np.full((len(rows), L), pad_id, dtype=np.int64)
            bl = np.full((len(rows), L), -100, dtype=np.int64)
            for k, (ids, mask) in enumerate(rows):
                bi[k, :len(ids)] = ids
                lab = make_labels(ids, mask)
                bl[k, :len(lab)] = lab
            ti = torch.tensor(bi, dtype=torch.long, device=dev)
            tl = torch.tensor(bl, dtype=torch.long, device=dev)
            loss = chunked_lm_loss(model, ti, tl, 512)
            tot += float(loss) * int((tl[:, 1:] != -100).sum())
            n += int((tl[:, 1:] != -100).sum())
        model.train()
        return tot / max(1, n)

    steps_per_epoch = max(1, len(samples) // a.batch)
    total = max(1, int(steps_per_epoch * a.epochs / max(1, a.accum)))
    if a.steps_cap:
        total = min(total, a.steps_cap)
    print(f"每轮 {steps_per_epoch} 批，共 {total} 步（等效 batch {a.batch*a.accum}）", flush=True)

    t0 = time.time()
    step = micro = 0
    seen = 0
    run = None
    peak = 0.0
    for epoch in range(max(1, math.ceil(a.epochs))):
        for idx in make_batches(samples, a.batch, rng):
            if step >= total:
                break
            rows = [samples[i] for i in idx]
            L = max(len(r[0]) for r in rows)
            bi = np.full((len(rows), L), pad_id, dtype=np.int64)
            bl = np.full((len(rows), L), -100, dtype=np.int64)
            for k, (ids, mask) in enumerate(rows):
                bi[k, :len(ids)] = ids
                lab = make_labels(ids, mask)
                bl[k, :len(lab)] = lab
            ti = torch.tensor(bi, dtype=torch.long, device=dev)
            tl = torch.tensor(bl, dtype=torch.long, device=dev)
            nl = int((tl[:, 1:] != -100).sum())
            loss = chunked_lm_loss(model, ti, tl, 512)
            (loss / a.accum).backward()
            seen += nl
            run = float(loss.detach()) if run is None else 0.98 * run + 0.02 * float(loss.detach())
            micro += 1
            if micro % a.accum == 0:
                step += 1
                lr = (a.lr * step / max(1, a.warmup) if step <= a.warmup else
                      a.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi *
                      (step - a.warmup) / max(1, total - a.warmup)))))
                for g in opt.param_groups:
                    g["lr"] = lr
                gn = float(torch.nn.utils.clip_grad_norm_(params, 1.0))
                opt.step()
                opt.zero_grad(set_to_none=True)
                if dev.type == "cuda":
                    peak = max(peak, torch.cuda.max_memory_allocated() / 2**30)
                el = time.time() - t0
                rec = {"step": step, "total": total, "epoch": epoch + 1,
                       "loss": round(run, 4), "lr": lr, "grad_norm": round(gn, 3),
                       "seen_tokens": seen, "elapsed_min": round(el / 60, 2),
                       "peak_gib": round(peak, 3)}
                logf.write(json.dumps(rec, ensure_ascii=False) + "\n")
                logf.flush()
                if step % 20 == 0 or step == total:
                    print(f"  step {step}/{total} ep{epoch+1} loss={run:.4f} "
                          f"gn={gn:.2f} {el/60:.1f}min peak={peak:.2f}GiB", flush=True)
                if a.save_every and step % a.save_every == 0:
                    save(out / "rolling" / f"step{step}")
                if step >= total:
                    break

    vloss = eval_val()
    final = out / "final"
    if final.exists():
        shutil.rmtree(final)
    save(final)
    summary = {"init": a.init, "data": a.data, "block": a.block, "group": a.group,
               "keep_lm_head": a.keep_lm_head, "samples": len(samples),
               "steps": step, "epochs": a.epochs, "lr": a.lr, "batch": a.batch,
               "accum": a.accum, "max_len": a.max_len,
               "loss_last": round(run, 4) if run is not None else None,
               "val_loss": None if vloss != vloss else round(vloss, 4),
               "seen_tokens": seen, "peak_gib": round(peak, 3),
               "elapsed_min": round((time.time() - t0) / 60, 2), "final": str(final)}
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                                      encoding="utf-8")
    logf.close()
    print(f"\n完成：{step} 步 / {summary['elapsed_min']} min / 峰值 {peak:.2f} GiB"
          + (f" / val_loss {vloss:.4f}" if vloss == vloss else ""), flush=True)
    print(f"权重 -> {final}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
