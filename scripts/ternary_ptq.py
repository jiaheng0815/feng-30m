"""三值量化 PTQ 实验（分支 `ternary`，不发布）。

参考 Bonsai 2 的三值配方（公开资料）：
  - 权重 {-d, 0, +d}，每组一个 fp16 scale，scale = 组内 max|w|；
  - 量化前对权重做 **blockwise Walsh–Hadamard 旋转**（只转输入维 K），
    运行时对激活做同样的变换：x' = FWHT(x ⊙ s) / sqrt(B)，W' = (W ⊙ s) H / sqrt(B)；
  - lm_head / embedding / norm 等保持较高精度（selective precision）。

本脚本是**纯 PyTorch 模拟**：不产生引擎格式，只回答“30M 模型在 PTQ 下能保住多少”。
真正的板端实现要等这里的 PPL/留出结果过关后再做（2-bit 打包 + C23 内核）。

用法：
    python scripts\\ternary_ptq.py --model v4/base --limit 0
    python scripts\\ternary_ptq.py --model v4/base --no-rotate
    python scripts\\ternary_ptq.py --model v4/patch_name_e5_lr5e-6 --rotate --block 64 --group 64
"""
import argparse
import json
import math
import sys
from pathlib import Path

import torch

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = Path(__file__).resolve().parents[1]


def build_hadamard(n: int) -> torch.Tensor:
    """Sylvester 序的 H_n（n 为 2 的幂），H H^T = n I。"""
    h = torch.ones(1, 1)
    while h.shape[0] < n:
        top = torch.cat([h, h], dim=1)
        bot = torch.cat([h, -h], dim=1)
        h = torch.cat([top, bot], dim=0)
    return h


def block_rotate(x: torch.Tensor, signs: torch.Tensor, H: torch.Tensor,
                 block: int) -> torch.Tensor:
    """对最后一维按 block 做 (x ⊙ s) H / sqrt(B)。x 可为 [..., K]。"""
    K = x.shape[-1]
    nb = K // block
    y = x.reshape(*x.shape[:-1], nb, block) * signs.view(nb, block)
    y = y @ H
    return (y / math.sqrt(block)).reshape(*x.shape)


def ternarize(w: torch.Tensor, group: int) -> torch.Tensor:
    """按最后一维分组，absmax scale + round/clamp 到 {-1,0,1}。"""
    out, K = w.shape
    if K % group:
        raise ValueError(f"K={K} 不能被 group={group} 整除")
    wg = w.reshape(out, K // group, group)
    d = wg.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8)
    q = (wg / d).round().clamp(-1, 1)
    return (q * d).reshape_as(w)


@torch.no_grad()
def eval_ppl(model, tok, rows, device, max_len: int) -> float:
    nll, n = 0.0, 0
    for r in rows:
        ids, mask = [], []
        for m in r["messages"]:
            t = tok.encode(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n",
                           add_special_tokens=False)
            ids += t
            mask += [1 if m["role"] == "assistant" else 0] * len(t)
        if len(ids) < 2:
            continue
        ids, mask = ids[-max_len:], mask[-max_len:]
        inp = torch.tensor([ids[:-1]], dtype=torch.long, device=device)
        lab = torch.tensor([ids[1:]], dtype=torch.long, device=device)
        m = torch.tensor([mask[1:]], dtype=torch.bool, device=device)
        logits = model(input_ids=inp).logits.float()
        logp = torch.log_softmax(logits, dim=-1)
        tok_lp = logp.gather(-1, lab.unsqueeze(-1)).squeeze(-1)
        if m.any():
            nll += float(-tok_lp[m].sum())
            n += int(m.sum())
    return math.exp(nll / max(1, n))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", default=str(ROOT / "data" / "v4_val.jsonl"))
    ap.add_argument("--out", default="")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--rotate", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--block", type=int, default=64, help="Hadamard 块（2 的幂，需整除 K）")
    ap.add_argument("--group", type=int, default=64, help="scale 分组（需整除 K）")
    ap.add_argument("--keep-lm-head", action=argparse.BooleanOptionalAction, default=True,
                    help="lm_head/embedding 保持 fp16（selective precision）")
    ap.add_argument("--keep", default="", help="逗号分隔的名字片段，命中的 Linear 不量化（如 o_proj,down_proj）")
    ap.add_argument("--seed", type=int, default=20261018)
    a = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch.manual_seed(a.seed)
    dev = torch.device(a.device)
    tok = AutoTokenizer.from_pretrained(a.model)
    rows = [json.loads(l) for l in Path(a.data).read_text(encoding="utf-8").splitlines()
            if l.strip()]
    if a.limit:
        rows = rows[:a.limit]
    model = AutoModelForCausalLM.from_pretrained(
        a.model, dtype=torch.float32, attn_implementation="sdpa").to(dev).eval()

    base = eval_ppl(model, tok, rows, dev, a.max_len)
    print(f"基线 PPL（assistant）：{base:.4f}", flush=True)

    # 每个输入宽度共享一条 ±1 符号向量（对应 Bonsai 的 per-K sign vector）
    linears = []
    widths = set()
    keep = [s.strip() for s in a.keep.split(",") if s.strip()]
    for name, mod in model.named_modules():
        if not isinstance(mod, torch.nn.Linear):
            continue
        if a.keep_lm_head and name.endswith("lm_head"):
            continue
        if any(s in name for s in keep):
            print(f"保留 {name}（--keep）")
            continue
        if mod.in_features % a.block or mod.in_features % a.group:
            print(f"跳过 {name}（K={mod.in_features} 不能被 block/group 整除）")
            continue
        linears.append((name, mod))
        widths.add(mod.in_features)

    gen = torch.Generator().manual_seed(a.seed)
    signs = {K: (torch.rand(K, generator=gen).to(dev) > 0.5).float() * 2 - 1
             for K in widths}
    hs = {B: build_hadamard(B).to(dev) for B in {a.block}}

    tern_params = kept_params = 0
    for name, mod in linears:
        K = mod.in_features
        w = mod.weight.detach()
        if a.rotate:
            w = block_rotate(w, signs[K], hs[a.block], a.block)
        wq = ternarize(w, a.group)
        mod.weight.data.copy_(wq)
        tern_params += mod.weight.numel()
        if a.rotate:
            s, H, B = signs[K], hs[a.block], a.block
            mod.register_forward_pre_hook(
                lambda m, args, s=s, H=H, B=B: (block_rotate(args[0], s, H, B),))
    kept_params = sum(p.numel() for p in model.parameters()) - tern_params
    # 估算：2-bit 码 + 每 group 一个 fp16 scale；保留参数按 Q4 4.25 bpw
    bpw = 2.0 + 16.0 / a.group
    est_bytes = tern_params * bpw / 8 + kept_params * 4.25 / 8
    print(f"三值权重 {tern_params/1e6:.2f}M ｜ 保留 {kept_params/1e6:.2f}M ｜ "
          f"估算 {est_bytes/1048576:.2f} MiB（{bpw:.2f} bpw）", flush=True)

    ppl = eval_ppl(model, tok, rows, dev, a.max_len)
    print(f"三值 PPL（assistant）：{ppl:.4f} ｜ 相对基线 {ppl/base:.2f}×", flush=True)
    out = {"model": a.model, "data": a.data, "rows": len(rows),
           "rotate": a.rotate, "block": a.block, "group": a.group,
           "keep_lm_head": a.keep_lm_head,
           "ppl_base": base, "ppl_ternary": ppl, "ratio": ppl / base,
           "ternary_params": tern_params, "kept_params": kept_params,
           "ternary_bpw": 2.0 + 16.0 / a.group, "est_mib": est_bytes / 1048576}
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
