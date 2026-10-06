"""KV 量化方案对比实验（PC/GPU）：在同一模型上模拟各种 KV 压缩方案，测长文召回与续写一致性。

目的：现有板端 q2 方案（对称 2bit、每 16 值一块 scale）在几百 token 就崩，
先在这里把候选方案逐一试掉，再决定值不值得写进 C 引擎。

方案（K/V 分别配置）：
  int8      : 每 (head, token) 对称 int8（现役板端方案）
  sym2b16   : 对称 2bit，每 16 值一块 scale（现役 q2）
  asym2b8/16: 非对称 2bit（min/max），每 8/16 值一块
  q3b16     : 非对称 3bit，每 16 值一块
  kivi32/128: K 按「通道 + 32/128 token 分组」非对称 2bit（KIVI 思路），V 非对称 2bit
  k8v2/k2v8 : K int8 + V q2 / K q2 + V int8 混合精度

测试：
  1) 长文取件码召回（不同长度/插入位置，贪心 20 token，答出 6 位数字算命中）
  2) 续写一致性（与 fp32 的 token 序列首个分歧位置）

用法：
  python scripts/kv_quant_experiment.py --model archive/v3_6\\release [--ctx-chars 600,2400,4800] [--schemes ...]
"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer, Qwen3ForCausalLM

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import ROOT  # noqa: E402

FILLER = (ROOT / "esp32s3-feng-llm" / "pc" / "prompt_long.txt").read_text(encoding="utf-8")


# ---------------------------------------------------------------- 量化算子
def _asym(x: torch.Tensor, bits: int, block: int) -> torch.Tensor:
    """非对称均匀量化：把最后一维按 block 分组，min/max → 2^bits 级。"""
    shape = x.shape
    d = shape[-1]
    xb = x.reshape(*shape[:-1], d // block, block)
    lo = xb.amin(dim=-1, keepdim=True)
    hi = xb.amax(dim=-1, keepdim=True)
    step = (hi - lo).clamp_min(1e-8) / (2**bits - 1)
    q = torch.round((xb - lo) / step).clamp_(0, 2**bits - 1)
    return (lo + q * step).reshape(shape)


def _sym2(x: torch.Tensor, block: int) -> torch.Tensor:
    shape = x.shape
    d = shape[-1]
    xb = x.reshape(*shape[:-1], d // block, block)
    scale = xb.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8) / 1.5
    q = torch.round(xb / scale + 1.5).clamp_(0, 3)
    return ((q - 1.5) * scale).reshape(shape)


def _asym_f16(x: torch.Tensor, bits: int, block: int) -> torch.Tensor:
    """非对称量化，但 lo/step 用 fp16 存储（模拟 C 侧 2 字节/参数的存储精度）。"""
    shape = x.shape
    d = shape[-1]
    xb = x.reshape(*shape[:-1], d // block, block)
    lo = xb.amin(dim=-1, keepdim=True)
    hi = xb.amax(dim=-1, keepdim=True)
    step = ((hi - lo).clamp_min(1e-8) / (2**bits - 1)).half().float()
    lo = lo.half().float()
    q = torch.round((xb - lo) / step).clamp_(0, 2**bits - 1)
    return (lo + q * step).reshape(shape)


def _int8_head(x: torch.Tensor) -> torch.Tensor:
    scale = x.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8) / 127.0
    return torch.round(x / scale).clamp_(-127, 127) * scale


def _kivi_k(x: torch.Tensor, group: int) -> torch.Tensor:
    """K：按通道分组量化——每 group 个 token、每个通道单独 min/max 的 2bit 非对称。"""
    b, h, t, d = x.shape
    out = x.clone()
    for s in range(0, t, group):
        e = min(t, s + group)
        blk = x[:, :, s:e, :]
        lo = blk.amin(dim=2, keepdim=True)          # [b,h,1,d]
        hi = blk.amax(dim=2, keepdim=True)
        step = (hi - lo).clamp_min(1e-8) / 3.0
        q = torch.round((blk - lo) / step).clamp_(0, 3)
        out[:, :, s:e, :] = lo + q * step
    return out


def quant(t: torch.Tensor, spec: str, is_k: bool) -> torch.Tensor:
    if spec == "fp32":
        return t
    if spec == "int8":
        return _int8_head(t)
    if spec == "sym2b16":
        return _sym2(t, 16)
    if spec == "sym2b8":
        return _sym2(t, 8)
    if spec == "sym2b4":
        return _sym2(t, 4)
    if spec == "asym2b16":
        return _asym(t, 2, 16)
    if spec == "asym2b8":
        return _asym(t, 2, 8)
    if spec == "asym2b8h":
        return _asym_f16(t, 2, 8)
    if spec == "asym2b16h":
        return _asym_f16(t, 2, 16)
    if spec == "q3b16":
        return _asym(t, 3, 16)
    if spec.startswith("kivi"):
        if is_k:
            return _kivi_k(t, int(spec[4:]))
        return _asym(t, 2, 16)
    if spec == "k8v2":
        return _int8_head(t) if is_k else _asym(t, 2, 16)
    if spec == "k2v8":
        return _asym(t, 2, 16) if is_k else _int8_head(t)
    raise ValueError(spec)


def bytes_per_token(hidden: int, heads: int, head_dim: int, k_spec: str, v_spec: str) -> float:
    """KV 每 token 每层的字节数（值 + fp16 scale），用于比较内存。"""
    def one(spec: str, is_k: bool) -> float:
        if spec == "fp32":
            return hidden * 4
        if spec == "int8":
            return hidden + heads * 2
        if spec == "sym2b16" or spec == "asym2b16" or spec == "k2v8" or spec == "k8v2":
            if spec == "k8v2" and is_k:
                return hidden + heads * 2
            if spec == "k2v8" and not is_k:
                return hidden + heads * 2
            return hidden / 4 + (hidden / 16) * 2
        if spec == "sym2b8":
            return hidden / 4 + (hidden / 8) * 2
        if spec == "sym2b4":
            return hidden / 4 + (hidden / 4) * 2
        if spec == "asym2b8h":
            return hidden / 4 + (hidden / 8) * 4
        if spec == "asym2b16h":
            return hidden / 4 + (hidden / 16) * 4
        if spec == "asym2b8":
            return hidden / 4 + (hidden / 8) * 2
        if spec == "q3b16":
            return hidden * 3 / 8 + (hidden / 16) * 2
        if spec.startswith("kivi"):
            if is_k:
                return hidden / 4 + (hidden // int(spec[4:])) * 2   # 每组每通道一个 scale（近似）
            return hidden / 4 + (hidden / 16) * 2
        raise ValueError(spec)
    return one(k_spec, True) + one(v_spec, False)


# ---------------------------------------------------------------- 生成循环
def k_spec(scheme: str) -> str:
    return {"k8v2": "int8", "k2v8": "asym2b16", "kivi32": "kivi32", "kivi128": "kivi128"}.get(scheme, scheme)


def v_spec(scheme: str) -> str:
    return {"k8v2": "asym2b16", "k2v8": "int8"}.get(scheme, scheme)


@torch.no_grad()
def generate(model, tok, prompt: str, scheme: str, max_new: int = 20,
             prefill_chunk: int = 128) -> list[int]:
    ids = tok(prompt, add_special_tokens=False)["input_ids"]
    cache = None
    out_ids: list[int] = []
    ksp, vsp = k_spec(scheme), v_spec(scheme)

    def step(inp: torch.Tensor):
        nonlocal cache
        out = model(input_ids=inp, past_key_values=cache, use_cache=True)
        cache = out.past_key_values
        n_new = inp.shape[1]
        if scheme != "fp32":
            for layer in cache.layers:
                layer.keys[..., -n_new:, :] = quant(layer.keys[..., -n_new:, :], ksp, True)
                layer.values[..., -n_new:, :] = quant(layer.values[..., -n_new:, :], vsp, False)
        return out

    # 分块 prefill：每个 chunk 写完就量化，后面的 chunk 看到的是量化后的历史（更接近板端行为）
    for s in range(0, len(ids), prefill_chunk):
        chunk = torch.tensor([ids[s:s + prefill_chunk]], dtype=torch.long, device="cuda")
        out = step(chunk)
    for _ in range(max_new):
        nxt = int(out.logits[0, -1].argmax())
        if nxt in (tok.eos_token_id, 2):     # 0=eot, 2=im_end
            break
        out_ids.append(nxt)
        out = step(torch.tensor([[nxt]], dtype=torch.long, device="cuda"))
    return out_ids


# ---------------------------------------------------------------- 测试集
def needle_prompt(n_chars: int, pos_frac: float, code: str) -> str:
    body = (FILLER * (n_chars // len(FILLER) + 1))[:n_chars]
    at = int(len(body) * pos_frac)
    fact = f"（重要信息：快递柜取件码是 {code}。）"
    text = body[:at] + fact + body[at:]
    return (f"<|im_start|>user\n{text}\n\n上文提到的快递柜取件码是多少？请只回答数字。<|im_end|>\n"
            f"<|im_start|>assistant\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(ROOT / "archive" / "v3_6" / "release"))
    ap.add_argument("--schemes", default="fp32,int8,sym2b16,asym2b16,asym2b8,q3b16,kivi32,kivi128,k8v2,k2v8")
    ap.add_argument("--ctx-chars", default="600,2400,4800")
    ap.add_argument("--out", default=str(ROOT / "eval" / "kv_quant_experiment.json"))
    args = ap.parse_args()
    schemes = [s.strip() for s in args.schemes.split(",") if s.strip()]
    lens = [int(x) for x in args.ctx_chars.split(",")]

    tok = AutoTokenizer.from_pretrained(args.model)
    model = Qwen3ForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16).to("cuda").eval()
    cfg = model.config
    print(f"model {args.model}: {cfg.num_hidden_layers}L hidden={cfg.hidden_size} heads={cfg.num_attention_heads}x{cfg.head_dim}")
    for s in schemes:
        b = bytes_per_token(cfg.hidden_size, cfg.num_attention_heads, cfg.head_dim, k_spec(s), v_spec(s))
        print(f"  {s:9s} ~{b:7.1f} B/token/layer  (ctx1024 all-layers {b*1024*cfg.num_hidden_layers/2**20:.2f} MiB)")

    rows = []
    for n_chars in lens:
        for pos_frac in (0.25, 0.7):
            code = f"{abs(hash((n_chars, pos_frac))) % 900000 + 100000}"
            prompt = needle_prompt(n_chars, pos_frac, code)
            rec = {"chars": n_chars, "pos": pos_frac, "code": code, "schemes": {}}
            for s in schemes:
                t0 = time.time()
                ids = generate(model, tok, prompt, s)
                text = tok.decode(ids)
                ok = code in text
                rec["schemes"][s] = {"ok": ok, "text": text[:60], "sec": round(time.time() - t0, 1)}
                print(f"[needle {n_chars}c pos={pos_frac}] {s:9s} {'OK ' if ok else 'MISS'} "
                      f"({time.time()-t0:.1f}s) {text[:40]!r}")
            rows.append(rec)

    # 续写一致性：与 fp32 比 token 序列的首个分歧
    prose = FILLER[:600]
    p = f"<|im_start|>user\n{prose}<|im_end|>\n<|im_start|>assistant\n"
    base = generate(model, tok, p, "fp32", max_new=24)
    cont = {"schemes": {}}
    for s in schemes:
        ids = generate(model, tok, p, s, max_new=24)
        div = next((i for i, (a, b) in enumerate(zip(base, ids)) if a != b), min(len(base), len(ids)))
        cont["schemes"][s] = {"first_div": div, "match": div / max(1, len(base)),
                              "text": tok.decode(ids)[:60]}
        print(f"[prose] {s:9s} first_div={div}/{len(base)} match={div/max(1,len(base)):.2f} {tok.decode(ids)[:40]!r}")

    out = {"model": args.model, "needle": rows, "prose": cont,
           "bytes_per_token_per_layer": {s: bytes_per_token(cfg.hidden_size, cfg.num_attention_heads,
                                                            cfg.head_dim, k_spec(s), v_spec(s)) for s in schemes}}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
