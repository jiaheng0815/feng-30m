"""Export the distilled 30M model to the ESP32-S3 firmware format.

Outputs (into --out):
  model.bin      header + tensor directory + Q4/fp16 tensors
  tokenizer.bin  vocab / merges / special tokens for the on-device BPE
  ref_logits.bin float32 logits of a reference prompt (for the PC self-check)
  export_info.json
"""
import argparse
import hashlib
import json
import struct
import sys
from pathlib import Path

import numpy as np
import torch

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

MAGIC = 0x46574E31          # "FWN1"
DT_FP16 = 0
DT_Q4 = 1                   # block 64: fp16 scale + 32 bytes of 4-bit values
QK = 64


def quantize_q4(w: torch.Tensor):
    """row-major (out, in) tensor -> (scales fp16 array, packed nibbles uint8 array)."""
    w = w.detach().float()
    out_f, in_f = w.shape
    assert in_f % QK == 0, f"{w.shape} not divisible by {QK}"
    wb = w.reshape(out_f, in_f // QK, QK)
    scale = wb.abs().amax(dim=2, keepdim=True) / 7.0        # signed 4-bit range [-8,7] -> use 7
    scale = torch.clamp(scale, min=1e-8)
    q = torch.round(wb / scale).clamp_(-8, 7).to(torch.int8)
    q = (q + 8).to(torch.uint8)                              # 0..15
    hi = q[:, :, 0::2]
    lo = q[:, :, 1::2]
    packed = (hi | (lo << 4)).to(torch.uint8).contiguous()   # (out, blocks, 32)
    return scale.squeeze(2).to(torch.float16).contiguous(), packed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    # 以前是 store_true + default=True —— 永远关不掉，else 分支是死代码。
    # 改成 BooleanOptionalAction（--no-quantize-embedding），默认仍是量化，行为不变。
    ap.add_argument("--quantize-embedding", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--keep", type=int, default=200, help="tensors with fewer params stay fp16")
    ap.add_argument("--weights-dtype", choices=["q4", "fp16"], default="q4",
                    help="2D 矩阵权重的存储格式。fp16 只有手机端/PC 端能跑"
                         "（58.9 MB，板端 16 MB mmap 窗口放不下）")
    ap.add_argument("--ctx", type=int, default=2048,
                    help="写进 export_info.json 的运行时上下文（板端 q2 默认 2048）")
    ap.add_argument("--max-model-bytes", type=int, default=0,
                    help=">0 时检查 model.bin 不超过该字节数（板端 0xEF0000）")
    ap.add_argument("--max-tokenizer-bytes", type=int, default=0,
                    help=">0 时检查 tokenizer.bin 不超过该字节数（板端 0x80000）")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    from transformers import AutoTokenizer, Qwen3ForCausalLM
    tok = AutoTokenizer.from_pretrained(args.model)
    model = Qwen3ForCausalLM.from_pretrained(args.model, dtype=torch.float32)
    cfg = model.config
    n_layers = cfg.num_hidden_layers
    hidden = cfg.hidden_size
    n_heads = cfg.num_attention_heads
    n_kv = getattr(cfg, "num_key_value_heads", n_heads)
    head_dim = cfg.head_dim or hidden // n_heads
    ffn = cfg.intermediate_size
    vocab = cfg.vocab_size
    rope_theta = float(getattr(cfg, "rope_theta", None)
                       or (getattr(cfg, "rope_parameters", {}) or {}).get("rope_theta", 1000000.0))
    # C 引擎按 MHA（7 KV 头 = 7 头）和 tied head 实现；导出前先拒绝不兼容的模型，
    # 否则板端会按错误布局静默解码。
    if n_kv != n_heads:
        raise SystemExit(f"不支持的模型：num_key_value_heads={n_kv} != num_attention_heads={n_heads}"
                         "（C 引擎是 MHA）")
    if not getattr(cfg, "tie_word_embeddings", False):
        raise SystemExit("不支持的模型：tie_word_embeddings=False（C 引擎的 lm head 复用 tok_embd）")
    if n_heads * head_dim != hidden:
        raise SystemExit(f"不支持的模型：n_heads*head_dim={n_heads*head_dim} != hidden={hidden}")
    if vocab != len(tok):
        raise SystemExit(f"词表不一致：config.vocab_size={vocab}, tokenizer={len(tok)}")
    print(f"model: {n_layers}L hidden={hidden} heads={n_heads}x{head_dim} ffn={ffn} vocab={vocab} "
          f"rope_theta={rope_theta}")

    sd = model.state_dict()
    tensors = []           # (name, dtype, shape, bytes)

    def add_fp16(name, t):
        tensors.append((name, DT_FP16, tuple(t.shape), t.half().numpy().tobytes()))

    def add_q4(name, t):
        s, p = quantize_q4(t)
        # layout: per output row: [scale fp16]*blocks then packed nibbles
        rows = t.shape[0]
        blocks = t.shape[1] // QK
        buf = bytearray()
        s_np = s.numpy().reshape(rows, blocks)
        p_np = p.numpy().reshape(rows, blocks, 32)
        for r in range(rows):
            buf += s_np[r].tobytes()
            buf += p_np[r].tobytes()
        tensors.append((name, DT_Q4, tuple(t.shape), bytes(buf)))

    def maybe(name, t, force_fp16=False):
        n = t.numel()
        if (force_fp16 or args.weights_dtype == "fp16" or n < args.keep
                or t.dim() < 2 or t.shape[1] % QK != 0):
            add_fp16(name, t)
        else:
            add_q4(name, t)

    if args.quantize_embedding:
        maybe("tok_embd.weight", sd["model.embed_tokens.weight"])
    else:
        add_fp16("tok_embd.weight", sd["model.embed_tokens.weight"])
    for i in range(n_layers):
        p = f"model.layers.{i}."
        maybe(f"blk.{i}.attn_norm.weight", sd[p + "input_layernorm.weight"], True)
        maybe(f"blk.{i}.ffn_norm.weight", sd[p + "post_attention_layernorm.weight"], True)
        maybe(f"blk.{i}.attn_q.weight", sd[p + "self_attn.q_proj.weight"])
        maybe(f"blk.{i}.attn_k.weight", sd[p + "self_attn.k_proj.weight"])
        maybe(f"blk.{i}.attn_v.weight", sd[p + "self_attn.v_proj.weight"])
        maybe(f"blk.{i}.attn_output.weight", sd[p + "self_attn.o_proj.weight"])
        maybe(f"blk.{i}.attn_q_norm.weight", sd[p + "self_attn.q_norm.weight"], True)
        maybe(f"blk.{i}.attn_k_norm.weight", sd[p + "self_attn.k_norm.weight"], True)
        maybe(f"blk.{i}.ffn_gate.weight", sd[p + "mlp.gate_proj.weight"])
        maybe(f"blk.{i}.ffn_up.weight", sd[p + "mlp.up_proj.weight"])
        maybe(f"blk.{i}.ffn_down.weight", sd[p + "mlp.down_proj.weight"])
    maybe("output_norm.weight", sd["model.norm.weight"], True)

    # ---------- write binary
    n_t = len(tensors)
    dir_fmt = "<32sIIIIIQQ"         # name, dtype, shape[4], offset, nbytes  (68 bytes)
    # header must match feng_model.c byte-for-byte:
    #   u32 magic, version, n_tensors, n_layers, hidden, n_heads, head_dim, ffn, vocab
    #   f32 rms_eps, rope_theta                                          (44 bytes total)
    header = struct.pack("<9I2f", MAGIC, 1, n_t, n_layers, hidden, n_heads, head_dim, ffn, vocab,
                         float(cfg.rms_norm_eps), rope_theta)
    dir_entry_size = struct.calcsize(dir_fmt)          # 32+4+16+8+8 = 68
    data_off = (len(header) + dir_entry_size * n_t + 63) // 64 * 64
    entries, blobs, off = [], [], data_off
    for name, dt, shape, blob in tensors:
        shp = list(shape) + [0] * (4 - len(shape))
        entries.append(struct.pack(dir_fmt, name.encode(), dt, *shp, off, len(blob)))
        blobs.append(blob)
        off += len(blob)
        off = (off + 63) // 64 * 64
    with (out / "model.bin").open("wb") as f:
        f.write(header)
        f.write(b"".join(entries))
        f.write(b"\x00" * (data_off - len(header) - dir_entry_size * n_t))
        for blob in blobs:
            f.write(blob)
            p = (len(blob) + 63) // 64 * 64
            f.write(b"\x00" * (p - len(blob)))
    size_mb = (out / "model.bin").stat().st_size / 1024**2
    print(f"model.bin: {size_mb:.2f} MB, {n_t} tensors")

    # ---------- tokenizer.bin
    vocab_map = tok.get_vocab()
    id2tok = [None] * (max(vocab_map.values()) + 1)
    for t, i in vocab_map.items():
        id2tok[i] = t
    merges = []
    tj_path = Path(args.model) / "tokenizer.json"
    if tj_path.exists():
        merges = json.loads(tj_path.read_text(encoding="utf-8"))["model"].get("merges", [])
    specials = {t: i for t, i in vocab_map.items()
                if t.startswith("<|") and t.endswith("|>")}
    with (out / "tokenizer.bin").open("wb") as f:
        f.write(struct.pack("<II", len(id2tok), len(merges)))
        for t in id2tok:
            b = (t or "").encode("utf-8")
            f.write(struct.pack("<H", len(b)) + b)
        for m in merges:
            a, b = (m if isinstance(m, (list, tuple)) else m.split(" "))[:2]
            f.write(struct.pack("<HH", len(a.encode()), len(b.encode())))
            f.write(a.encode() + b.encode())
    print(f"tokenizer.bin: {(out / 'tokenizer.bin').stat().st_size / 1024:.0f} KB, "
          f"vocab {len(id2tok)}, merges {len(merges)}, specials {specials}")
    model_path = out / "model.bin"
    tok_path = out / "tokenizer.bin"
    if args.max_model_bytes and model_path.stat().st_size > args.max_model_bytes:
        raise SystemExit(f"model.bin {model_path.stat().st_size} B 超过 --max-model-bytes "
                         f"{args.max_model_bytes}")
    if args.max_tokenizer_bytes and tok_path.stat().st_size > args.max_tokenizer_bytes:
        raise SystemExit(f"tokenizer.bin {tok_path.stat().st_size} B 超过 --max-tokenizer-bytes "
                         f"{args.max_tokenizer_bytes}")
    model_sha = hashlib.sha256(model_path.read_bytes()).hexdigest()
    tok_sha = hashlib.sha256(tok_path.read_bytes()).hexdigest()

    # ---------- reference logits for the PC self-check
    prompt = "你好"
    text = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
    ids = tok(text, add_special_tokens=False)["input_ids"]
    with torch.no_grad():
        out_logits = model(torch.tensor([ids])).logits[0, -1].float().numpy()
    (out / "ref_logits.bin").write_bytes(out_logits.tobytes())
    (out / "ref_ids.json").write_text(json.dumps(ids), encoding="utf-8")
    (out / "export_info.json").write_text(json.dumps(
        {"model": args.model, "layers": n_layers, "hidden": hidden, "heads": n_heads,
         "head_dim": head_dim, "ffn": ffn, "vocab": vocab, "ctx": args.ctx,
         "rope_theta": rope_theta, "rms_eps": float(cfg.rms_norm_eps),
         "model_mb": round(size_mb, 2),
         "format_version": 1, "qk": QK,
         "weights_dtype": args.weights_dtype, "keep": args.keep,
         "quantize_embedding": bool(args.quantize_embedding),
         "model_sha256": model_sha, "tokenizer_sha256": tok_sha,
         "ref_prompt_ids": ids}, ensure_ascii=False, indent=2),
        encoding="utf-8")
    print("wrote ref_logits.bin / ref_ids.json / export_info.json")


if __name__ == "__main__":
    main()
