"""Isolate quantization error from implementation error.

1. loads the HF model, applies the *same* Q4 round-trip as the exporter,
2. runs the same prompt, compares logits against the C engine's dumped logits,
3. also reports Q4-round-trip vs fp32 (pure quantization error) for reference.
"""
import argparse
import json
import struct
import sys
from pathlib import Path

import numpy as np
import torch

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from export_model import DT_FP16, DT_Q4, QK, quantize_q4   # noqa: E402

MAGIC = 0x46574E31


def load_model_bin(path: Path):
    blob = path.read_bytes()
    magnetic, version, n_tensors, n_layers, hidden, n_heads, head_dim, ffn, vocab, rms_eps, rope = \
        struct.unpack_from("<9I2f", blob, 0)
    assert magnetic == MAGIC, hex(magnetic)
    tensors = {}
    off = 44
    for _ in range(n_tensors):
        name = blob[off:off + 32].split(b"\x00")[0].decode()
        dtype, s0, s1, s2, s3, t_off, nbytes = struct.unpack_from("<IIIIIQQ", blob, off + 32)
        shape = tuple(v for v in (s0, s1, s2, s3) if v > 0)
        tensors[name] = {"dtype": dtype, "shape": shape, "offset": t_off, "nbytes": nbytes}
        off += 68
    return {"cfg": dict(n_layers=n_layers, hidden=hidden, n_heads=n_heads, head_dim=head_dim,
                        ffn=ffn, vocab=vocab, rms_eps=rms_eps, rope=rope),
            "tensors": tensors, "blob": blob}


def dequant_tensor(mb, name):
    t = mb["tensors"][name]
    if len(t["shape"]) == 1:
        rows, cols = 1, t["shape"][0]
        flat = True
    else:
        rows, cols = t["shape"]
        flat = False
    raw = mb["blob"][t["offset"]:t["offset"] + t["nbytes"]]
    if t["dtype"] == DT_FP16:
        w = np.frombuffer(raw, dtype=np.float16).astype(np.float32).reshape(rows, cols)
        return w.reshape(-1) if flat else w
    blocks = cols // QK
    out = np.zeros((rows, cols), dtype=np.float32)
    for r in range(rows):
        base = r * blocks * 34
        scales = np.frombuffer(raw[base:base + blocks * 2], dtype=np.float16).astype(np.float32)
        packed = np.frombuffer(raw[base + blocks * 2:base + blocks * 34], dtype=np.uint8)
        packed = packed.reshape(blocks, 32)
        lo = (packed & 0x0F).astype(np.int16) - 8
        hi = (packed >> 4).astype(np.int16) - 8
        vals = np.empty((blocks, QK), dtype=np.float32)
        vals[:, 0::2] = lo
        vals[:, 1::2] = hi
        out[r] = (vals * scales[:, None]).reshape(-1)
    return out.reshape(-1) if flat else out


def rmsnorm(x, w, eps):
    v = x.pow(2).mean(-1, keepdim=True)
    return x * torch.rsqrt(v + eps) * w


def forward_torch(mb, ids, deq):
    """Same math as feng_llm.c, in torch (fp32), using dequantized weights."""
    cfg = mb["cfg"]
    hidden, nh, hd, ffn, vocab = (cfg["hidden"], cfg["n_heads"], cfg["head_dim"], cfg["ffn"],
                                  cfg["vocab"])
    eps, rope_theta = cfg["rms_eps"], cfg["rope"]
    emb = deq("tok_embd.weight")
    kcache, vcache = {}, {}
    logits = None
    for pos, tok in enumerate(ids):
        x = torch.from_numpy(emb[tok].copy())
        for l in range(cfg["n_layers"]):
            p = f"blk.{l}."
            xn = rmsnorm(x, torch.from_numpy(deq(p + "attn_norm.weight")), eps)
            q = xn @ torch.from_numpy(deq(p + "attn_q.weight")).T
            k = xn @ torch.from_numpy(deq(p + "attn_k.weight")).T
            v = xn @ torch.from_numpy(deq(p + "attn_v.weight")).T
            qn = torch.from_numpy(deq(p + "attn_q_norm.weight"))
            kn = torch.from_numpy(deq(p + "attn_k_norm.weight"))
            qh = rmsnorm(q.view(nh, hd), qn, eps)
            kh = rmsnorm(k.view(nh, hd), kn, eps)
            for h in range(nh):
                for i in range(hd // 2):
                    inv = rope_theta ** (-2.0 * i / hd)
                    ang = pos * inv
                    c, s = np.cos(ang), np.sin(ang)
                    x1, x2 = qh[h, i].item(), qh[h, i + hd // 2].item()
                    qh[h, i] = x1 * c - x2 * s
                    qh[h, i + hd // 2] = x2 * c + x1 * s
                    x1, x2 = kh[h, i].item(), kh[h, i + hd // 2].item()
                    kh[h, i] = x1 * c - x2 * s
                    kh[h, i + hd // 2] = x2 * c + x1 * s
            kcache[l] = torch.cat([kcache.get(l, torch.zeros(0, nh * hd)), kh.reshape(1, -1)], 0)
            vcache[l] = torch.cat([vcache.get(l, torch.zeros(0, nh * hd)), v.reshape(1, -1)], 0)
            K = kcache[l].view(-1, nh, hd)
            V = vcache[l].view(-1, nh, hd)
            attn = torch.zeros(nh, hd)
            scale = 1.0 / np.sqrt(hd)
            for h in range(nh):
                sc = (K[:, h, :] @ qh[h]) * scale
                pr = torch.softmax(sc, dim=0)
                attn[h] = pr @ V[:, h, :]
            x = x + attn.reshape(-1) @ torch.from_numpy(deq(p + "attn_output.weight")).T
            xn = rmsnorm(x, torch.from_numpy(deq(p + "ffn_norm.weight")), eps)
            g = xn @ torch.from_numpy(deq(p + "ffn_gate.weight")).T
            u = xn @ torch.from_numpy(deq(p + "ffn_up.weight")).T
            x = x + (g * torch.sigmoid(g) * u) @ torch.from_numpy(deq(p + "ffn_down.weight")).T
        xn = rmsnorm(x, torch.from_numpy(deq("output_norm.weight")), eps)
        logits = xn @ torch.from_numpy(emb).T
    return logits.numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--export", required=True)
    ap.add_argument("--model", required=True, help="HF model dir (for the fp32 reference)")
    ap.add_argument("--c-logits", required=True)
    args = ap.parse_args()

    mb = load_model_bin(Path(args.export) / "model.bin")
    ids = json.loads((Path(args.export) / "ref_ids.json").read_text(encoding="utf-8"))
    c_logits = np.fromfile(args.c_logits, dtype=np.float32)
    cache = {}

    def deq(name):
        if name not in cache:
            cache[name] = dequant_tensor(mb, name)
        return cache[name]

    t_q4 = forward_torch(mb, ids, deq)
    d = np.abs(t_q4 - c_logits)
    print(f"[C vs torch(Q4)] max|diff|={d.max():.4f} mean={d.mean():.5f} "
          f"argmax C={int(c_logits.argmax())} torch={int(t_q4.argmax())}")

    from transformers import AutoModelForCausalLM
    ref = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32)
    with torch.no_grad():
        r = ref(torch.tensor([ids])).logits[0, -1].numpy()
    d2 = np.abs(r - t_q4)
    print(f"[torch(Q4) vs torch(fp32)] max|diff|={d2.max():.4f} mean={d2.mean():.5f} "
          f"  <- pure quantization error")
    d3 = np.abs(r - c_logits)
    print(f"[C vs torch(fp32)] max|diff|={d3.max():.4f} mean={d3.mean():.5f}")


if __name__ == "__main__":
    main()
