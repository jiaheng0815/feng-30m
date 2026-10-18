"""把板端 `model.bin`（Q4/fp16）反量化回 HF 目录，供后训练使用。

板端权重是 Q4 权重 + q2 KV 双 QAT 的产物，仓库里没有对应的 HF 源目录；
本脚本按 `board/tools/export_model.py` 的格式逐张量反量化，得到
`v4/base_v319b6_deq/`（fp32，Q4 舍入误差保留）。它只用于继续训练/评测，
不改变板端导出本身。

用法：
    python tools/dequant_model_bin.py --export board/model_export_v3_19b6 `
        --config v4/base --out v4/base_v319b6_deq
"""
import argparse
import struct
import sys
from pathlib import Path

import numpy as np
import torch

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = Path(__file__).resolve().parents[1]

MAGIC = 0x46574E31
DT_FP16, DT_Q4 = 0, 1
QK = 64

NAME_MAP = {
    "tok_embd.weight": "model.embed_tokens.weight",
    "output_norm.weight": "model.norm.weight",
}


def hf_name(name: str) -> str:
    if name in NAME_MAP:
        return NAME_MAP[name]
    if name.startswith("blk."):
        _, i, rest = name.split(".", 2)
        p = f"model.layers.{i}."
        m = {
            "attn_norm.weight": "input_layernorm.weight",
            "ffn_norm.weight": "post_attention_layernorm.weight",
            "attn_q.weight": "self_attn.q_proj.weight",
            "attn_k.weight": "self_attn.k_proj.weight",
            "attn_v.weight": "self_attn.v_proj.weight",
            "attn_output.weight": "self_attn.o_proj.weight",
            "attn_q_norm.weight": "self_attn.q_norm.weight",
            "attn_k_norm.weight": "self_attn.k_norm.weight",
            "ffn_gate.weight": "mlp.gate_proj.weight",
            "ffn_up.weight": "mlp.up_proj.weight",
            "ffn_down.weight": "mlp.down_proj.weight",
        }
        if rest not in m:
            raise SystemExit(f"未知张量 {name}")
        return p + m[rest]
    raise SystemExit(f"未知张量 {name}")


def dequant_q4(blob: bytes, shape: tuple[int, int]) -> np.ndarray:
    rows, cols = shape
    blocks = cols // QK
    assert cols % QK == 0
    stride = blocks * (2 + 32)
    assert len(blob) == rows * stride, (len(blob), rows, stride)
    out = np.empty((rows, cols), dtype=np.float32)
    for r in range(rows):
        base = r * stride
        scales = np.frombuffer(blob, dtype=np.float16, count=blocks,
                               offset=base).astype(np.float32)
        packed = np.frombuffer(blob, dtype=np.uint8, count=blocks * 32,
                               offset=base + blocks * 2).reshape(blocks, 32)
        # 导出器：packed = even_index | (odd_index << 4)
        # 所以低 nibble 是偶数下标、高 nibble 是奇数下标。
        even = (packed & 0x0F).astype(np.int16) - 8
        odd = (packed >> 4).astype(np.int16) - 8
        q = np.empty((blocks, QK), dtype=np.int16)
        q[:, 0::2] = even
        q[:, 1::2] = odd
        out[r] = (q.astype(np.float32) * scales[:, None]).reshape(cols)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--export", default=str(ROOT / "board" / "model_export_v3_19b6"))
    ap.add_argument("--config", default=str(ROOT / "v4" / "base"))
    ap.add_argument("--out", default=str(ROOT / "v4" / "base_v319b6_deq"))
    args = ap.parse_args()

    export = Path(args.export)
    blob = (export / "model.bin").read_bytes()
    (magic, ver, n_t, n_layers, hidden, n_heads, head_dim, ffn, vocab,
     rms_eps, rope_theta) = struct.unpack_from("<9I2f", blob, 0)
    if magic != MAGIC or ver != 1:
        raise SystemExit(f"不是 feng model.bin: magic={magic:#x} ver={ver}")
    if (n_heads * head_dim != hidden):
        raise SystemExit("head 维度不自洽")
    print(f"header: {n_layers}L h={hidden} heads={n_heads}x{head_dim} "
          f"ffn={ffn} vocab={vocab} rope={rope_theta:g} tensors={n_t}")

    dir_fmt = "<32sIIIIIQQ"
    esz = struct.calcsize(dir_fmt)
    state = {}
    off = 44
    for _ in range(n_t):
        name_b, dt, s0, s1, s2, s3, offset, nbytes = struct.unpack_from(dir_fmt, blob, off)
        off += esz
        name = name_b.split(b"\0", 1)[0].decode()
        shape = tuple(x for x in (s0, s1, s2, s3) if x)
        raw = blob[offset:offset + nbytes]
        if dt == DT_FP16:
            a = np.frombuffer(raw, dtype=np.float16, count=int(np.prod(shape)))
            a = a.reshape(shape).astype(np.float32)
        elif dt == DT_Q4:
            if len(shape) != 2:
                raise SystemExit(f"Q4 张量不是 2D：{name} {shape}")
            a = dequant_q4(raw, shape)
        else:
            raise SystemExit(f"未知 dtype {dt} @ {name}")
        key = hf_name(name)
        state[key] = torch.from_numpy(a.copy())
        print(f"  {name:28s} -> {key:45s} {shape} dt={dt}")

    from transformers import AutoConfig, AutoTokenizer, Qwen3ForCausalLM
    cfg = AutoConfig.from_pretrained(args.config)
    model = Qwen3ForCausalLM(cfg).float()
    missing, unexpected = model.load_state_dict(state, strict=False)
    # tied embedding：lm_head 与 embed_tokens 共享，state_dict 里本来就没有 lm_head。
    if missing not in ([], ["lm_head.weight"]) or unexpected:
        raise SystemExit(f"state_dict 不匹配：missing={missing} unexpected={unexpected}")
    if not cfg.tie_word_embeddings:
        raise SystemExit("config 不是 tied embedding，无法从板端权重恢复 lm_head")
    model.tie_weights()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out, safe_serialization=True)
    AutoTokenizer.from_pretrained(args.config).save_pretrained(out)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
