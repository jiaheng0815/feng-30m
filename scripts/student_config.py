"""Student model config (Qwen3 architecture, ~30M params, 32k context)."""
from transformers import Qwen3Config

EOS_ID = 0          # <|endoftext|>
IM_START_ID = 1
IM_END_ID = 2
PAD_ID = 3


def build_config(vocab_size=32768, max_pos=32768, rope_scaling=None):
    """Native long-context config: no RoPE interpolation, rope_theta 1e6."""
    return Qwen3Config(
        vocab_size=vocab_size,
        hidden_size=448,
        num_hidden_layers=8,
        num_attention_heads=7,
        # MHA (7 = 7): this torch build's memory-efficient SDPA kernel does not support GQA,
        # and the fp32 math fallback materialises an 8 GiB attention matrix at 8192 tokens.
        num_key_value_heads=7,
        head_dim=64,
        intermediate_size=896,
        hidden_act="silu",
        max_position_embeddings=max_pos,
        rms_norm_eps=1e-6,
        rope_theta=1000000.0,
        rope_scaling=rope_scaling,
        attention_bias=False,
        attention_dropout=0.0,
        tie_word_embeddings=True,
        initializer_range=0.02,
        use_cache=True,
        bos_token_id=IM_START_ID,
        eos_token_id=EOS_ID,
        pad_token_id=PAD_ID,
    )


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    from transformers import Qwen3ForCausalLM
    cfg = build_config()
    m = Qwen3ForCausalLM(cfg)
    n = sum(p.numel() for p in m.parameters())
    emb = m.get_input_embeddings().weight.numel()
    print(f"params: {n/1e6:.2f}M (embedding {emb/1e6:.2f}M, tied)")
    print(f"layers {cfg.num_hidden_layers}, hidden {cfg.hidden_size}, heads {cfg.num_attention_heads}"
          f"/{cfg.num_key_value_heads}, head_dim {cfg.head_dim}, ffn {cfg.intermediate_size}")
    print("attn impl:", cfg._attn_implementation)
