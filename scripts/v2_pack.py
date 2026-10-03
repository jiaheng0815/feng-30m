"""Train the 16k Chinese BPE and pack v2 corpora (pretrain stream + SFT conversations)."""
import json
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import ROOT  # noqa: E402
V2 = ROOT / "v2"
DATA = V2 / "data"
VOCAB = 16384
SPECIALS = ["<|endoftext|>", "<|im_start|>", "<|im_end|>", "<|pad|>"]


def main():
    from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
    (V2 / "tokenizer").mkdir(parents=True, exist_ok=True)
    tok_path = V2 / "tokenizer" / "tokenizer.json"
    if not tok_path.exists():
        tk = Tokenizer(models.BPE(unk_token=None))
        tk.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        tk.decoder = decoders.ByteLevel()
        trainer = trainers.BpeTrainer(vocab_size=VOCAB, special_tokens=SPECIALS,
                                      initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
                                      show_progress=True)
        t0 = time.time()
        tk.train([str(DATA / "tok_sample.txt")], trainer=trainer)
        tk.save(str(tok_path))
        print(f"tokenizer {VOCAB} trained in {time.time()-t0:.0f}s")
    tk = Tokenizer.from_file(str(tok_path))
    print("vocab:", tk.get_vocab_size())
    eot = tk.token_to_id("<|endoftext|>")

    # ---------- pretrain stream (packed, all-token loss)
    t0 = time.time()
    chunks = []
    total = 0
    buf = []
    with (DATA / "pretrain_text.txt").open(encoding="utf-8", errors="replace") as f:
        for line in f:
            buf.append(line)
            if sum(len(x) for x in buf) >= 1_000_000:      # ~1M chars per encode call
                ids_c = tk.encode("".join(buf)).ids
                chunks.append(np.asarray(ids_c, dtype=np.uint16))
                total += len(ids_c)
                buf = []
                if total % 20_000_000 < 1_000_000:
                    print(f"  encoded {total/1e6:.1f}M tokens ...", flush=True)
    if buf:
        ids_c = tk.encode("".join(buf)).ids
        chunks.append(np.asarray(ids_c, dtype=np.uint16))
        total += len(ids_c)
    ids = np.concatenate(chunks)
    del chunks
    print(f"pretrain tokens: {len(ids)/1e6:.1f}M in {time.time()-t0:.0f}s")
    L = 2049
    n = len(ids) // L
    arr = np.asarray(ids[:n * L], dtype=np.uint16).reshape(n, L)
    np.save(V2 / "pretrain_ids.npy", arr)
    print(f"pretrain sequences: {n} ({n*L/1e6:.1f}M tokens)")
    del ids, arr

    # ---------- SFT conversations (unpacked)
    cid, cmask, off = [], [], [0]
    n_conv, skipped = 0, 0
    with (DATA / "sft_convs.jsonl").open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            ids_i, mask_i = [], []
            for m in rec["messages"]:
                if m["role"] == "user":
                    t = tk.encode(f"<|im_start|>user\n{m['content']}<|im_end|>\n").ids
                    ids_i += t; mask_i += [0] * len(t)
                elif m["role"] == "assistant":
                    t = tk.encode(f"<|im_start|>assistant\n{m['content']}<|im_end|>\n").ids
                    ids_i += t; mask_i += [1] * len(t)
                else:
                    t = tk.encode(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n").ids
                    ids_i += t; mask_i += [0] * len(t)
            if not ids_i or sum(mask_i) == 0 or len(ids_i) > 2048:
                skipped += 1
                continue
            ids_i.append(eot); mask_i.append(1)
            cid.extend(ids_i); cmask.extend(mask_i); off.append(len(cid))
            n_conv += 1
    np.save(V2 / "sft_ids.npy", np.asarray(cid, dtype=np.uint16))
    np.save(V2 / "sft_mask.npy", np.asarray(cmask, dtype=np.uint8))
    np.save(V2 / "sft_offsets.npy", np.asarray(off, dtype=np.int64))
    print(f"SFT conversations: {n_conv} (skipped {skipped}), tokens {len(cid)/1e6:.1f}M, "
          f"supervised {sum(cmask)/1e6:.1f}M")
    (V2 / "pack_stats.json").write_text(json.dumps(
        {"pretrain_sequences": n, "pretrain_tokens": n * L,
         "sft_conversations": n_conv, "sft_tokens": len(cid), "vocab": tk.get_vocab_size()},
        ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
