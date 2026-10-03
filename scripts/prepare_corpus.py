"""Train a 32k BPE tokenizer and pack the SFT corpus into fixed-length training sequences."""
import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import DATA_DIR, ROOT  # noqa: E402
DATA = ROOT / "data"
TOK_DIR = ROOT / "student" / "tokenizer"
SEED = 20261008

SPECIALS = ["<|endoftext|>", "<|im_start|>", "<|im_end|>", "<|pad|>"]


def render(messages, tok):
    """Return (ids, loss_mask) with loss only on assistant tokens (including its <|im_end|>)."""
    ids, mask = [], []
    for m in messages:
        if m["role"] == "user":
            piece = f"<|im_start|>user\n{m['content']}<|im_end|>\n"
            ids += tok.encode(piece).ids
            mask += [0] * len(tok.encode(piece).ids)
        elif m["role"] == "assistant":
            piece = f"<|im_start|>assistant\n{m['content']}<|im_end|>\n"
            t = tok.encode(piece).ids
            ids += t
            mask += [1] * len(t)
        elif m["role"] == "system":
            piece = f"<|im_start|>system\n{m['content']}<|im_end|>\n"
            t = tok.encode(piece).ids
            ids += t
            mask += [0] * len(t)
    return ids, mask


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vocab", type=int, default=32768)
    ap.add_argument("--seq-len", type=int, default=2048)
    ap.add_argument("--include-teacher", action="store_true", default=True)
    ap.add_argument("--tag", default="_8k", help="suffix for the packed arrays (e.g. _8k / _32k)")
    ap.add_argument("--unpacked", action="store_true",
                    help="also emit per-conversation arrays (no packing; keeps prompt->reply clean)")
    ap.add_argument("--val-frac", type=float, default=0.005)
    ap.add_argument("--identity-dup", type=int, default=15,
                    help="oversample feng identity conversations (they are <1% of the raw mix)")
    ap.add_argument("--identity-extra", nargs="*",
                    default=[str(DATA_DIR / "train_feng2.jsonl"),
                             str(DATA_DIR / "polish_feng.jsonl")])
    args = ap.parse_args()

    from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders, processors

    TOK_DIR.mkdir(parents=True, exist_ok=True)
    tok_path = TOK_DIR / "tokenizer.json"

    # ---------- gather text for tokenizer training
    rng = random.Random(SEED)
    texts = []
    for name in ("aux_sft.jsonl", "teacher_distill.jsonl"):
        p = DATA / name
        if not p.exists():
            continue
        for line in p.read_text(encoding="utf-8").splitlines():
            if line.strip():
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                msgs = rec.get("messages")
                if not msgs and "prompt" in rec:
                    msgs = [{"role": "user", "content": rec["prompt"]},
                            {"role": "assistant", "content": rec.get("response", "")}]
                if not msgs:
                    continue
                texts.append("\n".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>"
                                       for m in msgs))
    print(f"documents for tokenizer: {len(texts)}")

    if not tok_path.exists():
        tk = Tokenizer(models.BPE(unk_token=None))
        tk.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        tk.decoder = decoders.ByteLevel()
        trainer = trainers.BpeTrainer(vocab_size=args.vocab, special_tokens=SPECIALS,
                                      initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
                                      show_progress=True)
        sample = texts if len(texts) < 400000 else rng.sample(texts, 400000)
        t0 = time.time()
        tk.train_from_iterator(sample, trainer=trainer)
        tk.save(str(tok_path))
        print(f"tokenizer trained ({args.vocab}) in {time.time() - t0:.0f}s -> {tok_path}")
    tk = Tokenizer.from_file(str(tok_path))
    vocab_size = tk.get_vocab_size()
    print("vocab size:", vocab_size)

    # ---------- tokenize + pack
    stream_ids, stream_mask = [], []
    stats = {}
    skipped = 0
    identity_msgs = []
    for p in args.identity_extra:
        p = Path(p)
        if not p.exists():
            continue
        for line in p.read_text(encoding="utf-8").splitlines():
            if line.strip():
                try:
                    identity_msgs.append(json.loads(line)["messages"])
                except Exception:
                    pass
    print(f"identity conversations (extra): {len(identity_msgs)} x{args.identity_dup} dup")
    for name in ("aux_sft.jsonl", "teacher_distill.jsonl"):
        p = DATA / name
        if not p.exists():
            continue
        n = 0
        for line in p.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            msgs = rec.get("messages")
            if not msgs and "prompt" in rec:  # teacher distill format
                msgs = [{"role": "user", "content": rec["prompt"]},
                        {"role": "assistant", "content": rec["response"]}]
            if not msgs:
                continue
            ids, mask = render(msgs, tk)
            if not ids or sum(mask) == 0:
                skipped += 1
                continue
            ids.append(tk.token_to_id("<|endoftext|>"))
            mask.append(1)
            stream_ids += ids
            stream_mask += mask
            n += 1
        stats[name] = n
        print(f"{name}: {n} conversations, stream now {len(stream_ids)/1e6:.2f}M tokens")

    arr_ids = np.asarray(stream_ids, dtype=np.uint16)
    arr_mask = np.asarray(stream_mask, dtype=np.uint8)
    # append oversampled identity conversations
    ident_ids, ident_mask = [], []
    for _ in range(args.identity_dup):
        for msgs in identity_msgs:
            ids_i, mask_i = render(msgs, tk)
            if not ids_i:
                continue
            ids_i.append(tk.token_to_id("<|endoftext|>"))
            mask_i.append(1)
            ident_ids += ids_i
            ident_mask += mask_i
    if ident_ids:
        arr_ids = np.concatenate([arr_ids, np.asarray(ident_ids, dtype=np.uint16)])
        arr_mask = np.concatenate([arr_mask, np.asarray(ident_mask, dtype=np.uint8)])
        stats["identity_oversampled"] = len(ident_ids)
        print(f"identity oversample added {len(ident_ids)/1e6:.2f}M tokens")
    print(f"packed stream: {len(arr_ids)/1e6:.2f}M tokens, supervised "
          f"{arr_mask.sum()/1e6:.2f}M ({100*arr_mask.mean():.1f}%)")

    L = args.seq_len + 1
    n_seq = len(arr_ids) // L
    arr_ids = arr_ids[:n_seq * L].reshape(n_seq, L)
    arr_mask = arr_mask[:n_seq * L].reshape(n_seq, L)
    keep = arr_mask[:, 1:].sum(axis=1) > 0
    arr_ids, arr_mask = arr_ids[keep], arr_mask[keep]
    n_val = max(1, int(len(arr_ids) * args.val_frac))
    perm = np.random.default_rng(SEED).permutation(len(arr_ids))
    val_idx, train_idx = perm[:n_val], perm[n_val:]
    (ROOT / "student").mkdir(exist_ok=True)
    tag = args.tag
    if args.unpacked:
        # rebuild the ragged conversation list from the same sources
        cid, cmask, off = [], [], [0]
        def emit(msgs):
            ids_i, mask_i = render(msgs, tk)
            if not ids_i or sum(mask_i) == 0:
                return
            ids_i.append(tk.token_to_id("<|endoftext|>")); mask_i.append(1)
            cid.extend(ids_i); cmask.extend(mask_i); off.append(len(cid))
        for name in ("aux_sft.jsonl", "teacher_distill.jsonl"):
            p = DATA / name
            if not p.exists():
                continue
            for line in p.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                msgs = rec.get("messages")
                if not msgs and "prompt" in rec:
                    msgs = [{"role": "user", "content": rec["prompt"]},
                            {"role": "assistant", "content": rec.get("response", "")}]
                if msgs:
                    emit(msgs)
        for _ in range(args.identity_dup):
            for msgs in identity_msgs:
                emit(msgs)
        np.save(ROOT / "student" / f"conv_ids{tag}.npy", np.asarray(cid, dtype=np.uint16))
        np.save(ROOT / "student" / f"conv_mask{tag}.npy", np.asarray(cmask, dtype=np.uint8))
        np.save(ROOT / "student" / f"conv_offsets{tag}.npy", np.asarray(off, dtype=np.int64))
        print(f"unpacked conversations: {len(off)-1}, tokens {len(cid)/1e6:.2f}M, "
              f"supervised {sum(cmask)/1e6:.2f}M")
    np.save(ROOT / "student" / f"train_ids{tag}.npy", arr_ids[train_idx])
    np.save(ROOT / "student" / f"train_mask{tag}.npy", arr_mask[train_idx])
    np.save(ROOT / "student" / f"val_ids{tag}.npy", arr_ids[val_idx])
    np.save(ROOT / "student" / f"val_mask{tag}.npy", arr_mask[val_idx])
    print(f"sequences: train {len(train_idx)}, val {len(val_idx)}, each {L} tokens "
          f"-> {ROOT / 'student'}")
    (ROOT / "student" / f"corpus_stats{tag}.json").write_text(json.dumps(
        {"vocab_size": vocab_size, "stream_tokens": int(len(stream_ids)),
         "supervised_tokens": int(arr_mask.sum()), "seq_len": L,
         "train_sequences": int(len(train_idx)), "val_sequences": int(len(val_idx)),
         "sources": stats, "skipped": skipped}, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
