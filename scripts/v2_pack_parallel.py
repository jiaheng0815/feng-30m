"""Parallel packing: pretrain text + SFT conversations (multiprocessing)."""
import argparse
import json
import os
import sys
from multiprocessing import Pool
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import ROOT  # noqa: E402
V2 = ROOT / "v2"
DATA = V2 / "data"
TOK = V2 / "tokenizer" / "tokenizer.json"
_tk = None


def _init():
    global _tk
    from tokenizers import Tokenizer
    _tk = Tokenizer.from_file(str(TOK))


def enc_text(lines):
    """lines: list[str] -> np.uint16 ids"""
    global _tk
    if _tk is None:
        _init()
    ids = _tk.encode("".join(lines)).ids
    return np.asarray(ids, dtype=np.uint16)


def enc_conv(rec):
    global _tk
    if _tk is None:
        _init()
    ids_i, mask_i = [], []
    for m in rec["messages"]:
        if m["role"] == "user":
            t = _tk.encode(f"<|im_start|>user\n{m['content']}<|im_end|>\n").ids
            ids_i += t; mask_i += [0] * len(t)
        elif m["role"] == "assistant":
            t = _tk.encode(f"<|im_start|>assistant\n{m['content']}<|im_end|>\n").ids
            ids_i += t; mask_i += [1] * len(t)
        else:
            t = _tk.encode(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n").ids
            ids_i += t; mask_i += [0] * len(t)
    if not ids_i or sum(mask_i) == 0 or len(ids_i) > 2048:
        return None
    ids_i.append(_tk.token_to_id("<|endoftext|>")); mask_i.append(1)
    return (np.asarray(ids_i, dtype=np.uint16), np.asarray(mask_i, dtype=np.uint8))


def split_lines(path, n_parts):
    with open(path, encoding="utf-8", errors="replace") as f:
        buf, size, part = [], 0, 0
        for line in f:
            buf.append(line)
            size += len(line)
            if size >= 2_000_000:               # ~2M chars per batch
                yield (part, buf)
                part += 1
                buf, size = [], 0
        if buf:
            yield (part, buf)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--convs", default="sft_convs.jsonl", help="SFT conversations file under v2/data")
    ap.add_argument("--tag", default="", help="suffix for the packed arrays")
    args = ap.parse_args()
    L = 2049
    tag = args.tag

    # ---------- pretrain
    batches = list(split_lines(DATA / "pretrain_text.txt", args.workers))
    print(f"pretrain batches: {len(batches)}", flush=True)
    with Pool(args.workers, initializer=_init) as pool:
        parts = []
        for k, arr in enumerate(pool.imap(enc_text, (b for _, b in batches), chunksize=1)):
            parts.append(arr)
            if (k + 1) % 20 == 0:
                print(f"  {k+1}/{len(batches)} batches, {sum(len(p) for p in parts)/1e6:.0f}M tokens",
                      flush=True)
    ids = np.concatenate(parts)
    del parts
    print(f"pretrain tokens: {len(ids)/1e6:.1f}M", flush=True)
    n = len(ids) // L
    np.save(V2 / "pretrain_ids.npy", ids[:n * L].reshape(n, L))
    del ids
    print(f"pretrain sequences: {n}", flush=True)

    # ---------- SFT
    recs = []
    with (DATA / args.convs).open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                recs.append(json.loads(line))
    print(f"SFT conversations to tokenize: {len(recs)}", flush=True)
    cid, cmask, off = [], [], [0]
    with Pool(args.workers, initializer=_init) as pool:
        for k, r in enumerate(pool.imap(enc_conv, recs, chunksize=64)):
            if r is None:
                continue
            cid.extend(r[0]); cmask.extend(r[1]); off.append(len(cid))
            if (k + 1) % 50000 == 0:
                print(f"  {k+1}/{len(recs)} convs, {len(cid)/1e6:.1f}M tokens", flush=True)
    np.save(V2 / f"sft_ids{tag}.npy", np.asarray(cid, dtype=np.uint16))
    np.save(V2 / f"sft_mask{tag}.npy", np.asarray(cmask, dtype=np.uint8))
    np.save(V2 / f"sft_offsets{tag}.npy", np.asarray(off, dtype=np.int64))
    print(f"SFT: {len(off)-1} convs, {len(cid)/1e6:.1f}M tokens, "
          f"supervised {sum(cmask)/1e6:.1f}M", flush=True)
    (V2 / "pack_stats.json").write_text(json.dumps(
        {"pretrain_sequences": int(n), "pretrain_tokens": int(n * L),
         "sft_conversations": len(off) - 1, "sft_tokens": len(cid)}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
