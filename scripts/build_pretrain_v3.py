"""Expand the pretraining corpus for the 1.5B-token run:
   existing conversation text + all zh-wiki shards + firefly instruction text.
Then tokenize (parallel) into packed sequences.
"""
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np
import pandas as pd

ROOT = Path(r"D:\wt\feng-distill-30m")
DATA = ROOT / "data"
V2 = ROOT / "v2"
TOK = V2 / "tokenizer" / "tokenizer.json"
L = 2049
_tk = None


def _init():
    global _tk
    from tokenizers import Tokenizer
    _tk = Tokenizer.from_file(str(TOK))


def enc(text):
    global _tk
    if _tk is None:
        _init()
    return np.asarray(_tk.encode(text).ids, dtype=np.uint16)


def chunks(text, size=3_000_000):
    for i in range(0, len(text), size):
        yield text[i:i + size]


def main():
    t0 = time.time()
    parts = []
    # 1) existing conversation text (as used by the 200M-token run)
    p = DATA / "pretrain_text.txt"
    if p.exists():
        t = p.read_text(encoding="utf-8", errors="replace")
        parts.append(t)
        print(f"conversation text: {len(t)/1e6:.1f}M chars")

    # 2) all available zh-wiki shards
    wiki_chars = 0
    for f in sorted((DATA / "wiki" / "20231101.zh").glob("*.parquet")):
        df = pd.read_parquet(f)
        txt = "\n".join((x or "").strip() for x in df["text"] if x and len(x) > 300)
        wiki_chars += len(txt)
        parts.append(txt)
        print(f"  {f.name}: {len(txt)/1e6:.1f}M chars")
    print(f"wiki total: {wiki_chars/1e6:.1f}M chars")

    # 3) firefly instruction text (input + target), sampled for balance
    ff_chars = 0
    ff = DATA / "firefly" / "firefly-train-1.1M.jsonl"
    if ff.exists():
        buf, n = [], 0
        with ff.open(encoding="utf-8", errors="replace") as fh:
            for line in fh:
                if not line.strip():
                    continue
                try:
                    d = json.loads(line)
                except Exception:
                    continue
                u = (d.get("input") or "").strip()
                a = (d.get("target") or "").strip()
                if not u or not a:
                    continue
                buf.append(f"<|im_start|>user\n{u}<|im_end|>\n<|im_start|>assistant\n{a}<|im_end|>\n")
                n += 1
                if n >= 700000:
                    break
        t = "".join(buf)
        ff_chars = len(t)
        parts.append(t)
        print(f"firefly: {n} samples, {ff_chars/1e6:.1f}M chars")

    text = "\n".join(parts)
    out_txt = DATA / "pretrain_text_v3.txt"
    out_txt.write_text(text, encoding="utf-8")
    total_chars = len(text)
    print(f"total: {total_chars/1e6:.1f}M chars -> {out_txt} ({time.time()-t0:.0f}s)")

    # 4) parallel tokenize + pack
    t0 = time.time()
    with Pool(16, initializer=_init) as pool:
        pieces = list(pool.imap(enc, chunks(text), chunksize=1))
    ids = np.concatenate(pieces)
    del pieces
    print(f"tokens: {len(ids)/1e6:.1f}M in {time.time()-t0:.0f}s")
    n = len(ids) // L
    arr = ids[:n * L].reshape(n, L)
    np.save(V2 / "pretrain_ids_v3.npy", arr)
    (V2 / "corpus_v3_stats.json").write_text(json.dumps(
        {"chars": total_chars, "tokens": int(n * L), "sequences": int(n), "seq_len": L},
        indent=2), encoding="utf-8")
    print(f"packed: {n} sequences x {L} = {n * L / 1e9:.3f}B tokens -> v2/pretrain_ids_v3.npy")


if __name__ == "__main__":
    main()
