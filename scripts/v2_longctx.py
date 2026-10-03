"""Build the v2 native-32k corpus: wiki docs + long chat + needle retrieval (parallel encode)."""
import json
import random
import sys
from multiprocessing import Pool
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import ROOT  # noqa: E402
V2 = ROOT / "v2"
DATA = ROOT / "data"
TOK = V2 / "tokenizer" / "tokenizer.json"
L = 32768
SEED = 20261015
_tk = None


def _init():
    global _tk
    from tokenizers import Tokenizer
    _tk = Tokenizer.from_file(str(TOK))


def enc(lines):
    global _tk
    if _tk is None:
        _init()
    return np.asarray(_tk.encode("".join(lines)).ids, dtype=np.uint16)


def main():
    rng = random.Random(SEED)
    docs = []
    for p in sorted((DATA / "wiki" / "20231101.zh").glob("*.parquet")):
        df = pd.read_parquet(p)
        for t in df["text"]:
            t = (t or "").strip()
            if len(t) >= 400:
                docs.append(t)
    rng.shuffle(docs)
    print("wiki docs:", len(docs))

    # wiki text -> one long stream
    wiki_text = "\n".join(docs)
    conv_stream = []
    with (V2 / "data" / "sft_convs.jsonl").open(encoding="utf-8") as f:
        lines = f.readlines()
    rng.shuffle(lines)
    for line in lines[:250000]:
        rec = json.loads(line)
        conv_stream.append("".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n"
                                   for m in rec["messages"]))
    conv_text = "\n".join(conv_stream)
    print(f"wiki chars {len(wiki_text)/1e6:.1f}M | chat chars {len(conv_text)/1e6:.1f}M")

    # parallel encode both streams in ~4M-char chunks
    def chunks(text, size=4_000_000):
        for i in range(0, len(text), size):
            yield text[i:i + size]

    with Pool(16, initializer=_init) as pool:
        wiki_ids = np.concatenate(list(pool.imap(enc, chunks(wiki_text), chunksize=1)))
    print(f"wiki tokens {len(wiki_ids)/1e6:.1f}M", flush=True)
    with Pool(16, initializer=_init) as pool:
        chat_ids = np.concatenate(list(pool.imap(enc, chunks(conv_text), chunksize=1)))
    print(f"chat tokens {len(chat_ids)/1e6:.1f}M", flush=True)

    rows = []
    n_wiki = int(len(wiki_ids) * 0.25 / L)     # ~25% wiki
    for i in range(n_wiki):
        s = rng.randrange(0, len(wiki_ids) - L)
        rows.append((wiki_ids[s:s + L], np.ones(L, dtype=np.uint8)))
    n_chat = int(len(chat_ids) * 0.70 / L)     # ~70% chat
    for i in range(n_chat):
        s = rng.randrange(0, len(chat_ids) - L)
        rows.append((chat_ids[s:s + L], np.ones(L, dtype=np.uint8)))
    print(f"wiki seqs {n_wiki}, chat seqs {n_chat}", flush=True)

    # needle retrieval tasks (5%)
    eot = 0
    n_needle = max(8, int((n_wiki + n_chat) * 0.07))
    facts = ["保险柜密码是 {c}", "钥匙编号是 {c}", "暗号是 {c}", "解锁码是 {c}"]
    for _ in range(n_needle):
        fact = rng.choice(facts).format(c=f"{rng.randint(10000, 99999)}")
        code = fact.split("是")[1].strip()
        q = "上文提到的" + fact.split("是")[0] + "是什么？请只回答数字。"
        s = rng.randrange(0, len(wiki_ids) - L)
        body = wiki_ids[s:s + L - 200].copy()
        pos = rng.randrange(1000, len(body) - 1000)
        fact_ids = _enc_one(f"（重要信息：{fact}。）")
        body = np.concatenate([body[:pos], fact_ids, body[pos:]])[:L - 120]
        tail = _enc_one(f"<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n")
        ans = _enc_one(code + "。<|im_end|>\n")
        seq = np.concatenate([body, tail, ans])
        mask = np.zeros(len(seq), dtype=np.uint8)
        mask[len(body) + len(tail):] = 1
        if len(seq) < L:
            pad = L - len(seq)
            seq = np.concatenate([seq, np.full(pad, eot, dtype=np.uint16)])
            mask = np.concatenate([mask, np.zeros(pad, dtype=np.uint8)])
        rows.append((seq[:L], mask[:L]))

    rng.shuffle(rows)
    ids = np.stack([r[0] for r in rows])
    mask = np.stack([r[1] for r in rows])
    n_val = max(2, int(len(rows) * 0.02))
    np.save(V2 / "long_ids.npy", ids[n_val:])
    np.save(V2 / "long_mask.npy", mask[n_val:])
    np.save(V2 / "long_val_ids.npy", ids[:n_val])
    np.save(V2 / "long_val_mask.npy", mask[:n_val])
    print(f"long ctx: {len(rows)} seqs x {L} = {len(rows)*L/1e6:.1f}M tokens "
          f"(needle {n_needle}) -> {V2}")


def _enc_one(text):
    _init() if _tk is None else None
    return np.asarray(_tk.encode(text).ids, dtype=np.uint16)


if __name__ == "__main__":
    main()
