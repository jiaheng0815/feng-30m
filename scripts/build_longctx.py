"""Build long-context training data: Wikipedia docs + concatenated chats + needle QA."""
import argparse
import json
import random
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np
import pandas as pd

ROOT = Path(r"D:\wt\feng-distill-30m")
DATA = ROOT / "data"
STU = ROOT / "student"
SEED = 20261010


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-tokens", type=int, default=9_000_000)
    ap.add_argument("--len", type=int, default=32768, help="sequence length (tokens)")
    ap.add_argument("--wiki-docs", type=int, default=4000)
    ap.add_argument("--needle-frac", type=float, default=0.06)
    ap.add_argument("--wiki-frac", type=float, default=0.60)
    ap.add_argument("--chat-frac", type=float, default=0.34)
    args = ap.parse_args()
    rng = random.Random(SEED)

    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(STU / "tokenizer" / "tokenizer.json"))
    eot = tok.token_to_id("<|endoftext|>")
    im_s = tok.token_to_id("<|im_start|>")
    im_e = tok.token_to_id("<|im_end|>")

    L = args.len
    out_rows = []
    n_model, n_needle = 0, 0

    def pack(text_ids, kind, mask=None):
        nonlocal n_model
        if len(text_ids) < L:
            return
        start = rng.randrange(0, len(text_ids) - L + 1)
        seq = np.asarray(text_ids[start:start + L], dtype=np.uint16)
        m = np.ones(L, dtype=np.uint8) if mask is None else np.asarray(mask[start:start + L], dtype=np.uint8)
        out_rows.append((seq, m))
        n_model += 1

    # ---- 1) Wikipedia zh long documents (raw LM, supervise everything)
    wiki_dir = DATA / "wiki" / "20231101.zh"
    docs = []
    for p in sorted(wiki_dir.glob("*.parquet")):
        df = pd.read_parquet(p)
        for t in df["text"]:
            t = (t or "").strip()
            if len(t) < 400:
                continue
            docs.append(t)
            if len(docs) >= args.wiki_docs:
                break
        if len(docs) >= args.wiki_docs:
            break
    rng.shuffle(docs)
    buf, buf_len = [], 0
    target_wiki = int(args.target_tokens * args.wiki_frac)
    wiki_ids = 0
    for t in docs:
        ids = tok.encode(t).ids + [eot]
        buf += ids
        buf_len += len(ids)
        while buf_len >= L * 2:
            pack(buf[:L * 2], "wiki")
            del buf[:L]
            buf_len -= L
            wiki_ids += L
            if wiki_ids >= target_wiki:
                break
        if wiki_ids >= target_wiki:
            break
    print(f"wiki: {n_model} sequences")

    # ---- 2) concatenated chat conversations (keeps chat behaviour at long ctx)
    target_chat = int(args.target_tokens * args.chat_frac)
    chat_ids = 0
    stream, stream_mask = [], []
    for name in ("aux_sft.jsonl", "teacher_distill.jsonl"):
        p = DATA / name
        if not p.exists():
            continue
        lines = p.read_text(encoding="utf-8").splitlines()
        rng.shuffle(lines)
        for line in lines:
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
            if not msgs:
                continue
            for m in msgs:
                if m["role"] == "user":
                    piece = f"<|im_start|>user\n{m['content']}<|im_end|>\n"
                    t = tok.encode(piece).ids
                    stream += t; stream_mask += [0] * len(t)
                elif m["role"] == "assistant":
                    piece = f"<|im_start|>assistant\n{m['content']}<|im_end|>\n"
                    t = tok.encode(piece).ids
                    stream += t; stream_mask += [1] * len(t)
            stream.append(eot); stream_mask.append(1)
            while len(stream) >= L * 2:
                pack(stream[:L * 2], "chat", stream_mask[:L * 2])
                del stream[:L]; del stream_mask[:L]
                chat_ids += L
            if chat_ids >= target_chat:
                break
        if chat_ids >= target_chat:
            break
    print(f"after chat: {n_model} sequences")

    # ---- 3) needle-in-haystack QA (teaches long-range retrieval)
    target_needle = int(args.target_tokens * args.needle_frac)
    hay = docs if docs else ["。".join(str(i) for i in range(2000))]
    needle_seq = 0
    facts = ["密码是 {c}", "钥匙编号是 {c}", "暗号是 {c}", "解锁码是 {c}", "编号是 {c}"]
    while needle_seq * L < target_needle:
        fact = rng.choice(facts).format(c=f"{rng.randint(1000, 99999)}")
        q = "上文提到的" + fact.split("是")[0] + "是什么？请只回答数字。"
        # build haystack from random wiki text
        ids = []
        while len(ids) < L - 200:
            d = rng.choice(hay)
            ids += tok.encode(d[:600]).ids + [eot]
        ids = ids[:L - 200]
        pos = rng.randrange(int(0.05 * L), int(0.9 * L))
        fact_ids = tok.encode(f"（重要信息：{fact}。）").ids
        ids = ids[:pos] + fact_ids + ids[pos:]
        ids = ids[:L - 120]
        tail = f"<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n"
        q_ids = tok.encode(tail).ids
        seq = ids + q_ids
        mask = [0] * len(seq)
        ans = tok.encode(fact.split("是")[1].strip("。") + "。<|im_end|>\n").ids
        seq = (seq + ans)[:L]
        mask = (mask + [1] * len(ans))[:L]
        if len(seq) < L:
            pad = L - len(seq)
            seq = seq + [eot] * pad
            mask = mask + [0] * pad
        out_rows.append((np.asarray(seq, dtype=np.uint16), np.asarray(mask, dtype=np.uint8)))
        needle_seq += 1
    print(f"needle: {needle_seq} sequences")

    rng.shuffle(out_rows)
    tot = len(out_rows)
    ids_arr = np.stack([r[0] for r in out_rows])
    mask_arr = np.stack([r[1] for r in out_rows])
    n_val = max(1, int(tot * 0.02))
    np.save(STU / "long_ids.npy", ids_arr[n_val:])
    np.save(STU / "long_mask.npy", mask_arr[n_val:])
    np.save(STU / "long_val_ids.npy", ids_arr[:n_val])
    np.save(STU / "long_val_mask.npy", mask_arr[:n_val])
    print(json.dumps({"sequences": tot, "len": L, "tokens": tot * L,
                      "wiki": n_model - needle_seq - (n_model - needle_seq), "needle": needle_seq,
                      "val": n_val}, ensure_ascii=False))
    print(f"saved -> {STU}/long_*.npy")


if __name__ == "__main__":
    main()
