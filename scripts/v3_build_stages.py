"""Build the progressive long-context packs for v3.
Stages are cut from the same 569M-token stream the v2 model was pretrained on, so
the only thing that changes is the sequence length.  Windows are spread evenly over
the whole stream (stride + jitter) so every stage sees varied content, and the token
budget shrinks as the context grows (that is the point of a progressive schedule:
long stages are much more expensive per token, so they get fewer tokens).
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import ROOT  # noqa: E402
STREAM = ROOT / "v2" / "pretrain_ids_v3.npy"
OUT = ROOT / "archive" / "v3" / "data"

# (name, seq len, token budget) -- budget halves-ish as the context doubles
STAGES = [
    ("ctx4096", 4096, 40_000_000),
    ("ctx8192", 8192, 20_000_000),
    ("ctx16384", 16384, 8_000_000),
    ("ctx32768", 32768, 4_000_000),
]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    packed = np.load(STREAM, mmap_mode="r")
    total = int(packed.size)
    flat = packed.reshape(-1)
    print(f"source stream: {total/1e6:.1f}M tokens ({packed.shape[0]} x {packed.shape[1]})")
    meta = {"source": str(STREAM), "source_tokens": total, "stages": []}
    for name, L, budget in STAGES:
        n = budget // L
        arr = np.empty((n, L + 1), dtype=np.uint16)
        stride = total // n
        jitter = np.random.default_rng(1234).integers(0, max(1, stride - (L + 2)), size=n)
        for i in range(n):
            start = i * stride + int(jitter[i])
            arr[i] = flat[start:start + L + 1]
        path = OUT / f"{name}.npy"
        np.save(path, arr)
        meta["stages"].append({"name": name, "seq": L, "sequences": int(n),
                               "tokens": int(n * L), "path": str(path)})
        print(f"{name}: {n} seqs x {L} = {n*L/1e6:.1f}M tokens -> {path.name} "
              f"({path.stat().st_size/1048576:.0f} MB)")
    (OUT / "stages.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    tot = sum(s["tokens"] for s in meta["stages"])
    print(f"total {tot/1e6:.1f}M tokens across {len(STAGES)} stages")


if __name__ == "__main__":
    main()
