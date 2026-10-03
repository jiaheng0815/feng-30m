"""Pack the Plan A chat corpus into long windows for the v3 long-context SFT.
Conversations are concatenated (assistant-token mask preserved) until the window is
full, so the model keeps seeing the chat format at 8192 tokens instead of being
pulled back to short sequences by a normal short-context SFT.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import ROOT  # noqa: E402
V2 = ROOT / "v2"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="_planA2")
    ap.add_argument("--seq", type=int, default=8192)
    ap.add_argument("--out", default=str(ROOT / "v3" / "data"))
    args = ap.parse_args()

    ids = np.load(V2 / f"sft_ids{args.tag}.npy")
    mask = np.load(V2 / f"sft_mask{args.tag}.npy")
    off = np.load(V2 / f"sft_offsets{args.tag}.npy")
    n_conv = len(off) - 1
    L = args.seq
    windows, wm, cur, cm = [], [], 0, False
    rows_i, rows_m = [], []
    for c in range(n_conv):
        i0, i1 = int(off[c]), int(off[c + 1])
        if i1 - i0 < 8:
            continue
        for k in range(i0, i1):
            rows_i.append(int(ids[k]))
            rows_m.append(int(mask[k]))
            cur += 1
            if cur == L + 1:
                windows.append(rows_i)
                wm.append(rows_m)
                rows_i, rows_m, cur = [], [], 0
    if rows_i:                                  # pad the tail with mask 0
        pad = (L + 1) - len(rows_i)
        rows_i += [0] * pad
        rows_m += [0] * pad
        windows.append(rows_i)
        wm.append(rows_m)
    ia = np.asarray(windows, dtype=np.uint16)
    ma = np.asarray(wm, dtype=np.uint8)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / f"sft{args.seq}_ids.npy", ia)
    np.save(out / f"sft{args.seq}_mask.npy", ma)
    sup = int(ma[:, 1:].sum())
    stats = {"windows": int(ia.shape[0]), "seq": L, "tokens": int(ia.size),
             "supervised_tokens": sup, "from_tag": args.tag, "conversations": n_conv}
    (out / f"sft{args.seq}_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(f"packed {n_conv} conversations -> {ia.shape[0]} windows x {L+1} = "
          f"{ia.size/1e6:.1f}M tokens ({sup/1e6:.1f}M supervised)")


if __name__ == "__main__":
    main()
