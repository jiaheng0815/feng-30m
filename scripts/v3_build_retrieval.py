"""Synthesise long-context retrieval training data (what Qwen2.5-1M calls
"long data synthesis"): a long body of varied filler with one planted fact, then a
question about that fact.  Loss is masked to the answer only.

Counts shrink as the window grows (800 / 400 / 200 / 80), matching the progressive
schedule, because long samples cost far more compute per sample.
"""
import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(r"D:\wt\feng-distill-30m")
V3 = ROOT / "v3"
SPECS = [(4096, 800), (8192, 400), (16384, 200), (32768, 80)]

FILLER = [
    "在遥远的山谷里，风穿过松林，村里的老人说时间像河水一样一去不回。",
    "孩子们在田野上奔跑，数着天上的云，直到晚霞把山坡染成金色。",
    "集市上卖菜的妇人把青菜摆得整整齐齐，一边招呼客人一边算账。",
    "河边洗衣的人把布拧干，晾在竹竿上，风吹过时发出轻响。",
    "老木匠把榫头削得刚好，敲进去以后，桌子就再也晃不动了。",
    "夜里下了一场小雨，第二天早上，石阶上还留着浅浅的水痕。",
    "火车经过小站时只停两分钟，站台上的人匆匆递上包袱又匆匆道别。",
    "讲台上老师用粉笔写下公式，粉笔灰落在袖口，学生们抄得沙沙响。",
    "渔夫收网时发现网里缠着几根水草，便把水草一根根挑出来。",
    "山坡上的羊群散开又聚拢，牧羊人吹了声口哨，狗就跑上前去。",
    "茶馆里说书先生一拍醒木，满堂的喧闹忽然就安静下来。",
    "她把手里的毛线绕成球，针尖上下翻飞，围巾一点点长起来。",
    "修车铺门口停着一辆旧自行车，链条上还挂着没干的雨珠。",
    "胡同深处的院墙上长满了爬山虎，夏天的蝉声从墙头一直漫到屋里。",
    "米店里新到的米带着清香，掌柜抓起一把让客人看颗粒。",
    "图书馆的窗边有位老人，翻书的动作很轻，像怕吵醒了谁。",
]
FACT_TPL = [
    "（重要信息：保险柜密码是 {c}。）",
    "（请记住：仓库编号为 {c}。）",
    "（重要信息：取件码是 {c}。）",
    "（备注：会议室预订号 {c}。）",
]
Q_TPL = [
    "上文的保险柜密码是什么？请只回答数字。",
    "上文的仓库编号是什么？请只回答数字。",
    "上文的取件码是什么？请只回答数字。",
    "上文的会议室预订号是什么？请只回答数字。",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(V3 / "data"))
    ap.add_argument("--seed", type=int, default=20261003)
    args = ap.parse_args()
    from tokenizers import Tokenizer
    tk = Tokenizer.from_file(str(ROOT / "v2" / "tokenizer" / "tokenizer.json"))
    enc = lambda s: tk.encode(s).ids          # noqa: E731
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    meta = []
    for L, n in SPECS:
        rng = random.Random(args.seed + L)
        ids = np.zeros((n, L), dtype=np.uint16)
        mask = np.zeros((n, L), dtype=np.uint8)
        filler_pool = [enc(s) for s in FILLER]
        for i in range(n):
            code = f"{rng.randint(10000, 99999)}"
            fact = enc(rng.choice(FACT_TPL).format(c=code))
            q = enc("<|im_start|>user\n" + rng.choice(Q_TPL) +
                    "<|im_end|>\n<|im_start|>assistant\n")
            a = enc(code + "<|im_end|>")
            body_len = L - len(q) - len(a) - len(fact)
            if body_len < 64:
                raise SystemExit(f"sequence too short for L={L}")
            body = []
            while len(body) < body_len:
                body.extend(filler_pool[rng.randrange(len(filler_pool))])
            body = body[:body_len]
            pos = rng.randrange(int(0.1 * body_len), int(0.9 * body_len))
            row = body[:pos] + fact + body[pos:] + q + a
            row = row[:L]
            if len(row) < L:
                row = row + [0] * (L - len(row))
            ids[i] = row
            mask[i, -len(a):] = 1
        np.save(out / f"retr{L}_ids.npy", ids)
        np.save(out / f"retr{L}_mask.npy", mask)
        meta.append({"seq": L, "samples": n, "tokens": int(ids.size)})
        print(f"retr{L}: {n} samples x {L} = {ids.size/1e6:.2f}M tokens "
              f"(answer tokens {int(mask.sum()/1000)}k)")
    (out / "retrieval_stats.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"total {sum(m['tokens'] for m in meta)/1e6:.1f}M tokens")


if __name__ == "__main__":
    main()
