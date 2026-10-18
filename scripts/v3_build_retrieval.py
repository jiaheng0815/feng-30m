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

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import ROOT  # noqa: E402
V3 = ROOT / "archive" / "v3"
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
# 诱饵：正文里**有**这些信息，但被问的那一类才是目标。
# 与 scripts/eval_longctx_many.py 的 OTHER_FACTS 保持一致——**正负样本都要撒**。
# 2026-10-17 实测：之前负样本是"纯填充文本、一条事实都没有"，跟评估端
# "撒 1~3 条诱饵、只缺目标那类"完全不是同一个任务，训了也学不会，拒答只有 4/64。
DECOYS = [
    "（登记：快递柜取件码 {c}。）",
    "（备注：会议室预订号 {c}。）",
    "（记录：快递单号 {c}。）",
    "（登记信息：设备序列号 {c}。）",
]
REFUSAL = "文中没有提到。"      # 必须包含 eval 的 REFUSAL_HINTS 之一


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(V3 / "data"))
    ap.add_argument("--seed", type=int, default=20261003)
    ap.add_argument("--specs", default="4096:800,8192:400,16384:200,32768:80",
                    help="长度:样本数，逗号分隔；板端 2048 可用 --specs 2048:800")
    ap.add_argument("--depth", type=float, default=0.0,
                    help="事实固定埋藏深度（0=随机 10%~90%%；0.25=固定 25%%）")
    # 「文中没有该信息」的负样本比例。**绝不能是 0**：2026-10-17 实测，纯正样本训完
    # 之后模型变成"永远硬抽"，留出题里"文中没有"的拒答从 v3.19 的 61/64 塌到 0/64。
    # 越长越容易编，所以按长度递增（v3.1 的 `--negative-fracs` 也是这个思路）。
    ap.add_argument("--negative-fracs", default="0.20,0.25,0.30,0.40",
                    help="按长度分别指定负样本比例（逗号分隔，对应 SPECS 顺序）")
    ap.add_argument("--negative-frac", type=float, default=0.15,
                    help="--negative-fracs 未给时的统一比例")
    args = ap.parse_args()
    from tokenizers import Tokenizer
    tk = Tokenizer.from_file(str(ROOT / "v4" / "base" / "tokenizer.json"))
    enc = lambda s: tk.encode(s).ids          # noqa: E731
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    neg_fracs = [float(x) for x in args.negative_fracs.split(",") if x.strip()]
    specs = [tuple(int(x) for x in item.split(":")) for item in args.specs.split(",") if item.strip()]
    meta = []
    for li, (L, n) in enumerate(specs):
        neg_frac = neg_fracs[li] if li < len(neg_fracs) else args.negative_frac
        rng = random.Random(args.seed + L)
        ids = np.zeros((n, L), dtype=np.uint16)
        mask = np.zeros((n, L), dtype=np.uint8)
        filler_pool = [enc(s) for s in FILLER]
        n_neg = 0
        for i in range(n):
            is_neg = rng.random() < neg_frac
            if is_neg:
                n_neg += 1
            # 事实类型与问法**必须同一个下标**——以前两句各用一次 rng.choice，
            # 会出现"埋了仓库编号、却问保险柜密码"的错配对样本。
            ki = rng.randrange(len(FACT_TPL))
            code = f"{rng.randint(10000, 99999)}"
            if is_neg:
                fact = []                       # 目标那类不埋
                ans_text = REFUSAL
            else:
                fact = enc(FACT_TPL[ki].format(c=code))
                ans_text = code
            # 诱饵 1~3 条，**正负样本都撒**，与评估端同分布
            decoys = [enc(f.format(c=f"{rng.randint(10000, 99999)}"))
                      for f in rng.sample(DECOYS, rng.randint(1, 3))]
            q = enc("<|im_start|>user\n" + Q_TPL[ki] +
                    "<|im_end|>\n<|im_start|>assistant\n")
            a = enc(ans_text + "<|im_end|>")
            body_len = L - len(q) - len(a) - len(fact) - sum(len(d) for d in decoys)
            if body_len < 64:
                raise SystemExit(f"sequence too short for L={L}")
            body = []
            while len(body) < body_len:
                body.extend(filler_pool[rng.randrange(len(filler_pool))])
            body = body[:body_len]
            for d in decoys:                    # 诱饵随机撒在正文里
                p = rng.randrange(len(body))
                body[p:p] = d
            if not is_neg:                       # 目标事实埋在 10%~90% 深度
                frac = args.depth if args.depth > 0 else rng.uniform(0.1, 0.9)
                pos = int(frac * len(body))
                body[pos:pos] = fact
            row = body + q + a
            row = row[:L]
            if len(row) < L:
                row = row + [0] * (L - len(row))
            ids[i] = row
            mask[i, -len(a):] = 1
        np.save(out / f"retr{L}_ids.npy", ids)
        np.save(out / f"retr{L}_mask.npy", mask)
        meta.append({"seq": L, "samples": n, "tokens": int(ids.size), "neg": n_neg})
        print(f"retr{L}: {n} samples x {L} = {ids.size/1e6:.2f}M tokens "
              f"(负样本 {n_neg}，{100*n_neg/n:.0f}%；answer tokens {int(mask.sum()/1000)}k)")
    (out / "retrieval_stats.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"total {sum(m['tokens'] for m in meta)/1e6:.1f}M tokens")


if __name__ == "__main__":
    main()
