"""v3.1 检索数据：在 v3 的基础上专门提高"上下文里的**精确性**"。

对照 32 题复测发现的三个弱点做针对性改进（v3 的旧数据只有单一 5 位数字、
单一问法、无干扰项、位置均匀）：

1. **答案更要精确**：v3 的失败大多是"漏一位数字"（44702→4702），所以答案类型扩到
   5/6/8 位数字、字母数字编号、姓名+日期，并混合多种问法（4~8 种改写），
   逼模型逐字拷贝而不是猜个大概。
2. **加干扰项**：正文里额外埋 1~3 条同类型但不同值的事实，杜绝"全文只有一个数字"的捷径。
3. **位置覆盖**：30% 强制埋在前 10%、30% 强制埋在后 10%，其余均匀——
   复测显示开头 20% 位置命中率最低（75%）。
4. **填充文本换成真实语料**：从 v2 预训练流（569M token）里随机抽窗口，替掉 v3 那 16 句固定
   循环文本，分布更接近真实长文。

另外 15% 的样本埋 2~3 条事实，问题只问其中一条（多针定位）。
loss 仍然只算答案部分（与 v3 一致）。
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

STREAM = ROOT / "v2" / "pretrain_ids_v3.npy"      # v2 预训练 token 流（真实中文语料）
OUT = ROOT / "v3_1" / "data"

# 长度 → 样本数（v3.1b：短长度加权，修 4k/8k 回退）
SPECS = [(4096, 2800), (8192, 1600), (16384, 700), (32768, 220)]

# 事实类型：(模板, 问题模板列表, 取值生成器名)
KINDS = [
    ("数字5", "（重要信息：保险柜密码是 {c}。）",
     ["上文的保险柜密码是什么？请只回答数字。",
      "请找出文中提到的保险柜密码，只回答数字。",
      "文中保险柜的密码是多少？直接给数字。",
      "根据上文，保险柜密码是几号？只回答数字。"], "digits5"),
    ("数字6", "（备注：图书借阅证号 {c}。）",
     ["上文的图书借阅证号是多少？只回答数字。",
      "请从文中找出图书借阅证号，只回答数字。",
      "文中提到的借阅证号是几号？直接给数字。"], "digits6"),
    ("数字8", "（登记信息：设备序列号 {c}。）",
     ["上文的设备序列号是什么？只回答数字。",
      "请找出文中设备序列号，只回答数字。",
      "文中设备的序列号是多少？直接给数字。"], "digits8"),
    ("编号", "（重要信息：工单编号是 {c}。）",
     ["上文的工单编号是什么？",
      "请从文中找出工单编号。",
      "文中提到的工单编号是多少？"], "alnum"),
    ("姓名日期", "（记录：值班人是 {c}。）",
     ["上文的值班人是谁？请连日期一起回答。",
      "请找出文中记录的值班人和日期。",
      "文中提到的值班人是谁、哪天值班？"], "namedate"),
]
# 近似干扰项：同位数、只改一位数字，但**换了对象名称**（所以问题没有歧义）
NEAR_MISS = {
    "digits5": ("（登记：快递柜取件码 {c}。）", "digits5"),
    "digits6": ("（备注：会议室预订号 {c}。）", "digits6"),
    "digits8": ("（记录：快递单号 {c}。）", "digits8"),
}


def make_value(rng: random.Random, kind: str) -> str:
    if kind == "digits5":
        return f"{rng.randint(10000, 99999)}"
    if kind == "digits6":
        return f"{rng.randint(100000, 999999)}"
    if kind == "digits8":
        return f"{rng.randint(10000000, 99999999)}"
    if kind == "alnum":
        letters = "ABCDEFGHJKLMNPQRSTUVWXY"          # 去掉易混的 I/O/Z
        return (f"{rng.choice(letters)}{rng.randint(10,99)}-"
                f"{rng.choice(letters)}{rng.choice(letters)}{rng.randint(100,999)}")
    surname = "赵钱孙李周吴郑王冯陈褚卫蒋沈韩杨朱秦许何吕施张孔曹严华金魏陶姜"
    return (f"{rng.choice(surname)}{rng.choice('明华强丽芳娟军磊洋勇艳杰涛超霞平刚')} "
            f"{rng.randint(2026, 2030)}年{rng.randint(1,12)}月{rng.randint(1,28)}日")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--seed", type=int, default=20261004)
    ap.add_argument("--specs", default="",
                    help="覆盖默认规格，如 4096:3200,8192:1800,16384:800,32768:260")
    ap.add_argument("--near-miss-frac", type=float, default=0.30,
                    help="干扰项里有多大比例是'同位数、只差一位'的近似值（练数字精度）")
    ap.add_argument("--negative-frac", type=float, default=0.08,
                    help="多大比例的样本是'文中没有该信息'，正确答案是说明没有提到")
    ap.add_argument("--negative-fracs", default="",
                    help="按长度分别指定负样本比例（如 0.10,0.15,0.20,0.30），优先级高于 --negative-frac")
    ap.add_argument("--repeat-frac", type=float, default=0.0,
                    help="多大比例的正样本把目标事实**重复两遍**（练逐位精确拷贝）")
    ap.add_argument("--decoys", type=int, default=0,
                    help="正样本固定埋 N 条其它类别的事实（0 = 随机 1~3 条）")
    ap.add_argument("--kind-weights", default="",
                    help="5 类事实的抽样权重（如 2,3,4,1,1 表示数字类过采样）")
    args = ap.parse_args()
    specs = SPECS
    if args.specs:
        specs = [(int(a), int(b)) for a, b in (s.split(":") for s in args.specs.split(","))]
    neg_fracs = ([float(x) for x in args.negative_fracs.split(",")]
                 if args.negative_fracs else [])
    kind_w = ([float(x) for x in args.kind_weights.split(",")]
              if args.kind_weights else None)

    from tokenizers import Tokenizer
    tk = Tokenizer.from_file(str(ROOT / "v2" / "tokenizer" / "tokenizer.json"))
    enc = lambda s: tk.encode(s).ids                                  # noqa: E731

    stream = np.load(STREAM, mmap_mode="r")
    flat = stream.reshape(-1)
    total = int(flat.size)
    print(f"填充文本来源：{STREAM.name}，{total/1e6:.1f}M tokens")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    meta = []
    for li, (L, n) in enumerate(specs):
        neg_frac = neg_fracs[li] if neg_fracs else args.negative_frac
        rng = random.Random(args.seed + L)
        ids = np.zeros((n, L), dtype=np.uint16)
        mask = np.zeros((n, L), dtype=np.uint8)
        for i in range(n):
            kind, tpl, q_tpls, gen = rng.choices(KINDS, weights=kind_w, k=1)[0]

            # 「文中没有这条信息」的负样本：问题照问，正确答案是说明没有提到
            if rng.random() < neg_frac:
                q = enc("<|im_start|>user\n" + rng.choice(q_tpls) +
                        "<|im_end|>\n<|im_start|>assistant\n")
                a = enc("文中没有提到。<|im_end|>")
                others = [k for k in KINDS if k[0] != kind]
                # **同位数诱饵**：文中一定存在一个和所问同类位数、但对象不同的数字，
                # 逼模型学会"所问的那条不存在 → 说明没有提到"，而不是抓最近似的数字
                if gen in NEAR_MISS:
                    t2, g2 = NEAR_MISS[gen]
                    distract = [enc(t2.format(c=make_value(rng, g2)))]
                else:
                    distract = []
                while len(distract) < rng.randint(3, 5):
                    k2, t2, _, g2 = KINDS[rng.randrange(len(KINDS))]
                    distract.append(enc(t2.format(c=make_value(rng, g2))))
                body_len = L - len(q) - len(a) - sum(map(len, distract))
                start = rng.randrange(0, max(1, total - body_len - 1))
                row = flat[start:start + body_len].astype(np.uint16).tolist()
                for d in distract:
                    p = rng.randrange(0, max(1, len(row)))
                    row = row[:p] + d + row[p:]
                row = (row + q + a)[:L]
                if len(row) < L:
                    row = row + [0] * (L - len(row))
                ids[i] = row
                mask[i, -len(a):] = 1
                continue

            code = make_value(rng, gen)
            fact = enc(tpl.format(c=code))
            q = enc("<|im_start|>user\n" + rng.choice(q_tpls) +
                    "<|im_end|>\n<|im_start|>assistant\n")
            a = enc(code + "<|im_end|>")

            # 干扰项：1~3 条**其它类型**的事实（杜绝"全文唯一数字"的捷径，
            # 同时避免 v3.1 犯过的错——给同一类事实再塞一条，会让问题变得有歧义）
            distract = []
            n_dec = args.decoys or rng.randint(1, 3)
            for _ in range(n_dec):
                k2, t2, _, g2 = KINDS[rng.randrange(len(KINDS))]
                while k2 == kind:                       # 必须换一个类别
                    k2, t2, _, g2 = KINDS[rng.randrange(len(KINDS))]
                if gen in NEAR_MISS and rng.random() < args.near_miss_frac:
                    # 近似干扰项：**同位数、只改一位**，但对象名称不同（无歧义）
                    # —— 直接练"逐位精确拷贝"，针对复测里剩下的"差一位数字"错误
                    t2, g2 = NEAR_MISS[gen]
                    v2 = code
                    pos = rng.randrange(len(v2))
                    d = rng.choice([c for c in "0123456789" if c != v2[pos]])
                    v2 = v2[:pos] + d + v2[pos + 1:]
                else:
                    v2 = make_value(rng, g2)
                distract.append(enc(t2.format(c=v2)))     # 模板本身已写明是别的东西

            # 15% 的样本额外埋 1~2 条其它类型的事实（多针）
            extra = []
            if rng.random() < 0.15:
                for _ in range(rng.randint(1, 2)):
                    k2, t2, _, g2 = KINDS[rng.randrange(len(KINDS))]
                    extra.append(enc(t2.format(c=make_value(rng, g2))))

            # 冗余：把**同一条事实**在文中再放一次（真实文档里信息常有重复，
            # 给模型第二次读准的机会，专门压"差一位数字"）
            repeat = [list(fact)] if rng.random() < args.repeat_frac else []

            body_len = (L - len(q) - len(a) - len(fact)
                        - sum(map(len, distract + extra + repeat)))
            if body_len < 256:
                raise SystemExit(f"sequence too short for L={L}")

            # 真实语料窗口当填充
            start = rng.randrange(0, max(1, total - body_len - 1))
            body = flat[start:start + body_len].astype(np.uint16).tolist()

            # 位置：30% 前 10%、30% 后 10%、40% 均匀（复测显示边缘位置最弱）
            r = rng.random()
            if r < 0.30:
                pos = rng.randrange(int(0.02 * body_len), max(int(0.02 * body_len) + 1,
                                                              int(0.10 * body_len)))
            elif r < 0.60:
                pos = rng.randrange(int(0.90 * body_len), max(int(0.90 * body_len) + 1,
                                                              int(0.98 * body_len)))
            else:
                pos = rng.randrange(int(0.05 * body_len), int(0.95 * body_len))

            row = body[:pos] + fact + body[pos:]
            for d in distract:                      # 干扰项撒在正文其它位置
                p = rng.randrange(0, max(1, len(row)))
                row = row[:p] + d + row[p:]
            for e in extra + repeat:
                p = rng.randrange(0, max(1, len(row)))
                row = row[:p] + e + row[p:]
            row = row + q + a
            row = row[:L]
            if len(row) < L:
                row = row + [0] * (L - len(row))
            ids[i] = row
            mask[i, -len(a):] = 1
        np.save(out / f"retr{L}_ids.npy", ids)
        np.save(out / f"retr{L}_mask.npy", mask)
        meta.append({"seq": L, "samples": n, "tokens": int(ids.size),
                     "kinds": [k[0] for k in KINDS]})
        print(f"retr{L}: {n} 样本 x {L} = {ids.size/1e6:.2f}M tokens "
              f"（答案 token {int(mask.sum()/1000)}k）")
    (out / "retrieval_stats.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"合计 {sum(m['tokens'] for m in meta)/1e6:.1f}M tokens -> {out}")


if __name__ == "__main__":
    main()
