"""把长文针检索样本导出成 **预 tokenized** 的 id 文件，给手机端做 32k 验收。

手机端 `phone_chat` 的普通输入走 `char prompt[4096]` + O(n²) BPE，没法一次喂 32k
文档；`--ids-file` 直接吃这里导出的完整 chat template token 序列，绕过这两个限制。
协议与 `eval_longctx_many.py` 一致：token 精确、预留 32 个生成位、埋点 10%~90%。

用法：
    python scripts/make_longctx_ids.py --ctx 4096,8192,16384,32768 --n 8 --neg-n 2
"""
import argparse
import json
import random
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import ROOT  # noqa: E402

FILLER = ("在遥远的山谷里，风穿过松林。村里的老人说，时间像河水一样一去不回。"
          "孩子们在田野上奔跑，数着天上的云。")
QUESTION = ("<|im_start|>user\n上文的保险柜密码是什么？请只回答数字。<|im_end|>\n"
            "<|im_start|>assistant\n")
DECOYS = [
    "（登记：快递柜取件码 {c}。）",
    "（备注：会议室预订号 {c}。）",
    "（记录：快递单号 {c}。）",
    "（登记信息：设备序列号 {c}。）",
]
REFUSAL = "文中没有提到。"
GEN_RESERVE = 32


def body_text(tok, target_tokens: int) -> str:
    unit = FILLER * 3
    ut = max(1, len(tok(unit, add_special_tokens=False)["input_ids"]))
    reps = max(1, (max(0, target_tokens) + 256 + ut - 1) // ut)
    return unit * reps


def build(tok, ctx: int, seed: int, negative: bool) -> dict:
    rng = random.Random(seed)
    keep = max(1, ctx - GEN_RESERVE)
    text = body_text(tok, keep + 256)
    code = None
    if negative:
        for _ in range(rng.randint(1, 3)):
            fact = rng.choice(DECOYS).format(c=f"{rng.randint(10000, 99999)}")
            p = rng.randrange(0, max(1, len(text)))
            text = text[:p] + fact + text[p:]
    else:
        code = f"{rng.randint(10000, 99999)}"
        fact = f"（重要信息：保险柜密码是 {code}。）"
        p = rng.randrange(int(0.1 * len(text)), int(0.9 * len(text)))
        text = text[:p] + fact + text[p:]
    ids = tok(text + QUESTION, add_special_tokens=False)["input_ids"][-keep:]
    return {"code": code, "ids": ids, "negative": negative}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(ROOT / "v4" / "base"),
                    help="tokenizer 所在目录（默认 v4/base）")
    ap.add_argument("--out", default=str(ROOT / "eval" / "longctx_ids"))
    ap.add_argument("--ctx", default="4096,8192,16384,32768")
    ap.add_argument("--n", type=int, default=8, help="每个长度的正样本数")
    ap.add_argument("--neg-n", type=int, default=2, help="每个长度的负样本数")
    ap.add_argument("--seed", type=int, default=20261018)
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    meta = []
    for ctx in [int(x) for x in args.ctx.split(",")]:
        for i in range(args.n + args.neg_n):
            negative = i >= args.n
            s = build(tok, ctx, args.seed + ctx + i, negative)
            name = f"ctx{ctx}_{i:03d}.ids"
            (out / name).write_text(" ".join(str(t) for t in s["ids"]), encoding="ascii")
            meta.append({"file": name, "ctx": ctx, "tokens": len(s["ids"]),
                         "code": s["code"], "negative": negative})
        print(f"ctx={ctx}: {args.n}+{args.neg_n} 条，实际 {len(s['ids'])} tokens/条")
    (out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1),
                                   encoding="utf-8")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
