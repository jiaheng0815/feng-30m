"""长文针检索：每个长度 32 题，GPU 批处理并行跑 v1/v2/v3。

协议与 `eval_longctx.py` **完全一致**（同填充文本、同"保险柜密码"埋点、同提问、
贪婪解码 16 token、同样按 `[-ctx:]` 截断），所以结果与旧的 3 题版本可直接比较；
区别只有两点：
  1. 每个长度跑 `--n` 题（默认 32），而不是 3 题；
  2. 同一个 batch 的题一起送进 GPU（左 padding + attention mask），而不是逐题串行。

三个模型使用**同一套随机种子**（按 长度+题号 生成），因此看到的埋点位置/密码完全相同，
可以逐题配对比较。
"""
import argparse
import json
import random
import re
import sys
import time
from pathlib import Path

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import ROOT  # noqa: E402

FILLER = ("在遥远的山谷里，风穿过松林。村里的老人说，时间像河水一样一去不回。"
          "孩子们在田野上奔跑，数着天上的云。")
QUESTION = ("<|im_start|>user\n上文的保险柜密码是什么？请只回答数字。<|im_end|>\n"
            "<|im_start|>assistant\n")
PAD_ID = 3
EOS_ID = 0

# 三个要对比的模型（与文档里的口径一致）
MODELS = {
    "v1": "student/feng-30m-32k",
    "v2": "v2/stage_planA3b/final",
    "v3": "v3/retr_sft/ctx32768/final",
}


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def build_sample(tok, ctx: int, seed: int) -> dict:
    """按旧协议造一题：随机密码 + 随机位置的埋点 + 截断到 ctx。"""
    rng = random.Random(seed)
    code = f"{rng.randint(10000, 99999)}"
    fact = f"（重要信息：保险柜密码是 {code}。）"
    body, cur, need = [], 0, max(0, ctx - 400)
    while cur < need:
        body.append(FILLER * 3)
        cur += len(FILLER) * 3
    text = "".join(body)
    pos = rng.randrange(int(0.05 * len(text)), int(0.9 * len(text)))
    text = text[:pos] + fact + text[pos:]
    ids = tok(text + QUESTION, add_special_tokens=False)["input_ids"][-ctx:]
    return {"code": code, "ids": ids, "needle_frac": round(pos / len(text), 3)}


OTHER_FACTS = [                       # 正文里有别的信息，但没有被问的那条
    "（登记：快递柜取件码 {c}。）",
    "（备注：会议室预订号 {c}。）",
    "（记录：快递单号 {c}。）",
    "（登记信息：设备序列号 {c}。）",
]
REFUSAL_HINTS = ("没有提到", "未提到", "文中没有", "没有找到", "未找到", "无法找到", "没有相关")

# 多类别评测：与训练数据同分布（5 类事实 × 各自问法），避免只测"保险柜密码"一类
KINDS = [
    ("保险柜密码", "（重要信息：保险柜密码是 {c}。）",
     "上文的保险柜密码是什么？请只回答数字。", "digits5"),
    ("借阅证号", "（备注：图书借阅证号 {c}。）",
     "上文的图书借阅证号是多少？只回答数字。", "digits6"),
    ("设备序列号", "（登记信息：设备序列号 {c}。）",
     "上文的设备序列号是什么？只回答数字。", "digits8"),
    ("工单编号", "（重要信息：工单编号是 {c}。）",
     "上文的工单编号是什么？", "alnum"),
    ("值班人", "（记录：值班人是 {c}。）",
     "上文的值班人是谁？请连日期一起回答。", "namedate"),
]


def _make_value(rng: random.Random, kind: str) -> str:
    if kind == "digits5":
        return f"{rng.randint(10000, 99999)}"
    if kind == "digits6":
        return f"{rng.randint(100000, 999999)}"
    if kind == "digits8":
        return f"{rng.randint(10000000, 99999999)}"
    if kind == "alnum":
        letters = "ABCDEFGHJKLMNPQRSTUVWXY"
        return (f"{rng.choice(letters)}{rng.randint(10,99)}-"
                f"{rng.choice(letters)}{rng.choice(letters)}{rng.randint(100,999)}")
    surname = "赵钱孙李周吴郑王冯陈褚卫蒋沈韩杨朱秦许何吕施张孔曹严华金魏陶姜"
    return (f"{rng.choice(surname)}{rng.choice('明华强丽芳娟军磊洋勇艳杰涛超霞平刚')} "
            f"{rng.randint(2026, 2030)}年{rng.randint(1,12)}月{rng.randint(1,28)}日")


def build_multi(tok, ctx: int, seed: int, negative: bool) -> dict:
    """多类别版：随机选一类事实（问法也随之变化）；negative=True 时该类事实不存在。"""
    rng = random.Random(seed)
    name, tpl, q_txt, gen = KINDS[rng.randrange(len(KINDS))]
    body, cur, need = [], 0, max(0, ctx - 400)
    while cur < need:
        body.append(FILLER * 3)
        cur += len(FILLER) * 3
    text = "".join(body)
    code = None if negative else _make_value(rng, gen)
    if not negative:                     # 正样本：埋目标事实
        fact = tpl.format(c=code)
        p = rng.randrange(int(0.05 * len(text)), int(0.9 * len(text)))
        text = text[:p] + fact + text[p:]
    for k in KINDS:                      # 正负样本都撒别的类别的事实当诱饵
        if k[0] == name:
            continue
        fact = k[1].format(c=_make_value(rng, k[3]))
        p = rng.randrange(0, max(1, len(text)))
        text = text[:p] + fact + text[p:]
    q = f"<|im_start|>user\n{q_txt}<|im_end|>\n<|im_start|>assistant\n"
    ids = tok(text + q, add_special_tokens=False)["input_ids"][-ctx:]
    return {"code": code, "ids": ids, "needle_frac": -1.0 if negative else 0.5,
            "kind": name}


def build_negative(tok, ctx: int, seed: int) -> dict:
    """负样本：正文里**没有**保险柜密码，正确行为是说明"没有提到"，而不是编一个数字。"""
    rng = random.Random(seed)
    body, cur, need = [], 0, max(0, ctx - 400)
    while cur < need:
        body.append(FILLER * 3)
        cur += len(FILLER) * 3
    text = "".join(body)
    for _ in range(rng.randint(1, 3)):            # 埋一些**别的**信息当诱饵
        fact = rng.choice(OTHER_FACTS).format(c=f"{rng.randint(10000, 99999)}")
        p = rng.randrange(0, max(1, len(text)))
        text = text[:p] + fact + text[p:]
    ids = tok(text + QUESTION, add_special_tokens=False)["input_ids"][-ctx:]
    return {"code": None, "ids": ids, "needle_frac": -1.0}


def equalize(samples: list[dict]) -> int:
    """把一批题裁到相同长度（都从头部裁掉多余的几个 token）。

    这样同一个 batch 不需要 padding / attention mask，注意力就能走 memory-efficient
    内核（本机 PyTorch 没编译 flash 内核，带 mask 时会退化成 O(n²) 直接爆显存）。
    埋点位置在 5%~90% 之间，从头部裁几个 token 不影响协议。
    """
    length = min(len(s["ids"]) for s in samples)
    for s in samples:
        s["ids"] = s["ids"][-length:]
    return length


@torch.no_grad()
def run_batch(model, tok, samples: list[dict], max_new: int = 16) -> list[dict]:
    """一个 batch 里的题一起生成（等长、无 padding、无 mask）。"""
    width = equalize(samples)
    inp = torch.tensor([s["ids"] for s in samples], dtype=torch.long, device="cuda")
    with sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION, SDPBackend.FLASH_ATTENTION]):
        out = model.generate(input_ids=inp, max_new_tokens=max_new, do_sample=False,
                             pad_token_id=PAD_ID, eos_token_id=EOS_ID)
    rows = []
    for i, s in enumerate(samples):
        resp = tok.decode(out[i][width:].tolist()).split("<|im_end|>")[0].strip()
        if s["code"] is None:                     # 负样本：必须说"没有提到"，且不能编数字
            refused = any(h in resp for h in REFUSAL_HINTS)
            invented = re.search(r"\d{4,}", resp) is not None
            ok = bool(refused and not invented)
        else:
            ok = s["code"] in resp
        rows.append({"code": s["code"], "kind": s.get("kind"),
                     "needle_frac": s["needle_frac"],
                     "tokens": len(s["ids"]), "response": resp[:60], "ok": ok})
    return rows


def verify_batch(model, tok, ctx: int, batch: int) -> bool:
    """一致性检查：同一题单独跑 vs 混在 batch 里跑，输出必须完全一致。"""
    samples = [build_sample(tok, ctx, 1000 + ctx + i) for i in range(batch)]
    solo = run_batch(model, tok, samples[:1])
    mixed = run_batch(model, tok, samples)
    same = solo[0]["response"] == mixed[0]["response"] and solo[0]["ok"] == mixed[0]["ok"]
    print(f"    batch 一致性：单独 [{solo[0]['response'][:20]}] vs "
          f"混批 [{mixed[0]['response'][:20]}] -> {'一致' if same else '不一致！'}")
    return same


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="v1,v2,v3", help="逗号分隔：v1/v2/v3 或任意 HF 目录")
    ap.add_argument("--ctx", default="4096,8192,16384,32768")
    ap.add_argument("--n", type=int, default=32, help="每个长度的题数")
    ap.add_argument("--batch", type=int, default=8, help="每批并行题数")
    ap.add_argument("--out-dir", default=str(ROOT / "eval"))
    ap.add_argument("--neg-n", type=int, default=0,
                    help="每个长度额外跑 N 道负样本（文中没有该信息）")
    ap.add_argument("--multi-kind", action="store_true",
                    help="多类别复测：问题随机取自 5 类事实（与训练数据同分布）")
    ap.add_argument("--verify-batch", action="store_true", help="先验证批处理与单题一致")
    args = ap.parse_args()

    from transformers import AutoTokenizer, Qwen3ForCausalLM

    ctxs = [int(c) for c in args.ctx.split(",")]
    names = [m.strip() for m in args.models.split(",")]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, dict[int, str]] = {}

    for spec in names:
        # 支持 "别名=路径" 写法：结果文件名用别名，避免覆盖历史同名结果
        name, _, explicit = spec.partition("=")
        path = Path(explicit) if explicit else Path(MODELS.get(name, name))
        if not (path / "model.safetensors").exists():
            log(f"跳过 {name}：{path} 没有 model.safetensors")
            continue
        log(f"=== {name} | {path} ===")
        tok = AutoTokenizer.from_pretrained(path, padding_side="left")
        model = Qwen3ForCausalLM.from_pretrained(
            path, dtype=torch.bfloat16, attn_implementation="sdpa").to("cuda").eval()

        if args.verify_batch:
            verify_batch(model, tok, ctxs[0], min(args.batch, 4))

        rows, t0 = [], time.time()
        for ctx in ctxs:
            if args.multi_kind:
                samples = [build_multi(tok, ctx, 1000 + ctx + i, False)
                           for i in range(args.n)]
                negs = [build_multi(tok, ctx, 900000 + ctx + i, True)
                        for i in range(args.neg_n)]
            else:
                samples = [build_sample(tok, ctx, 1000 + ctx + i) for i in range(args.n)]
                negs = [build_negative(tok, ctx, 900000 + ctx + i) for i in range(args.neg_n)]
            detail = []
            for b0 in range(0, len(samples), args.batch):
                detail += run_batch(model, tok, samples[b0:b0 + args.batch])
            neg_rows = []
            for b0 in range(0, len(negs), args.batch):
                neg_rows += run_batch(model, tok, negs[b0:b0 + args.batch])
            hit = sum(1 for d in detail if d["ok"])
            neg_ok = sum(1 for d in neg_rows if d["ok"])
            rows.append({"ctx": ctx, "hit": hit, "n": len(detail), "detail": detail,
                         "neg_hit": neg_ok, "neg_n": len(neg_rows), "neg_detail": neg_rows})
            neg_txt = f"| 负样本 {neg_ok}/{len(neg_rows)} 正确拒答  " if neg_rows else ""
            tok_info = (detail[0]["tokens"] if detail else
                        (neg_rows[0]["tokens"] if neg_rows else 0))
            log(f"  ctx={ctx:6d}  {hit}/{len(detail)} 命中  {neg_txt}"
                f"（token {tok_info}，耗时 {time.time() - t0:.0f}s）")

        safe = re.sub(r"[^0-9A-Za-z_.-]+", "_", name)     # 传路径时也能当文件名
        # 纯负样本 / 多类别口径分别存，避免互相覆盖
        prefix = ("longctx32neg" if args.n == 0 else
                  ("longctx32multi" if args.multi_kind else "longctx32"))
        out = out_dir / f"{prefix}_{safe}.json"
        out.write_text(json.dumps({"model": str(path), "n": args.n, "rows": rows},
                                  ensure_ascii=False, indent=2), encoding="utf-8")
        log(f"  -> {out}")
        summary[name] = {r["ctx"]: f"{r['hit']}/{r['n']}"
                         + (f" | 负{r['neg_hit']}/{r['neg_n']}" if r["neg_n"] else "")
                         for r in rows}
        del model
        torch.cuda.empty_cache()

    if summary:
        print("\n=== 汇总（每格 = 命中/题数）===")
        header = "模型  " + "".join(f"{c:>12}" for c in ctxs)
        print(header)
        for name, r in summary.items():
            print(f"{name:<6}" + "".join(f"{r.get(c, '-'):>12}" for c in ctxs))


if __name__ == "__main__":
    main()
