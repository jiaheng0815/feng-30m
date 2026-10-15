"""把 v4.0 的教师输出合并成训练用的 SFT 数据集。

输入（都来自 `data/`）：
  - `v4_teacher.jsonl`   教师生成：单轮是 `{prompt, response}`，多轮是 `{turns:[{user,assistant}]}`
  - `v4_identity.jsonl`  身份池（**手写答案，不经教师**，已经是 messages 格式）

输出：`data/v4_sft.jsonl`，统一成 `{"messages": [{"role","content"}, ...]}`，
与仓库既有的训练脚本（`v3_6_sft_patch.py` 的 `load_convs`）口径一致。

为什么要**二次质检**：生成时已经过一遍闸门（`v4_teacher_gen.quality_ok`），
但这里是训练前最后一道，值得再拦一次——尤其是身份泄漏（提到 Qwen/通义/厂商），
这类样本一旦进训练集，模型就会学会自称别的模型。代价很低（纯 CPU 正则）。

用法：
    python scripts\\v4_build_sft.py                        # 合到 data\\v4_sft.jsonl
    python scripts\\v4_build_sft.py --val-ratio 0.02       # 再切一份验证集
"""
import argparse
import hashlib
import json
import random
import sys
from collections import Counter
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).resolve().parent))

from paths import ROOT  # noqa: E402
from v4_teacher_gen import quality_ok, RE_IDENTITY  # noqa: E402
import chat_probe_heldout as H1  # noqa: E402
import chat_probe_heldout2 as H2  # noqa: E402

DATA = ROOT / "data"
# 身份句锚定：官方口径，出现偏差就说明样本被污染了（AGENTS.md 硬性要求）
IDENTITY_ANCHOR = "我是 feng"


def norm(s: str) -> str:
    return " ".join((s or "").split())


def held_prompts() -> set[str]:
    out = set()
    for mod in (H1, H2):
        for row in mod.CASES:
            if len(row) > 1 and row[1]:
                out.add(norm(row[1]))
    return out


def h(s: str) -> str:
    return hashlib.md5(s.encode()).hexdigest()[:16]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", default=str(DATA / "v4_teacher.jsonl"))
    ap.add_argument("--identity", default=str(DATA / "v4_identity.jsonl"))
    ap.add_argument("--out", default=str(DATA / "v4_sft.jsonl"))
    ap.add_argument("--val-out", default="", help="非空则按 --val-ratio 切验证集")
    ap.add_argument("--val-ratio", type=float, default=0.0)
    ap.add_argument("--min-chars", type=int, default=2)
    ap.add_argument("--max-chars", type=int, default=600)
    ap.add_argument("--seed", type=int, default=20261015)
    a = ap.parse_args()

    held = held_prompts()
    seen: set[str] = set()
    rows: list[dict] = []
    drop = Counter()
    src = Counter()

    def keep(prompt: str, answer: str, source: str) -> bool:
        """返回 True 表示这条可以进训练集。"""
        p, ans = norm(prompt), (answer or "").strip()
        if norm(p) in held:
            drop["留出题污染"] += 1
            return False
        if not (a.min_chars <= len(ans) <= a.max_chars):
            drop["长度越界"] += 1
            return False
        # 二次质检：生成时过了，这里再拦一次（重点是身份泄漏）
        ok, why = quality_ok(ans, p, "stop")
        if not ok:
            drop[f"质检:{why}"] += 1
            return False
        key = h(p + "\x00" + ans)
        if key in seen:
            drop["重复"] += 1
            return False
        seen.add(key)
        src[source] += 1
        return True

    # ---- 教师数据 ----
    n_t, n_mt = 0, 0
    tp = Path(a.teacher)
    if not tp.exists():
        print(f"!! 找不到 {tp}")
    else:
        for line in tp.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except Exception:
                drop["解析失败"] += 1
                continue
            s = r.get("source", "?")
            if r.get("kind") == "multiturn":
                turns = r.get("turns") or []
                if not turns:
                    drop["空多轮"] += 1
                    continue
                # 逐轮质检：任何一轮不合格就整段丢弃（训练时整段是一个样本）
                msgs, bad = [], False
                for t in turns:
                    u, ans = t.get("user", ""), t.get("assistant", "")
                    if not keep(u, ans, s):
                        bad = True
                        break
                    msgs += [{"role": "user", "content": norm(u)},
                             {"role": "assistant", "content": ans.strip()}]
                if bad or len(msgs) < 4:
                    continue
                rows.append({"messages": msgs, "source": s, "kind": "multiturn"})
                n_mt += 1
            else:
                p, ans = r.get("prompt", ""), r.get("response", "")
                if not p:
                    drop["空提示"] += 1
                    continue
                if not keep(p, ans, s):
                    continue
                rows.append({"messages": [{"role": "user", "content": norm(p)},
                                          {"role": "assistant", "content": ans.strip()}],
                             "source": s, "kind": "single"})
                n_t += 1

    # ---- 身份池（手写答案，逐条校验锚定句）----
    n_id = 0
    ip = Path(a.identity)
    if ip.exists():
        for line in ip.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            msgs = r.get("messages") or []
            asst = " ".join(m["content"] for m in msgs if m["role"] == "assistant")
            if not asst:
                continue
            # 唯一要拦的是**答案里出现别家模型/厂商**。AGENTS.md 明令过滤这类样本：
            # 对一个 29M 模型来说，把"通义千问""OpenAI"这些词写进训练数据，风险大于
            # "教会它否认"的收益（v3.19 的身份串名就是这么来的）。
            #
            # 不要再去校验"答案里必须出现『我是 feng』"——池子里大量正确说法是
            # 「我是个人开发者 jiaheng 独立开发训练的 feng」「feng 是 jiaheng 给我起的
            # 名字」「不是微调，是 jiaheng 从零训练的我，我叫 feng」，逐字锚定会误伤
            # 三分之一（实测 197/600）。否定行为另有 DENY 那 4 条干净答案兜着。
            if RE_IDENTITY.search(asst):
                drop["答案提到别家模型"] += 1
                continue
            rows.append({"messages": msgs, "source": "identity", "kind": "identity"})
            n_id += 1

    rng = random.Random(a.seed)
    rng.shuffle(rows)

    val = []
    if a.val_out and a.val_ratio > 0:
        k = max(1, int(len(rows) * a.val_ratio))
        val, rows = rows[:k], rows[k:]

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps({"messages": r["messages"]}, ensure_ascii=False) + "\n")
    if val:
        vp = Path(a.val_out)
        with vp.open("w", encoding="utf-8") as f:
            for r in val:
                f.write(json.dumps({"messages": r["messages"]}, ensure_ascii=False) + "\n")

    # ---- 统计 ----
    n_turns = sum(len(r["messages"]) // 2 for r in rows)
    chars = [sum(len(m["content"]) for m in r["messages"] if m["role"] == "assistant")
             for r in rows]
    chars.sort()
    print(f"\n合并完成 -> {out}")
    print(f"  单轮 {n_t} | 多轮 {n_mt} 段 | 身份 {n_id} 条  =>  {len(rows)} 条样本（{n_turns} 轮对话）")
    if val:
        print(f"  验证集 {len(val)} 条 -> {a.val_out}")
    if chars:
        print(f"  回答长度 中位 {chars[len(chars)//2]} / 均值 {sum(chars)//len(chars)} 字符")
    print("\n  来源分布:")
    for s, c in src.most_common():
        print(f"    {s:20s} {c}")
    print(f"\n  剔除 {sum(drop.values())} 条:")
    for k, c in drop.most_common():
        print(f"    {k:20s} {c}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
