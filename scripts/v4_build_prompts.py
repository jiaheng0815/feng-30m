"""feng-30m 4.0 提示词池：为 ~50k 条大规模教师蒸馏构建提示词。

与 v3.x 的单轮定向补丁不同，4.0 要的是**广度优先的优质对话**，所以主体是真实中文
对话语料，定向模板只占很小一份用来兜已知短板。

组成（默认 50,000 条，每条 = 一次教师调用；多轮对话按轮数计）：
  sharegpt_zh 多轮  8,000 组 × ≤3 轮  → 磨多轮（v3.x 最弱的一环）
  sharegpt_zh 单轮 16,000            → 中文对话广度
  evol_zh           6,000            → 指令多样性
  alpaca_zh         4,000            → 任务型指令
  ultrachat_en      1,500            → 英文开放对话
  dolly_en          1,000            → 英文基础能力
  定向扩量          2,500            → v3.19/v3.21 模板扩量，兜已知短板
  身份池              500            → 身份锚定（必须，缺了会串名）

去污染：与两套留出题（chat_probe_heldout.CASES / chat_probe_heldout2.CASES）归一化后
逐字比对，命中即剔除；公开语料之间也做全局去重。

用法：
    python scripts\\v4_build_prompts.py --out data\\v4_prompts.jsonl
    python scripts\\v4_build_prompts.py --scale 0.01 --out data\\v4_prompts_pilot.jsonl
"""
import argparse
import hashlib
import json
import random
import re
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).resolve().parent))

from paths import ROOT  # noqa: E402
from build_prompts import clean, acceptable, is_cjk, iter_json_objects  # noqa: E402
import chat_probe_heldout as H1  # noqa: E402
import chat_probe_heldout2 as H2  # noqa: E402
import v3_19_build_target_prompts as T19  # noqa: E402
import v3_21_build_target_prompts as T21  # noqa: E402
from v3_2_build_identity_mix import build_identity_pool  # noqa: E402

SEED = 20261015
DATA = ROOT / "data"

# 各来源的目标条数（--scale 会等比缩放，用来做小规模试跑）。
#
# 注意：这些是**新增**条数。v3.19 Release 包里已经带了 12,000 条现成蒸馏
# （datasets/teacher_distill.jsonl）+ planA 行为数据 + v3.19 定向，共约 13.8k 条，
# 4.0 直接复用，本脚本只负责生成**没做过的那部分**，并在去重时避开它们
# （省下约 8.4 小时教师时间）。
# 注意：**身份提问不在这里**。身份答案必须来自 v3_2_build_identity_mix.py 里硬编码的
# 手写答案，绝不能交给教师——2026-10-15 实测 v3.19 的定向数据里 7 条身份提问有 4 条
# 被教师答成「我是通义千问，由阿里巴巴集团通义实验室开发」，那批脏数据进了 v3.19 训练。
QUOTA = {
    "sgpt_mt":     10000,  # sharegpt_zh 多轮（组）/ 轮数另算
    "sgpt_single": 20000,
    "evol_zh":     8000,
    "alpaca_zh":   6000,
    "ultrachat_en": 2000,
    "dolly_en":    1200,
    "targeted":    3000,
}

# v3.19 Release 包里已蒸馏过的提示词，避免重复生成
REUSE_DIR = ROOT / "v3_19" / "feng-30m-v3.19" / "datasets"
MT_MAX_TURNS = 3          # 多轮最多取几轮，控制生成成本
ID_TARGETED = 7_000_000   # v4 定向提示词的 id 起点（v3.19=5e6, v3.21=6e6，避开）


def norm(s: str) -> str:
    return " ".join((s or "").split())


# ---------------------------------------------------------------- 输入侧适配闸门
# 公开语料里大量提示词是 30M 中文对话模型**结构上做不到**的：写代码、写长文、
# 专业建议。把它们喂给教师不但白烧调用（质量闸门一定会刷掉代码回答，实测每条
# 浪费 3 次重采），还会让模型去学它永远学不会的能力。这里在提示词阶段就删掉。
RE_TECH = re.compile(
    r"代码|编程|程序|脚本|函数|算法|调试|报错|编译|变量|数组|循环|调试器|"
    r"[Pp]ython|PYTHON|[Jj]ava|[Jj]ava[Ss]cript|C\+\+|C#|SQL|sql|[Rr]egex|正则|"
    r"爬虫|接口|API|api|数据库|服务器|部署|docker|git|Linux|linux|命令行|终端|"
    r"pandas|numpy|DataFrame|lightgbm|sklearn|[Ee]xcel|JSON|json|HTML|CSS|XML|"
    r"前端|后端|框架|依赖|单元测试|重构|并发|线程|数据类型|面向对象|机器学习|深度学习|"
    r"神经网络|模型训练|数据集")
RE_LONG_TASK = re.compile(
    r"写一篇|撰写|写一份|写一首|论文|报告|方案|计划书|策划|大纲|摘要|"
    r"详细(说明|介绍|阐述|解释)|全面|详尽|一步一步|一步步|"
    r"列出所有|所有可能|尽可能多|不少于\d+|至少\d+字")
RE_PRO = re.compile(
    r"法律|律师|合同|诉讼|诊断|处方|用药|药物|症状|治疗|"
    r"投资|理财|股票|基金|证券|税务|保险|房贷|信用卡|体检报告|"
    r"商业计划|市场分析|营销策略|盈利|收益|商业模式")
# 纯算式交给板内 C++ tool（v3.14 起训练数据不再含算术），别让模型学
RE_ARITH = re.compile(r"[\d０-９]+\s*(?:[+\-*/×÷]|乘以|除以|加上|减去|乘|除)\s*[\d０-９]+|"
                      r"计算|求解|方程|几加几|几减几|等于多少|算一下|算算")
# 要求特定输出格式/机械复述的任务：30M 模型学这些只会变成格式杂音
RE_FORMAT = re.compile(r"markdown|Markdown|用表格|分点|条目式|项目符号|编号列出|"
                       r"只需回复|只回复|请回复[“\"]|重复我说的|跟着我说|照我说的念")


def prompt_suitable(t: str) -> tuple[bool, str]:
    """判断这条提示词适不适合 30M 中文对话模型。"""
    if RE_TECH.search(t):
        return False, "tech"
    if RE_LONG_TASK.search(t):
        return False, "long_task"
    if RE_PRO.search(t):
        return False, "pro_domain"
    if RE_ARITH.search(t):
        return False, "arith"
    if RE_FORMAT.search(t):
        return False, "format"
    # 太长的提示词：30M 模型上下文有限，且 prefill 是教师侧的时间大头
    if len(t) > 200:
        return False, "too_long_prompt"
    return True, ""


def load_heldout() -> set[str]:
    """两套留出题的题面，归一化后用于去污染。"""
    out = set()
    for mod in (H1, H2):
        for row in mod.CASES:
            q = row[1] if len(row) > 1 else ""
            if q:
                out.add(norm(q))
    return out


def md5(s: str) -> str:
    return hashlib.md5(s.lower().encode()).hexdigest()[:12]


def load_existing(distill_dir: Path = REUSE_DIR) -> set[str]:
    """已经蒸馏过的提示词（Release 包里那 12k + planA + v3.19 定向）。

    这些数据 4.0 直接复用，不能再花教师时间重生成，所以要挡在提示词池外面。
    """
    out: set[str] = set()
    if not distill_dir.exists():
        print(f"  ! 未找到已蒸馏数据集 {distill_dir}，将不做复用去重（教师耗时会多约 8 小时）")
        return out
    for name, keys in (("teacher_distill.jsonl", ("prompt",)),
                       ("planA_teacher.jsonl", ("prompt",)),
                       ("planA_teacher_v2.jsonl", ("prompt",)),
                       ("planA_teacher_partial.jsonl", ("prompt",)),
                       ("v3_19_teacher.jsonl", ("prompt",))):
        p = distill_dir / name
        if not p.exists():
            continue
        for line in p.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            for k in keys:
                if rec.get(k):
                    out.add(norm(rec[k]))
            # 多轮记录：把每个 user 轮都挡掉
            for t in (rec.get("turns") or []):
                if isinstance(t, dict) and t.get("user"):
                    out.add(norm(t["user"]))
    print(f"已蒸馏提示词（复用，不再生成）: {len(out)} 条")
    return out


class Pool:
    """统一的去重 + 污染过滤容器。

    三层过滤：格式/语言 → 留出题（评测污染，绝不可进训练集）→ 已蒸馏过的提示词
    （不是污染，只是没必要再花教师时间），最后是池内全局去重。
    """

    def __init__(self, held: set[str], existing: set[str]):
        self.held = held
        self.existing = existing
        self.seen: set[str] = set()
        self.rows: list[dict] = []
        self.dropped_held = 0
        self.dropped_existing = 0
        self.dropped_dup = 0
        self.dropped_unsuitable = 0

    def ok(self, text: str, lang: str = "zh", check_existing: bool = True) -> bool:
        t = norm(text)
        if not t or not acceptable(t, 6 if lang == "zh" else 10, 420):
            return False
        if lang == "zh" and not is_cjk(t):
            return False
        if lang == "en" and is_cjk(t):
            return False
        if not prompt_suitable(t)[0]:
            self.dropped_unsuitable += 1
            return False
        if t in self.held:
            self.dropped_held += 1
            return False
        if check_existing and t in self.existing:
            self.dropped_existing += 1
            return False
        h = md5(t)
        if h in self.seen:
            self.dropped_dup += 1
            return False
        self.seen.add(h)
        return True

    def add(self, item: dict) -> bool:
        self.rows.append(item)
        return True


def add_single(pool: Pool, text: str, source: str, lang: str = "zh",
               system: str = "casual", iid: int | None = None) -> bool:
    t = norm(text)
    if not pool.ok(t, lang):
        return False
    pool.add({"id": iid if iid is not None else len(pool.rows),
              "kind": "single", "turns": [t], "system": system, "source": source})
    return True


def add_multiturn(pool: Pool, users: list[str], source: str) -> bool:
    us = [norm(u) for u in users if u and norm(u)]
    us = [u for u in us if 2 <= len(u) <= 420]
    if len(us) < 2:
        return False
    us = us[:MT_MAX_TURNS]
    # 任何一轮落到做不到的类别上，整段对话都不要（否则教师那一轮会被闸门刷 3 次）
    for u in us:
        if not prompt_suitable(u)[0]:
            return False
    # 只对首轮查重（同一段对话的后续轮必然不同，重的是组）。
    # 首轮若撞上「已蒸馏」不算致命（第 2、3 轮仍是新数据），所以放过 existing 检查，
    # 但留出题一律拦死。
    if not pool.ok(us[0], check_existing=False):
        return False
    pool.add({"id": len(pool.rows), "kind": "multiturn", "turns": us,
              "system": "casual", "source": source})
    return True


# ---------------------------------------------------------------- 公开语料

def take_sharegpt(pool: Pool, rng: random.Random, n_single: int, n_mt: int) -> tuple[int, int]:
    """sharegpt_zh：同一份文件同时产出单轮（取首问）与多轮（取整段 user 轮）。"""
    path = DATA / "sgpt" / "sharegpt_zh_38K_format.jsonl"
    if not path.exists():
        print(f"  ! 缺 {path}，跳过 sharegpt_zh")
        return 0, 0
    items = list(iter_json_objects(path))
    rng.shuffle(items)
    n_s, n_m = 0, 0
    for it in items:
        conv = it.get("conversations") or it.get("conversation") or []
        users = [t.get("value") or t.get("content") or ""
                 for t in conv
                 if (t.get("from") or t.get("role") or "").lower() in ("human", "user")]
        if not users:
            continue
        if n_m < n_mt:
            if add_multiturn(pool, users, "sharegpt_zh_mt"):
                n_m += 1
        if n_s < n_single and add_single(pool, users[0], "sharegpt_zh"):
            n_s += 1
        if n_s >= n_single and n_m >= n_mt:
            break
    return n_s, n_m


def take_json_questions(pool: Pool, rng: random.Random, path: Path, source: str,
                        n: int, lang: str) -> int:
    """evol / alpaca 这类「一个问题一个对象」的 json。"""
    if not path.exists():
        print(f"  ! 缺 {path}，跳过 {source}")
        return 0
    try:
        items = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        items = list(iter_json_objects(path))
    if not isinstance(items, list):
        items = list(items)
    rng.shuffle(items)
    got = 0
    for it in items:
        if got >= n:
            break
        q = it.get("instruction") or it.get("question") or it.get("prompt")
        if not q:
            # evol_* 系列没有 instruction 字段，问题藏在 conversations 里
            for turn in (it.get("conversations") or []):
                if (turn.get("from") or turn.get("role") or "").lower() in ("human", "user"):
                    q = turn.get("value") or turn.get("content")
                    break
        if not q:
            continue
        # alpaca 带 input 时拼进问题，否则教师看不到上下文
        inp = (it.get("input") or "").strip()
        if inp and inp not in q:
            q = f"{q}\n{inp}" if len(inp) < 200 else q
        if add_single(pool, q, source, lang):
            got += 1
    return got


def take_dolly(pool: Pool, rng: random.Random, n: int) -> int:
    path = DATA / "dolly" / "databricks-dolly-15k.jsonl"
    if not path.exists():
        print(f"  ! 缺 {path}，跳过 dolly")
        return 0
    items = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    rng.shuffle(items)
    got = 0
    for it in items:
        if got >= n:
            break
        q = (it.get("instruction") or "").strip()
        ctx = (it.get("context") or "").strip()
        if ctx and len(ctx) < 300:
            q = f"{q}\n{ctx}"
        if add_single(pool, q, "dolly_en", "en"):
            got += 1
    return got


def take_ultrachat(pool: Pool, rng: random.Random, n: int) -> int:
    path = DATA / "ultrachat" / "train_sft-00000-of-00003-a3ecf92756993583.parquet"
    if not path.exists():
        print(f"  ! 缺 {path}，跳过 ultrachat")
        return 0
    import pandas as pd
    df = pd.read_parquet(path)
    idx = list(range(len(df)))
    rng.shuffle(idx)
    got = 0
    for i in idx:
        if got >= n:
            break
        msgs = df.iloc[i]["messages"]
        users = [m["content"] for m in msgs if m.get("role") == "user"]
        if users and add_single(pool, users[0], "ultrachat_en", "en"):
            got += 1
    return got


# ---------------------------------------------------------------- 定向扩量

def take_targeted(pool: Pool, rng: random.Random, n: int) -> int:
    """v3.19 + v3.21 的模板素材扩量。

    这两套脚本原本是「一次性定向补丁」，素材是硬编码清单（FACT_TOPICS 110 个、
    KNOWLEDGE 36 条等）。4.0 里把它们当**素材库**，用更多模板组合抽更多变体，
    把 440+555 条扩到几千条，覆盖面拉宽（包括 v3.19 只随机训了 2/6 句式的事实题）。
    """
    cands: list[str] = []
    # 事实题：v3.21 的教训是 v3.19 只训了 2/6 句式 → 这里全句式铺开
    for topic in T19.FACT_TOPICS:
        for tmpl in T19.FACT_TMPL:
            cands.append(tmpl.format(t=topic))
    for cat in T19.LIST_CATS:
        for tmpl in T19.LIST_TMPL:
            cands.append(tmpl.format(c=cat))
    for w in T19.CREATE_WORDS:
        for tmpl in T19.CREATE_TMPL:
            cands.append(tmpl.format(w=w))
    cands += T19.EMO_POS + T19.EMO_NEG + T19.EMO_NEU
    cands += T19.GREET
    cands += T19.SENTENCES
    cands += T19.IDENTITY
    for a, b, c, want in T19.CATEGORY_TRIOS:
        cands.append(f"{a}、{b}和{c}，哪个是{want}？")
        cands.append(f"{a}、{b}和{c}，哪个不是{want}？")
    cands += list(T21.KNOWLEDGE)
    cands += [f"推荐一种{c}。" for c in T21.RECOMMEND_CATS]
    cands += list(T21.RECOMMEND_EXTRA) + list(T21.WISHES)
    cands += T21.HONESTY + T21.PRAISE + T21.UPSET

    rng.shuffle(cands)
    got = 0
    for i, q in enumerate(cands):
        if got >= n:
            break
        if add_single(pool, q, "v4_target", "zh", iid=ID_TARGETED + i):
            got += 1
    return got


def export_identity(out_dir: Path, rng: random.Random, n: int) -> int:
    """身份池**直接导出成最终训练格式**，不经过教师。

    v3.2 的 FAQ/DENY/GENERAL 是手写答案（含锚定句「我是 feng，一个由个人开发者 jiaheng
    独立开发训练的 AI。」），这是我们唯一能保证身份不漂移的来源。教师一被问身份就会
    自报通义千问，所以这条链路必须绕开它。
    """
    convs = build_identity_pool(rng, n)
    rows = []
    for conv in convs:
        msgs = [{"role": r, "content": t} for r, t in conv if t]
        if sum(1 for m in msgs if m["role"] == "assistant") == 0:
            continue
        rows.append({"messages": msgs, "source": "identity"})
    p = out_dir / "v4_identity.jsonl"
    with p.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return len(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(DATA / "v4_prompts.jsonl"))
    ap.add_argument("--scale", type=float, default=1.0, help="配额等比缩放（试跑用 0.01）")
    ap.add_argument("--identity-n", type=int, default=600, help="手写身份池条数（不经教师）")
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    held = load_heldout()
    existing = load_existing()
    pool = Pool(held, existing)
    q = {k: max(1, int(v * args.scale)) for k, v in QUOTA.items()}

    print(f"配额（新增）: {q}")
    print(f"留出题去污染表: {len(held)} 条")

    n_mt = take_sharegpt(pool, rng, q["sgpt_single"], q["sgpt_mt"])
    print(f"  sharegpt_zh 单轮 {n_mt[0]} / 多轮 {n_mt[1]} 组")
    print(f"  evol_zh     {take_json_questions(pool, rng, DATA/'evol'/'evol-instruct-chinese.json', 'evol_zh', q['evol_zh'], 'zh')}")
    print(f"  alpaca_zh   {take_json_questions(pool, rng, DATA/'alpaca_zh'/'alpaca_gpt4_data_zh.json', 'alpaca_zh', q['alpaca_zh'], 'zh')}")
    print(f"  ultrachat   {take_ultrachat(pool, rng, q['ultrachat_en'])}")
    print(f"  dolly       {take_dolly(pool, rng, q['dolly_en'])}")
    print(f"  定向扩量    {take_targeted(pool, rng, q['targeted'])}")
    out_dir = Path(args.out).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    n_id = export_identity(out_dir, rng, args.identity_n)
    print(f"  身份池      {n_id}（手写答案，不经教师）-> {out_dir/'v4_identity.jsonl'}")

    rng.shuffle(pool.rows)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    n_turns = 0
    with out.open("w", encoding="utf-8") as f:
        for i, r in enumerate(pool.rows):
            r["id"] = i          # 重编 id，保证唯一且连续（断点续跑靠它）
            n_turns += len(r["turns"]) if r["kind"] == "multiturn" else 1
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    from collections import Counter
    by_src = Counter(r["source"] for r in pool.rows)
    print(f"\n写出 {len(pool.rows)} 条提示词（约 {n_turns} 次教师调用）-> {out}")
    print(f"剔除：不适配 {pool.dropped_unsuitable} 条（编程/长文/专业领域/算术）、"
          f"留出题 {pool.dropped_held} 条、已蒸馏过 {pool.dropped_existing} 条、"
          f"池内重复 {pool.dropped_dup} 条")
    for s, c in by_src.most_common():
        print(f"   {s:18s} {c}")
    # 按 59.3 tok/s、每轮 ~150 token 估时
    est_h = n_turns * 150 / 59.3 / 3600
    print(f"\n预估教师耗时 ~{est_h:.1f} 小时（59.3 tok/s、每轮 150 token）")
    print("（4.0 数据集全部由这批新生成，不复用 v3.19 的旧蒸馏数据；"
          "身份数据走 v4_identity.jsonl 的手写答案）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
