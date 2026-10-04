"""文档事实校验：把全部 markdown 与仓库里的实际产物对齐。

分三层：
  [1] 结构   —— 代码围栏配对、反引号路径存在、无过时表述、无硬编码盘符
  [2] 事实   —— 参数量 / 层数 / 词表 / 上下文 与 config.json 实测一致；
                GGUF 体积与 chat template 内嵌情况与文件实测一致；
                评测数字（范围内 10/10、针检索 3/3 3/3 2/3 2/3）与 eval/*.json 一致
  [3] 一致性 —— 已删除的表述不得回流（如教师型号的旧说法）

用法：python tools/check_docs.py        # 退出码 0 = 全部通过
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[1]
TICK = chr(96)


def _tracked_docs() -> list[Path]:
    """只审 git 跟踪的 markdown（.Codex/ 等运行时目录不算项目文档）。"""
    try:
        out = subprocess.run(["git", "ls-files", "-z", "*.md"], cwd=ROOT,
                             capture_output=True, check=True).stdout
        names = [n.decode("utf-8") for n in out.split(b"\0") if n]
        if names:
            return sorted(ROOT / n for n in names)
    except Exception as exc:
        print(f"[警告] git ls-files 失败（{exc}），退化为全目录扫描")
    return sorted(p for p in ROOT.rglob("*.md")
                  if ".git" not in p.parts and ".Codex" not in p.parts
                  and "__pycache__" not in p.parts)


DOCS = _tracked_docs()

# 只存在于 Release 压缩包里的路径（合法引用，不在仓库中）
RELEASE_PREFIXES = ("weights/", "datasets/", "feng-30m-v3/", "feng-30m-v3.6/")
# 已废弃的表述，不得再出现在文档里
STALE = [
    "0x310000", "0x1A10000", "stage_32k", "29.66", "MQA(1 KV) / head", "3 MB 分区",
    "COM5 ", "you: / feng:", "MAX_CTX=256", "每 +512 需 +10 MB",
    "三值量化", "Qwen 3.8", "不是 git 仓库",
    "R16N32",                                      # 非官方型号写法，正确为 N32R16V
    "只能映射前 16MB flash", "只能映射前 16 MB flash",   # 易被误读成"模块只有 16MB flash"
]
PATH_RE = re.compile(TICK + r"([^" + TICK + r"\n]+)" + TICK)
FILE_RE = re.compile(r"^[\w./\\-]+\.(md|py|c|exe|json|npy|gguf|bin|csv|ps1|txt|safetensors|jinja|example\.json)$")
DRIVE_RE = re.compile(r"[A-Za-z]:\\")

fail: list[str] = []
warn: list[str] = []


def check_structure() -> None:
    print(f"[1] 结构检查（{len(DOCS)} 个 markdown）")
    for doc in DOCS:
        rel = doc.relative_to(ROOT)
        text = doc.read_text(encoding="utf-8")
        if len(re.findall(r"^```", text, re.M)) % 2:
            fail.append(f"{rel}: 代码围栏不配对")
        for s in STALE:
            if s in text:
                fail.append(f"{rel}: 出现过时表述 {s!r}")
        for m in DRIVE_RE.finditer(text):
            line = text[:m.start()].count("\n") + 1
            fail.append(f"{rel}:{line}: 出现硬编码盘符 {m.group(0)}")
        for m in PATH_RE.finditer(text):
            s = m.group(1).strip()
            if not s or " " in s or s.startswith(("$", "<")) or "<" in s or ">" in s:
                continue
            if "/" not in s and "\\" not in s:
                continue                       # 纯文件名，不当作路径
            cand = s.replace("\\", "/")
            if cand.startswith(RELEASE_PREFIXES):
                continue                       # Release 包内路径
            cand = cand.rstrip("/")
            if not (s.endswith("/") or FILE_RE.match(cand)):
                continue
            probes = [ROOT, ROOT / "esp32s3-feng-llm", ROOT.parent, doc.parent]
            if "*" in cand or "?" in cand:
                # 通配写法（如 v3/retr_sft/ctx*/final/）：至少要能匹配到真实路径
                hit = any(len(list(base.glob(cand))) > 0 for base in probes)
            else:
                hit = any((base / cand).exists() for base in probes)
            if not hit:
                fail.append(f"{rel}: 引用了不存在的路径 {s}")


def _params(config: dict) -> int:
    """按 Qwen3 结构算参数量（tied embedding 只计一次）。"""
    h = config["hidden_size"]
    v = config["vocab_size"]
    n_head = config["num_attention_heads"]
    n_kv = config["num_key_value_heads"]
    hd = config["head_dim"]
    ffn = config["intermediate_size"]
    per_layer = (h * n_head * hd + 2 * h * n_kv * hd + n_head * hd * h         # q/k/v/o
                 + 2 * hd + 3 * h * ffn + 2 * h)                                # norms + mlp
    return v * h + config["num_hidden_layers"] * per_layer + h


def check_facts() -> None:
    print("[2] 事实检查（模型规格 / GGUF / 评测数字）")

    # --- 参数量与结构：文档声称 vs config.json ---
    specs = {
        "v3": (ROOT / "v3" / "retr_sft" / "ctx32768" / "final", 29.43, "11 层", 32768),
        "v1": (ROOT / "student" / "feng-30m-chat", 30.75, "8 层", 8192),
    }
    for name, (d, want_m, want_layers, want_ctx) in specs.items():
        cfg_path = d / "config.json"
        if not cfg_path.exists():
            warn.append(f"{name}: {cfg_path.relative_to(ROOT)} 不存在，跳过参数校验")
            continue
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        got_m = _params(cfg) / 1e6
        if abs(got_m - want_m) > 0.05:
            fail.append(f"{name}: 参数量实测 {got_m:.2f}M != 文档声称 {want_m}M")
        if f"{cfg['num_hidden_layers']} 层" != want_layers:
            fail.append(f"{name}: 层数实测 {cfg['num_hidden_layers']} != 文档 {want_layers}")
        if cfg["max_position_embeddings"] != want_ctx:
            fail.append(f"{name}: 上下文实测 {cfg['max_position_embeddings']} != 文档 {want_ctx}")
        print(f"    {name}: {got_m:.2f}M / {cfg['num_hidden_layers']} 层 / "
              f"vocab {cfg['vocab_size']} / ctx {cfg['max_position_embeddings']}")

    # --- GGUF：体积与 chat template 内嵌 ---
    gguf_expect = {"feng-30m-Q4_K_M.gguf": 23.7, "feng-30m-Q8_0.gguf": 30.5,
                   "feng-30m-f16.gguf": 56.8}
    for fname, want_mb in gguf_expect.items():
        f = ROOT / "v3" / "gguf" / fname
        if not f.exists():
            warn.append(f"v3/gguf/{fname} 不存在（本地未导出？），跳过")
            continue
        got_mb = f.stat().st_size / 1024 ** 2
        if abs(got_mb - want_mb) > 0.5:
            fail.append(f"v3/gguf/{fname}: 实测 {got_mb:.1f} MB != 文档 {want_mb} MB")
        blob = f.read_bytes()
        if b"tokenizer.chat_template" not in blob:
            fail.append(f"v3/gguf/{fname}: 未内嵌 chat template（文档声称已内嵌）")
    print(f"    已校验 {len(gguf_expect)} 个 GGUF（体积 + chat template）")

    # --- 评测数字：以 eval/*.json 为唯一真值 ---
    scope = json.loads((ROOT / "eval" / "v3_scope.json").read_text(encoding="utf-8"))
    judged = [r for r in scope if r["ok"] is not None]
    passed = [r for r in judged if r["ok"]]
    scope_txt = f"{len(passed)}/{len(judged)}"
    if scope_txt != "10/10":
        fail.append(f"eval/v3_scope.json: 判分项 {scope_txt}，文档声称 10/10")

    lc = json.loads((ROOT / "eval" / "longctx_v3.json").read_text(encoding="utf-8"))
    needle = "、".join(f"{r['hit']}/{r['n']}" for r in lc["rows"])
    if needle != "3/3、3/3、2/3、2/3":
        fail.append(f"eval/longctx_v3.json: 针检索 {needle}，文档声称 3/3、3/3、2/3、2/3")
    print(f"    范围内 {scope_txt}（判分项）；针检索 {needle}")

    # --- 文档里出现的 v3 针检索数字必须与真值一致 ---
    for doc in DOCS:
        t = doc.read_text(encoding="utf-8")
        for m in re.finditer(r"(\d)/3、(\d)/3、(\d)/3、(\d)/3", t):
            got = "、".join(f"{m.group(i)}/3" for i in range(1, 5))
            if got != "3/3、3/3、2/3、2/3":
                fail.append(f"{doc.relative_to(ROOT)}: 针检索写法 {got} 与真值不符")

    # --- 检索 SFT 的 loss 序列必须与 v3/retr_sft/summary.json 一致 ---
    retr = json.loads((ROOT / "v3" / "retr_sft" / "summary.json").read_text(encoding="utf-8"))
    want_loss = [f"{r['loss']:.2f}" for r in retr]          # ['0.76', '0.43', '0.22', '0.33']

    # --- v3.6：日常对话探针与针检索真值 ---
    probe_path = ROOT / "eval" / "chat_probe_v3_6r.json"
    if probe_path.exists():
        rows = json.loads(probe_path.read_text(encoding="utf-8"))["rows"]
        bad = [r["prompt"] for r in rows
               if r["blurb_leak"] or r["loop"] or r["empty"] or r["topic_miss"] is True]
        if bad:
            fail.append(f"eval/chat_probe_v3_6r.json: {len(bad)} 题未过（{bad[:3]}…），"
                        f"文档声称 42/42")
        else:
            print(f"    v3.6 日常探针 {len(rows)}/{len(rows)}（与文档一致）")
    probe34 = ROOT / "eval" / "chat_probe_v3_4.json"
    if probe34.exists():
        rows = json.loads(probe34.read_text(encoding="utf-8"))["rows"]
        miss = sum(1 for r in rows if r["topic_miss"] is True)
        if len(rows) != 42 or miss != 15:
            fail.append(f"eval/{probe34.name}: 通过 {len(rows)-miss}/42（未命中 {miss}），"
                        f"文档声称 27/42（15 处未命中）")
        else:
            print(f"    v3.4 日常探针 {len(rows)-miss}/{len(rows)}（与文档一致）")
    lc6 = ROOT / "eval" / "longctx32_v3_6r_final_ctx32768_final.json"
    if lc6.exists():
        rows = json.loads(lc6.read_text(encoding="utf-8"))["rows"]
        hits = "/".join(str(r["hit"]) for r in rows)
        if hits != "27/29/24/22":
            fail.append(f"eval/{lc6.name}: 针检索 {hits}，README 声称 27/29/24/22")
        else:
            print(f"    v3.6 针检索 {hits}（与文档一致）")
        neg = sum(r.get("neg_hit", 0) for r in rows)
        if neg != 62:
            fail.append(f"eval/{lc6.name}: 拒答合计 {neg}/64，文档声称 62/64")
    lc6m = ROOT / "eval" / "longctx32multi_v3_6r_final_ctx32768_final.json"
    if lc6m.exists():
        rows = json.loads(lc6m.read_text(encoding="utf-8"))["rows"]
        hits = "/".join(str(r["hit"]) for r in rows)
        neg = sum(r.get("neg_hit", 0) for r in rows)
        if hits != "28/26/28/17" or neg != 63:
            fail.append(f"eval/{lc6m.name}: 多类别 {hits} 拒答 {neg}/64，文档声称 99/128 与 63/64")
        else:
            print(f"    v3.6 多类别针检索 {hits}（合计 {sum(r['hit'] for r in rows)}/128）、"
                  f"拒答 {neg}/64（与文档一致）")
    for fname, want in [("v3_6_scope.json", "8"), ("v3_5_release_scope.json", "7"),
                        ("v3_4_release_scope.json", "9")]:
        p = ROOT / "eval" / fname
        if not p.exists():
            warn.append(f"eval/{fname} 不存在，跳过范围评测校验")
            continue
        rows = json.loads(p.read_text(encoding="utf-8"))
        got = sum(1 for r in rows if r.get("ok") is True)
        if str(got) != want:
            fail.append(f"eval/{fname}: 范围评测 {got}/10 != 文档 {want}/10")
        else:
            print(f"    {fname}: 范围评测 {got}/10（与文档一致）")
    gguf6 = {"feng-30m-Q4_K_M.gguf": 23.7, "feng-30m-Q8_0.gguf": 30.5,
             "feng-30m-f16.gguf": 56.8}
    for fname, want_mb in gguf6.items():
        f = ROOT / "v3_6" / "gguf" / fname
        if not f.exists():
            warn.append(f"v3_6/gguf/{fname} 不存在（本地未导出？），跳过")
            continue
        got_mb = f.stat().st_size / 1024 ** 2
        if abs(got_mb - want_mb) > 0.5:
            fail.append(f"v3_6/gguf/{fname}: 实测 {got_mb:.1f} MB != 文档 {want_mb} MB")
        if b"tokenizer.chat_template" not in f.read_bytes():
            fail.append(f"v3_6/gguf/{fname}: 未内嵌 chat template")
    print("    已校验 v3.6 GGUF（体积 + chat template）")

    # --- v3.7：嵌入式 32 题矩阵 + 范围评测 + 检索口径 ---
    suite = ROOT / "logs" / "pc_kv_suite32_v3_7soupd_q2b8.txt"
    if suite.exists():
        t = suite.read_text(encoding="utf-8", errors="replace")
        if "短任务 27/27" not in t or "长文召回 4/4" not in t:
            fail.append(f"logs/{suite.name}: C 矩阵不是 27/27 + 4/4，文档声称满分")
        else:
            print("    v3.7 嵌入式 32 题矩阵 27/27 + 4/4（与文档一致）")
    lc7 = ROOT / "eval" / "longctx32_v3_7.json"
    if lc7.exists():
        rows = json.loads(lc7.read_text(encoding="utf-8"))["rows"]
        hits = "/".join(str(r["hit"]) for r in rows)
        neg = sum(r.get("neg_hit", 0) for r in rows)
        if hits != "27/30/28/16" or neg != 63:
            fail.append(f"eval/{lc7.name}: {hits} 拒答 {neg}/64，文档声称 27/30/28/16 与 63/64")
        else:
            print(f"    v3.7 针检索 {hits}（拒答 {neg}/64，与文档一致）")
    lc7m = ROOT / "eval" / "longctx32multi_v3_7.json"
    if lc7m.exists():
        rows = json.loads(lc7m.read_text(encoding="utf-8"))["rows"]
        hits = "/".join(str(r["hit"]) for r in rows)
        if hits != "28/26/30/11":
            fail.append(f"eval/{lc7m.name}: 多类别 {hits}，文档声称 28/26/30/11")
        else:
            print(f"    v3.7 多类别针检索 {hits}（与文档一致）")
    p37 = ROOT / "eval" / "v3_7_scope.json"
    if p37.exists():
        rows = json.loads(p37.read_text(encoding="utf-8"))
        got = sum(1 for r in rows if r.get("ok") is True)
        if got != 10:
            fail.append(f"eval/{p37.name}: 范围评测 {got}/10 != 文档 10/10")
        else:
            print("    v3.7 范围评测 10/10（与文档一致）")
    for fname, want_mb in {"feng-30m-Q4_K_M.gguf": 23.7, "feng-30m-Q8_0.gguf": 30.5,
                           "feng-30m-f16.gguf": 56.8}.items():
        f = ROOT / "v3_7" / "gguf" / fname
        if not f.exists():
            warn.append(f"v3_7/gguf/{fname} 不存在，跳过")
            continue
        got_mb = f.stat().st_size / 1024 ** 2
        if abs(got_mb - want_mb) > 0.5 or b"tokenizer.chat_template" not in f.read_bytes():
            fail.append(f"v3_7/gguf/{fname}: 体积/模板与文档不一致")
    print("    已校验 v3.7 GGUF（体积 + chat template）")

    # --- v3.8：上下文专项（单/多类别、拒答、对话），真值来自 eval/*.json ---
    lc8 = ROOT / "eval" / "longctx32_v3_8cr10.json"
    if lc8.exists():
        rows = json.loads(lc8.read_text(encoding="utf-8"))["rows"]
        hits = "/".join(str(r["hit"]) for r in rows)
        neg = sum(r.get("neg_hit", 0) for r in rows)
        if hits != "29/29/23/27" or neg != 61:
            fail.append(f"eval/{lc8.name}: {hits} 拒答 {neg}/64，文档声称 29/29/23/27 与 61/64")
        else:
            print(f"    v3.8 针检索 {hits}（拒答 {neg}/64，与文档一致）")
    lc8m = ROOT / "eval" / "longctx32multi_v3_8cr10.json"
    if lc8m.exists():
        rows = json.loads(lc8m.read_text(encoding="utf-8"))["rows"]
        hits = "/".join(str(r["hit"]) for r in rows)
        neg = sum(r.get("neg_hit", 0) for r in rows)
        if hits != "28/25/32/23" or neg != 62:
            fail.append(f"eval/{lc8m.name}: 多类别 {hits} 拒答 {neg}/64，文档声称 28/25/32/23 与 62/64")
        else:
            print(f"    v3.8 多类别针检索 {hits}（拒答 {neg}/64，与文档一致）")
    probe8 = ROOT / "eval" / "chat_probe_v3_8cr10.json"
    if probe8.exists():
        rows = json.loads(probe8.read_text(encoding="utf-8"))["rows"]
        miss = sum(1 for r in rows if r["topic_miss"] is True)
        if len(rows) != 42 or miss != 0:
            fail.append(f"eval/{probe8.name}: {len(rows)-miss}/42，文档声称 42/42")
        else:
            print("    v3.8 日常探针 42/42（与文档一致）")
    scope8 = ROOT / "eval" / "v3_8_scope.json"
    if scope8.exists():
        rows = json.loads(scope8.read_text(encoding="utf-8"))
        got = sum(1 for r in rows if r.get("ok") is True)
        if got != 8:
            fail.append(f"eval/{scope8.name}: 范围评测 {got}/10 != 文档 8/10")
        else:
            print("    v3.8 范围评测 8/10（与文档一致）")
    for f in (ROOT / "v3_8" / "gguf").glob("*.gguf"):
        if b"tokenizer.chat_template" not in f.read_bytes():
            fail.append(f"v3_8/gguf/{f.name}: 未内嵌 chat template")

    # --- v3.9：末层微调后的范围 10/10 + 上下文保持 ---
    scope9 = ROOT / "eval" / "v3_9_scope_sf2.json"
    if scope9.exists():
        rows = json.loads(scope9.read_text(encoding="utf-8"))
        got = sum(1 for r in rows if r.get("ok") is True)
        if got != 10:
            fail.append(f"eval/{scope9.name}: 范围评测 {got}/10 != 文档 10/10")
        else:
            print("    v3.9 范围评测 10/10（与文档一致）")
    for fname, want_hits, want_neg in [("longctx32_v3_9sf2.json", "29/29/23/27", 61),
                                       ("longctx32multi_v3_9sf2.json", "28/25/32/23", 62)]:
        p = ROOT / "eval" / fname
        if not p.exists():
            warn.append(f"eval/{fname} 不存在，跳过 v3.9 检索校验")
            continue
        rows = json.loads(p.read_text(encoding="utf-8"))["rows"]
        hits = "/".join(str(r["hit"]) for r in rows)
        neg = sum(r.get("neg_hit", 0) for r in rows)
        if hits != want_hits or neg != want_neg:
            fail.append(f"eval/{fname}: {hits} 拒答 {neg}/64，文档声称 {want_hits} 与 {want_neg}/64")
        else:
            print(f"    v3.9 {fname.split('_')[0]} {hits}（拒答 {neg}/64，与文档一致）")
    probe9 = ROOT / "eval" / "chat_probe_v3_9_sf2.json"
    if probe9.exists():
        rows = json.loads(probe9.read_text(encoding="utf-8"))["rows"]
        miss = sum(1 for r in rows if r["topic_miss"] is True)
        if len(rows) != 42 or miss != 0:
            fail.append(f"eval/{probe9.name}: {len(rows)-miss}/42，文档声称 42/42")
        else:
            print("    v3.9 日常探针 42/42（与文档一致）")
    for f in (ROOT / "v3_9" / "gguf").glob("*.gguf"):
        if b"tokenizer.chat_template" not in f.read_bytes():
            fail.append(f"v3_9/gguf/{f.name}: 未内嵌 chat template")

    # --- v3.10：板端版（v3.9 底座 + q2 KV-QAT + 日常回补） ---
    scope10 = ROOT / "eval" / "v3_10p3_scope.json"
    if scope10.exists():
        rows = json.loads(scope10.read_text(encoding="utf-8"))
        got = sum(1 for r in rows if r.get("ok") is True)
        if got != 10:
            fail.append(f"eval/{scope10.name}: 范围评测 {got}/10 != 文档 10/10")
        else:
            print("    v3.10 范围评测 10/10（与文档一致）")
    for fname, want_hits, want_neg in [("longctx32_v3_10p3.json", "28/30/28/21", 62),
                                       ("longctx32multi_v3_10p3.json", "30/28/28/8", 64)]:
        p = ROOT / "eval" / fname
        if not p.exists():
            warn.append(f"eval/{fname} 不存在，跳过 v3.10 检索校验")
            continue
        rows = json.loads(p.read_text(encoding="utf-8"))["rows"]
        hits = "/".join(str(r["hit"]) for r in rows)
        neg = sum(r.get("neg_hit", 0) for r in rows)
        if hits != want_hits or neg != want_neg:
            fail.append(f"eval/{fname}: {hits} 拒答 {neg}/64，文档声称 {want_hits} 与 {want_neg}/64")
        else:
            print(f"    v3.10 {fname.split('_')[0]} {hits}（拒答 {neg}/64，与文档一致）")
    probe10 = ROOT / "eval" / "chat_probe_v3_10p3.json"
    if probe10.exists():
        rows = json.loads(probe10.read_text(encoding="utf-8"))["rows"]
        miss = sum(1 for r in rows if r["topic_miss"] is True)
        loop = sum(1 for r in rows if r["loop"] is True)
        if len(rows) != 42 or miss != 0 or loop != 0:
            fail.append(f"eval/{probe10.name}: {len(rows)-miss}/42（复读 {loop}），文档声称 42/42 且 0 复读")
        else:
            print("    v3.10 日常探针 42/42、0 复读（与文档一致）")
    ident10 = ROOT / "eval" / "identity_v3_10p3.json"
    if ident10.exists():
        d = json.loads(ident10.read_text(encoding="utf-8"))
        if d.get("score") != "12/12":
            fail.append(f"eval/{ident10.name}: 身份 {d.get('score')} != 文档 12/12")
        else:
            print("    v3.10 身份 12/12（与文档一致）")
    for mode, fname in [("q2b8", "pc_kv_suite32_v3_10p3_q2b8.txt"),
                        ("i8", "pc_kv_suite32_v3_10p3_i8.txt")]:
        p = ROOT / "logs" / fname
        if not p.exists():
            warn.append(f"logs/{fname} 不存在，跳过 v3.10 C 矩阵校验")
            continue
        t = p.read_text(encoding="utf-8", errors="replace")
        if "短任务 27/27" not in t or "长文召回 4/4" not in t:
            fail.append(f"logs/{fname}: 不是 27/27 + 4/4，文档声称（{mode}）满分")
        else:
            print(f"    v3.10 C 引擎 {mode} 27/27 + 4/4（与文档一致）")
    for fname in ("board_v3_10p3_multi.txt", "board_v3_10p3_chat10.txt"):
        p = ROOT / "logs" / fname
        if not p.exists():
            warn.append(f"logs/{fname} 不存在，跳过 v3.10 板端校验")
            continue
        t = p.read_text(encoding="utf-8", errors="replace")
        if "10 成功 / 0 失败" not in t:
            fail.append(f"logs/{fname}: 未记录 10/10 成功")
        else:
            print(f"    v3.10 板端 {fname} 10/10（与文档一致）")
    for f in (ROOT / "v3_10" / "gguf").glob("*.gguf"):
        if b"tokenizer.chat_template" not in f.read_bytes():
            fail.append(f"v3_10/gguf/{f.name}: 未内嵌 chat template")

    # --- v3.11：板端当前版（算术边界 + Q4 权重/q2 KV 双 QAT） ---
    arith11 = ROOT / "eval" / "arith_v3_11pol8.json"
    if arith11.exists():
        rows = json.loads(arith11.read_text(encoding="utf-8"))["rows"]
        per = {}
        for r in rows:
            per.setdefault(r["kind"], [0, 0])
            per[r["kind"]][1] += 1
            per[r["kind"]][0] += int(r["ok"])
        want = {"加": (100, 100), "减(结果>0)": (43, 45), "减(结果=0)": (9, 10),
                "减(结果<0)": (44, 45), "乘": (81, 81)}
        got = {k: tuple(v) for k, v in per.items()}
        if got != want:
            fail.append(f"eval/{arith11.name}: 算术网格 {got} != 文档 {want}")
        else:
            print("    v3.11 算术网格 277/281（与文档一致）")
    for fname, want_hits, want_neg in [("longctx32_v3_11p8.json", "29/30/27/20", 62),
                                       ("longctx32multi_v3_11p8.json", "29/27/27/7", 62)]:
        p = ROOT / "eval" / fname
        if not p.exists():
            warn.append(f"eval/{fname} 不存在，跳过 v3.11 检索校验")
            continue
        rows = json.loads(p.read_text(encoding="utf-8"))["rows"]
        hits = "/".join(str(r["hit"]) for r in rows)
        neg = sum(r.get("neg_hit", 0) for r in rows)
        if hits != want_hits or neg != want_neg:
            fail.append(f"eval/{fname}: {hits} 拒答 {neg}/64，文档声称 {want_hits} 与 {want_neg}/64")
        else:
            print(f"    v3.11 {fname.split('_')[0]} {hits}（拒答 {neg}/64，与文档一致）")
    scope11 = ROOT / "eval" / "v3_11p8_scope.json"
    if scope11.exists():
        rows = json.loads(scope11.read_text(encoding="utf-8"))
        got = sum(1 for r in rows if r.get("ok") is True)
        if got != 10:
            fail.append(f"eval/{scope11.name}: 范围评测 {got}/10 != 文档 10/10")
        else:
            print("    v3.11 范围评测 10/10（与文档一致）")
    probe11 = ROOT / "eval" / "chat_probe_v3_11p8.json"
    if probe11.exists():
        rows = json.loads(probe11.read_text(encoding="utf-8"))["rows"]
        miss = sum(1 for r in rows if r["topic_miss"] is True)
        loop = sum(1 for r in rows if r["loop"] is True)
        if len(rows) != 42 or miss != 0 or loop != 0:
            fail.append(f"eval/{probe11.name}: {len(rows)-miss}/42（复读 {loop}），文档声称 42/42 且 0 复读")
        else:
            print("    v3.11 日常探针 42/42、0 复读（与文档一致）")
    ident11 = ROOT / "eval" / "identity_v3_11p8.json"
    if ident11.exists():
        d = json.loads(ident11.read_text(encoding="utf-8"))
        if d.get("score") != "12/12":
            fail.append(f"eval/{ident11.name}: 身份 {d.get('score')} != 文档 12/12")
        else:
            print("    v3.11 身份 12/12（与文档一致）")
    for mode, fname in [("q2b8", "pc_kv_suite32_v3_11p8_q2b8.txt"),
                        ("i8", "pc_kv_suite32_v3_11p8_i8.txt")]:
        p = ROOT / "logs" / fname
        if not p.exists():
            warn.append(f"logs/{fname} 不存在，跳过 v3.11 C 矩阵校验")
            continue
        t = p.read_text(encoding="utf-8", errors="replace")
        if "短任务 27/27" not in t or "长文召回 4/4" not in t:
            fail.append(f"logs/{fname}: 不是 27/27 + 4/4，文档声称（{mode}）满分")
        else:
            print(f"    v3.11 C 引擎 {mode} 27/27 + 4/4（与文档一致）")
    arith_suite = ROOT / "logs" / "pc_arith_suite_v3_11p8_q2b8.txt"
    if arith_suite.exists():
        t = arith_suite.read_text(encoding="utf-8", errors="replace")
        if "短任务 21/21" not in t:
            fail.append("logs/pc_arith_suite_v3_11p8_q2b8.txt: 不是 21/21，文档声称算术子集满分")
        else:
            print("    v3.11 C 引擎算术子集 21/21（与文档一致）")
    else:
        warn.append("logs/pc_arith_suite_v3_11p8_q2b8.txt 不存在，跳过 v3.11 算术子集校验")
    for fname in ("board_v3_11p8_multi.txt", "board_v3_11p8_chat10.txt", "board_v3_11p8_arith.txt"):
        p = ROOT / "logs" / fname
        if not p.exists():
            warn.append(f"logs/{fname} 不存在，跳过 v3.11 板端校验")
            continue
        t = p.read_text(encoding="utf-8", errors="replace")
        if "10 成功 / 0 失败" not in t:
            fail.append(f"logs/{fname}: 未记录 10/10 成功")
        else:
            print(f"    v3.11 板端 {fname} 10/10（与文档一致）")
    for f in (ROOT / "v3_11" / "gguf").glob("*.gguf"):
        if b"tokenizer.chat_template" not in f.read_bytes():
            fail.append(f"v3_11/gguf/{f.name}: 未内嵌 chat template")

    # --- v3.12：PC 当前版（末层算术微调） ---
    arith12 = ROOT / "eval" / "arith_v3_12a2l3.json"
    if arith12.exists():
        rows = json.loads(arith12.read_text(encoding="utf-8"))["rows"]
        per = {}
        for r in rows:
            per.setdefault(r["kind"], [0, 0])
            per[r["kind"]][1] += 1
            per[r["kind"]][0] += int(r["ok"])
        want = {"加": (100, 100), "减(结果>0)": (45, 45), "减(结果=0)": (10, 10),
                "减(结果<0)": (39, 45), "乘": (81, 81)}
        got = {k: tuple(v) for k, v in per.items()}
        if got != want:
            fail.append(f"eval/{arith12.name}: 算术网格 {got} != 文档 {want}")
        else:
            print("    v3.12 算术网格 275/281（与文档一致）")
    for fname, want_hits, want_neg in [("longctx32_v3_12a2l3.json", "28/29/26/27", 61),
                                       ("longctx32multi_v3_12a2l3.json", "28/25/32/23", 62)]:
        p = ROOT / "eval" / fname
        if not p.exists():
            warn.append(f"eval/{fname} 不存在，跳过 v3.12 检索校验")
            continue
        rows = json.loads(p.read_text(encoding="utf-8"))["rows"]
        hits = "/".join(str(r["hit"]) for r in rows)
        neg = sum(r.get("neg_hit", 0) for r in rows)
        if hits != want_hits or neg != want_neg:
            fail.append(f"eval/{fname}: {hits} 拒答 {neg}/64，文档声称 {want_hits} 与 {want_neg}/64")
        else:
            print(f"    v3.12 {fname.split('_')[0]} {hits}（拒答 {neg}/64，与文档一致）")
    scope12 = ROOT / "eval" / "v3_12a2l3_scope.json"
    if scope12.exists():
        rows = json.loads(scope12.read_text(encoding="utf-8"))
        got = sum(1 for r in rows if r.get("ok") is True)
        if got != 10:
            fail.append(f"eval/{scope12.name}: 范围评测 {got}/10 != 文档 10/10")
        else:
            print("    v3.12 范围评测 10/10（与文档一致）")
    probe12 = ROOT / "eval" / "chat_probe_v3_12a2l3.json"
    if probe12.exists():
        rows = json.loads(probe12.read_text(encoding="utf-8"))["rows"]
        miss = sum(1 for r in rows if r["topic_miss"] is True)
        loop = sum(1 for r in rows if r["loop"] is True)
        if len(rows) != 42 or miss != 0 or loop != 0:
            fail.append(f"eval/{probe12.name}: {len(rows)-miss}/42（复读 {loop}），文档声称 42/42 且 0 复读")
        else:
            print("    v3.12 日常探针 42/42、0 复读（与文档一致）")
    ident12 = ROOT / "eval" / "identity_v3_12a2l3.json"
    if ident12.exists():
        d = json.loads(ident12.read_text(encoding="utf-8"))
        if d.get("score") != "12/12":
            fail.append(f"eval/{ident12.name}: 身份 {d.get('score')} != 文档 12/12")
        else:
            print("    v3.12 身份 12/12（与文档一致）")
    for f in (ROOT / "v3_12" / "gguf").glob("*.gguf"):
        if b"tokenizer.chat_template" not in f.read_bytes():
            fail.append(f"v3_12/gguf/{f.name}: 未内嵌 chat template")

    # --- v3.13：记忆版（PC + 板端两套权重） ---
    mem13 = ROOT / "eval" / "memory_v3_13pc3.json"
    if mem13.exists():
        d = json.loads(mem13.read_text(encoding="utf-8"))
        if d.get("score") != "21/24":
            fail.append(f"eval/{mem13.name}: 记忆 {d.get('score')} != 文档 21/24")
        else:
            print("    v3.13 记忆评测 21/24（与文档一致）")
    arith13 = ROOT / "eval" / "arith_v3_13pc3.json"
    if arith13.exists():
        rows = json.loads(arith13.read_text(encoding="utf-8"))["rows"]
        per = {}
        for r in rows:
            per.setdefault(r["kind"], [0, 0])
            per[r["kind"]][1] += 1
            per[r["kind"]][0] += int(r["ok"])
        want = {"加": (100, 100), "减(结果>0)": (45, 45), "减(结果=0)": (10, 10),
                "减(结果<0)": (38, 45), "乘": (81, 81)}
        got = {k: tuple(v) for k, v in per.items()}
        if got != want:
            fail.append(f"eval/{arith13.name}: 算术网格 {got} != 文档 {want}")
        else:
            print("    v3.13 算术网格 274/281（与文档一致）")
    for fname, want_hits, want_neg in [("longctx32_v3_13pc3.json", "28/29/29/27", 61),
                                       ("longctx32multi_v3_13pc3.json", "28/26/32/22", 62)]:
        p = ROOT / "eval" / fname
        if not p.exists():
            warn.append(f"eval/{fname} 不存在，跳过 v3.13 检索校验")
            continue
        rows = json.loads(p.read_text(encoding="utf-8"))["rows"]
        hits = "/".join(str(r["hit"]) for r in rows)
        neg = sum(r.get("neg_hit", 0) for r in rows)
        if hits != want_hits or neg != want_neg:
            fail.append(f"eval/{fname}: {hits} 拒答 {neg}/64，文档声称 {want_hits} 与 {want_neg}/64")
        else:
            print(f"    v3.13 {fname.split('_')[0]} {hits}（拒答 {neg}/64，与文档一致）")
    scope13 = ROOT / "eval" / "v3_13pc3_scope.json"
    if scope13.exists():
        rows = json.loads(scope13.read_text(encoding="utf-8"))
        got = sum(1 for r in rows if r.get("ok") is True)
        if got != 10:
            fail.append(f"eval/{scope13.name}: 范围评测 {got}/10 != 文档 10/10")
        else:
            print("    v3.13 范围评测 10/10（与文档一致）")
    probe13 = ROOT / "eval" / "chat_probe_v3_13pc3.json"
    if probe13.exists():
        rows = json.loads(probe13.read_text(encoding="utf-8"))["rows"]
        miss = sum(1 for r in rows if r["topic_miss"] is True)
        loop = sum(1 for r in rows if r["loop"] is True)
        if len(rows) != 42 or miss != 0 or loop != 0:
            fail.append(f"eval/{probe13.name}: {len(rows)-miss}/42（复读 {loop}），文档声称 42/42 且 0 复读")
        else:
            print("    v3.13 日常探针 42/42、0 复读（与文档一致）")
    ident13 = ROOT / "eval" / "identity_v3_13pc3.json"
    if ident13.exists():
        d = json.loads(ident13.read_text(encoding="utf-8"))
        if d.get("score") != "12/12":
            fail.append(f"eval/{ident13.name}: 身份 {d.get('score')} != 文档 12/12")
        else:
            print("    v3.13 身份 12/12（与文档一致）")
    for mode, fname in [("q2b8", "pc_kv_suite32_v3_13b_q2b8.txt")]:
        p = ROOT / "logs" / fname
        if not p.exists():
            warn.append(f"logs/{fname} 不存在，跳过 v3.13 C 矩阵校验")
            continue
        t = p.read_text(encoding="utf-8", errors="replace")
        if "短任务 27/27" not in t or "长文召回 4/4" not in t:
            fail.append(f"logs/{fname}: 不是 27/27 + 4/4，文档声称（{mode}）满分")
        else:
            print(f"    v3.13 C 引擎 {mode} 27/27 + 4/4（与文档一致）")
    arith_suite13 = ROOT / "logs" / "pc_arith_suite_v3_13b_q2b8.txt"
    if arith_suite13.exists():
        t = arith_suite13.read_text(encoding="utf-8", errors="replace")
        if "短任务 21/21" not in t:
            fail.append("logs/pc_arith_suite_v3_13b_q2b8.txt: 不是 21/21，文档声称算术子集满分")
        else:
            print("    v3.13 C 引擎算术子集 21/21（与文档一致）")
    else:
        warn.append("logs/pc_arith_suite_v3_13b_q2b8.txt 不存在，跳过 v3.13 算术子集校验")
    for fname in ("board_v3_13b_multi.txt", "board_v3_13b_chat10.txt", "board_v3_13b_arith.txt"):
        p = ROOT / "logs" / fname
        if not p.exists():
            warn.append(f"logs/{fname} 不存在，跳过 v3.13 板端校验")
            continue
        t = p.read_text(encoding="utf-8", errors="replace")
        if "10 成功 / 0 失败" not in t:
            fail.append(f"logs/{fname}: 未记录 10/10 成功")
        else:
            print(f"    v3.13 板端 {fname} 10/10（与文档一致）")
    memboard = ROOT / "logs" / "board_v3_13b_memory.txt"
    if memboard.exists():
        t = memboard.read_text(encoding="utf-8", errors="replace")
        if "6 成功 / 0 失败" not in t or "你叫小明" not in t or "你最喜欢蓝色" not in t:
            fail.append("logs/board_v3_13b_memory.txt: 未记录跨轮记忆 6/6（小明/蓝色）")
        else:
            print("    v3.13 板端跨轮记忆 6/6（与文档一致）")
    else:
        warn.append("logs/board_v3_13b_memory.txt 不存在，跳过 v3.13 记忆校验")
    for d in ("gguf_pc", "gguf_board"):
        for f in (ROOT / "v3_13" / d).glob("*.gguf"):
            if b"tokenizer.chat_template" not in f.read_bytes():
                fail.append(f"v3_13/{d}/{f.name}: 未内嵌 chat template")

    # --- v3.14：tool 版（算术/时间/随机数交给 C 引擎；GGUF 取消发行） ---
    mem14 = ROOT / "eval" / "memory_v3_14pc2.json"
    if mem14.exists():
        d = json.loads(mem14.read_text(encoding="utf-8"))
        if d.get("score") != "24/24":
            fail.append(f"eval/{mem14.name}: 记忆 {d.get('score')} != 文档 24/24")
        else:
            print("    v3.14 PC 记忆 24/24（与文档一致）")
    for fname, want_hits, want_neg in [("longctx32_v3_14pc2.json", "28/28/25/27", 61),
                                       ("longctx32multi_v3_14pc2.json", "28/25/32/22", 62)]:
        p = ROOT / "eval" / fname
        if not p.exists():
            warn.append(f"eval/{fname} 不存在，跳过 v3.14 检索校验")
            continue
        rows = json.loads(p.read_text(encoding="utf-8"))["rows"]
        hits = "/".join(str(r["hit"]) for r in rows)
        neg = sum(r.get("neg_hit", 0) for r in rows)
        if hits != want_hits or neg != want_neg:
            fail.append(f"eval/{fname}: {hits} 拒答 {neg}/64，文档声称 {want_hits} 与 {want_neg}/64")
        else:
            print(f"    v3.14 {fname.split('_')[0]} {hits}（拒答 {neg}/64，与文档一致）")
    for fname, want, tag in [("v3_14pc2_scope.json", 10, "范围"),
                             ("chat_probe_v3_14pc2.json", 42, "探针")]:
        p = ROOT / "eval" / fname
        if not p.exists():
            warn.append(f"eval/{fname} 不存在，跳过 v3.14 {tag}校验")
            continue
        d = json.loads(p.read_text(encoding="utf-8"))
        if tag == "范围":
            got = sum(1 for r in d if r.get("ok") is True)
            if got != want:
                fail.append(f"eval/{fname}: {tag} {got}/10 != 文档 10/10")
            else:
                print("    v3.14 范围评测 10/10（与文档一致）")
        else:
            rows = d["rows"]
            miss = sum(1 for r in rows if r["topic_miss"] is True)
            loop = sum(1 for r in rows if r["loop"] is True)
            if len(rows) != want or miss or loop:
                fail.append(f"eval/{fname}: 探针 {len(rows)-miss}/42（复读 {loop}）!= 文档 42/42")
            else:
                print("    v3.14 日常探针 42/42、0 复读（与文档一致）")
    ident14 = ROOT / "eval" / "identity_v3_14pc2.json"
    if ident14.exists():
        d = json.loads(ident14.read_text(encoding="utf-8"))
        if d.get("score") != "12/12":
            fail.append(f"eval/{ident14.name}: 身份 {d.get('score')} != 文档 12/12")
        else:
            print("    v3.14 身份 12/12（与文档一致）")
    suite14 = ROOT / "logs" / "pc_kv_suite32_v3_14b6_calc_q2b8.txt"
    if suite14.exists():
        t = suite14.read_text(encoding="utf-8", errors="replace")
        if "短任务 27/27" not in t or "长文召回 4/4" not in t:
            fail.append("logs/pc_kv_suite32_v3_14b6_calc_q2b8.txt: 不是 27/27 + 4/4")
        else:
            print("    v3.14 C 引擎 q2 27/27 + 4/4（与文档一致）")
    arith14 = ROOT / "logs" / "pc_arith_suite_v3_14b_q2b8.txt"
    if arith14.exists():
        t = arith14.read_text(encoding="utf-8", errors="replace")
        if "短任务 21/21" not in t:
            fail.append("logs/pc_arith_suite_v3_14b_q2b8.txt: 不是 21/21")
        else:
            print("    v3.14 C 引擎算术子集 21/21（tool 回答，与文档一致）")
    for fname, needle in [("board_v3_14b6_tools.txt", "8 成功 / 0 失败"),
                          ("board_v3_14b6_tools2.txt", "10 成功 / 0 失败"),
                          ("board_v3_14b6_multi.txt", "10 成功 / 0 失败"),
                          ("board_v3_14b6_chat10.txt", "10 成功 / 0 失败"),
                          ("board_v3_14b6_memory12.txt", "板端记忆 10/12")]:
        p = ROOT / "logs" / fname
        if not p.exists():
            warn.append(f"logs/{fname} 不存在，跳过 v3.14 板端校验")
            continue
        t = p.read_text(encoding="utf-8", errors="replace")
        if needle not in t:
            fail.append(f"logs/{fname}: 未记录「{needle}」")
        else:
            print(f"    v3.14 板端 {fname}（{needle}，与文档一致）")
    a8 = ROOT / "logs" / "pc_kv_suite32_v3_14b6_a8_q2b8.txt"
    if a8.exists():
        t = a8.read_text(encoding="utf-8", errors="replace")
        if "短任务 27/27" not in t or "长文召回 4/4" not in t:
            fail.append("logs/pc_kv_suite32_v3_14b6_a8_q2b8.txt: A8 激活量化不是 27/27 + 4/4")
        else:
            print("    v3.14 A8（int8 激活）27/27 + 4/4（与文档一致）")
    # 附录 4：KV 访问优化（逐位一致）+ 长对话压力 + 上下文写满
    vseq = ROOT / "logs" / "pc_kv_suite32_v3_14b6_vseq_q2b8.txt"
    vbase = ROOT / "logs" / "pc_kv_suite32_v3_14b6_calc_q2b8.txt"
    if vseq.exists() and vbase.exists():
        t1 = vseq.read_text(encoding="utf-8", errors="replace")
        t2 = vbase.read_text(encoding="utf-8", errors="replace")
        if "短任务 27/27" not in t1 or "长文召回 4/4" not in t1:
            fail.append("logs/pc_kv_suite32_v3_14b6_vseq_q2b8.txt: 不是 27/27 + 4/4")
        elif t1 != t2:
            fail.append("KV 访问优化后的 q2 套件输出与优化前不一致（文档声称逐行零差异）")
        else:
            print("    v3.14 KV 优化：q2 套件 27/27+4/4 且与优化前逐行一致（与文档一致）")
    stress = ROOT / "logs" / "board_v3_14b6_stress64_v2_console.txt"
    if stress.exists():
        t = stress.read_text(encoding="utf-8", errors="replace")
        if "64 成功 / 0 失败" not in t:
            fail.append("logs/board_v3_14b6_stress64_v2_console.txt: 未记录 64/64 成功")
        else:
            print("    v3.14 板端 64 轮长对话 64/64（与文档一致）")
    ctxreset = ROOT / "logs" / "board_v3_14b6_ctxreset_test256.txt"
    if ctxreset.exists():
        t = ctxreset.read_text(encoding="utf-8", errors="replace")
        if "CONTEXT-FULL" not in t or "ctx 0->10/256" not in t:
            fail.append("logs/board_v3_14b6_ctxreset_test256.txt: 未记录上下文写满自动重置")
        else:
            print("    v3.14 上下文写满自动开新对话（256 测试版，与文档一致）")
    tools3 = ROOT / "logs" / "board_v3_14b6_tools3.txt"
    if tools3.exists():
        t = tools3.read_text(encoding="utf-8", errors="replace")
        if "8 成功 / 0 失败" not in t or "3天后是几号" not in t:
            fail.append("logs/board_v3_14b6_tools3.txt: 未记录工具扩展 8/8")
        else:
            print("    v3.14 工具扩展（百分号/平方/根号/日期）8/8（与文档一致）")
    # --- v3.15-embed：板端身份漂移修复 ---
    ci4 = ROOT / "logs" / "pc_kv_suite32_v3_15ci4_q2b8.txt"
    if ci4.exists():
        t = ci4.read_text(encoding="utf-8", errors="replace")
        if "短任务 27/27" not in t or "长文召回 4/4" not in t:
            fail.append("logs/pc_kv_suite32_v3_15ci4_q2b8.txt: 不是 27/27 + 4/4")
        else:
            print("    v3.15-embed C 引擎 q2 27/27 + 4/4（与文档一致）")
    ci4a = ROOT / "logs" / "pc_arith_suite_v3_15ci4_q2b8.txt"
    if ci4a.exists() and "短任务 21/21" not in ci4a.read_text(encoding="utf-8", errors="replace"):
        fail.append("logs/pc_arith_suite_v3_15ci4_q2b8.txt: 不是 21/21")
    for fname, needle in [("board_v3_15ci4_memory12.txt", "板端记忆 10/12"),
                          ("board_v3_15ci4_tools.txt", "8 成功 / 0 失败"),
                          ("board_v3_15ci4_multi.txt", "10 成功 / 0 失败"),
                          ("board_v3_15ci4_chat10.txt", "10 成功 / 0 失败")]:
        p = ROOT / "logs" / fname
        if not p.exists():
            warn.append(f"logs/{fname} 不存在，跳过 v3.15-embed 校验")
            continue
        if needle not in p.read_text(encoding="utf-8", errors="replace"):
            fail.append(f"logs/{fname}: 未记录「{needle}」")
        else:
            print(f"    v3.15-embed {fname}（{needle}，与文档一致）")
    idctx = ROOT / "logs" / "board_v3_15ci4_identity_ctx.txt"
    if idctx.exists():
        t = idctx.read_text(encoding="utf-8", errors="replace")
        if "我叫 feng，由个人开发者 jiaheng 开发训练" not in t:
            fail.append("logs/board_v3_15ci4_identity_ctx.txt: 未记录上下文身份修复结果")
        else:
            print("    v3.15-embed 上下文身份修复（与文档一致）")
    tools4 = ROOT / "logs" / "board_v3_15ci4_tools4.txt"
    if tools4.exists():
        t = tools4.read_text(encoding="utf-8", errors="replace")
        if "把59+1算一下" not in t or "100的15%是多少钱" not in t or "0.5s" not in t:
            fail.append("logs/board_v3_15ci4_tools4.txt: 未记录工具外壳扩展 6/6")
        else:
            print("    v3.15-embed 工具外壳扩展（与文档一致）")
    pctools = ROOT / "logs" / "pc_tools_test.txt"
    if pctools.exists():
        t = pctools.read_text(encoding="utf-8", errors="replace")
        if "全部通过（0 个失败）" not in t or "第一个确实被丢弃" not in t:
            fail.append("logs/pc_tools_test.txt: 未记录时间/随机数单测全部通过")
        else:
            print("    时间/随机数 C 单测 53 项全部通过（与文档一致）")
    btools = ROOT / "logs" / "board_v3_15ci4_tools_time_rand.txt"
    if btools.exists():
        t = btools.read_text(encoding="utf-8", errors="replace")
        if "板端时间与网络时间一致" not in t or "13 成功 / 0 失败" not in t:
            fail.append("logs/board_v3_15ci4_tools_time_rand.txt: 未记录 13/13 或时间一致性")
        else:
            print("    板端时间/随机数 tool 专项 13/13（与文档一致）")
    # --- v3.15-embed 附录：长上下文注意力优化（位精确） ---
    bench = ROOT / "logs" / "board_bench_attn_inl.txt"
    if bench.exists():
        t = bench.read_text(encoding="utf-8", errors="replace")
        if "ctx= 256: cold 808 ms/forward" not in t or "ctx=2048: cold 2719 ms/forward" not in t:
            fail.append("logs/board_bench_attn_inl.txt: 与文档的 808/2719 ms 不一致")
        else:
            print("    板端注意力基准 808/1631/2719 ms（与文档一致）")
    prof = ROOT / "logs" / "board_bench_attn_prof.txt"
    if prof.exists():
        t = prof.read_text(encoding="utf-8", errors="replace")
        if "K=894 ms" not in t or "softmax=169 ms" not in t or "V=1109 ms" not in t:
            fail.append("logs/board_bench_attn_prof.txt: 与文档的 K/softmax/V 分解不一致")
        else:
            print("    板端注意力分解 K894 / softmax169 / V1109（与文档一致）")
    pairb = ROOT / "logs" / "board_bench_attn_pair.txt"
    if pairb.exists():
        t = pairb.read_text(encoding="utf-8", errors="replace")
        if "ctx= 256: cold 777 ms/forward" not in t or "ctx=2048: cold 2470 ms/forward" not in t:
            fail.append("logs/board_bench_attn_pair.txt: 与文档的 777/2470 ms 不一致")
        elif "K=835 ms" not in t or "V=912 ms" not in t:
            fail.append("logs/board_bench_attn_pair.txt: 与文档的 K835/V912 不一致")
        else:
            print("    板端注意力 2-token 展开 777/1506/2470 ms、K835/V912（与文档一致）")
    nohdb = ROOT / "logs" / "board_bench_attn_nohd.txt"
    if nohdb.exists():
        t = nohdb.read_text(encoding="utf-8", errors="replace")
        if "with 532 ms | without 407 ms" not in t or "省 124 ms" not in t:
            fail.append("logs/board_bench_attn_nohd.txt: 与文档的 lm head 532/407/124 ms 不一致")
        else:
            print("    板端 lm head 532/407 ms、每 prefill token 省 124 ms（与文档一致）")
    ab = [ROOT / "logs" / f"pc_kv_suite32_v3_15ci4_q2b8_{k}.txt"
          for k in ("nolut", "lut", "lin", "inl", "pair", "nohd", "endtok", "madd", "fexp", "vfold")]
    texts = [p.read_text(encoding="utf-8", errors="replace") for p in ab if p.exists()]
    if len(texts) >= 2:
        if any(t != texts[0] for t in texts[1:]):
            fail.append("q2 注意力优化各版 PC 套件输出不一致（应为逐字节一致）")
        elif "短任务 27/27" not in texts[0] or "长文召回 4/4" not in texts[0]:
            fail.append("q2 注意力优化 PC 套件不是 27/27 + 4/4")
        else:
            print(f"    q2 注意力优化 {len(texts)} 版输出逐字节一致（27/27+4/4，与文档一致）")
    for fname, needle, label in [
            ("pc_check_v3_15ci4_inl.txt", "MATCH", "fp32 参考 logits MATCH"),
            ("board_v3_15ci4_tools_after_attnopt.txt", "13 成功 / 0 失败", "板端 tool 13/13"),
            ("board_v3_15ci4_memory_after_attnopt.txt", "5 成功 / 0 失败", "板端多轮记忆 5/5"),
            ("board_v3_15ci4_tools_after_pair.txt", "13 成功 / 0 失败", "2-token 展开后 tool 13/13"),
            ("board_v3_15ci4_memory_after_pair.txt", "4 成功 / 0 失败", "2-token 展开后记忆 4/4"),
            ("board_v3_15ci4_tools_after_nohd.txt", "13 成功 / 0 失败", "跳过 lm head 后 tool 13/13"),
            ("board_v3_15ci4_memory_after_nohd.txt", "4 成功 / 0 失败", "跳过 lm head 后记忆 4/4"),
            ("board_v3_15ci4_memory_after_endtok.txt", "4 成功 / 0 失败", "轮末 token 也跳过后的记忆 4/4"),
            ("pc_check_v3_15ci4_nohd.txt", "MATCH", "跳过 lm head 后 fp32 参考 logits MATCH")]:
        p = ROOT / "logs" / fname
        if p.exists():
            if needle not in p.read_text(encoding="utf-8", errors="replace"):
                fail.append(f"logs/{fname}: 未记录「{needle}」")
            else:
                print(f"    注意力优化后 {label}（与文档一致）")
    # --- v3.15-embed 附录：A8 整数 GEMV 的负结果 ---
    a8 = [ROOT / "logs" / f"pc_kv_suite32_v3_15ci4_q2b8_{k}.txt" for k in ("a8old", "a8h")]
    at = [p.read_text(encoding="utf-8", errors="replace") for p in a8 if p.exists()]
    if len(at) == 2 and at[0] != at[1]:
        fail.append("A8 提升版与旧版输出不一致（应逐字节一致）")
    abench = ROOT / "logs" / "board_bench_attn_a8.txt"
    if abench.exists():
        t = abench.read_text(encoding="utf-8", errors="replace")
        if "with 721 ms" not in t or "attn bench ctx= 256: cold 967 ms/forward" not in t:
            fail.append("logs/board_bench_attn_a8.txt: 与文档的 A8 负结果数字不一致")
        else:
            print("    A8 在 S3 更慢（967 ms / lm head 721 ms，与文档一致）")
    for fname, needle, label in [
            ("board_v3_15ci4_tools_after_a8revert.txt", "13 成功 / 0 失败", "刷回默认后 tool 13/13"),
            ("board_v3_15ci4_memory_after_a8revert.txt", "4 成功 / 0 失败", "刷回默认后记忆 4/4")]:
        p = ROOT / "logs" / fname
        if p.exists() and needle not in p.read_text(encoding="utf-8", errors="replace"):
            fail.append(f"logs/{fname}: 未记录「{needle}」")
        elif p.exists():
            print(f"    {label}（与文档一致）")
    # --- v3.15-embed 附录：纯 madd 累加链（默认开启） ---
    mbench = ROOT / "logs" / "board_bench_attn_madd.txt"
    if mbench.exists():
        t = mbench.read_text(encoding="utf-8", errors="replace")
        if "ctx= 256: cold 738 ms/forward" not in t or "ctx=2048: cold 2434 ms/forward" not in t:
            fail.append("logs/board_bench_attn_madd.txt: 与文档的 738/2434 ms 不一致")
        elif "with 492 ms | without 378 ms" not in t:
            fail.append("logs/board_bench_attn_madd.txt: 与文档的 lm head 492/378 ms 不一致")
        else:
            print("    madd 累加链 738/1469/2434 ms、lm head 492/378 ms（与文档一致）")
    mlogits = ROOT / "logs" / "pc_logits_madd_vs_old.txt"
    if mlogits.exists():
        t = mlogits.read_text(encoding="utf-8", errors="replace")
        if "max|diff|=3.815e-06" not in t:
            fail.append("logs/pc_logits_madd_vs_old.txt: 与文档的 3.8e-6 不一致")
        else:
            print("    madd vs 旧累加：logits max|diff|=3.8e-6（与文档一致）")
    fbench = ROOT / "logs" / "board_bench_attn_fexp.txt"
    if fbench.exists():
        t = fbench.read_text(encoding="utf-8", errors="replace")
        if "ctx= 256: cold 716 ms/forward" not in t or "ctx=2048: cold 2303 ms/forward" not in t:
            fail.append("logs/board_bench_attn_fexp.txt: 与文档的 716/2303 ms 不一致")
        elif "softmax=48 ms" not in t:
            fail.append("logs/board_bench_attn_fexp.txt: 与文档的 softmax 48 ms 不一致")
        else:
            print("    快速 exp：716/1399/2303 ms、softmax 48 ms（与文档一致）")
    flogits = ROOT / "logs" / "pc_logits_fexp_vs_newlib.txt"
    if flogits.exists():
        t = flogits.read_text(encoding="utf-8", errors="replace")
        if "max|diff|=3.815e-06" not in t:
            fail.append("logs/pc_logits_fexp_vs_newlib.txt: 与文档的 3.8e-6 不一致")
        else:
            print("    快速 exp vs newlib：logits max|diff|=3.8e-6（与文档一致）")
    for fname, needle, label in [
            ("board_v3_15ci4_tools_after_fexp.txt", "13 成功 / 0 失败", "快速 exp 固件 tool 13/13"),
            ("board_v3_15ci4_memory_after_fexp.txt", "5 成功 / 0 失败", "快速 exp 固件记忆 5/5")]:
        p = ROOT / "logs" / fname
        if p.exists() and needle not in p.read_text(encoding="utf-8", errors="replace"):
            fail.append(f"logs/{fname}: 未记录「{needle}」")
        elif p.exists():
            print(f"    {label}（与文档一致）")
    vbench = ROOT / "logs" / "board_bench_attn_vfold.txt"
    if vbench.exists():
        t = vbench.read_text(encoding="utf-8", errors="replace")
        if "ctx=2048: cold 2224 ms/forward" not in t or "V=832 ms" not in t:
            fail.append("logs/board_bench_attn_vfold.txt: 与文档的 2224 ms / V 832 ms 不一致")
        else:
            print("    V 段折叠：2048 ctx 2224 ms、V 832 ms（与文档一致）")
    allon = ROOT / "logs" / "pc_logits_allon_vs_session_start.txt"
    if allon.exists():
        t = allon.read_text(encoding="utf-8", errors="replace")
        if "max|diff|=4.768e-06" not in t:
            fail.append("logs/pc_logits_allon_vs_session_start.txt: 与文档的 4.8e-6 不一致")
        else:
            print("    三重优化累计 logits 差 4.8e-6（与文档一致）")
    for fname, needle, label in [
            ("board_v3_15ci4_tools_after_vfold.txt", "13 成功 / 0 失败", "V 折叠固件 tool 13/13"),
            ("board_v3_15ci4_memory_after_vfold.txt", "5 成功 / 0 失败", "V 折叠固件记忆 5/5")]:
        p = ROOT / "logs" / fname
        if p.exists() and needle not in p.read_text(encoding="utf-8", errors="replace"):
            fail.append(f"logs/{fname}: 未记录「{needle}」")
        elif p.exists():
            print(f"    {label}（与文档一致）")
    for fname, needle, label in [
            ("board_v3_15ci4_tools_after_madd.txt", "13 成功 / 0 失败", "madd 默认固件 tool 13/13"),
            ("board_v3_15ci4_memory_after_madd.txt", "5 成功 / 0 失败", "madd 默认固件记忆 5/5")]:
        p = ROOT / "logs" / fname
        if p.exists() and needle not in p.read_text(encoding="utf-8", errors="replace"):
            fail.append(f"logs/{fname}: 未记录「{needle}」")
        elif p.exists():
            print(f"    {label}（与文档一致）")
    # --- v3.16-embed：报名字后的身份串名修复 ---
    for fname, needle, label in [
            ("board_v3_16p3_tools.txt", "13 成功 / 0 失败", "v3.16-embed tool 13/13"),
            ("board_v3_16p3_memory12.txt", "板端记忆 10/12", "v3.16-embed 记忆 10/12"),
            ("pc_kv_suite32_v3_16p3_q2b8.txt", "短任务 27/27", "v3.16-embed PC 32 题 27/27+4/4"),
            ("pc_arith_suite_v3_16p3_q2b8.txt", "短任务 21/21", "v3.16-embed 算术 21/21")]:
        p = ROOT / "logs" / fname
        if p.exists() and needle not in p.read_text(encoding="utf-8", errors="replace"):
            fail.append(f"logs/{fname}: 未记录「{needle}」")
        elif p.exists():
            print(f"    {label}（与文档一致）")
    nl = ROOT / "logs" / "board_v3_16p3_nameleak.txt"
    if nl.exists():
        t = nl.read_text(encoding="utf-8", errors="replace")
        if "我叫 feng，由个人开发者 jiaheng 开发训练" not in t:
            fail.append("logs/board_v3_16p3_nameleak.txt: 未记录报名字后的身份回答")
        else:
            print("    v3.16-embed 报名字后答 feng（与文档一致）")
    mts = ROOT / "logs" / "pc_mtsuite_model_export_v3_16p3.txt"
    if mts.exists():
        t = mts.read_text(encoding="utf-8", errors="replace")
        if "SUMMARY mt-suite 8/10" not in t or "SUMMARY mem12 12/12" not in t:
            fail.append("logs/pc_mtsuite_model_export_v3_16p3.txt: 不是 8/10 + 12/12")
        else:
            print("    v3.16-embed 多轮套件 8/10、PC 12 题 12/12（与文档一致）")
    # --- v3.16-embed 附录：修残余链的代价（p4–p6，未采用） ---
    for fname, needle in [("pc_mtsuite_v3_16p4_seq.txt", "SUMMARY seq 10/10"),
                          ("pc_mtsuite_v3_16p5_seq.txt", "SUMMARY seq 10/10"),
                          ("pc_mtsuite_v3_16p6_seq.txt", "SUMMARY seq 10/10"),
                          ("pc_mtsuite_v3_16p7_seq.txt", "SUMMARY seq 10/10"),
                          ("pc_kv_suite32_v3_16p7_q2b8.txt", "长文召回 4/4"),
                          ("pc_kv_suite32_v3_16p6_q2b8.txt", "长文召回 4/4"),
                          ("board_v3_16p6_identity_ctx.txt", "8 成功 / 0 失败"),
                          ("board_v3_16p6_memory12.txt", "板端记忆 9/12")]:
        p = ROOT / "logs" / fname
        if p.exists() and needle not in p.read_text(encoding="utf-8", errors="replace"):
            fail.append(f"logs/{fname}: 未记录「{needle}」")
        elif p.exists():
            print(f"    p4–p6 实验记录 {fname}（与文档一致）")
    # --- v3.16-embed 工程附录：长文输入 4KB + prefill 成本实测 ---
    for fname, needle, label in [
            ("board_longprompt_2turn_491tok.txt", "包含 7391: True", "长文 491 tokens 答对取件码"),
            ("board_longprompt_2turn_491tok.txt", "第1轮（正文）290.6s", "长文 491 tokens 290.6 s"),
            ("board_longprompt_2turn.txt", "包含 2468: False", "217 tokens 同类测试失败（不稳定）")]:
        p = ROOT / "logs" / fname
        if p.exists() and needle not in p.read_text(encoding="utf-8", errors="replace"):
            fail.append(f"logs/{fname}: 未记录「{needle}」")
        elif p.exists():
            print(f"    {label}（与文档一致）")
    # --- PIE 第二版：正确的内核也只有 1.05×（负结果） ---
    pie1 = ROOT / "logs" / "board_bench_pie.txt"
    pie2 = ROOT / "logs" / "board_bench_pie_gemv2.txt"
    if pie1.exists():
        t = pie1.read_text(encoding="utf-8", errors="replace")
        if "0.629 周期/MAC" not in t and "0.63" not in t:
            fail.append("logs/board_bench_pie.txt: 未记录 PIE 裸吞吐 0.63 周期/MAC")
        else:
            print("    PIE 裸吞吐 0.63 周期/MAC（与文档一致）")
    if pie2.exists():
        t = pie2.read_text(encoding="utf-8", errors="replace")
        if "PIE dot self-check: MATCH" not in t or "1.05x" not in t:
            fail.append("logs/board_bench_pie_gemv2.txt: 未记录自检 MATCH 与 1.05×")
        else:
            print("    PIE 内核自检 MATCH、1.05×（与文档一致）")
    # --- v3.16-embed：板端回复尾行（ctx / tok/s） ---
    tail = ROOT / "logs" / "board_v3_16p3_tail_trailer.txt"
    if tail.exists():
        t = tail.read_text(encoding="utf-8", errors="replace")
        if ">>END (ctx " not in t or "tok/s)" not in t:
            fail.append("logs/board_v3_16p3_tail_trailer.txt: 未记录新的 >>END 尾行")
        else:
            print("    板端回复尾行 ctx/tok/s（与文档一致）")
# --- v3.17 引擎记忆 tool ---
    for fname, needle, label in [
            ("board_v3_16p3_memory12_engmem.txt", "板端记忆 12/12", "v3.17 记忆 12/12"),
            ("board_v3_17_memory12.txt", "板端记忆 12/12", "v3.17 通用槽后记忆 12/12"),
            ("board_v3_17_generic_slots2.txt", "7 成功 / 0 失败", "v3.17 通用键值槽 7/7"),
            ("board_v3_17_patterns.txt", "8 成功 / 0 失败", "v3.17 新问法 8/8"),
            ("board_v3_17_memory12b.txt", "板端记忆 12/12", "v3.17 补问法后记忆 12/12"),
            ("board_v3_17_tools_final.txt", "13 成功 / 0 失败", "v3.17 终版固件工具 13/13"),
            ("board_v3_17_final_smoke.txt", "4 成功 / 0 失败", "v3.17 终版冒烟 4/4"),
            ("board_v3_17_mem_list_forget2.txt", "9 成功 / 0 失败", "v3.17 列出/遗忘/墓碑 9/9"),
            ("board_v3_17_stress64.txt", "64 成功 / 0 失败", "v3.17 当前固件 64 轮压测 64/64"),
            ("board_v3_16p3_identity_ctx_engmem.txt", "8 成功 / 0 失败", "v3.17 身份 8/8"),
            ("board_v3_16p3_tools_engmem.txt", "13 成功 / 0 失败", "v3.17 工具专项 13/13"),
            ("python_tools_selftest_engmem.txt", "你叫小雨", "v3.17 Python 同口径")]:
        p = ROOT / "logs" / fname
        if p.exists() and needle not in p.read_text(encoding="utf-8", errors="replace"):
            fail.append(f"logs/{fname}: 未记录「{needle}」")
        elif p.exists():
            print(f"    {label}（与文档一致）")
    # --- v3.17：留出题泛化评测 + 时钟推算 tool ---
    for path, key, want, label in [
            ("eval/heldout_v3_14pc2.json", "ok", 16, "留出题 PC v3.14 16/30"),
            ("eval/heldout_v3_16p3.json", "ok", 17, "留出题板端 v3.16-embed 17/30")]:
        p = ROOT / path
        if p.exists():
            d = json.loads(p.read_text(encoding="utf-8"))
            if d.get(key) != want or d.get("n") != 30:
                fail.append(f"{path}: 应为 {want}/30，实际 {d.get(key)}/{d.get('n')}")
            else:
                print(f"    {label}（与文档一致）")
    clock = ROOT / "logs" / "board_v3_17_clock_math.txt"
    if clock.exists():
        t = clock.read_text(encoding="utf-8", errors="replace")
        if "5 成功 / 0 失败" not in t or "明天 1 点" not in t:
            fail.append("logs/board_v3_17_clock_math.txt: 未记录时钟推算 5/5")
        else:
            print("    时间推算 tool 板端 5/5（与文档一致）")
    # --- v3.18：陈述模板对照数据（洗牌实验，未采用） ---
    for path, want, label in [("eval/heldout_v3_18s1.json", 16, "v3.18-s1 留出 16/30"),
                              ("eval/heldout_v3_18s2.json", 17, "v3.18-s2 留出 17/30")]:
        p = ROOT / path
        if p.exists():
            d = json.loads(p.read_text(encoding="utf-8"))
            if d.get("ok") != want:
                fail.append(f"{path}: 应为 {want}/30，实际 {d.get('ok')}/30")
            else:
                print(f"    {label}（与文档一致）")
    for fname, needle, label in [
            ("pc_kv_suite32_v3_18s2_q2b8.txt", "长文召回 4/4", "v3.18-s2 32 题 4/4"),
            ("pc_mtsuite_v3_18s2.txt", "SUMMARY seq 10/10", "v3.18-s2 seq 10/10")]:
        p = ROOT / "logs" / fname
        if p.exists() and needle not in p.read_text(encoding="utf-8", errors="replace"):
            fail.append(f"logs/{fname}: 未记录「{needle}」")
        elif p.exists():
            print(f"    {label}（与文档一致）")

    # --- 身份表述：写了"身份自述"的文档必须是 v3.2 的新说法 ---
    new_identity = "独立开发训练的 AI"
    for doc in DOCS:
        t = doc.read_text(encoding="utf-8")
        if "身份自述" in t and new_identity not in t:
            fail.append(f"{doc.relative_to(ROOT)}: 身份自述不是 v3.2 的"
                        f"「{new_identity}」表述")
    for doc in DOCS:
        t = doc.read_text(encoding="utf-8")
        for m in re.finditer(r"0\.76\s*/\s*0\.43\s*/\s*([\d.]+)\s*/\s*([\d.]+)", t):
            got = ["0.76", "0.43", m.group(1), m.group(2)]
            if got != want_loss:
                fail.append(f"{doc.relative_to(ROOT)}: 检索 SFT loss 写为 "
                            f"{'/'.join(got)}，实测为 {'/'.join(want_loss)}")
    print(f"    检索 SFT loss 序列 {'/'.join(want_loss)}（与 summary.json 一致）")


def main() -> int:
    check_structure()
    check_facts()
    print("[3] 汇总")
    for w in warn:
        print(f"    WARN {w}")
    for f in fail:
        print(f"    FAIL {f}")
    print(f"    结论：{'全部通过' if not fail else f'{len(fail)} 处失败'}"
          + (f"，{len(warn)} 处警告" if warn else ""))
    return 1 if fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
