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
RELEASE_PREFIXES = ("weights/", "datasets/", "feng-30m-v3/")
# 已废弃的表述，不得再出现在文档里
STALE = [
    "0x310000", "0x1A10000", "stage_32k", "29.66", "MQA(1 KV) / head", "3 MB 分区",
    "COM5 ", "you: / feng:", "MAX_CTX=256", "每 +512 需 +10 MB",
    "三值量化", "Qwen 3.8", "不是 git 仓库",
    "R16N32",                                      # 非官方型号写法，正确为 N32R16V
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
