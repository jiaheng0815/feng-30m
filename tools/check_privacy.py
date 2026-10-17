#!/usr/bin/env python
"""隐私自检：防止真实姓名等个人身份信息混进仓库。

背景（2026-10-17）：作者的真名以拼音 `jiaheng` 形式公开是**允许的**，
但**中文全名不能出现在项目的任何地方**。这个脚本把这条约束变成可执行的检查，
而不是靠人记着。

两层：

1. **黑名单硬失败**：把要保护的字符串写进 `scripts/local_paths.json` 的
   `private_names`（那个文件已 gitignore），或用环境变量
   `FENG_PRIVATE_NAMES="名字1,名字2"`。命中即退出码 1。
   —— **名字本身不进仓库**，这正是它能起作用的前提。

2. **启发式告警**：扫描「作者 / 开发者 / 版权 / Copyright / 署名」后面紧跟的
   2~4 个汉字，像是人名的就报出来（不失败，只提醒人看一眼）。
   公开语料（`data/`）里有大量历史人物名（冯·诺依曼之类），不扫。

用法：
    python tools/check_privacy.py            # 检查（CI 也用这个）
    python tools/check_privacy.py --list     # 只显示当前生效的黑名单

CI 只跑第 2 层（黑名单是私密的，clean clone 上没有）。
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[1]

# 这些目录是「别人的文字」，不扫：公开语料里人名多如牛毛
SKIP_DIRS = ("data/", "node_modules/", "build/", ".git/", "esp32s3-feng-llm/build/")
# 二进制/大文件不扫
SKIP_EXT = {".safetensors", ".pt", ".pth", ".gguf", ".bin", ".npy", ".npz",
            ".o", ".obj", ".exe", ".dll", ".lib", ".pyc", ".so", ".a"}

# 「作者：张三」这类的上下文。故意收窄，减少误报。
ATTR = re.compile(
    r"(?:作者|开发者|版权人|著作权人|署名|维护者|Copyright\s*(?:\(c\))?|"
    r"Copyright\s*©|©)\s*[:：]?\s*([一-鿿]{2,4})",
    re.IGNORECASE,
)
# 这些是「上下文词」不是人名，过滤掉
NOT_NAME = {
    "个人", "独立", "开发", "训练", "作者", "版权", "所有", "保留", "权利",
    "本机", "产物", "说明", "文档", "仓库", "项目", "模型", "代码", "放进去的",
    "把工作开",
    # 实测误报（2026-10-17 首跑 11 处全是这些）
    "本机产物", "开发并发", "微调的", "环境", "终端里", "是谁", "开发者",
}


def tracked_files() -> list[Path]:
    try:
        out = subprocess.run(["git", "ls-files", "-z"], cwd=ROOT,
                             capture_output=True, check=True).stdout
        names = [n for n in out.decode("utf-8", "replace").split("\0") if n]
    except Exception:
        names = [str(p.relative_to(ROOT)) for p in ROOT.rglob("*") if p.is_file()]
    out = []
    for n in names:
        n = n.replace("\\", "/")
        if any(n.startswith(d) or f"/{d}" in f"/{n}" for d in SKIP_DIRS):
            continue
        if Path(n).suffix.lower() in SKIP_EXT:
            continue
        out.append(ROOT / n)
    return out


def load_private_names() -> list[str]:
    """黑名单来源：环境变量 > scripts/local_paths.json（已 gitignore）> 空。"""
    raw = os.environ.get("FENG_PRIVATE_NAMES", "")
    if not raw:
        lp = ROOT / "scripts" / "local_paths.json"
        if lp.exists():
            try:
                raw = ",".join(json.loads(lp.read_text(encoding="utf-8"))
                               .get("private_names", []))
            except Exception:
                pass
    return [s.strip() for s in raw.split(",") if s.strip()]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true", help="只打印生效的黑名单")
    ap.add_argument("--verbose", action="store_true", help="列出所有告警上下文")
    args = ap.parse_args()

    names = load_private_names()
    if args.list:
        print(f"黑名单 {len(names)} 条：{'（已设置，不显示）' if names else '空'}")
        print("设置方法：环境变量 FENG_PRIVATE_NAMES，或 scripts/local_paths.json 的 private_names")
        return 0

    files = tracked_files()
    hard_hits, soft_hits = [], []

    for p in files:
        try:
            text = p.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        for i, line in enumerate(text.splitlines(), 1):
            for nm in names:                      # 第 1 层：硬失败
                if nm and nm in line:
                    hard_hits.append((p.relative_to(ROOT), i, nm))
            for m in ATTR.finditer(line):         # 第 2 层：告警
                cand = m.group(1)
                if cand in NOT_NAME or cand in names:
                    continue
                soft_hits.append((p.relative_to(ROOT), i, cand, line.strip()[:80]))

    print(f"隐私自检：扫描 {len(files)} 个跟踪文件")
    if names:
        print(f"  黑名单 {len(names)} 条已生效（内容不打印）")
    else:
        print("  黑名单为空——本机想启用硬检查，请设置 FENG_PRIVATE_NAMES 或 "
              "scripts/local_paths.json 的 private_names")

    if hard_hits:
        print(f"\n✗ 命中黑名单 {len(hard_hits)} 处：")
        for p, i, nm in hard_hits[:20]:
            print(f"    {p}:{i}")
        print("  这些必须删掉/改写后才能提交。")
    else:
        print("  ✓ 黑名单：无命中")

    if soft_hits:
        print(f"\n? 疑似人名的署名上下文 {len(soft_hits)} 处（人工确认，不阻塞）：")
        for p, i, cand, ctx in (soft_hits if args.verbose else soft_hits[:8]):
            print(f"    {p}:{i}  「{cand}」  {ctx}")
        if not args.verbose and len(soft_hits) > 8:
            print(f"    …还有 {len(soft_hits) - 8} 处，用 --verbose 全看")
    else:
        print("  ✓ 署名上下文：未发现疑似中文人名")

    return 1 if hard_hits else 0


if __name__ == "__main__":
    raise SystemExit(main())
