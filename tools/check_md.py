"""Sanity checks for the project's markdown files.
1. code fences must be balanced
2. file / directory paths written in `backticks` must exist
3. flags numbers that look stale (old partition offsets, old baud in protocol sections...)
"""
import re
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[1]
TICK = chr(96)
DOCS = [ROOT / "README.md", ROOT / "DELIVERY.md", ROOT / "COMPARISON.md",
        ROOT / "CHANGELOG.md", ROOT / "esp32s3-feng-llm" / "README.md"]
DOCS += sorted((ROOT / "student").glob("*/MODEL_CARD.md"))

# strings that must not appear any more (superseded facts)
STALE = ["0x310000", "0x1A10000", "stage_32k", "29.66", "MQA(1 KV) / head", "3 MB 分区",
         "COM5 ", "you: / feng:", "MAX_CTX=256", "每 +512 需 +10 MB"]

# 这些路径只存在于 Release 压缩包里，不在仓库中，文档引用它们是合法的
RELEASE_PREFIXES = ("weights/", "datasets/", "feng-30m-v3/", "feng-30m-v3.6/")

PATH_RE = re.compile(TICK + r"([^" + TICK + r"\n]+)" + TICK)
FILE_RE = re.compile(r"^[\w./\\-]+\.(md|py|c|exe|json|npy|gguf|bin|csv|ps1|txt|safetensors)$")


def main():
    bad_paths, bad_fences, stale_hits = [], [], []
    for d in DOCS:
        txt = d.read_text(encoding="utf-8")
        rel = d.relative_to(ROOT)
        fences = len(re.findall(r"^```", txt, re.M))
        if fences % 2:
            bad_fences.append((rel, fences))
        for s in STALE:
            if s in txt:
                stale_hits.append((rel, s))
        for m in PATH_RE.finditer(txt):
            s = m.group(1).strip()
            if s.startswith("$") or " " in s or "<" in s or ">" in s:
                continue
            if "/" not in s and "\\" not in s:
                continue          # bare filename in prose, not a project path
            is_dir = s.endswith("/")
            if not (is_dir or FILE_RE.match(s)):
                continue
            cand = s.replace("\\", "/")
            if cand.startswith(RELEASE_PREFIXES):
                continue                      # Release 包内路径，不在仓库里
            cand = cand.rstrip("/")
            probes = [ROOT / cand, ROOT / "esp32s3-feng-llm" / cand, ROOT.parent / cand,
                      (d.parent / cand)]          # relative to the doc itself
            if not any(p.exists() for p in probes):
                bad_paths.append((rel, s))
    print(f"checked {len(DOCS)} markdown files")
    print(f"code fences : {'OK' if not bad_fences else bad_fences}")
    print(f"stale facts : {'none' if not stale_hits else stale_hits}")
    if bad_paths:
        print("missing paths:")
        for rel, s in bad_paths:
            print(f"   {rel} -> {s}")
    else:
        print("referenced paths: all exist")
    return 1 if (bad_fences or stale_hits or bad_paths) else 0


if __name__ == "__main__":
    raise SystemExit(main())
