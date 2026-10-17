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
# v4.0 起全部说明文档归到 docs/：根目录只留 README.md（仓库首页）与 AGENTS.md（代理约定）。
# 模型卡跟着模型目录走（archive/v1/*/MODEL_CARD.md 等）。
DOCS = [ROOT / "README.md", ROOT / "AGENTS.md", ROOT / "phone" / "README.md"]
DOCS += sorted((ROOT / "docs").glob("*.md"))
# v4.0 起历史目录已删；模型卡跟着当前模型目录走。
# 用 rglob 而不是 `*/*/MODEL_CARD.md`：v4 的模型目录是 `v4/<阶段>/`（两层），
# 旧 glob 按 archive/v1/<名字>/ 的三层写的，会把 `v4/base/MODEL_CARD.md` 整个漏掉——
# 2026-10-17 就是这么漏掉的（本地自检过、CI 才炸）。别再改回固定层数。
DOCS += sorted((ROOT / "v4").rglob("MODEL_CARD.md"))

# strings that must not appear any more (superseded facts)
STALE = ["0x310000", "0x1A10000", "stage_32k", "29.66", "MQA(1 KV) / head", "3 MB 分区",
         "COM5 ", "you: / feng:", "MAX_CTX=256", "每 +512 需 +10 MB"]

# 这些路径只存在于 Release 压缩包里，不在仓库中，文档引用它们是合法的
RELEASE_PREFIXES = ("weights/", "datasets/", "feng-30m-v3/", "feng-30m-v3.6/")

# 本机训练/编译产物：干净 clone 里没有是正常的（跟踪文件缺失仍会报错）
#
# **v4.0 起仓库只保留 v4 一条线**：v1（原 student/）、v2、v3.19 与全部 v3 历史版本目录
# （原 archive/，52 个）已删除，所需权重收进 v4/base/。因此文档里对这些目录的引用
# **是历史记录，不再指向本机存在的路径**——CHANGELOG 正文按约定保留撰写当时的原文，
# 不去改写它们（改写历史记录比留一个失效路径更糟）。新写的文档请不要再引用这些前缀。
LOCAL_PREFIXES = ("archive/", "v2/", "student/", "v3_", "v3/", "data/", "v4/sft/",
                  "v4/qat/", "model_export_",
                  "esp32s3-feng-llm/model_export_", "logs/", "scripts/local_paths.json",
                  "feng_cont/",            # 续训/恢复实验的训练产物目录（本机外置盘）
                  # 编辑器与代码提示（v4.0）：.vscode/ 是每台机器自己的配置，
                  # compile_commands.json 是 tools/gen_compile_commands.py 的派生数据
                  # （从 IDF 构建产物生成），两者都在 .gitignore 里，
                  # 干净 clone 上必然没有——AGENTS.md 的「编辑器配置」一节会引用它们。
                  ".vscode/", "compile_commands.json", "build/compile_commands.json",
                  "esp32s3-feng-llm/build/",
                  # 一键发布脚本：**故意 gitignore**（内嵌魔搭令牌），干净 clone 上必然没有。
                  # AGENTS.md 的目录地图会提到它——不列进来 CI 就会报"引用了不存在的路径"。
                  "tools/publish_all.py",
                  # 手机端（phone/）的构建产物：干净 clone 里没有是正常的
                  "phone/cli/build/", "phone/cli/build-baseline/", "phone/cli/build-aarch64/",
                  "phone/cli/build-android/", "phone/cli/build-clang/",
                  "phone/android/app/build/", "phone/android/build/",
                  "phone/android/.gradle/", "phone/android/local.properties")

PATH_RE = re.compile(TICK + r"([^" + TICK + r"\n]+)" + TICK)
FILE_RE = re.compile(r"^[\w./\\-]+\.(md|py|c|exe|json|npy|gguf|bin|csv|ps1|txt|safetensors)$")


def normalize(cand: str) -> str:
    """把文档里的路径规整成「相对仓库根」的形式。

    **别用 `lstrip("./")`**：它会把 `.vscode/` 的前导点一起吃掉（变成 "vscode"），
    于是 LOCAL_PREFIXES 里的 `.vscode/` 就永远匹配不上，干净 clone 上会误报缺失。
    这里只逐段剥掉 `../` 与 `./`，保留目录名自己的前导点。
    """
    c = cand.replace("\\", "/").strip().lstrip("/").rstrip("/")
    while c.startswith("../"):
        c = c[3:]
    while c.startswith("./"):
        c = c[2:]
    return c


def tracked_files(root: Path) -> set:
    import subprocess
    try:
        out = subprocess.run(["git", "-C", str(root), "ls-files", "-z"],
                             capture_output=True, check=False).stdout
    except Exception:
        return set()
    return {x.decode("utf-8", "replace") for x in out.split(b"\0") if x}


def is_local_artifact(cand: str) -> bool:
    c = normalize(cand)
    if c.endswith((".exe", ".o", ".dll")):
        return True
    for p in LOCAL_PREFIXES:
        if c == p.rstrip("/") or c.startswith(p):
            return True
    return False


def main():
    bad_paths, bad_fences, stale_hits = [], [], []
    tracked = tracked_files(ROOT)
    skipped = 0
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
            if "*" in s:
                continue          # 通配写法（如 v3_5*/）不逐字校验
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
                if normalize(cand) not in tracked and is_local_artifact(cand):
                    skipped += 1
                    continue          # 本机产物（权重/日志/exe），干净 clone 里允许缺
                bad_paths.append((rel, s))
    print(f"checked {len(DOCS)} markdown files")
    print(f"code fences : {'OK' if not bad_fences else bad_fences}")
    print(f"stale facts : {'none' if not stale_hits else stale_hits}")
    print(f"local-only  : {skipped} 条本机产物路径（允许缺）")
    if bad_paths:
        print("missing paths:")
        for rel, s in bad_paths:
            print(f"   {rel} -> {s}")
    else:
        print("referenced paths: all exist")
    return 1 if (bad_fences or stale_hits or bad_paths) else 0


if __name__ == "__main__":
    raise SystemExit(main())
