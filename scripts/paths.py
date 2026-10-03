"""集中解析项目路径，脚本里不再出现硬编码盘符。

解析优先级：**环境变量 > scripts/local_paths.json > 从本文件位置推导的默认值**。

环境变量：
    FENG_ROOT           项目根目录（默认：本文件的上一级目录）
    FENG_DATA_DIR       同级项目 feng-ai-qwen35 的数据目录（v1 蒸馏用的原始数据）
    FENG_LLAMA_DIR      llama.cpp 仓库目录（GGUF 转换 / 量化 / benchmark）
    FENG_PY             Python 解释器（默认：当前解释器 sys.executable）
    FENG_TEACHER_GGUF   v1 教师模型 feng-0.8b 的 GGUF 路径

也可以不设环境变量，改为在仓库里放一份机器本地配置（已 gitignore，不会提交）：
    scripts/local_paths.json
    {"llama_dir": "...", "data_dir": "...", "python": "...", "teacher_gguf": "..."}
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

try:                                              # 让中文报错在 UTF-8 终端里正常显示
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

HERE = Path(__file__).resolve()
ROOT = Path(os.environ.get("FENG_ROOT") or HERE.parents[1])

# 同级项目：v1 的教师权重与原始数据都在那里
SIBLING = ROOT.parent / "feng-ai-qwen35"

_LOCAL_CONFIG = HERE.parent / "local_paths.json"
_local: dict[str, str] = {}
if _LOCAL_CONFIG.exists():
    try:
        _local = json.loads(_LOCAL_CONFIG.read_text(encoding="utf-8"))
    except Exception as exc:                      # 配置写坏了要提示，而不是静默忽略
        print(f"[paths] 警告：{_LOCAL_CONFIG} 解析失败（{exc}），改用默认路径", file=sys.stderr)


def _resolve(env: str, key: str, *candidates: Path) -> Path:
    """按 环境变量 > 本地配置 > 候选路径（取第一个存在的）的顺序解析。"""
    if os.environ.get(env):
        return Path(os.environ[env])
    if _local.get(key):
        return Path(_local[key])
    for cand in candidates:
        if cand.exists():
            return cand
    return candidates[0]                          # 都不存在时返回首选，便于报错时给出提示


DATA_DIR = _resolve("FENG_DATA_DIR", "data_dir", SIBLING / "data")

# llama.cpp：优先环境变量 / 本地配置，其次找常见的同级目录
LLAMA_DIR = _resolve(
    "FENG_LLAMA_DIR", "llama_dir",
    ROOT.parent / "llama.cpp",                    # <父目录>/llama.cpp
    ROOT / "llama.cpp",                           # <仓库>/llama.cpp
    ROOT.parent / "llama.cpp-master",
)
LLAMA_BIN = LLAMA_DIR / "build" / "bin"
CONVERT_HF_TO_GGUF = LLAMA_DIR / "convert_hf_to_gguf.py"

# Python 解释器：默认就用当前解释器（用 venv 的 python 跑脚本时天然正确）
PY = Path(os.environ.get("FENG_PY") or _local.get("python") or sys.executable)

# v1 教师权重（feng-0.8b bf16 GGUF）
TEACHER_GGUF = _resolve(
    "FENG_TEACHER_GGUF", "teacher_gguf",
    SIBLING / "quant" / "gguf" / "feng-bf16.gguf",
)


def require(path: Path, what: str, env: str = "") -> Path:
    """路径不存在时给出可操作的报错（含环境变量提示）。"""
    if not path.exists():
        hint = f"，或用环境变量 {env} 指定" if env else ""
        raise FileNotFoundError(f"{what} 不存在：{path}\n请检查路径{hint}（见 scripts/paths.py 顶部说明）")
    return path


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    rows = [
        ("FENG_ROOT (ROOT)", ROOT, "FENG_ROOT"),
        ("数据目录", DATA_DIR, "FENG_DATA_DIR"),
        ("llama.cpp", LLAMA_DIR, "FENG_LLAMA_DIR"),
        ("llama.cpp/bin", LLAMA_BIN, ""),
        ("Python", PY, "FENG_PY"),
        ("v1 教师 GGUF", TEACHER_GGUF, "FENG_TEACHER_GGUF"),
    ]
    print(f"本地配置 {_LOCAL_CONFIG}：" + ("已加载" if _local else "无"))
    for name, path, env in rows:
        mark = "OK  " if Path(path).exists() else "缺失"
        print(f"{mark} {name:18} {path}" + (f"    [env {env}]" if env else ""))


if __name__ == "__main__":
    main()
