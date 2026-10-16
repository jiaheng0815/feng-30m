#!/usr/bin/env python
"""生成仓库根目录的 compile_commands.json —— 给 clangd / C/C++ 扩展做**精确**的代码提示。

为什么要有这个脚本
------------------
C23 核心（`esp32s3-feng-llm/main/*.c`）要看见几十条 ESP-IDF 的 include 路径
（freertos / driver / esp_system / newlib…）才能正确解析，还要拿到
`build/config/sdkconfig.h` 和 `-DFENG_KV_Q2` 这类只在构建期存在的宏。
手写进 `c_cpp_properties.json` 一定会随 IDF 升级而失效；而 ESP-IDF 自己生成的
`esp32s3-feng-llm/build/compile_commands.json` 里已经有**逐文件精确**的命令。

本脚本把它的 `main/` 条目抽出来，补上 PC 驱动 / 手机引擎 / CUDA / Android JNI
四组手工条目，合并成**一份**放仓库根：clangd 从被打开的文件逐级向上找
`compile_commands.json`，根目录是所有子树的共同祖先，一份就覆盖全部代码。

生成物 `compile_commands.json` 不入库（是派生数据，且有几 MB）。

用法
----
    python tools/gen_compile_commands.py           # 生成
    python tools/gen_compile_commands.py --check   # 只报告会生成什么，不写盘

前提
----
固件侧需要先在 `esp32s3-feng-llm/` 下跑过一次 `idf.py build`；没有那次构建，
固件条目会退化成"只有 -Imain"，其余四组不受影响。
"""
from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "scripts"))
import paths  # noqa: E402  （和全项目一致：路径解析只走 scripts/paths.py）

CORE = ROOT / "esp32s3-feng-llm" / "main"
PC = ROOT / "esp32s3-feng-llm" / "pc"
ENGINE = ROOT / "phone" / "engine"
JNI = ROOT / "phone" / "android" / "app" / "src" / "main" / "cpp"
IDF_CC = ROOT / "esp32s3-feng-llm" / "build" / "compile_commands.json"
OUT = ROOT / "compile_commands.json"

# 与 build_pc_chat.ps1 / pc_kv_suite 的验收口径一致（发布配置是 q2 / block8）
Q2 = ["-DFENG_KV_Q2=1", "-DFENG_KV_Q2_BLOCK=8"]


def _p(path: Path) -> str:
    """compile_commands.json 里一律用正斜杠：反斜杠在命令行里是转义字符。"""
    return path.as_posix()


def find_msys2_include() -> Path | None:
    """从 PATH 上的 gcc 反推 MSYS2 的 include 目录。

    手机引擎的 `feng_smp.c` 用 pthread、`feng_phone.c` 用 semaphore.h，
    在 Windows 上这些头由 MSYS2 的 ucrt64 提供（winpthreads）。
    """
    for exe in ("gcc", "g++", "cc"):
        found = shutil.which(exe)
        if not found:
            continue
        inc = Path(found).resolve().parent.parent / "include"
        if (inc / "pthread.h").exists():
            return inc
    return None


def find_cuda_include() -> Path | None:
    """定位 CUDA 的 include 目录（`feng_cuda.cu` 需要 cuda_runtime.h）。"""
    cands: list[Path] = []
    env = os.environ.get("CUDA_PATH")
    if env:
        cands.append(Path(env) / "include")
    # NVIDIA 的固定安装位置；有多个版本时取版本号最大的那个
    base = Path("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA")
    if base.is_dir():
        try:
            vers = sorted((d for d in base.iterdir() if d.is_dir()), reverse=True)
        except OSError:
            vers = []
        cands += [d / "include" for d in vers]
    for c in cands:
        if (c / "cuda_runtime.h").exists():
            return c
    return None


def find_ndk() -> Path | None:
    """可选的 Android NDK（只有手机端 JNI 层用得到 jni.h）。"""
    for key in ("ANDROID_NDK_HOME", "NDK_HOME", "ANDROID_NDK_ROOT"):
        v = os.environ.get(key)
        if v and Path(v).is_dir():
            return Path(v)
    return None


def idf_entries() -> tuple[list[dict], int]:
    """抽取 ESP-IDF 生成的 main/ 条目。

    保留 IDF 原始 `command` 字符串而不拆成 arguments：里面有
    `-DIDF_VER=\\"v5.5.5-dirty\\"` 这类带转义的参数，自己拆一定拆错。
    """
    if not IDF_CC.exists():
        return [], 0
    try:
        data = json.loads(IDF_CC.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"  ! 读不了 {IDF_CC}: {exc}")
        return [], 0

    core = CORE.resolve()
    out, skipped = [], 0
    for e in data:
        f = Path(e["file"])
        if not f.is_absolute():
            f = Path(e["directory"]) / f
        try:
            f = f.resolve()
        except OSError:
            continue
        # 只留 main/ 下的——另外 900 多条是 IDF 组件自己的源码，与本仓库无关
        if f.parent != core:
            continue
        if not f.exists():
            # 构建后删过文件（比如换了 SRCS），跳过而不是留一条指向空路径的坏条目
            skipped += 1
            continue
        out.append({
            "directory": _p(Path(e["directory"]).resolve()),
            "file": _p(f),
            "command": e["command"],
        })
    return out, skipped


def manual_entries() -> list[dict]:
    """四组手工条目：PC 驱动 / 手机引擎 / Android JNI / CUDA。"""
    out: list[dict] = []

    def add(files: list[Path], argv: list[str], note: str) -> None:
        for f in files:
            out.append({
                "directory": _p(f.parent),
                "file": _p(f),
                "arguments": argv,
                "_note": note,
            })

    # --- PC 驱动（C++23）---------------------------------------------------
    # 只需要 -I 核心目录：feng.h 在非 ESP_PLATFORM 下不拉任何 IDF 头
    # （`esp_attr.h` 那一行在 `#if defined(ESP_PLATFORM)` 里）。
    add(sorted(PC.glob("*.cpp")),
        ["clang++", "-std=c++23", "-fno-exceptions", "-fno-rtti",
         "-fno-threadsafe-statics", "-O2", *Q2, f"-I{_p(CORE)}"],
        "pc")

    # --- 手机引擎（C23 + pthread）------------------------------------------
    msys2_inc = find_msys2_include()
    eng_argv = ["clang", "-std=c23", "-O2", f"-I{_p(ENGINE)}"]
    if msys2_inc:
        eng_argv += [f"-I{_p(msys2_inc)}"]
    eng_files = sorted(ENGINE.glob("*.c"))
    add(eng_files, eng_argv, "phone/engine")

    # feng_neon.c 的 `<arm_neon.h>` 在 `#if defined(__aarch64__)` 里，
    # x86 主机上这一整块被预处理跳过，所以不需要 arm 的 sysroot 也能正常提示。

    # --- Android JNI 层（复用手机引擎的头）----------------------------------
    jni = JNI / "feng_jni.c"
    if jni.exists():
        ndk = find_ndk()
        jni_argv = ["clang", "-std=c23", "-O2", f"-I{_p(ENGINE)}"]
        if ndk:
            for abi in ("arm64-v8a",):
                sysroot = ndk / "toolchains" / "llvm" / "prebuilt"
                for pre in sysroot.glob("*"):
                    inc = pre / "sysroot" / "usr" / "include"
                    if inc.is_dir():
                        jni_argv += [f"-I{_p(inc)}"]
                        break
            jni_argv.append("--target=aarch64-linux-android")
        add([jni], jni_argv, "android/jni")

    # --- CUDA 后端（nvcc 侧是 C++）-----------------------------------------
    cuda_inc = find_cuda_include()
    cu = CORE / "feng_cuda.cu"
    if cu.exists() and cuda_inc:
        add([cu],
            ["clang++", "-x", "cuda", "-std=c++20", "--cuda-path=" + _p(cuda_inc.parent),
             f"-I{_p(cuda_inc)}", f"-I{_p(CORE)}"],
            "cuda")

    return out


def main() -> int:
    check_only = "--check" in sys.argv[1:]

    print(f"仓库根: {ROOT}")
    print()

    idf, skipped = idf_entries()
    if idf:
        print(f"  固件 C23 核心   : {len(idf):3d} 条（来自 IDF 的 build/compile_commands.json）")
    else:
        print("  固件 C23 核心   :   0 条 —— 还没跑过 idf.py build，")
        print("                    固件侧只能退化成基础提示（其余组不受影响）")

    manual = manual_entries()
    by_note: dict[str, int] = {}
    for e in manual:
        by_note[e["_note"]] = by_note.get(e["_note"], 0) + 1
    for note, n in by_note.items():
        print(f"  {note:<16}: {n:3d} 条")

    if skipped:
        print(f"  （跳过 {skipped} 条指向已删除文件的旧条目）")

    # _note 只是给人看的，不写进产物
    for e in manual:
        e.pop("_note", None)
    merged = idf + manual

    print()
    if check_only:
        print(f"--check：将生成 {len(merged)} 条到 {OUT.name}（未写盘）")
        return 0

    OUT.write_text(json.dumps(merged, ensure_ascii=False, indent=1), encoding="utf-8")
    size_mb = OUT.stat().st_size / 1024 / 1024
    print(f"已写入 {OUT}（{len(merged)} 条，{size_mb:.1f} MB）")
    print("重开一次 VS Code（或命令面板 → clangd: Restart language server）即可生效。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
