"""一键验收：把 AGENTS.md 第 7 节的检查链跑一遍（文档 → PC 构建 → 单测 → 套件 → 一致性）。

用法：
    python tools/check_all.py [--model-export esp32s3-feng-llm/model_export_v3_19b6]

检查项：
  1. tools/check_md.py + tools/check_docs.py（文档数字必须对得上产物）
  2. esp32s3-feng-llm/build_pc_chat.ps1（PC C 引擎构建 + 两个 tool 单测）
  3. 套件（需要 model_export 目录，缺省自动跳过）：
     pc_mt_suite（报名字→身份/名字、12 题连续记忆）
     pc_kv_suite 32 题矩阵 + FENG_SUITE=arith 算术子集
     pc_check（fp32 参考 logits MATCH）
退出码非 0 表示有环节失败。
"""
import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = Path(__file__).resolve().parents[1]
PC = ROOT / "esp32s3-feng-llm" / "pc"


def run(name, cmd, env=None):
    print(f"\n=== {name} ===", flush=True)
    r = subprocess.run(cmd, cwd=str(ROOT), env=env, capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    out = (r.stdout or "") + (r.stderr or "")
    tail = "\n".join(out.strip().splitlines()[-6:])
    if tail:
        print(tail, flush=True)
    ok = r.returncode == 0
    print(f"--- {name}: {'PASS' if ok else 'FAIL'} (exit {r.returncode})", flush=True)
    return ok


def run_expect(name, cmd, needles, env=None):
    """跑一个套件：退出码忽略（模型质量分不等于仓库完整性），但输出里必须出现期望串。"""
    print(f"\n=== {name} ===", flush=True)
    r = subprocess.run(cmd, cwd=str(ROOT), env=env, capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    out = (r.stdout or "") + (r.stderr or "")
    lines = [l for l in out.splitlines() if any(k in l for k in ("SUMMARY", "MATCH", "MISMATCH"))]
    for l in lines[-4:]:
        print("  " + l, flush=True)
    missing = [n for n in needles if n not in out]
    ok = not missing
    print(f"--- {name}: {'PASS' if ok else 'FAIL'}"
          + ("" if ok else f"（缺：{', '.join(missing)}）"), flush=True)
    return ok


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-export",
        default=str(ROOT / "esp32s3-feng-llm" / "model_export_v3_19b6"))
    ap.add_argument("--skip-docs", action="store_true")
    ap.add_argument("--skip-build", action="store_true")
    args = ap.parse_args()

    results = []
    py = sys.executable

    if not args.skip_docs:
        results.append(("check_md", run("文档结构/路径", [py, "tools/check_md.py"])))
        results.append(("check_docs", run("文档数字与产物一致性", [py, "tools/check_docs.py"])))
    if not args.skip_build:
        ps = shutil.which("pwsh") or "powershell"
        results.append(("build_pc", run(
            "PC C 引擎构建 + tool 单测",
            [ps, "-NoProfile", "-ExecutionPolicy", "Bypass", "-File",
             str(ROOT / "esp32s3-feng-llm" / "build_pc_chat.ps1")])))

    exp = Path(args.model_export)
    if not (exp / "model.bin").exists():
        print(f"\n[跳过套件] 找不到 {exp}/model.bin（模型导出产物不入库，属正常）")
    else:
        mt = PC / "pc_mt_suite.exe"
        if mt.exists():
            results.append(("mt_suite", run_expect(
                "多轮回归套件", [str(mt), str(exp)],
                ["SUMMARY mt-suite", "SUMMARY mem12 12/12"])))
        suite = PC / "pc_kv_suite_q2b8.exe"
        prompt = ROOT / "esp32s3-feng-llm" / "pc" / "prompt_long.txt"
        if suite.exists() and prompt.exists():
            results.append(("kv_suite32", run_expect(
                "32 题矩阵（q2 block8）", [str(suite), str(exp), str(prompt), "5200"],
                ["短任务 27/27", "长文召回 4/4"])))
            env = dict(os.environ, FENG_SUITE="arith")
            results.append(("arith21", run_expect(
                "算术子集", [str(suite), str(exp), str(prompt), "5200"],
                ["短任务 21/21"], env=env)))
        check = PC / "pc_check.exe"
        if check.exists() and (exp / "ref_logits.bin").exists():
            results.append(("pc_check", run_expect(
                "fp32 参考 logits",
                [str(check), str(exp), str(ROOT / "logs" / "c_logits_check_all.bin")],
                ["argmax c=5331 ref=5331 MATCH"])))

    print("\n=== 汇总 ===")
    for name, ok in results:
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    bad = [n for n, ok in results if not ok]
    print("结论：" + ("全部通过" if not bad else f"{len(bad)} 项失败：" + ", ".join(bad)))
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main())
