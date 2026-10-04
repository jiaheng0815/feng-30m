"""模型质量套餐：一条命令跑完 5 个既有评测，落 JSON 到 eval/ 并打印记分卡。

跑的评测（全部是仓库里已有的口径，不新增标准）：
  scope      范围/身份/寒暄/任务（eval_planA_scope.py）
  identity   身份 12 题（eval_identity.py）
  probe42    日常对话探针 42 题（chat_probe.py）
  heldout30  留出 30 题（chat_probe_heldout.py）
  memory24   多轮记忆 24 题（eval_memory.py）

用法：
    python scripts\\eval_suite.py --model v3_19\\pc1 --tag v3_19pc1
    python scripts\\eval_suite.py --model v3_19\\pc1 --tag v3_19pc1 --only heldout30,probe42

结果 JSON：eval/<tag>_<suite>.json；汇总：eval/<tag>_suite.json
注意：留出 30 题已被用于多轮迭代，只当"开发集"看，不能当无偏测试集宣传。
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import ROOT  # noqa: E402

EVAL = ROOT / "eval"
PROBE_FLAGS = ("blurb_leak", "topic_miss", "empty", "loop")


def parse_score(name: str, path: Path):
    """把各评测的 JSON 统一成 (ok, n)。"""
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):                      # scope：行列表
        judged = [r for r in data if r.get("ok") is not None]
        return sum(1 for r in judged if r.get("ok")), len(judged)
    if "score" in data:                             # "12/12" 形式
        a, b = data["score"].split("/")
        return int(a), int(b)
    if "ok" in data and "n" in data:                # heldout
        return int(data["ok"]), int(data["n"])
    rows = data.get("rows", [])                     # probe42
    ok = sum(1 for r in rows if not any(r.get(f) for f in PROBE_FLAGS))
    return ok, len(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tag", required=True, help="输出文件名前缀，如 v3_19pc1")
    ap.add_argument("--only", default="", help="只跑其中几个（逗号分隔）")
    args = ap.parse_args()
    only = {s for s in args.only.split(",") if s}
    py = sys.executable
    def dest(suffix: str) -> str:
        return str(EVAL / f"{args.tag}_{suffix}.json")

    suites = [
        ("scope", [str(ROOT / "scripts" / "eval_planA_scope.py"), args.model,
                   dest("scope")], dest("scope")),
        ("identity", [str(ROOT / "scripts" / "eval_identity.py"), args.model,
                      dest("identity")], dest("identity")),
        ("probe42", [str(ROOT / "scripts" / "chat_probe.py"), "--model", args.model,
                     "--out", dest("probe42")], dest("probe42")),
        ("heldout30", [str(ROOT / "scripts" / "chat_probe_heldout.py"), "--model", args.model,
                       "--out", dest("heldout30")], dest("heldout30")),
        ("memory24", [str(ROOT / "scripts" / "eval_memory.py"), "--model", args.model,
                      "--out", dest("memory24"), "--n", "24"], dest("memory24")),
    ]
    summary, failed = {}, []
    print(f"=== 评测套餐：{args.tag}（model={args.model}）===", flush=True)
    for name, cmd, out_path in suites:
        if only and name not in only:
            continue
        t0 = time.time()
        r = subprocess.run([py, *cmd], cwd=str(ROOT), capture_output=True, text=True,
                           encoding="utf-8", errors="replace")
        out = (r.stdout or "") + (r.stderr or "")
        dt = time.time() - t0
        score = parse_score(name, Path(out_path))
        if r.returncode != 0 or score is None:
            failed.append(name)
            tail = "\n".join(out.strip().splitlines()[-4:])
            print(f"  {name:10s} FAIL ({dt:.0f}s)\n{tail}", flush=True)
            continue
        ok, n = score
        summary[name] = {"ok": ok, "n": n, "json": str(Path(out_path).relative_to(ROOT))}
        print(f"  {name:10s} {ok}/{n}  ({dt:.0f}s)", flush=True)
    dest = EVAL / f"{args.tag}_suite.json"
    dest.write_text(json.dumps({"model": args.model, "suites": summary,
                                "failed": failed}, ensure_ascii=False, indent=2),
                    encoding="utf-8")
    print(f"-> {dest.relative_to(ROOT)}")
    if failed:
        print(f"失败：{', '.join(failed)}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
