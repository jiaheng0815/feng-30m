"""标准基准评测（lm-evaluation-harness）：8 个内置英文任务 + 自定义 SciCloze-900。

任务：**只跑 sciq / piqa**。

2026-10-15 起删掉了 arc_easy / arc_challenge / hellaswag / winogrande / openbookqa /
boolq / scicloze_900 —— 它们在 v3.19/pc4 上全部贴近或低于随机线（BoolQ 37.8% 甚至低于
二选一随机，只有 yes/no 偏置），对这个 30M 中文对话模型**测不出任何能力差异**。
留着一堆没信号的分数只会掩盖真实水平。原始结果仍在 eval/lm_eval_feng_v3_19_pc4.json 里。

口径：0-shot、dtype=float32、prefix_token_id=0（自训 tokenizer 没有 bos/eos，
lm-eval 默认会取到 None 而崩溃）。

用法：
    python scripts/bench_standard.py --model v3_19/pc4 --tag v3_19_pc4
    python scripts/bench_standard.py --model v3_19/board6 --tag v3_19_board6 --limit 100

结果写 eval/lm_eval_<tag>.json；命令与版本一并打印，便于复现。
"""
import argparse
import subprocess
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = Path(__file__).resolve().parents[1]

TASKS = "sciq,piqa"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="HF 模型目录（相对仓库根或绝对路径）")
    ap.add_argument("--tag", required=True, help="输出文件名后缀：eval/lm_eval_<tag>.json")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--fewshot", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None, help="每任务只跑前 N 题（冒烟用）")
    args = ap.parse_args()

    model = Path(args.model)
    if not model.is_absolute():
        model = (ROOT / model).resolve()
    out = ROOT / "eval" / f"lm_eval_{args.tag}.json"

    cmd = [
        sys.executable, "-m", "lm_eval",
        "--model", "hf",
        "--model_args", f"pretrained={model},dtype=float32,prefix_token_id=0",
        "--tasks", TASKS,
        "--num_fewshot", str(args.fewshot),
        "--batch_size", str(args.batch_size),
        "--output_path", str(out),
    ]
    # 自建任务目录（曾放 scicloze_900.yaml，2026-10-15 随该任务一并删除）。
    # 现在跑的都是 lm-eval 内置任务，但保留这个分支：以后再造自定义任务时把 yaml
    # 放进 eval/lm_eval_tasks/ 就会自动生效。
    include_dir = ROOT / "eval" / "lm_eval_tasks"
    if include_dir.is_dir():
        cmd += ["--include_path", str(include_dir)]
    if args.limit:
        cmd += ["--limit", str(args.limit)]

    print("复现命令：", flush=True)
    print("  " + " ".join(cmd), flush=True)
    print("lm-eval 版本：", flush=True)
    subprocess.call([sys.executable, "-m", "pip", "show", "lm-eval"])
    return subprocess.call(cmd)


if __name__ == "__main__":
    raise SystemExit(main())
