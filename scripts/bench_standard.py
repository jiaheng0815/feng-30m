"""标准基准评测（lm-evaluation-harness）：英文 0-shot。

默认只跑 **sciq / piqa**（README 头条数字用这两项）。`--full` 会加回
arc_easy / arc_challenge / hellaswag / winogrande / openbookqa / boolq；
这些任务在 v3.19 上的结果大多贴随机线，其中 BoolQ 37.8%、OpenBookQA 14.0%、
ARC-Challenge 18.8% 是**显著低于随机线**的答案位置/是否偏置，不是“没信号”，
删除它们不应该被理解成“这些能力没问题”。

**SciQ 是开卷**：lm-eval 的 0-shot prompt 把 `support`（含答案句的证据段）拼在
问题前面，所以 71% 是“带证据的阅读理解”，不是闭卷知识分。README/模型卡必须
按这个口径描述，不能与闭卷 SciQ 直接比较。

口径：0-shot、dtype=float32、prefix_token_id=0（自训 tokenizer 没有 bos/eos，
lm-eval 默认会取到 None 而崩溃）；**不套 chat template**，所以测的是 base-LM
续写概率，不是部署时的对话分布。

用法：
    python scripts/bench_standard.py --model v3_19/pc4 --tag v3_19_pc4
    python scripts/bench_standard.py --model v3_19/board6 --tag v3_19_board6 --limit 100
    python scripts/bench_standard.py --model v3_19/pc4 --tag v3_19_full --full

结果写 eval/lm_eval_<tag>*.json（lm-eval 会自动加时间戳）；命令与版本一并打印。
"""
import argparse
import subprocess
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = Path(__file__).resolve().parents[1]

TASKS = "sciq,piqa"
TASKS_FULL = ("sciq,piqa,arc_easy,arc_challenge,hellaswag,winogrande,openbookqa,boolq")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="HF 模型目录（相对仓库根或绝对路径）")
    ap.add_argument("--tag", required=True, help="输出文件名后缀：eval/lm_eval_<tag>.json")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--fewshot", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None, help="每任务只跑前 N 题（冒烟用）")
    ap.add_argument("--full", action="store_true",
                    help="加回已删除的 6 项（结果会显著低于随机线，用于如实留档）")
    args = ap.parse_args()

    model = Path(args.model)
    if not model.is_absolute():
        model = (ROOT / model).resolve()
    out = ROOT / "eval" / f"lm_eval_{args.tag}.json"

    cmd = [
        sys.executable, "-m", "lm_eval",
        "--model", "hf",
        "--model_args", f"pretrained={model},dtype=float32,prefix_token_id=0",
        "--tasks", TASKS_FULL if args.full else TASKS,
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
