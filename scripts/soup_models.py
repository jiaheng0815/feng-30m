"""把两个同源微调模型的权重平均（model soup），取长补短。

用途：v3.1c 的正样本检索最好、v3.1d 的"不胡说"（负样本拒答）最好，
两者从同一份权重出发，权重插值通常能同时保留两种能力。

用法：
    python scripts/soup_models.py --models a,b --weights 0.5,0.5 --out out_dir
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import ROOT  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", required=True, help="逗号分隔的 HF 目录（同源微调）")
    ap.add_argument("--weights", default="", help="逗号分隔的权重，默认等权")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    dirs = [Path(m.strip()) for m in args.models.split(",")]
    ws = ([float(w) for w in args.weights.split(",")] if args.weights
          else [1.0 / len(dirs)] * len(dirs))
    if len(ws) != len(dirs):
        raise SystemExit("--weights 数量必须与 --models 相同")
    for d in dirs:
        if not (d / "model.safetensors").exists():
            raise SystemExit(f"{d} 缺少 model.safetensors")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    states = [load_file(str(d / "model.safetensors")) for d in dirs]
    keys = list(states[0].keys())
    for st in states[1:]:
        if list(st.keys()) != keys:
            raise SystemExit("两个模型的张量列表不一致，无法插值")

    merged = {}
    for k in keys:
        t = states[0][k]
        if t.dtype.is_floating_point:
            acc = torch.zeros_like(t, dtype=torch.float32)
            for st, w in zip(states, ws):
                acc += st[k].to(torch.float32) * w
            merged[k] = acc.to(t.dtype)
        else:
            merged[k] = t.clone()
    save_file(merged, str(out / "model.safetensors"))

    for extra in ("config.json", "generation_config.json", "tokenizer.json",
                  "tokenizer_config.json", "chat_template.jinja"):
        src = dirs[0] / extra
        if src.exists():
            shutil.copy2(src, out / extra)
    (out / "SOUP.json").write_text(json.dumps(
        {"models": [str(d) for d in dirs], "weights": ws}, ensure_ascii=False, indent=2),
        encoding="utf-8")
    print(f"已合并 {len(dirs)} 个模型 -> {out}（权重 {ws}）")


if __name__ == "__main__":
    main()
