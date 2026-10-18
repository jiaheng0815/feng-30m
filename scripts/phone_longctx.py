"""在手机上跑预 tokenized 的长文针检索（`--ids-file` 路径）。

前置：
  1. `python scripts/make_longctx_ids.py` 生成 `eval/longctx_ids/`；
  2. 手机上有编译好的 `phone_chat`（`FENG_PHONE_MAX_CTX=32768`），
     且 `/data/local/tmp/feng/` 里已有 model.bin / tokenizer.bin；
  3. `adb push eval/longctx_ids /data/local/tmp/feng/ids`。

注意：prefill 是 O(n²)。P30 Pro 上 4k 约 107 秒/题，32k 约 2 小时/题；
默认只跑 4k，32k 请显式 `--ctx 32768` 并自己安排时间。

用法：
    python scripts/phone_longctx.py --ctx 4096 --limit 4
"""
import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adb", default="adb")
    ap.add_argument("--device-dir", default="/data/local/tmp/feng")
    ap.add_argument("--binary", default="./phone_chat32k")
    ap.add_argument("--ids-dir", default=str(ROOT / "eval" / "longctx_ids"))
    ap.add_argument("--ctx", default="4096")
    ap.add_argument("--limit", type=int, default=4, help="每个 ctx 最多跑几条（0=全部）")
    ap.add_argument("--max-new", type=int, default=16)
    args = ap.parse_args()

    meta = json.loads((Path(args.ids_dir) / "meta.json").read_text(encoding="utf-8"))
    want = {int(x) for x in args.ctx.split(",")}
    rows = [m for m in meta if m["ctx"] in want]
    if args.limit > 0:
        out, seen = [], {}
        for m in rows:
            k = m["ctx"]
            if seen.get(k, 0) < args.limit:
                out.append(m)
                seen[k] = seen.get(k, 0) + 1
        rows = out

    for m in rows:
        cmd = (f"cd {args.device_dir} && {args.binary} --ctx {m['ctx']} "
               f"--max-new {args.max_new} --ids-file ids/{m['file']} .")
        r = subprocess.run([args.adb, "shell", cmd], capture_output=True, text=True,
                           encoding="utf-8", errors="replace", timeout=9000)
        text = (r.stdout or "") + (r.stderr or "")
        if m["negative"]:
            ok = any(h in text for h in ("没有提到", "未提到", "文中没有", "没有找到"))
        else:
            ok = m["code"] in text
        tm = re.search(r"prefill\s+(\d+)\s+ms", text)
        pre = f"{int(tm.group(1))/1000:.1f}s" if tm else "?"
        print(f"[{'OK' if ok else 'FAIL'}] ctx={m['ctx']} {m['file']} "
              f"prefill={pre} code={m['code']}")
        if not ok:
            print("  " + text.strip().splitlines()[-1][:160])


if __name__ == "__main__":
    main()
