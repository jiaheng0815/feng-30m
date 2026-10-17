#!/usr/bin/env python
"""用**真实提示词**测教师服务的吞吐，逐个配置对比。

为什么不用 scripts/bench_teacher_conc.py：那个脚本用的是 8 条五个字的单轮问题
（"介绍一下杭州"），prefill 几乎为零，测出来的是纯 decode 上限（59.3 tok/s）；
生产负载是 65~170 token 的提示词 + max_tokens 200，固定开销摊不掉，只有 37.8。
本脚本直接读 data/v4_prompts.jsonl 的前 N 条，按 v4_teacher_gen.py 完全一样的
方式组消息、同样的 4 个 worker、同样的 max_tokens=200——测出来的数字和生产
日志里的 tok/s 可以直接比。

用法：
    python tools/bench_teacher_real.py                 # 跑全部配置
    python tools/bench_teacher_real.py --only cur      # 只跑一个
    python tools/bench_teacher_real.py --n 40 --workers 4
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent
EXE = Path(os.environ.get("NINFER_EXE") or r"D:\ai-ninfer\build\b1\apps\ninfer-serve.exe")
MODEL = Path(os.environ.get("NINFER_MODEL") or r"D:\ai-ninfer\artifacts\swift-bonsai2-27b-pruned.ninfer")
LOG_DIR = ROOT / "logs"
BASE = "http://127.0.0.1:8080/v1"

PROMPTS = ROOT / "data" / "v4_prompts.jsonl"

MAX_CTX = 8192
COMMON = [
    "--host", "127.0.0.1", "--port", "8080", "--model-id", "bonsai2-27b",
    "--max-context", str(MAX_CTX), "--kv-dtype", "nvfp4",
    "--prefill-chunk", "8192", "--max-pending-requests", "32",
    "--vision", "--cors",
]
SPEC = ["--spec", "mtp", "--draft-tokens", "3", "--lm-head-draft"]


def conc_args(conc: int, host_kv: int = 0, slots: int | None = None) -> list[str]:
    """并发相关参数。引擎约束：kv_capacity <= max_context * max_concurrency。"""
    s = conc if slots is None else slots
    return ["--max-concurrency", str(conc),
            "--kv-capacity", str(MAX_CTX * conc),
            "--host-kv-mib", str(host_kv), "--host-state-slots", "0",
            "--device-state-slots", str(s),
            "--max-private-continuations", str(s), "--max-shared-prefixes", "4"]


# 生产实际在跑的（从 ninfer-serve.exe 的命令行抄下来）——基准线
PROD = [
    "--host", "127.0.0.1", "--port", "8080", "--model-id", "bonsai2-27b",
    "--max-context", "8192", "--kv-capacity", "8192", "--kv-dtype", "nvfp4",
    "--prefill-chunk", "8192", "--max-concurrency", "1", "--max-pending-requests", "32",
    "--spec", "mtp", "--draft-tokens", "3", "--lm-head-draft",
    "--host-kv-mib", "1024", "--device-state-slots", "8",
    "--max-private-continuations", "8", "--max-shared-prefixes", "4",
    "--vision", "--cors",
]

CONFIGS: dict[str, list[str]] = {
    # 现在生产在跑的：1 GiB host KV + 8 个 state 槽（agent 用配置）
    "cur": PROD,
    # 10/15 扫出 59.3 的那个：host KV 关掉、槽位收到 1
    "sweep1": COMMON + conc_args(1) + SPEC,
    # 同上但开并发，让 prefill 和 decode 重叠
    "sweep2": COMMON + conc_args(2) + SPEC,
    "sweep4": COMMON + conc_args(4) + SPEC,
}


# ------------------------------------------------------------------ 服务

def kill_server(wait_idle: bool = True) -> None:
    subprocess.run(["taskkill", "/IM", "ninfer-serve.exe", "/F"], capture_output=True)
    if not wait_idle:
        return
    for _ in range(60):                      # 等显存回落 / 端口放开
        time.sleep(0.5)
        try:
            urllib.request.urlopen(BASE + "/models", timeout=1)
        except Exception:
            return


def start_server(args: list[str], logname: str) -> subprocess.Popen | None:
    out = open(LOG_DIR / f"teacher_real_{logname}.err.log", "w", encoding="utf-8")
    p = subprocess.Popen([str(EXE), str(MODEL), *args],
                         stdout=subprocess.DEVNULL, stderr=out)
    for _ in range(180):                     # 最多等 90 s
        time.sleep(0.5)
        if p.poll() is not None:
            out.close()
            tail = (LOG_DIR / f"teacher_real_{logname}.err.log").read_text(
                encoding="utf-8", errors="replace")[-400:]
            print(f"    ✗ 启动失败：{tail.strip().splitlines()[-1] if tail.strip() else '?'}")
            return None
        try:
            urllib.request.urlopen(BASE + "/models", timeout=2)
            t = time.time()
            print(f"    · 服务就绪（{t - t0:.1f}s）")
            return p
        except Exception:
            pass
    p.kill()
    out.close()
    print("    ✗ 启动超时")
    return None


# ------------------------------------------------------------------ 压测

def load_prompts(n: int) -> list[list[dict]]:
    """按 v4_teacher_gen.py 的单轮路径组消息：system + 第一个 user turn。"""
    msgs = []
    for line in PROMPTS.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        it = json.loads(line)
        turns = it.get("turns") or []
        if not turns:
            continue
        msgs.append([{"role": "system", "content": it.get("system", "")},
                     {"role": "user", "content": turns[0]}])
        if len(msgs) >= n:
            break
    return msgs


def call(msgs: list[dict], max_tokens: int) -> tuple[int, int]:
    """返回 (completion_tokens, prompt_tokens)。"""
    body = {"model": "bonsai2-27b", "messages": msgs, "max_tokens": max_tokens,
            "temperature": 0.8, "reasoning_effort": "none"}
    req = urllib.request.Request(BASE + "/chat/completions",
                                 data=json.dumps(body, ensure_ascii=False).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=900) as f:
        d = json.loads(f.read().decode())
    u = d.get("usage", {})
    return u.get("completion_tokens", 0), u.get("prompt_tokens", 0)


def bench(batch: list[list[dict]], workers: int, max_tokens: int) -> dict:
    idx = [0]
    lock = threading.Lock()
    acc = {"out": 0, "in": 0, "n": 0, "fail": 0}

    def worker():
        while True:
            with lock:
                if idx[0] >= len(batch):
                    return
                i = idx[0]
                idx[0] += 1
            try:
                o, p = call(batch[i], max_tokens)
                with lock:
                    acc["out"] += o
                    acc["in"] += p
                    acc["n"] += 1
            except Exception:
                with lock:
                    acc["fail"] += 1

    t0 = time.time()
    ths = [threading.Thread(target=worker) for _ in range(workers)]
    for t in ths:
        t.start()
    for t in ths:
        t.join()
    dt = time.time() - t0
    return {"wall_s": round(dt, 1), **acc,
            "out_tok_s": round(acc["out"] / dt, 1) if dt else 0.0,
            "req_s": round(acc["n"] / dt, 2) if dt else 0.0,
            "out_per_req": round(acc["out"] / acc["n"], 1) if acc["n"] else 0.0,
            "in_per_req": round(acc["in"] / acc["n"], 1) if acc["n"] else 0.0}


# ------------------------------------------------------------------ main

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40, help="请求条数")
    ap.add_argument("--workers", type=int, default=4, help="客户端并发（生产是 4）")
    ap.add_argument("--max-tokens", type=int, default=200, help="生产是 200")
    ap.add_argument("--only", default="", help="只跑指定配置，逗号分隔")
    args = ap.parse_args()

    batch = load_prompts(args.n)
    print(f"真实提示词 {len(batch)} 条，客户端 {args.workers} 线程，max_tokens {args.max_tokens}")
    print(f"（生产实际是 4 线程 / 200，所以下面的 out_tok_s 可以直接跟 37.8 比）\n")

    names = [s for s in (args.only.split(",") if args.only else CONFIGS) if s]
    results = {}
    for name in names:
        cfg = CONFIGS[name]
        print(f"[{name}]")
        kill_server()
        global t0
        t0 = time.time()
        p = start_server(cfg, name)
        if p is None:
            results[name] = {"error": "startup_failed"}
            continue
        r = bench(batch, args.workers, args.max_tokens)
        results[name] = r
        print(f"    ✓ out {r['out_tok_s']} tok/s ｜ {r['req_s']} req/s ｜ "
              f"每请求 in {r['in_per_req']} / out {r['out_per_req']} tok ｜ "
              f"{r['wall_s']}s ｜ 失败 {r['fail']}")
        kill_server()

    print("\n=== 汇总（按 out tok/s 排序）===")
    for name, r in sorted(results.items(), key=lambda kv: -kv[1].get("out_tok_s", -1)):
        if "error" in r:
            print(f"  {name:<8} {r['error']}")
        else:
            print(f"  {name:<8} {r['out_tok_s']:>6} tok/s   "
                  f"（{r['n']} 请求 / {r['wall_s']}s / 失败 {r['fail']}）")
    out = LOG_DIR / "teacher_real_bench.json"
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n明细已写 {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
