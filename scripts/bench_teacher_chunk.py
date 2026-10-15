#!/usr/bin/env python3
"""测 --prefill-chunk 对**短提示**（蒸馏真实负载）的影响。

背景：v4.0 蒸馏时服务端日志显示 prompt 140~270 token 的 prefill 只有 250~300 tok/s，
而 TUNING.md 在 20k 长提示下是 654 tok/s —— 差在**每次请求的固定开销**。prefill-chunk
控制的是单次 prefill 的 token 上限（默认 8192），短提示下它可能是纯浪费。

用法：
    python scripts/bench_teacher_chunk.py --chunks 512,2048,8192 --n 40
结果追加到 logs/teacher_chunk_sweep.jsonl。
"""
import argparse
import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = Path(os.environ.get("FENG_ROOT") or Path(__file__).resolve().parents[1])
EXE = Path(os.environ.get("NINFER_EXE") or r"D:\ai-ninfer\build\b1\apps\ninfer-serve.exe")
MODEL = Path(os.environ.get("NINFER_MODEL") or r"D:\ai-ninfer\artifacts\swift-bonsai2-27b-pruned.ninfer")
BASE = "http://127.0.0.1:8080/v1"
SYS = ("你是一个中文 AI 助手，说话自然、口语化、简短（一般 1-3 句）。"
       "不要提及自己的身份、名字、厂商或训练信息；不要用列表；直接回答问题。")


def kill_server():
    subprocess.run(["taskkill", "/IM", "ninfer-serve.exe", "/F"], capture_output=True)
    for _ in range(60):
        time.sleep(0.5)
        try:
            urllib.request.urlopen(BASE + "/models", timeout=1)
        except Exception:
            return


def start_server(chunk, logname):
    args = [str(EXE), str(MODEL), "--host", "127.0.0.1", "--port", "8080",
            "--model-id", "bonsai2-27b", "--max-context", "8192", "--kv-capacity", "8192",
            "--kv-dtype", "nvfp4", "--prefill-chunk", str(chunk),
            "--max-concurrency", "1", "--max-pending-requests", "32",
            "--spec", "mtp", "--draft-tokens", "3", "--lm-head-draft",
            "--host-kv-mib", "1024", "--device-state-slots", "8",
            "--max-private-continuations", "8", "--max-shared-prefixes", "4",
            "--vision", "--cors"]
    out = open(ROOT / "logs" / f"teacher_chunk_{logname}.err.log", "w", encoding="utf-8")
    p = subprocess.Popen(args, stdout=subprocess.DEVNULL, stderr=out)
    for _ in range(240):
        time.sleep(0.5)
        if p.poll() is not None:
            out.close()
            return None, (ROOT / "logs" / f"teacher_chunk_{logname}.err.log").read_text(
                encoding="utf-8", errors="replace")[-400:]
        try:
            urllib.request.urlopen(BASE + "/models", timeout=2)
            return p, None
        except Exception:
            pass
    p.kill()
    out.close()
    return None, "启动超时"


def call(user, mt=200, temp=0.8, timeout=900):
    body = {"model": "bonsai2-27b",
            "messages": [{"role": "system", "content": SYS},
                         {"role": "user", "content": user}],
            "max_tokens": mt, "temperature": temp, "reasoning_effort": "none"}
    req = urllib.request.Request(BASE + "/chat/completions",
                                 data=json.dumps(body, ensure_ascii=False).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as f:
        d = json.loads(f.read().decode())
    dt = time.time() - t0
    return d.get("usage", {}).get("completion_tokens", 0), dt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", default="512,2048,8192")
    ap.add_argument("--n", type=int, default=40, help="每个配置跑多少条")
    ap.add_argument("--prompts", default=str(ROOT / "data" / "v4_prompts.jsonl"))
    a = ap.parse_args()

    # 用真实提示词，避免合成数据不代表负载
    pool = []
    with open(a.prompts, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            it = json.loads(line)
            t = it["turns"][0] if it.get("turns") else None
            if t and 10 <= len(t) <= 200:
                pool.append(t)
            if len(pool) >= 5000:
                break
    probes = pool[: a.n]
    print(f"用 {len(probes)} 条真实提示词（来自 {Path(a.prompts).name}）", flush=True)

    rows = []
    for chunk in [int(x) for x in a.chunks.split(",")]:
        print(f"\n=== prefill-chunk={chunk} ===", flush=True)
        kill_server()
        p, err = start_server(chunk, str(chunk))
        if p is None:
            print(f"  启动失败: {err}", flush=True)
            rows.append({"chunk": chunk, "error": "startup_failed"})
            continue
        # 预热
        try:
            call("你好", mt=16)
        except Exception:
            pass
        toks = 0
        t0 = time.time()
        fails = 0
        for q in probes:
            try:
                k, _ = call(q)
                toks += k
            except Exception:
                fails += 1
        dt = time.time() - t0
        r = {"chunk": chunk, "n": len(probes), "tokens": toks,
             "wall_s": round(dt, 1), "tok_s": round(toks / dt, 1) if dt else 0,
             "req_s": round(len(probes) / dt, 3) if dt else 0, "failed": fails}
        print(f"  {toks} tok / {dt:.1f}s = {r['tok_s']} tok/s | "
              f"{r['req_s']} 请求/s | 失败 {fails}", flush=True)
        rows.append(r)
        p.kill()

    kill_server()
    with open(ROOT / "logs" / "teacher_chunk_sweep.jsonl", "a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\n结果: {ROOT / 'logs' / 'teacher_chunk_sweep.jsonl'}")
    print(f"\n⚠️ 服务已被杀掉，跑完记得用生产配置重启（见 logs/ninfer_teacher_v40 的启动命令）")


if __name__ == "__main__":
    main()
