#!/usr/bin/env python3
"""扫教师服务的并发/投机解码配置，找蒸馏吞吐最优解。

背景：v4.0 蒸馏要大量独立单轮样本，聚合 tok/s 才是关键指标（不是单请求延迟）。
TUNING.md 只测过 --max-concurrency 1 的单用户场景，多并发该配什么没人测过。

用法：
    python scripts/bench_teacher_conc.py                    # 跑内置配置表
    python scripts/bench_teacher_conc.py --concs 1,8        # 只测这些并发度
    python scripts/bench_teacher_conc.py --configs spec8    # 只测指定配置

结果同时打印到 stdout 并追加进 logs/teacher_conc_sweep.jsonl。
"""
import sys, os, json, time, subprocess, urllib.request, urllib.error, threading
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(os.environ.get("FENG_ROOT") or Path(__file__).resolve().parents[1])
EXE = Path(os.environ.get("NINFER_EXE") or r"D:\ai-ninfer\build\b1\apps\ninfer-serve.exe")
MODEL = Path(os.environ.get("NINFER_MODEL") or r"D:\ai-ninfer\artifacts\swift-bonsai2-27b-pruned.ninfer")
LOG_DIR = ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)
BASE = "http://127.0.0.1:8080/v1"

# 公共底座：短上下文蒸馏负载（提示词上游截断在 1200 字符）。
# 注意 --lm-head-draft 属于投机解码，必须跟 --spec 一起出现，否则引擎直接打 usage 退出。
MAX_CTX = 8192
COMMON = [
    "--host", "127.0.0.1", "--port", "8080", "--model-id", "bonsai2-27b",
    "--max-context", str(MAX_CTX), "--kv-dtype", "nvfp4",
    "--prefill-chunk", "8192", "--max-pending-requests", "32",
    "--vision", "--cors",
]


def conc_args(conc):
    """并发相关参数。引擎实测约束：kv_capacity 必须 <= max_context * max_concurrency，
    且 max_private_continuations 必须覆盖每个活动请求。"""
    return ["--max-concurrency", str(conc),
            "--kv-capacity", str(MAX_CTX * conc),
            "--host-kv-mib", "0", "--host-state-slots", "0",
            "--device-state-slots", str(conc),
            "--max-private-continuations", str(conc), "--max-shared-prefixes", "4"]


# name -> (额外参数, 并发度)
SPEC = ["--spec", "mtp", "--draft-tokens", "3", "--lm-head-draft"]
CONFIGS = {
    "spec8":            (SPEC + ["--no-cuda-graph"], 8),
    "spec8_cudagraph":  (SPEC, 8),
    "nospec8":          (["--no-cuda-graph"], 8),
    "nospec8_cudagraph": ([], 8),
    "spec1":            (SPEC + ["--no-cuda-graph"], 1),
    "spec1_cudagraph":  (SPEC, 1),
    "spec2":            (SPEC + ["--no-cuda-graph"], 2),
    "spec4":            (SPEC + ["--no-cuda-graph"], 4),
    "nospec1":          (["--no-cuda-graph"], 1),
    "spec2_nolmd":      (["--spec", "mtp", "--draft-tokens", "2", "--no-cuda-graph"], 1),
}

PROMPTS = ["介绍一下杭州", "为什么天是蓝的", "给我讲个笑话", "推荐一部电影",
           "怎么做番茄炒蛋", "解释一下什么是光合作用", "写一句鼓励的话", "今天心情不太好"]


def kill_server():
    subprocess.run(["taskkill", "/IM", "ninfer-serve.exe", "/F"],
                   capture_output=True)
    for _ in range(60):                      # 等显存回落
        time.sleep(0.5)
        try:
            urllib.request.urlopen(BASE + "/models", timeout=1)
        except Exception:
            return


def start_server(extra, conc, logname):
    args = [str(EXE), str(MODEL)] + COMMON + conc_args(conc) + extra
    out = open(LOG_DIR / f"teacher_sweep_{logname}.err.log", "w", encoding="utf-8")
    p = subprocess.Popen(args, stdout=subprocess.DEVNULL, stderr=out)
    for _ in range(180):                     # 最多等 90 s
        time.sleep(0.5)
        if p.poll() is not None:
            out.close()
            return None, (LOG_DIR / f"teacher_sweep_{logname}.err.log").read_text(
                encoding="utf-8", errors="replace")[-600:]
        try:
            urllib.request.urlopen(BASE + "/models", timeout=2)
            return p, None
        except Exception:
            pass
    p.kill()
    out.close()
    return None, "启动超时"


def call(msgs, mt=128, temp=0.8, timeout=900):
    body = {"model": "bonsai2-27b", "messages": msgs, "max_tokens": mt,
            "temperature": temp, "reasoning_effort": "none"}
    req = urllib.request.Request(BASE + "/chat/completions",
                                 data=json.dumps(body, ensure_ascii=False).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as f:
        d = json.loads(f.read().decode())
    return d.get("usage", {}).get("completion_tokens", 0)


def bench(conc, n_req, mt=128):
    idx = [0]; lock = threading.Lock(); toks = [0]; lat = []; fail = [0]
    def worker():
        while True:
            with lock:
                if idx[0] >= n_req:
                    return
                i = idx[0]; idx[0] += 1
            t = time.time()
            try:
                k = call([{"role": "user", "content": PROMPTS[i % len(PROMPTS)]}], mt=mt)
                with lock:
                    toks[0] += k; lat.append(time.time() - t)
            except Exception as e:
                with lock:
                    fail[0] += 1; lat.append(float("nan"))
    t0 = time.time()
    ths = [threading.Thread(target=worker) for _ in range(conc)]
    [t.start() for t in ths]
    [t.join() for t in ths]
    dt = time.time() - t0
    ok = [x for x in lat if x == x]
    return {"requests": n_req, "tokens": toks[0], "wall_s": round(dt, 1),
            "agg_tok_s": round(toks[0] / dt, 1) if dt else 0.0,
            "lat_mean_s": round(sum(ok) / len(ok), 2) if ok else None,
            "failed": fail[0]}


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", default=",".join(CONFIGS))
    ap.add_argument("--concs", default="")
    ap.add_argument("--requests", type=int, default=16, help="每个并发度发多少条")
    ap.add_argument("--max-tokens", type=int, default=128)
    a = ap.parse_args()

    names = [c.strip() for c in a.configs.split(",") if c.strip()]
    if not EXE.exists():
        sys.exit(f"引擎不存在: {EXE}")
    if not MODEL.exists():
        sys.exit(f"模型不存在: {MODEL}")

    out_path = LOG_DIR / "teacher_conc_sweep.jsonl"
    rows = []
    for name in names:
        extra, cfg_conc = CONFIGS[name]
        concs = [int(x) for x in a.concs.split(",")] if a.concs else [cfg_conc]
        for conc in concs:
            label = f"{name}_c{conc}"
            print(f"\n=== {label} ===", flush=True)
            kill_server()
            p, err = start_server(extra, conc, label)
            if p is None:
                print(f"  启动失败: {err}", flush=True)
                rows.append({"config": name, "conc": conc, "error": "startup_failed"})
                continue
            r = bench(conc, a.requests, a.max_tokens)
            r.update({"config": name, "conc": conc,
                      "extra": " ".join(extra)})
            print(f"  聚合 {r['agg_tok_s']} tok/s | 墙钟 {r['wall_s']}s | "
                  f"单请求均值 {r['lat_mean_s']}s | 失败 {r['failed']}", flush=True)
            rows.append(r)
            p.kill()

    kill_server()
    with open(out_path, "a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\n结果已追加: {out_path}")


if __name__ == "__main__":
    main()
