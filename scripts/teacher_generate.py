"""Generate distillation data with the feng-0.8b bf16 teacher via llama-server (continuous batching)."""
import argparse
import json
import queue
import socket
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import ROOT  # noqa: E402
from paths import LLAMA_BIN as BIN  # noqa: E402
from paths import TEACHER_GGUF as TEACHER  # noqa: E402
from paths import require  # noqa: E402

PROMPT_TMPL = ("<|im_start|>user\n{user}<|im_end|>\n"
               "<|im_start|>assistant\n<think>\n\n</think>\n\n")


def free_port():
    s = socket.socket(); s.bind(("127.0.0.1", 0)); p = s.getsockname()[1]; s.close(); return p


def post(url, payload, timeout=900):
    req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


class Stats:
    def __init__(self):
        self.lock = threading.Lock()
        self.done = 0
        self.tokens = 0
        self.t0 = time.time()

    def add(self, n_tokens):
        with self.lock:
            self.done += 1
            self.tokens += n_tokens
            return self.done, self.tokens


def main():
    ap = argparse.ArgumentParser()
    require(TEACHER, "v1 教师 GGUF（feng-0.8b bf16）", "FENG_TEACHER_GGUF")
    ap.add_argument("--prompts", default=str(ROOT / "data" / "prompts.jsonl"))
    ap.add_argument("--out", default=str(ROOT / "data" / "distill.jsonl"))
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--parallel", type=int, default=24)
    ap.add_argument("--ctx", type=int, default=65536,
                    help="TOTAL context; per-slot context = ctx / parallel (llama.cpp splits it)")
    ap.add_argument("--batch-size", type=int, default=4096)
    ap.add_argument("--ubatch-size", type=int, default=1024)
    ap.add_argument("--n-predict", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--bench", type=int, default=0, help="only generate N samples for a throughput test")
    ap.add_argument("--log-every", type=int, default=200)
    args = ap.parse_args()

    prompts = [json.loads(l) for l in Path(args.prompts).read_text(encoding="utf-8").splitlines()
               if l.strip()]
    if args.limit:
        prompts = prompts[:args.limit]
    out_path = Path(args.out)
    done_ids = set()
    if out_path.exists() and not args.bench:
        for line in out_path.read_text(encoding="utf-8").splitlines():
            try:
                done_ids.add(json.loads(line)["id"])
            except Exception:
                pass
    todo = [p for p in prompts if p["id"] not in done_ids]
    if args.bench:
        todo = todo[:args.bench]
    print(f"prompts total {len(prompts)}, already done {len(done_ids)}, todo {len(todo)}", flush=True)

    port = free_port()
    log_path = ROOT / "logs" / "teacher_server.log"
    cmd = [str(BIN / "llama-server.exe"), "-m", str(TEACHER), "--host", "127.0.0.1",
           "--port", str(port), "-c", str(args.ctx), "--parallel", str(args.parallel),
           "-b", str(args.batch_size), "-ub", str(args.ubatch_size),
           "-ngl", "99", "-t", "8", "-fa", "on", "--cache-type-k", "q8_0", "--cache-type-v", "q8_0"]
    print("server:", " ".join(cmd), flush=True)
    server = subprocess.Popen(cmd, stdout=open(log_path, "w", encoding="utf-8", errors="replace"),
                              stderr=subprocess.STDOUT)
    base = f"http://127.0.0.1:{port}"
    t0 = time.time()
    ready = False
    while time.time() - t0 < 300:
        if server.poll() is not None:
            print("server died:", log_path.read_text(encoding="utf-8", errors="replace")[-1000:])
            return 2
        try:
            with urllib.request.urlopen(base + "/health", timeout=5) as r:
                if json.loads(r.read()).get("status") == "ok":
                    ready = True
                    break
        except Exception:
            time.sleep(1.0)
    if not ready:
        print("server not ready")
        return 3
    print(f"teacher ready in {time.time() - t0:.1f}s | per-slot ctx = {args.ctx // args.parallel}",
          flush=True)

    stats = Stats()
    q = queue.Queue()
    for item in todo:
        q.put(item)
    write_lock = threading.Lock()
    fout = out_path.open("a", encoding="utf-8") if not args.bench else None
    stop_flag = threading.Event()

    def worker():
        while not stop_flag.is_set():
            try:
                item = q.get_nowait()
            except queue.Empty:
                return
            prompt = PROMPT_TMPL.format(user=item["text"])
            try:
                resp = post(base + "/completion", {
                    "prompt": prompt, "n_predict": args.n_predict, "temperature": args.temperature,
                    "top_p": args.top_p, "top_k": 40, "stop": ["<|im_end|>"], "cache_prompt": False})
                text = (resp.get("content") or "").strip()
                ntok = int(resp.get("tokens_predicted") or 0)
            except Exception as e:
                text, ntok = "", 0
                print(f"  ! request failed id={item['id']}: {str(e)[:80]}", flush=True)
            if text:
                rec = {"id": item["id"], "prompt": item["text"], "response": text,
                       "source": item["source"], "lang": item["lang"], "tokens": ntok}
                if fout:
                    with write_lock:
                        fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                        fout.flush()
            n_done, n_tok = stats.add(ntok if text else 0)
            if n_done % args.log_every == 0:
                el = time.time() - stats.t0
                rate = n_tok / max(el, 1e-6)
                eta_h = (len(todo) - n_done) * (n_tok / max(n_done, 1)) / max(rate, 1e-6) / 3600
                print(f"  {n_done}/{len(todo)} samples, {n_tok/1e6:.2f}M tokens, "
                      f"{rate:.0f} tok/s, ETA {eta_h:.1f}h", flush=True)

    threads = [threading.Thread(target=worker, daemon=True) for _ in range(args.workers)]
    for t in threads:
        t.start()
    try:
        for t in threads:
            t.join()
    except KeyboardInterrupt:
        stop_flag.set()
    el = time.time() - stats.t0
    print(f"\nDONE: {stats.done} samples, {stats.tokens} tokens in {el/60:.1f} min "
          f"({stats.tokens/max(el,1e-6):.0f} tok/s)", flush=True)
    if fout:
        fout.close()
    server.terminate()
    try:
        server.wait(timeout=20)
    except Exception:
        server.kill()
    return 0


if __name__ == "__main__":
    sys.exit(main())
