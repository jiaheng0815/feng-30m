"""Plan A data generation with the NInfer 27B teacher (thinking off, concurrency 8)."""
import argparse
import json
import queue
import sys
import threading
import time
import urllib.request
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import ROOT  # noqa: E402
BASE = "http://127.0.0.1:8080/v1"
MODEL = "bonsai2-27b"

SYS_CASUAL = ("你是一个中文 AI 助手，说话自然、口语化、简短（一般 1-3 句）。"
              "不要提及自己的身份、名字、厂商或训练信息；不要用列表；直接回答问题。")
SYS_REFUSAL = ("你是一个中文 AI 助手。对于实时信息、专业医疗/法律/投资建议、超长任务，"
               "用一句自然的话礼貌说明自己做不到，并建议用户找更合适的渠道，然后可以给一点力所能及的帮助。"
               "不要提及自己的身份或厂商，不要用列表。")


def call(messages, max_tokens=200, temperature=0.8):
    body = {"model": MODEL, "messages": messages, "max_tokens": max_tokens,
            "temperature": temperature, "reasoning_effort": "none"}
    req = urllib.request.Request(BASE + "/chat/completions",
                                 data=json.dumps(body, ensure_ascii=False).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=900) as r:
        d = json.loads(r.read().decode())
    m = d["choices"][0]["message"]
    return (m.get("content") or "").strip(), d.get("usage", {}).get("completion_tokens", 0)


class Stats:
    def __init__(self):
        self.lock = threading.Lock()
        self.done = 0
        self.tokens = 0
        self.t0 = time.time()

    def add(self, n):
        with self.lock:
            self.done += 1
            self.tokens += n
            return self.done, self.tokens


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", default=str(ROOT / "data" / "planA_prompts.jsonl"))
    ap.add_argument("--out", default=str(ROOT / "data" / "planA_teacher.jsonl"))
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=200)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    prompts = [json.loads(l) for l in Path(args.prompts).read_text(encoding="utf-8").splitlines()
               if l.strip()]
    if args.limit:
        prompts = prompts[:args.limit]
    out_path = Path(args.out)
    done_ids = set()
    if out_path.exists():
        for line in out_path.read_text(encoding="utf-8").splitlines():
            try:
                done_ids.add(json.loads(line)["id"])
            except Exception:
                pass
    todo = [p for p in prompts if p["id"] not in done_ids]
    print(f"prompts {len(prompts)}, done {len(done_ids)}, todo {len(todo)}", flush=True)

    stats = Stats()
    q = queue.Queue()
    for it in todo:
        q.put(it)
    lock = threading.Lock()
    fout = out_path.open("a", encoding="utf-8")

    def worker():
        while True:
            try:
                it = q.get_nowait()
            except queue.Empty:
                return
            sysp = SYS_REFUSAL if it["system"] == "refusal" else SYS_CASUAL
            msgs = [{"role": "system", "content": sysp}]
            turns = it["turns"]
            produced = []
            try:
                if it["kind"] == "multiturn":
                    for user in turns:
                        msgs.append({"role": "user", "content": user})
                        ans, ntok = call(msgs, args.max_tokens, 0.8)
                        if not ans:
                            break
                        msgs.append({"role": "assistant", "content": ans})
                        produced.append({"user": user, "assistant": ans})
                    ntok_total = sum(len(p["assistant"]) for p in produced) // 2
                    if len(produced) == len(turns):
                        with lock:
                            fout.write(json.dumps({"id": it["id"], "kind": it["kind"],
                                                   "turns": produced}, ensure_ascii=False) + "\n")
                            fout.flush()
                        stats.add(ntok_total)
                    else:
                        stats.add(0)
                else:
                    user = turns[0]
                    msgs.append({"role": "user", "content": user})
                    ans, ntok = call(msgs, args.max_tokens, 0.8)
                    if ans:
                        with lock:
                            fout.write(json.dumps({"id": it["id"], "kind": it["kind"],
                                                   "prompt": user, "response": ans},
                                                  ensure_ascii=False) + "\n")
                            fout.flush()
                        stats.add(ntok)
                    else:
                        stats.add(0)
            except Exception as e:
                print(f"  ! id={it['id']} {str(e)[:90]}", flush=True)
                stats.add(0)
            n_done, n_tok = stats.done, stats.tokens
            if n_done and n_done % 50 == 0:
                el = time.time() - stats.t0
                print(f"  {n_done}/{len(todo)} | {n_tok} tokens | "
                      f"{n_tok / max(el, 1e-6):.1f} tok/s | {el/60:.1f} min", flush=True)

    threads = [threading.Thread(target=worker, daemon=True) for _ in range(args.workers)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    fout.close()
    el = time.time() - stats.t0
    print(f"\nDONE {stats.done} samples, {stats.tokens} tokens in {el/60:.1f} min "
          f"({stats.tokens/max(el,1e-6):.1f} tok/s)", flush=True)


if __name__ == "__main__":
    main()
