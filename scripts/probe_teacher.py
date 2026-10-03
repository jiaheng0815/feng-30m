"""Probe the NInfer teacher: API shape, thinking behaviour, single & batch throughput."""
import json
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
BASE = "http://127.0.0.1:8080/v1"
MODEL = "bonsai2-27b"


def post(path, body, timeout=600):
    req = urllib.request.Request(BASE + path, data=json.dumps(body, ensure_ascii=False).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def get(path):
    with urllib.request.urlopen(BASE + path, timeout=60) as r:
        return json.loads(r.read().decode())


def main():
    print("models:", json.dumps(get("/models"), ensure_ascii=False)[:300])

    # --- single request: inspect the response shape (thinking vs answer)
    t0 = time.time()
    r = post("/chat/completions", {
        "model": MODEL,
        "messages": [{"role": "user", "content": "你是谁？请用一句话回答。"}],
        "max_tokens": 300, "temperature": 0.7})
    dt = time.time() - t0
    ch = r.get("choices", [{}])[0]
    msg = ch.get("message", {})
    usage = r.get("usage", {})
    print(f"\nsingle request: {dt:.1f}s | usage {usage}")
    print("keys:", list(r.keys()), "| message keys:", list(msg.keys()))
    print("finish_reason:", ch.get("finish_reason"))
    print("reasoning:", (msg.get("reasoning_content") or "")[:200].replace("\n", " "))
    print("content  :", (msg.get("content") or "")[:300].replace("\n", " "))

    # --- batch throughput (concurrency 8 like the server config)
    prompts = ["你好", "你是谁？", "今天心情不好，陪我聊聊天。", "推荐一部电影。", "1+1等于几？",
               "你会做什么？", "给我起个名字。", "晚安"] * 2

    def one(p):
        t = time.time()
        rr = post("/chat/completions", {"model": MODEL,
                                        "messages": [{"role": "user", "content": p}],
                                        "max_tokens": 160, "temperature": 0.8})
        u = rr.get("usage", {})
        return time.time() - t, u.get("completion_tokens", 0), u.get("prompt_tokens", 0)

    t0 = time.time()
    with ThreadPoolExecutor(8) as ex:
        res = list(ex.map(one, prompts))
    wall = time.time() - t0
    gen = sum(x[1] for x in res)
    print(f"\nbatch: {len(prompts)} reqs, concurrency 8 | wall {wall:.1f}s | "
          f"completion tokens {gen} -> {gen / wall:.1f} tok/s aggregate | "
          f"per-req {sum(x[0] for x in res) / len(res):.1f}s avg")


if __name__ == "__main__":
    main()
