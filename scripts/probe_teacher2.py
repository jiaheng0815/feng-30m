"""Quick probe: thinking off, measure single + batch throughput of the NInfer teacher."""
import json
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
BASE = "http://127.0.0.1:8080/v1"
MODEL = "bonsai2-27b"


def chat(prompt, max_tokens=160, effort="none", temp=0.8):
    body = {"model": MODEL, "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens, "temperature": temp, "reasoning_effort": effort}
    req = urllib.request.Request(BASE + "/chat/completions",
                                 data=json.dumps(body, ensure_ascii=False).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=900) as r:
        d = json.loads(r.read().decode())
    dt = time.time() - t0
    m = d["choices"][0]["message"]
    return dt, d.get("usage", {}), m.get("content", ""), m.get("reasoning_content") or ""


def main():
    dt, usage, content, reas = chat("你是谁？请用一句话回答。", 120, "none")
    print(f"[effort=none] {dt:.1f}s usage={usage}")
    print("  reasoning:", reas[:120].replace("\n", " "))
    print("  content  :", content[:200].replace("\n", " "))

    prompts = ["你好", "你是谁？", "今天心情不好", "推荐一部电影", "1+1等于几", "你会做什么",
               "给我起个名字", "晚安", "你叫什么名字", "讲个笑话", "你是什么模型", "在吗"] * 2
    for conc in (8, 16):
        t0 = time.time()
        with ThreadPoolExecutor(conc) as ex:
            res = list(ex.map(lambda p: chat(p, 160, "none"), prompts))
        wall = time.time() - t0
        gen = sum(r[1].get("completion_tokens", 0) for r in res)
        print(f"\n[conc {conc}] {len(prompts)} reqs | wall {wall:.1f}s | "
              f"{gen} completion tokens -> {gen / wall:.1f} tok/s aggregate | "
              f"avg per-req {sum(r[0] for r in res) / len(res):.1f}s")
        print("  sample:", res[0][2][:120].replace("\n", " "))


if __name__ == "__main__":
    main()
