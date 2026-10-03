"""Search GitHub for on-device LLM projects relevant to ESP32-S3."""
import json
import sys
import urllib.parse
import urllib.request

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

QUERIES = [
    "esp32 llm",
    "esp32 llama",
    "esp32-s3 llm",
    "llama2.c esp32",
    "tinystories esp32",
    "microcontroller llm inference",
    "esp32 bitnet",
    "tflite micro llm",
]


def search(q, n=8):
    url = ("https://api.github.com/search/repositories?q=" + urllib.parse.quote(q)
           + f"&sort=stars&per_page={n}")
    req = urllib.request.Request(url, headers={"User-Agent": "codex-search"})
    try:
        d = json.load(urllib.request.urlopen(req, timeout=40))
    except Exception as e:
        print(f"=== {q}: ERR {str(e)[:80]}")
        return
    print(f"=== {q}  (total {d.get('total_count')})")
    for r in d.get("items", []):
        print("  {:>6}*  {:<48} {}  {}".format(
            r["stargazers_count"], r["full_name"], r["pushed_at"][:10],
            (r.get("description") or "")[:85]))


if __name__ == "__main__":
    for q in QUERIES:
        search(q)
