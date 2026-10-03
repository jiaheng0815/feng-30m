"""Fetch key facts from the most relevant ESP32-S3 LLM repos."""
import json
import re
import sys
import urllib.request

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

REPOS = [
    "JARACH-209/esp32-30.7M",
    "loopDelicious/esp32-ai-octal",
    "andrisgauracs/needle-2-esp32",
    "mediacutlet/pocket-tank",
    "therezor/cardputer-ai",
    "ModenCn/tinyllamas-zh",
    "marcelpetrick/ESP32-Tiny-LLM",
]


def get(url):
    req = urllib.request.Request(url, headers={"User-Agent": "codex-search"})
    return urllib.request.urlopen(req, timeout=40).read().decode("utf-8", "replace")


for full in REPOS:
    try:
        d = json.loads(get(f"https://api.github.com/repos/{full}"))
        readme = get(f"https://raw.githubusercontent.com/{full}/{d['default_branch']}/README.md")
    except Exception as e:
        print(f"--- {full}: ERR {str(e)[:70]}")
        continue
    print(f"\n{'='*90}\n{full}  ({d['stargazers_count']}*, updated {d['pushed_at'][:10]})")
    print(f"desc: {(d.get('description') or '')[:150]}")
    facts = []
    for pat, label in [
        (r"[^\n]*\b\d[\d.,]*\s*(?:tok|token)s?\s*/\s*s[^\n]*", "speed"),
        (r"[^\n]*\b\d[\d.,]*\s*(?:MB|MiB)\b[^\n]*", "size"),
        (r"[^\n]*(?:param|parameters)[^\n]*", "params"),
        (r"[^\n]*(?:quant|int4|int8|q4|q8)[^\n]*", "quant"),
        (r"[^\n]*(?:psram|flash|sram|ram)[^\n]*", "memory"),
    ]:
        for m in re.findall(pat, readme, re.I)[:3]:
            line = " ".join(m.split())[:180]
            facts.append(f"  [{label}] {line}")
    seen = set()
    for f in facts:
        if f not in seen:
            print(f)
            seen.add(f)
