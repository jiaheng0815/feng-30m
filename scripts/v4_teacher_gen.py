"""feng-30m 4.0 教师数据生成：带质量闸门的 27B 蒸馏。

与 v3.x 的 `teacher_gen_ninfer.py` 的三点关键差别：

1) **质量闸门 + 重采样**。v3.x 是「教师说什么就收什么」，实测下来 v3.19 的定向数据里
   7 条身份提问有 4 条被答成「我是通义千问，由阿里巴巴集团通义实验室开发」，公开蒸馏
   数据里有「我是 feng，一个由个人开发者 jiaheng 微调的 Qwen 模型」——既提 Qwen 又
   自称"微调"。4.0 要求劣质数据全部剔除：每条回答过规则闸门，不合格就**升温重采**，
   采不满 `--max-tries` 次就整条丢掉并记进 rejects 日志。

2) **网络重试**。v3.x 版一次 HTTP 异常就丢整条（多轮还连带浪费前面几轮），
   ~65k 次调用的规模下这个损耗不可接受。

3) **按实测配置并发**。服务端单通道最快（1→2 并发聚合吞吐掉 58%），
   所以 `--workers` 只用来把流水线喂满，默认 4。

用法：
    python scripts\\v4_teacher_gen.py --prompts data\\v4_prompts.jsonl \\
        --out data\\v4_teacher.jsonl --rejects data\\v4_rejects.jsonl --target 50000
"""
import argparse
import json
import queue
import re
import sys
import threading
import time
import urllib.error
import urllib.request
from collections import Counter
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

# ---------------------------------------------------------------- 质量闸门
# 提到别的 AI / 厂商：这是 4.0 最不能容忍的一类
RE_IDENTITY = re.compile(
    r"通义|千问|[Qq]wen|阿里|阿里巴巴|淘宝|支付宝|"
    r"ChatGPT|GPT-?[345]|OpenAI|Anthropic|Claude|Gemini|DeepMind|Copilot|"
    r"文心|一言|ERNIE|讯飞|星火|豆包|抖音|字节跳动|智谱|GLM|"
    r"月之暗面|Kimi|DeepSeek|深度求索|零一万物|Llama|[Mm]istral|Grok|"
    r"[Mm]iniMax|百川|元宝|阶跃星辰")
# 「作为一个 AI 助手」这类元话术：30M 模型学这个只会变成口头禅
RE_META = re.compile(
    r"作为(一个)?(人工)?智能(助手|助理|体)|作为一个?AI|"
    r"我是(一个)?(人工智能|语言|大)模型|我是(一个)?AI|"
    r"训练数据|参数量|知识截止|我的知识|被训练|开发者训练")
# 代码：30M 中文对话模型永远做不到，学了只会污染输出格式
RE_CODE = re.compile(r"```|^\s{0,4}(import\s+\w|from\s+\w+\s+import|def\s+\w|class\s+\w|"
                     r"#include|public\s+(static|class)|function\s+\w|SELECT\s+.+\s+FROM)", re.I | re.M)
# 行首列表标记（句内的「苹果、香蕉」不算）
RE_LIST = re.compile(r"^\s*(?:[-*•·]\s|\d+[.、)]\s|[（(]\d+[)）]\s)", re.M)
RE_TERM = tuple("。！？…\n”\"）)】」』.!?")


def rep_ratio(s: str, n: int = 5) -> float:
    """重复 n-gram 占比，用来抓复读。"""
    if len(s) < n * 2:
        return 0.0
    grams = [s[i:i + n] for i in range(len(s) - n + 1)]
    c = Counter(grams)
    return sum(v - 1 for v in c.values() if v > 1) / len(grams)


def quality_ok(text: str, prompt: str, finish_reason: str | None) -> tuple[bool, str]:
    """返回 (是否合格, 不合格原因)。原因同时用于统计和 rejects 日志。"""
    a = (text or "").strip()
    if not a:
        return False, "empty"
    if len(a) < 4:
        return False, "too_short"
    if len(a) > 600:
        return False, "too_long"
    # 教师自报家门 / 提到别家
    if RE_IDENTITY.search(a):
        return False, "identity"
    if RE_META.search(a):
        return False, "meta"
    if RE_CODE.search(a):
        return False, "code"
    if RE_LIST.search(a):
        return False, "list"
    if rep_ratio(a) > 0.25:
        return False, "repeat"
    # 复述问题（把用户的话抄一遍当回答）
    p = (prompt or "").strip()
    if p and len(a) <= len(p) + 4 and p[:20] in a:
        return False, "prompt_echo"
    # 中文提问答成全英文（保留英文提问的正常情况）
    if re.search(r"[一-鿿]", p) and not re.search(r"[一-鿿]", a):
        return False, "no_cjk"
    # 截断：撞到 max_tokens 上限且收尾不干净
    if finish_reason == "length" and not a.endswith(RE_TERM):
        return False, "truncated"
    # 长回答必须有句读收尾（短的如「好」「晚安」放过）
    if len(a) > 20 and not a.endswith(RE_TERM):
        return False, "no_stop"
    return True, ""


# ---------------------------------------------------------------- HTTP

def call(messages, max_tokens=200, temperature=0.8, retries=3):
    """一次对话补全。返回 (text, completion_tokens, finish_reason)。网络异常按指数退避重试。"""
    body = {"model": MODEL, "messages": messages, "max_tokens": max_tokens,
            "temperature": temperature, "reasoning_effort": "none"}
    data = json.dumps(body, ensure_ascii=False).encode()
    last = None
    for k in range(retries):
        try:
            req = urllib.request.Request(BASE + "/chat/completions", data=data,
                                         headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=900) as r:
                d = json.loads(r.read().decode())
            ch = d["choices"][0]
            m = ch.get("message") or {}
            return ((m.get("content") or "").strip(),
                    d.get("usage", {}).get("completion_tokens", 0),
                    ch.get("finish_reason"))
        except Exception as e:                       # noqa: BLE001
            last = e
            time.sleep(min(2 ** k, 8))
    raise RuntimeError(f"{type(last).__name__}: {last}")


class Stats:
    def __init__(self):
        self.lock = threading.Lock()
        self.accepted = 0
        self.tokens = 0
        self.calls = 0
        self.rejects = Counter()
        self.failed = 0
        self.t0 = time.time()

    def snapshot(self):
        with self.lock:
            return self.accepted, self.tokens, self.calls, dict(self.rejects), self.failed


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", default=str(ROOT / "data" / "v4_prompts.jsonl"))
    ap.add_argument("--out", default=str(ROOT / "data" / "v4_teacher.jsonl"))
    ap.add_argument("--rejects", default=str(ROOT / "data" / "v4_rejects.jsonl"))
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--max-tokens", type=int, default=200)
    ap.add_argument("--max-tries", type=int, default=3, help="每条最多采几次（质量不合格会升温重采）")
    ap.add_argument("--target", type=int, default=0, help="凑够这么多条合格样本就停（0=全部）")
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 条提示词（试跑用）")
    ap.add_argument("--skip-rejects", action="store_true", default=True,
                    help="rejects 里已记过的 id 不再重试（断点续跑用）")
    args = ap.parse_args()

    prompts = [json.loads(l) for l in Path(args.prompts).read_text(encoding="utf-8").splitlines()
               if l.strip()]
    if args.limit:
        prompts = prompts[:args.limit]

    out_path, rej_path = Path(args.out), Path(args.rejects)
    done_ids, rej_ids = set(), set()
    if out_path.exists():
        for line in out_path.read_text(encoding="utf-8").splitlines():
            try:
                done_ids.add(json.loads(line)["id"])
            except Exception:
                pass
    if rej_path.exists() and args.skip_rejects:
        for line in rej_path.read_text(encoding="utf-8").splitlines():
            try:
                rej_ids.add(json.loads(line)["id"])
            except Exception:
                pass
    todo = [p for p in prompts if p["id"] not in done_ids and p["id"] not in rej_ids]
    n_turns = sum(len(p["turns"]) if p["kind"] == "multiturn" else 1 for p in todo)
    print(f"提示词 {len(prompts)}，已完成 {len(done_ids)}，已拒 {len(rej_ids)}，"
          f"待跑 {len(todo)} 条（约 {n_turns} 次调用）", flush=True)
    if not todo:
        print("没有待跑的，退出")
        return 0

    stats = Stats()
    q = queue.Queue()
    for it in todo:
        q.put(it)
    lock = threading.Lock()
    fout = out_path.open("a", encoding="utf-8")
    frej = rej_path.open("a", encoding="utf-8")
    stop = threading.Event()

    def emit(rec, rejected, reason=""):
        with lock:
            if rejected:
                frej.write(json.dumps({**rec, "reason": reason}, ensure_ascii=False) + "\n")
                frej.flush()
            else:
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fout.flush()

    def gen_one(msgs, prompt_text, sysp):
        """采一条合格回答；不合格升温重采。返回 (ans, tok) 或 None。"""
        for k in range(args.max_tries):
            temp = 0.8 + 0.1 * k
            ans, tok, fr = call(msgs, args.max_tokens, temp)
            with lock:
                stats.calls += 1
                stats.tokens += tok
            ok, why = quality_ok(ans, prompt_text, fr)
            if ok:
                return ans, tok
            with lock:
                stats.rejects[why] += 1
        return None

    def run_item(it):
        """处理一条提示词，返回记录 dict；全轮都失败则返回 None。

        多轮里某一轮采不满会把**前面已经成功的轮次**保留下来（截短成一段对话），
        而不是整条丢弃——否则 3 轮里第 3 轮失败就白白浪费前面 2 轮的时间和 token。
        """
        sysp = SYS_REFUSAL if it["system"] == "refusal" else SYS_CASUAL
        if it["kind"] == "multiturn":
            msgs = [{"role": "system", "content": sysp}]
            produced = []
            for user in it["turns"]:
                msgs.append({"role": "user", "content": user})
                got = gen_one(msgs, user, sysp)
                if got is None:
                    break
                msgs.append({"role": "assistant", "content": got[0]})
                produced.append({"user": user, "assistant": got[0]})
            if not produced:
                return None
            return {"id": it["id"], "kind": "multiturn", "turns": produced,
                    "source": it.get("source", "")}
        user = it["turns"][0]
        got = gen_one([{"role": "system", "content": sysp},
                       {"role": "user", "content": user}], user, sysp)
        if got is None:
            return None
        return {"id": it["id"], "kind": it["kind"], "prompt": user,
                "response": got[0], "source": it.get("source", "")}

    def worker():
        while not stop.is_set():
            try:
                it = q.get_nowait()
            except queue.Empty:
                return
            try:
                rec = run_item(it)
            except Exception as e:                    # noqa: BLE001
                with lock:
                    stats.failed += 1
                print(f"  ! id={it['id']} {str(e)[:100]}", flush=True)
                continue
            if rec is None:
                emit({"id": it["id"], "kind": it["kind"], "source": it.get("source", ""),
                      "prompt": it["turns"][0] if it["turns"] else ""},
                     True, "quality_failed")
                continue
            emit(rec, False)
            with lock:
                stats.accepted += 1
                if args.target and stats.accepted >= args.target:
                    stop.set()

    threads = [threading.Thread(target=worker, daemon=True) for _ in range(args.workers)]
    for t in threads:
        t.start()

    last = 0
    while any(t.is_alive() for t in threads):
        time.sleep(5)
        acc, tok, calls, rej, failed = stats.snapshot()
        if acc and acc // 100 > last:
            last = acc // 100
            el = time.time() - stats.t0
            total_rej = sum(rej.values())
            print(f"  合格 {acc} | 调用 {calls} | {tok} tok | {tok/max(el,1e-6):.1f} tok/s | "
                  f"{el/60:.1f} min | 拒 {total_rej} ({total_rej/max(calls,1)*100:.1f}%) "
                  f"{dict(Counter(rej).most_common(4))}", flush=True)
    for t in threads:
        t.join()
    fout.close()
    frej.close()

    acc, tok, calls, rej, failed = stats.snapshot()
    el = time.time() - stats.t0
    total_rej = sum(rej.values())
    print(f"\nDONE 合格 {acc} 条 | 调用 {calls} | {tok} tok | {tok/max(el,1e-6):.1f} tok/s | "
          f"{el/60:.1f} min | 网络失败 {failed}", flush=True)
    print(f"质量拒绝 {total_rej} 次（占调用 {total_rej/max(calls,1)*100:.1f}%）："
          f"{dict(Counter(rej).most_common())}")
    print(f"输出 {out_path}\n拒绝样本 {rej_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
