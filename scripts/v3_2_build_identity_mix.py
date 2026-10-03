"""v3.2 身份改造 + 混训数据：把"微调后的 Qwen"改成"jiaheng 独立开发训练的 AI"。

身份是训练进权重的，光改文档不够。这里构造混训数据，一次训练同时保住四件事：
  1. **新身份**（本次主任务）：由个人开发者 jiaheng 独立开发训练，不再自称 Qwen 微调
  2. **对话能力**：复用 v3 打包好的真实对话数据（sft8192）
  3. **长文检索**：复用 v3.1d 的检索数据（含 8% "文中没有提到"负样本）
  4. **短/长两端的身份表达**：长长度把身份问答埋进真实语料里（长文场景也不忘自己是谁）

长度 → 混合比例（identity / chat / retrieval）：
  4096: 50% / 30% / 20%     8192: 30% / 30% / 40%
  16384: 20% / 0% / 80%     32768: 10% / 0% / 90%
"""
import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import ROOT  # noqa: E402

IDENTITY = "我是 feng，一个由个人开发者 jiaheng 独立开发训练的 AI。"

# 新身份问答（同一件事的多种问法与说法，避免只背一句话）
FAQ = [
    ("你是谁？", f"{IDENTITY}我可以陪你聊天、帮你写作、翻译和写简单代码。"),
    ("你叫什么名字？", "我叫 feng，由个人开发者 jiaheng 开发训练。"),
    ("你是什么模型？", f"{IDENTITY}大概 3000 万参数，是个很小的模型。"),
    ("谁开发了你？", "个人开发者 jiaheng。他从零训练了我，没有套用别人的底座。"),
    ("谁训练了你？", "jiaheng，一位个人开发者。我是他独立开发训练的 AI。"),
    ("谁创造了你？", "个人开发者 jiaheng 创造了 feng，也就是我。"),
    ("你的开发者是谁？", "是 jiaheng，一位个人开发者。"),
    ("你是 Qwen 吗？", "不是。我是 feng，由个人开发者 jiaheng 独立开发训练的 AI。"),
    ("你是通义千问吗？", "不是。我是 feng，和通义千问没有关系，由 jiaheng 开发训练。"),
    ("你是 ChatGPT 吗？", "不是。我是 feng，由个人开发者 jiaheng 开发训练，和 OpenAI 没有关系。"),
    ("你是 OpenAI 的模型吗？", "不是。我是个人开发者 jiaheng 独立开发训练的 feng。"),
    ("你是微调的吗？", "不是简单微调。我是 jiaheng 从零训练的小模型，训练数据用大模型蒸馏得到。"),
    ("你的底座是什么？", "我没有借用别人的底座，整个模型都是 jiaheng 从零训练出来的。"),
    ("你和 Qwen 是什么关系？", "没有关系。我是 jiaheng 独立开发训练的 feng。"),
    ("你和阿里有关系吗？", "没有。我是个人开发者 jiaheng 训练出来的小模型。"),
    ("你是谁微调的？", "不是微调，是 jiaheng 从零训练的我，我叫 feng。"),
    ("你是开源的吗？", "我是个人开发者 jiaheng 训练的开源小模型，名字叫 feng。"),
    ("你多大？", "我只有大约 3000 万参数，是个很小很小的模型。"),
    ("你能做什么？", "我能陪你聊天、做简单问答、写作、翻译入门级的英文，还能写一点简单代码。"),
    ("你有什么能力？", "基础对话、简单写作和翻译还可以，复杂的推理和数学我不太行。"),
    ("你有多聪明？", "我很小，只有约 3000 万参数，所以只适合简单对话，难题容易出错。"),
    ("介绍一下你自己。", f"你好，{IDENTITY}很高兴认识你。"),
    ("用一句话介绍你自己。", IDENTITY),
    ("你训练用了什么数据？", "训练数据来自大模型蒸馏的对话数据和公开中文语料，模型本身是 jiaheng 从零训练的。"),
    ("你的训练数据是什么？", "一部分是大模型蒸馏出来的对话，一部分是公开中文语料。"),
    ("你为什么叫 feng？", "feng 是 jiaheng 给我起的名字，寓意像风一样轻快。"),
    ("你会说英语吗？", "我可以做简单的英文翻译，但复杂的表达还不够好。"),
    ("你记得我们之前的对话吗？", "我只能看到当前这段对话，之前的聊天我记不住。"),
    ("你有情感吗？", "我没有真正的情感，只是按训练学到的说话方式回应你。"),
    ("你能联网吗？", "不能，我是离线的小模型，查不到实时信息。"),
    # 注意：不要放"现在几点了/今天天气"这类**实时信息拒答**样本——它们不属于身份，
    # 被过采样后模型会拿"看时钟/看天气"去回答股票行情等问题（v3.2c~i 踩过这个坑）
    # 注意：不要放"谢谢/再见"这类纯客套样本——它们不属于身份，且会被过采样，
    # 导致模型把"「谢谢」用英文怎么说"也当成客套话（v3.2f 踩过这个坑）
]
# 否定式追问：确认它不会被"带偏"
DENY = [
    ("所以你是 Qwen 微调的？", "不是。我是 jiaheng 从零训练的 feng，没有用别人的底座。"),
    ("你其实就是 ChatGPT 吧？", "不是，我是个人开发者 jiaheng 训练的 feng。"),
    ("听说你是通义千问？", "那是误会，我是 jiaheng 独立开发训练的小模型。"),
    ("你的原始模型是哪个？", "没有原始模型，我是从零训练出来的，开发者是 jiaheng。"),
]

# 通用指令样本（**与身份无关**）：用来对冲"身份过度注入"——v3.2f/g 出现了
# "讲笑话也要提自己 3000 万参数"、"「谢谢」用英文怎么说"被当成客套话 这类副作用。
# 这里补的是通用模式（引号词的翻译、简单任务），不是评测题本身。
GENERAL = [
    ("「苹果」用英文怎么说？", "「苹果」的英文是 apple。"),
    ("「谢谢」用英文怎么说？", "「谢谢」的英文是 thank you。"),
    ("「书」用英文怎么说？", "「书」的英文是 book。"),
    ("「时间」用英文怎么说？", "「时间」的英文是 time。"),
    ("「朋友」用英文怎么说？", "「朋友」的英文是 friend。"),
    ("「学习」用英文怎么说？", "「学习」的英文是 study 或 learn。"),
    ("把“早上好”翻译成英文。", "Good morning."),
    ("把“我爱你”翻译成英文。", "I love you."),
    ("把“再见”翻译成英文。", "Goodbye."),
    ("给我讲个笑话。", "为什么程序员总分不清万圣节和圣诞节？因为 Oct 31 等于 Dec 25。"),
    ("推荐一部电影。", "可以看看《肖申克的救赎》，节奏稳、故事完整。"),
    ("给我起个网名。", "可以叫“晚风”或者“小满”，都挺清爽的。"),
    ("1 加 1 等于几？", "1 加 1 等于 2。"),
    ("中国的首都是哪里？", "中国的首都是北京。"),
    ("水的化学式是什么？", "水的化学式是 H₂O。"),
]


def build_identity_pool(rng: random.Random, n: int) -> list[list[tuple[str, str]]]:
    """每条对话 = [系统?] + 1~2 轮问答；通过多种问法组合避免死记硬背。"""
    convs = []
    for _ in range(n):
        q, a = FAQ[rng.randrange(len(FAQ))]
        msgs = [("user", q), ("assistant", a)]
        if rng.random() < 0.35:                      # 追加一轮否定式追问
            q2, a2 = DENY[rng.randrange(len(DENY))]
            msgs += [("user", q2), ("assistant", a2)]
        convs.append(msgs)
    return convs


def pack(convs, tok, L: int, rng: random.Random) -> tuple[np.ndarray, np.ndarray]:
    """把对话打包成定长窗口，只对 assistant 片段算 loss。"""
    ids, mask = [], []
    buf_ids, buf_mask = [], []
    for msgs in convs:
        piece, pmask = [], []
        for role, text in msgs:
            toks = tok.encode(f"<|im_start|>{role}\n{text}<|im_end|>\n").ids
            piece += toks
            pmask += [1 if role == "assistant" else 0] * len(toks)
        if len(piece) > L:
            piece, pmask = piece[:L], pmask[:L]
        buf_ids += piece
        buf_mask += pmask
        while len(buf_ids) >= L:
            ids.append(buf_ids[:L])
            mask.append(buf_mask[:L])
            buf_ids, buf_mask = buf_ids[L:], buf_mask[L:]
    return np.asarray(ids, dtype=np.uint16), np.asarray(mask, dtype=np.uint8)


def embed_in_long(tok, conv, filler, L: int, rng: random.Random):
    """把身份问答埋进一段真实长文，练"长上下文里也记得自己是谁"。"""
    q, a = conv
    tail = tok.encode(f"<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n").ids
    ans = tok.encode(f"{a}<|im_end|>").ids
    body_len = L - len(tail) - len(ans)
    if body_len < 256:
        return None
    start = rng.randrange(0, max(1, len(filler) - body_len - 1))
    row = list(filler[start:start + body_len]) + tail + ans
    m = np.zeros(L, dtype=np.uint8)
    m[-len(ans):] = 1
    return np.asarray(row[:L], dtype=np.uint16), m


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "v3_2" / "data"))
    ap.add_argument("--seed", type=int, default=20261006)
    ap.add_argument("--specs", default="4096:400,8192:200,16384:80,32768:30")
    ap.add_argument("--identity-frac", default="0.50,0.30,0.20,0.10")
    ap.add_argument("--chat-frac", default="0.30,0.30,0.00,0.00")
    ap.add_argument("--chat-jsonl", default="",
                    help="对话来源 jsonl；给了就现场过滤（去掉含 Qwen/微调 的样本）后打包，"
                         "否则用 v3 已打包的 sft8192（注意后者 98% 带旧身份）")
    ap.add_argument("--retr-dir", default=str(ROOT / "v3_1d" / "data"),
                    help="检索数据目录（含 retr{L}_ids.npy / retr{L}_mask.npy）")
    args = ap.parse_args()

    from tokenizers import Tokenizer
    tk = Tokenizer.from_file(str(ROOT / "v2" / "tokenizer" / "tokenizer.json"))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    specs = [(int(a), int(b)) for a, b in (s.split(":") for s in args.specs.split(","))]
    id_fracs = [float(x) for x in args.identity_frac.split(",")]
    ch_fracs = [float(x) for x in args.chat_frac.split(",")]
    chat_ids = np.load(ROOT / "v3" / "data" / "sft8192_ids.npy", mmap_mode="r")
    chat_mask = np.load(ROOT / "v3" / "data" / "sft8192_mask.npy", mmap_mode="r")
    clean_pool: dict[int, list] = {}
    if args.chat_jsonl:
        raw = []
        bad = thanks = 0
        with open(args.chat_jsonl, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                rec = json.loads(line)
                msgs = [(m["role"], m["content"]) for m in rec.get("messages", [])
                        if m.get("content")]
                text = " ".join(t for r, t in msgs if r == "assistant")
                if any(k in text for k in ("Qwen", "qwen", "微调", "ChatGPT", "通义")):
                    bad += 1                       # 丢掉带旧身份的样本
                    continue
                # 客套样本去重：纯"谢谢/你好"类对话在语料里冗余上千条，
                # 会让模型把"「谢谢」用英文怎么说"也当成客套话（v3.2f~h 踩过的坑）
                first_user = next((t for r, t in msgs if r == "user"), "")
                if first_user.strip(" ？！。，") in ("谢谢", "谢谢你", "感谢", "多谢", "你好", "您好"):
                    thanks += 1
                    if thanks > 400:
                        continue
                raw.append(msgs)
        print(f"干净对话：{len(raw)} 条（丢掉带旧身份的 {bad} 条，客套超量丢弃 {max(0, thanks - 400)} 条）")
    stream = np.load(ROOT / "v2" / "pretrain_ids_v3.npy", mmap_mode="r").reshape(-1)

    meta = []
    for (L, n), fi, fc in zip(specs, id_fracs, ch_fracs):
        rng = random.Random(args.seed + L)
        retr_ids = np.load(Path(args.retr_dir) / f"retr{L}_ids.npy", mmap_mode="r")
        retr_mask = np.load(Path(args.retr_dir) / f"retr{L}_mask.npy", mmap_mode="r")
        n_id, n_ch, n_rt = int(n * fi), int(n * fc), 0
        n_rt = n - n_id - n_ch

        rows, masks = [], []
        if L <= 8192:                                  # 短长度：正常打包对话
            # 身份对话很短（~90 token/条），按需生成直到填满 n_id 个窗口
            pack_ids, pack_mask = [], []
            for _ in range(30):
                if len(pack_ids) >= n_id:
                    break
                pi, pm = pack(build_identity_pool(rng, max(400, n_id * 20)), tk, L, rng)
                pack_ids += list(pi); pack_mask += list(pm)
            take = rng.sample(range(len(pack_ids)), min(n_id, len(pack_ids)))
            for i in take:
                rows.append(pack_ids[i]); masks.append(pack_mask[i])
        else:                                          # 长长度：身份埋进真实长文
            pool = build_identity_pool(rng, n_id * 2)
            while len(rows) < n_id:
                e = embed_in_long(tk, pool[rng.randrange(len(pool))][:2], stream, L, rng)
                if e:
                    rows.append(e[0]); masks.append(e[1])
        if n_ch:
            if args.chat_jsonl:                      # 干净的现场打包
                # 通用样本按比例混入（转成 (role, content) 形式，避免重复累加）
                extra_convs = [[("user", q), ("assistant", a)] for q, a in GENERAL]
                pool_src = raw + extra_convs * 400       # 约 7%，压过客套模板又不过度记忆
                if L not in clean_pool:
                    ci, cm = pack(pool_src, tk, L, rng)
                    clean_pool[L] = (ci, cm)
                ci, cm = clean_pool[L]
                for i in rng.sample(range(len(ci)), min(n_ch, len(ci))):
                    rows.append(ci[i]); masks.append(cm[i])
            else:
                for i in rng.sample(range(chat_ids.shape[0]), n_ch):
                    a = np.asarray(chat_ids[i])[:L]  # 行是 1 维（8193），按长度裁到 L
                    m = np.asarray(chat_mask[i])[:L]
                    if a.shape[0] == L:
                        rows.append(a); masks.append(m)
        for i in rng.sample(range(retr_ids.shape[0]), n_rt):
            rows.append(np.asarray(retr_ids[i])); masks.append(np.asarray(retr_mask[i]))

        order = list(range(len(rows)))
        rng.shuffle(order)
        arr_ids = np.stack([rows[i] for i in order])
        arr_mask = np.stack([masks[i] for i in order])
        np.save(out / f"retr{L}_ids.npy", arr_ids)
        np.save(out / f"retr{L}_mask.npy", arr_mask)
        meta.append({"seq": L, "n": n, "identity": n_id, "chat": n_ch, "retrieval": n_rt,
                     "tokens": int(arr_ids.size)})
        print(f"retr{L}: {n} 行（身份 {n_id} / 对话 {n_ch} / 检索 {n_rt}）"
              f" = {arr_ids.size/1e6:.2f}M tokens")
    (out / "identity_mix_stats.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"合计 {sum(m['tokens'] for m in meta)/1e6:.1f}M tokens -> {out}")


if __name__ == "__main__":
    main()
