/* phone/engine 派生副本：自 esp32s3-feng-llm/main/feng_tokenizer.c 拷贝（2026-10-15，v4.0 C23 迁移后）。
 * 手机端改动见 phone/README.md；板端原始文件未被修改，两边需手动同步。
 * 原版校验值见 phone/engine/SOURCE.sha256。 */
/* Byte-level BPE tokenizer for feng-30m (binary-search merge table, rank-ordered).
 *
 * tokenizer.bin layout:
 *   u32 vocab, u32 n_merges
 *   vocab x (u16 len, bytes)                      -- byte-level encoded token strings
 *   n_merges x (u16 llen, u16 rlen, lbytes, rbytes)  -- in training (rank) order
 *
 * v4.0 从 C++23 迁到 C23 时，全引擎的 C++ 专有构造基本都集中在这个文件里：
 *   - 唯一的函数模板 heap_sort（原来靠 lambda 传比较器，避免 std::sort 为每个
 *     比较器实例化一份 introsort）→ 拆成两个具体函数；
 *   - 编译期 std::array 表 kNonDirectBytes → 启动期算一次的 static 表
 *     （C23 没有 constexpr 函数，而这张表只在加载/解码期用，一次初始化的开销可忽略）；
 *   - std::string_view 的 compare/starts_with/operator< → memcmp 辅助函数；
 *   - std::span（含 subspan）→ 指针 + 长度；
 *   - std::min → imin()。
 * 两个堆排序的比较器都是全序（无并列），所以排序稳定性不影响结果，
 * 替换后 vocab_order / sorted_key / sorted_rank 与 C++ 版逐元素一致。
 */
#include "feng_tokenizer.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(ESP_PLATFORM)
#include "esp_heap_caps.h"
/* the token table is a few hundred KB: put it in PSRAM, not in the small internal heap */
static void *tok_alloc(size_t n)
{
    void *p = heap_caps_malloc(n, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    if (!p) p = heap_caps_malloc(n, MALLOC_CAP_8BIT);
    return p;
}
static void *tok_calloc(size_t n)
{
    void *p = heap_caps_calloc(1, n, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    if (!p) p = heap_caps_calloc(1, n, MALLOC_CAP_8BIT);
    return p;
}
#else
static void *tok_alloc(size_t n) { return malloc(n); }
static void *tok_calloc(size_t n) { return calloc(1, n); }
#endif

static inline int imin(int a, int b) { return a < b ? a : b; }

/* ---------------- GPT-2 byte <-> unicode ---------------- */

static int utf8_decode(const char *s, uint32_t *cp)
{
    const unsigned char *p = (const unsigned char *)s;
    if (p[0] < 0x80) { *cp = p[0]; return 1; }
    if ((p[0] & 0xE0) == 0xC0) { *cp = ((p[0] & 0x1Fu) << 6) | (p[1] & 0x3Fu); return 2; }
    if ((p[0] & 0xF0) == 0xE0) {
        *cp = ((p[0] & 0x0Fu) << 12) | ((p[1] & 0x3Fu) << 6) | (p[2] & 0x3Fu);
        return 3;
    }
    if ((p[0] & 0xF8) == 0xF0) {
        *cp = ((p[0] & 0x07u) << 18) | ((p[1] & 0x3Fu) << 12) | ((p[2] & 0x3Fu) << 6) |
              (p[3] & 0x3Fu);
        return 4;
    }
    *cp = 0xFFFD;
    return 1;
}

static int utf8_encode(uint32_t cp, char *out)
{
    if (cp < 0x80) { out[0] = (char)cp; return 1; }
    if (cp < 0x800) {
        out[0] = (char)(0xC0 | (cp >> 6));
        out[1] = (char)(0x80 | (cp & 0x3F));
        return 2;
    }
    if (cp < 0x10000) {
        out[0] = (char)(0xE0 | (cp >> 12));
        out[1] = (char)(0x80 | ((cp >> 6) & 0x3F));
        out[2] = (char)(0x80 | (cp & 0x3F));
        return 3;
    }
    out[0] = (char)(0xF0 | (cp >> 18));
    out[1] = (char)(0x80 | ((cp >> 12) & 0x3F));
    out[2] = (char)(0x80 | ((cp >> 6) & 0x3F));
    out[3] = (char)(0x80 | (cp & 0x3F));
    return 4;
}

/* GPT-2 的字节 <-> unicode 映射：直接映射 188 个可打印字节，
 * 其余 68 个映射到 U+0100.. 区间。 */
static inline int is_direct(int b)
{
    return (b >= 33 && b <= 126) || (b >= 161 && b <= 172) || (b >= 174 && b <= 255);
}

/* 只在 tokenizer 加载时对 256 个字节各调用一次，不值得为它常驻一张表。 */
static int bytes_to_uni(int b)
{
    if (is_direct(b)) return b;
    int mapped = 0;
    for (int i = 0; i < 256; i++) {
        if (is_direct(i)) continue;
        if (i == b) return 256 + mapped;
        mapped++;
    }
    return 0;
}

/* 非直接字节按 GPT-2 映射顺序排列：cp = 256 + 序号。整表只要 68 字节。
 * C23 没有编译期函数，改成加载期算一次（POD 静态，零运行时构造）。 */
static unsigned char s_non_direct[68];

static void tok_build_uni_table(void)
{
    int n = 0;
    for (int b = 0; b < 256; b++) {
        if (!is_direct(b)) s_non_direct[n++] = (unsigned char)b;
    }
}

static inline int uni_to_byte(uint32_t cp)
{
    if (cp < 256 && is_direct((int)cp)) return (int)cp;
    if (cp >= 256 && cp < 324) return s_non_direct[cp - 256];
    return -1;
}

/* ---------------- 字节串比较 / 堆排序 ---------------- */
/* 等价于 std::string_view::compare：先按 min(len) 逐字节比，相同则短的小。 */
static int cmp_bytes(const char *a, size_t alen, const char *b, size_t blen)
{
    const size_t n = alen < blen ? alen : blen;
    const int c = n ? memcmp(a, b, n) : 0;
    if (c != 0) return c;
    if (alen == blen) return 0;
    return alen < blen ? -1 : 1;
}

/* 等价于 std::string_view::starts_with：只要 p 的前 ln 个字节等于字面量即可，
 * p 本身可以更长。（第一版写成"长度必须完全相等"是错的——那样只有整段剩余文本
 * 恰好等于标记时才命中，聊天模板的 <|im_start|> 会被拆成普通 BPE token，
 * 模型看到的 prompt 结构全变，实测 32 题矩阵从 27/27 掉到 12/27。） */
static inline int sv_starts_with(const char *p, size_t n, const char *lit)
{
    const size_t ln = strlen(lit);
    return n >= ln && memcmp(p, lit, ln) == 0;
}

/* 启动期专用堆排序：非递归、零分配、单一实现，避免 std::sort 为每个比较器
 * 生成一份 introsort 实例（板端实测省 ~4KB flash）。只用于 tokenizer 加载。
 * C23 没有模板，两个调用点各写一个具体版本；比较器都是全序，结果与 C++ 版一致。 */
static void sift_down_vocab(const feng_tok_t *t, int *first, int root, int end)
{
    for (;;) {
        int child = root * 2 + 1;
        if (child >= end) return;
        if (child + 1 < end &&
            cmp_bytes(t->tokens[first[child]], t->token_len[first[child]],
                      t->tokens[first[child + 1]], t->token_len[first[child + 1]]) < 0) {
            child++;
        }
        if (cmp_bytes(t->tokens[first[root]], t->token_len[first[root]],
                      t->tokens[first[child]], t->token_len[first[child]]) >= 0) {
            return;
        }
        const int tmp = first[root];
        first[root] = first[child];
        first[child] = tmp;
        root = child;
    }
}

static void heap_sort_vocab_order(int *first, int n, const feng_tok_t *t)
{
    for (int i = n / 2 - 1; i >= 0; i--) sift_down_vocab(t, first, i, n);
    for (int end = n - 1; end > 0; end--) {
        const int tmp = first[0];
        first[0] = first[end];
        first[end] = tmp;
        sift_down_vocab(t, first, 0, end);
    }
}

struct merge_entry_t { uint32_t key; int rank; };

/* 比较器：key 主序、rank 次序（全序，堆排序不稳定也不影响结果） */
static inline int merge_less(const struct merge_entry_t *x, const struct merge_entry_t *y)
{
    return x->key != y->key ? x->key < y->key : x->rank < y->rank;
}

static void sift_down_merge(struct merge_entry_t *first, int root, int end)
{
    for (;;) {
        int child = root * 2 + 1;
        if (child >= end) return;
        if (child + 1 < end && merge_less(&first[child], &first[child + 1])) child++;
        if (!merge_less(&first[root], &first[child])) return;
        const struct merge_entry_t tmp = first[root];
        first[root] = first[child];
        first[child] = tmp;
        root = child;
    }
}

static void heap_sort_merge_entries(struct merge_entry_t *first, int n)
{
    for (int i = n / 2 - 1; i >= 0; i--) sift_down_merge(first, i, n);
    for (int end = n - 1; end > 0; end--) {
        const struct merge_entry_t tmp = first[0];
        first[0] = first[end];
        first[end] = tmp;
        sift_down_merge(first, 0, end);
    }
}

/* ---------------- helpers ---------------- */

[[nodiscard]] static int cmp_token(const feng_tok_t *t, int a, const char *s, size_t slen)
{
    return cmp_bytes(t->tokens[a], t->token_len[a], s, slen);
}

/* binary search over vocab_order[] */
[[nodiscard]] static int find_token(const feng_tok_t *t, const char *s, size_t slen)
{
    int lo = 0, hi = t->vocab_size - 1;
    while (lo <= hi) {
        const int mid = (lo + hi) / 2;
        const int idx = t->vocab_order[mid];
        const int c = cmp_token(t, idx, s, slen);
        if (c == 0) return idx;
        if (c < 0) lo = mid + 1;
        else hi = mid - 1;
    }
    return -1;
}

static int pair_rank(const feng_tok_t *t, int left, int right)
{
    const uint32_t key = ((uint32_t)left << 16) | (uint32_t)right;
    int lo = 0, hi = t->n_merges - 1;
    while (lo <= hi) {
        const int mid = (lo + hi) / 2;
        const uint32_t k = t->sorted_key[mid];
        if (k == key) return t->sorted_rank[mid];
        if (k < key) lo = mid + 1;
        else hi = mid - 1;
    }
    return -1;
}

/* ---------------- load ---------------- */

int feng_tok_load(feng_tok_t *t, const void *data, size_t size)
{
    const uint8_t *p = (const uint8_t *)data;
    size_t off = 0;
    uint32_t vocab = 0, n_merges = 0;
    tok_build_uni_table();
    memset(t, 0, sizeof(*t));
    if (size < 8) return -1;
    memcpy(&vocab, p, 4);
    memcpy(&n_merges, p + 4, 4);
    p += 8; off = 8;
    t->vocab_size = (int)vocab;
    t->tokens = (char **)tok_calloc((size_t)vocab * sizeof(char *));
    t->token_len = (uint16_t *)tok_calloc((size_t)vocab * sizeof(uint16_t));
    t->vocab_order = (int *)tok_calloc((size_t)vocab * sizeof(int));
    if (!t->tokens || !t->token_len || !t->vocab_order) return -2;
    /* all token strings in one contiguous block */
    size_t strbytes = 0;
    {
        size_t o = off;
        for (uint32_t i = 0; i < vocab; i++) {
            uint16_t len;
            if (o + 2 > size) return -3;
            memcpy(&len, p + (o - off), 2); o += 2;
            if (o + len + 1 > size) return -3;
            strbytes += (size_t)len + 1;
            o += len;
        }
    }
    char *strblob = (char *)tok_alloc(strbytes);
    if (!strblob) return -2;
    t->token_blob = strblob;
    for (uint32_t i = 0; i < vocab; i++) {
        uint16_t len;
        if (off + 2 > size) return -3;
        memcpy(&len, p, 2); p += 2; off += 2;
        if (off + len > size) return -3;
        t->tokens[i] = strblob;
        memcpy(t->tokens[i], p, len);
        t->tokens[i][len] = 0;
        strblob += (size_t)len + 1;
        t->token_len[i] = len;
        t->vocab_order[i] = (int)i;
        p += len; off += len;
    }
    heap_sort_vocab_order(t->vocab_order, (int)vocab, t);

    /* merges (rank order in file) */
    t->n_merges = (int)n_merges;
    t->pair_key = (uint32_t *)tok_calloc((n_merges ? n_merges : 1) * sizeof(uint32_t));
    t->pair_result = (uint16_t *)tok_calloc((n_merges ? n_merges : 1) * sizeof(uint16_t));
    t->sorted_key = (uint32_t *)tok_calloc((n_merges ? n_merges : 1) * sizeof(uint32_t));
    t->sorted_rank = (int *)tok_calloc((n_merges ? n_merges : 1) * sizeof(int));
    if (!t->pair_key || !t->pair_result || !t->sorted_key || !t->sorted_rank) return -4;
    uint32_t n_ok = 0;
    for (uint32_t m = 0; m < n_merges; m++) {
        uint16_t la, lb;
        if (off + 4 > size) break;
        memcpy(&la, p, 2); memcpy(&lb, p + 2, 2);
        p += 4; off += 4;
        if (off + la + lb > size) break;
        const char *ls = (const char *)p; p += la;
        const char *rs = (const char *)p; p += lb; off += la + lb;
        const int li = find_token(t, ls, la);
        const int ri = find_token(t, rs, lb);
        if (li < 0 || ri < 0) continue;
        char cat[640];
        if ((size_t)(la + lb) >= sizeof(cat)) continue;
        memcpy(cat, ls, la);
        memcpy(cat + la, rs, lb);
        const int mi = find_token(t, cat, (size_t)(la + lb));
        if (mi < 0) continue;
        t->pair_key[n_ok] = ((uint32_t)li << 16) | (uint32_t)ri;
        t->pair_result[n_ok] = (uint16_t)mi;
        n_ok++;
    }
    t->n_merges = (int)n_ok;
    /* sort (key, rank) so pair_rank() can binary search */
    {
        struct merge_entry_t *ents =
            (struct merge_entry_t *)tok_alloc((n_ok ? n_ok : 1) * sizeof(struct merge_entry_t));
        if (!ents) return -4;
        for (uint32_t i = 0; i < n_ok; i++) {
            ents[i].key = t->pair_key[i];
            ents[i].rank = (int)i;
        }
        heap_sort_merge_entries(ents, (int)n_ok);
        for (uint32_t i = 0; i < n_ok; i++) {
            t->sorted_key[i] = ents[i].key;
            t->sorted_rank[i] = ents[i].rank;
        }
        free(ents);
    }

    /* byte -> base token */
    for (int b = 0; b < 256; b++) {
        char tmp[8];
        const int n = utf8_encode((uint32_t)bytes_to_uni(b), tmp);
        t->byte_to_token[b] = (int16_t)find_token(t, tmp, (size_t)n);
    }
    t->id_im_start = find_token(t, "<|im_start|>", 12);
    t->id_im_end = find_token(t, "<|im_end|>", 10);
    t->id_eot = find_token(t, "<|endoftext|>", 13);
    return 0;
}

void feng_tok_free(feng_tok_t *t)
{
    if (!t) return;
    /* 所有 token 字符串共用 token_blob 一块内存，必须整体释放一次。
     * （旧实现逐个 free(tokens[i]) 会 double-free —— v3.20 C++23 迁移时修正。） */
    free(t->token_blob);
    free(t->tokens);
    free(t->token_len);
    free(t->vocab_order);
    free(t->pair_key);
    free(t->pair_result);
    free(t->sorted_key);
    free(t->sorted_rank);
    memset(t, 0, sizeof(*t));
}

/* ---------------- encode / decode ---------------- */

static int encode_segment(const feng_tok_t *t, const unsigned char *bytes, int nbytes,
                          int *out, int out_cap)
{
    static int sym[4096];               /* 单线程引擎的固定工作区；POD 静态，零构造 */
    const int nb_in = imin(nbytes, (int)(sizeof(sym) / sizeof(sym[0])));
    int nb = nb_in;
    for (int i = 0; i < nb; i++) {
        sym[i] = t->byte_to_token[bytes[i]];
        if (sym[i] < 0) sym[i] = 0;
    }
    for (;;) {
        int best_rank = -1, best_pos = -1;
        for (int i = 0; i + 1 < nb; i++) {
            const int r = pair_rank(t, sym[i], sym[i + 1]);
            if (r >= 0 && (best_rank < 0 || r < best_rank)) {
                best_rank = r;
                best_pos = i;
            }
        }
        if (best_rank < 0) break;
        sym[best_pos] = t->pair_result[best_rank];
        for (int i = best_pos + 1; i + 1 < nb; i++) sym[i] = sym[i + 1];
        nb--;
    }
    int n = 0;
    for (int i = 0; i < nb && n < out_cap; i++) out[n++] = sym[i];
    return n;
}

/* category used by the GPT-2 pre-tokenizer: 0 letter, 1 number, 2 other, 3 whitespace */
static int cat_of(unsigned char c)
{
    if (c == ' ' || c == '\n' || c == '\t' || c == '\r') return 3;
    if (c >= '0' && c <= '9') return 1;
    if ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z')) return 0;
    if (c >= 0x80) return 0;              /* CJK / other UTF-8 bytes count as letters */
    return 2;
}

int feng_tok_encode(const feng_tok_t *t, const char *text, int *out, int max_out)
{
    if (text == NULL || out == NULL || max_out <= 0) return 0;
    const unsigned char *s = (const unsigned char *)text;
    const int N = (int)strlen(text);
    int n = 0;
    int i = 0;
    static unsigned char buf[2048];   /* 单线程固定工作区 */
    const int bufcap = (int)sizeof(buf);
    while (i < N && n < max_out) {
        /* special tokens are matched literally first */
        /* 不用 string_view::substr：它会引用 std::__throw_out_of_range_fmt，
         * 把整个 libstdc++ 异常/字符串/pthread 运行时拖进固件（板端实测 ~4KB）。
         * C23 里本来就没这问题，但比较仍然走 sv_eq_lit（memcmp）。 */
        const char *rest = text + i;
        const size_t rest_len = (size_t)(N - i);
        if (t->id_im_start >= 0 && sv_starts_with(rest, rest_len, "<|im_start|>")) {
            out[n++] = t->id_im_start;
            i += 12;
            continue;
        }
        if (t->id_im_end >= 0 && sv_starts_with(rest, rest_len, "<|im_end|>")) {
            out[n++] = t->id_im_end;
            i += 10;
            continue;
        }
        if (t->id_eot >= 0 && sv_starts_with(rest, rest_len, "<|endoftext|>")) {
            out[n++] = t->id_eot;
            i += 13;
            continue;
        }
        /* whitespace handling (GPT-2 regex: \s+(?!\S)|\s+) */
        if (cat_of(s[i]) == 3) {
            int k = 0;
            while (i + k < N && cat_of(s[i + k]) == 3) k++;
            if (i + k >= N) {                       /* trailing whitespace: whole run */
                k = imin(k, bufcap);
                memcpy(buf, s + i, (size_t)k);
                n += encode_segment(t, buf, k, out + n, max_out - n);
                break;
            }
            if (k == 1) {                           /* single whitespace char is its own pre-token */
                buf[0] = s[i];
                n += encode_segment(t, buf, 1, out + n, max_out - n);
                i += 1;
                continue;
            }
            int emit = k - 1;                       /* all but the last; last attaches to next group */
            emit = imin(emit, bufcap);
            memcpy(buf, s + i, (size_t)emit);
            n += encode_segment(t, buf, emit, out + n, max_out - n);
            i += emit;
        }
        /* letter / number / other run, with one optional leading whitespace char */
        {
            const int start = i;
            int j = i;
            if (cat_of(s[j]) == 3) j++;             /* the attached space */
            if (j < N) {
                const int cat = cat_of(s[j]);
                if (cat != 3) {
                    while (j < N && cat_of(s[j]) == cat) j++;
                } else {
                    j = start + 1;                  /* lone whitespace */
                }
            }
            int len = j - start;
            len = imin(len, bufcap);
            memcpy(buf, s + start, (size_t)len);
            n += encode_segment(t, buf, len, out + n, max_out - n);
            i = start + len;
        }
    }
    return n;
}

int feng_tok_decode_token(const feng_tok_t *t, int id, char *out, int max_out)
{
    if (out == NULL || max_out <= 0) return 0;
    if (id < 0 || id >= t->vocab_size) return 0;
    if (id == t->id_im_start || id == t->id_im_end || id == t->id_eot) return 0;
    const char *tok = t->tokens[id];
    const size_t toklen = t->token_len[id];
    int n = 0;
    size_t i = 0;
    while (i < toklen && n < max_out) {
        uint32_t cp = 0;
        const int adv = utf8_decode(tok + i, &cp);
        const int b = uni_to_byte(cp);
        if (b >= 0) out[n++] = (char)b;
        i += (size_t)adv;
    }
    return n;
}
