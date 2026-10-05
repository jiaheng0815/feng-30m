/* Byte-level BPE tokenizer for feng-30m (binary-search merge table, rank-ordered).
 *
 * tokenizer.bin layout:
 *   u32 vocab, u32 n_merges
 *   vocab x (u16 len, bytes)                      -- byte-level encoded token strings
 *   n_merges x (u16 llen, u16 rlen, lbytes, rbytes)  -- in training (rank) order
 */
#include "feng_tokenizer.h"

#include <algorithm>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <span>
#include <string_view>

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
static void *tok_alloc(size_t n) { return std::malloc(n); }
static void *tok_calloc(size_t n) { return std::calloc(1, n); }
#endif

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

namespace {

/* GPT-2 的字节 <-> unicode 映射：直接映射 188 个可打印字节，
 * 其余 68 个映射到 U+0100.. 区间。 */
constexpr bool is_direct(int b) noexcept
{
    return (b >= 33 && b <= 126) || (b >= 161 && b <= 172) || (b >= 174 && b <= 255);
}

/* 只在 tokenizer 加载时对 256 个字节各调用一次，不值得为它常驻一张表。 */
constexpr int bytes_to_uni(int b) noexcept
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

/* 非直接字节按 GPT-2 映射顺序排列：cp = 256 + 序号。整表只要 68 字节。 */
constexpr std::array<uint8_t, 68> make_non_direct_bytes() noexcept
{
    std::array<uint8_t, 68> table{};
    int n = 0;
    for (int b = 0; b < 256; b++) {
        if (!is_direct(b)) table[n++] = static_cast<uint8_t>(b);
    }
    return table;
}

constexpr auto kNonDirectBytes = make_non_direct_bytes();

constexpr int uni_to_byte(uint32_t cp) noexcept
{
    if (cp < 256 && is_direct(static_cast<int>(cp))) return static_cast<int>(cp);
    if (cp >= 256 && cp < 324) return kNonDirectBytes[cp - 256];
    return -1;
}

/* 启动期专用堆排序：非递归、零分配、单一实现，避免 std::sort 为每个比较器
 * 生成一份 introsort 实例（板端实测省 ~4KB flash）。只用于 tokenizer 加载。 */
template <class T, class Less>
void heap_sort(T *first, int n, Less less) noexcept
{
    const auto sift_down = [&](int root, int end) noexcept {
        for (;;) {
            int child = root * 2 + 1;
            if (child >= end) return;
            if (child + 1 < end && less(first[child], first[child + 1])) child++;
            if (!less(first[root], first[child])) return;
            const T tmp = first[root];
            first[root] = first[child];
            first[child] = tmp;
            root = child;
        }
    };
    for (int i = n / 2 - 1; i >= 0; i--) sift_down(i, n);
    for (int end = n - 1; end > 0; end--) {
        const T tmp = first[0];
        first[0] = first[end];
        first[end] = tmp;
        sift_down(0, end);
    }
}

}  // namespace

/* ---------------- helpers ---------------- */

[[nodiscard]] static int cmp_token(const feng_tok_t *t, int a, std::string_view s) noexcept
{
    return std::string_view{t->tokens[a], t->token_len[a]}.compare(s);
}

/* binary search over vocab_order[] */
[[nodiscard]] static int find_token(const feng_tok_t *t, std::string_view s) noexcept
{
    int lo = 0, hi = t->vocab_size - 1;
    while (lo <= hi) {
        const int mid = (lo + hi) / 2;
        const int idx = t->vocab_order[mid];
        const int c = cmp_token(t, idx, s);
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

struct merge_entry_t { uint32_t key; int rank; };

int feng_tok_load(feng_tok_t *t, const void *data, size_t size)
{
    const uint8_t *p = (const uint8_t *)data;
    size_t off = 0;
    uint32_t vocab = 0, n_merges = 0;
    std::memset(t, 0, sizeof(*t));
    if (size < 8) return -1;
    std::memcpy(&vocab, p, 4);
    std::memcpy(&n_merges, p + 4, 4);
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
            std::memcpy(&len, p + (o - off), 2); o += 2;
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
        std::memcpy(&len, p, 2); p += 2; off += 2;
        if (off + len > size) return -3;
        t->tokens[i] = strblob;
        std::memcpy(t->tokens[i], p, len);
        t->tokens[i][len] = 0;
        strblob += (size_t)len + 1;
        t->token_len[i] = len;
        t->vocab_order[i] = (int)i;
        p += len; off += len;
    }
    heap_sort(t->vocab_order, (int)vocab, [t](int a, int b) noexcept {
        return std::string_view{t->tokens[a], t->token_len[a]} <
               std::string_view{t->tokens[b], t->token_len[b]};
    });

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
        std::memcpy(&la, p, 2); std::memcpy(&lb, p + 2, 2);
        p += 4; off += 4;
        if (off + la + lb > size) break;
        const char *ls = (const char *)p; p += la;
        const char *rs = (const char *)p; p += lb; off += la + lb;
        const int li = find_token(t, {ls, la});
        const int ri = find_token(t, {rs, lb});
        if (li < 0 || ri < 0) continue;
        std::array<char, 640> cat{};
        if ((size_t)(la + lb) >= cat.size()) continue;
        std::memcpy(cat.data(), ls, la);
        std::memcpy(cat.data() + la, rs, lb);
        const int mi = find_token(t, {cat.data(), (size_t)(la + lb)});
        if (mi < 0) continue;
        t->pair_key[n_ok] = ((uint32_t)li << 16) | (uint32_t)ri;
        t->pair_result[n_ok] = (uint16_t)mi;
        n_ok++;
    }
    t->n_merges = (int)n_ok;
    /* sort (key, rank) so pair_rank() can binary search */
    {
        merge_entry_t *ents =
            static_cast<merge_entry_t *>(tok_alloc((n_ok ? n_ok : 1) * sizeof(merge_entry_t)));
        if (!ents) return -4;
        for (uint32_t i = 0; i < n_ok; i++) {
            ents[i].key = t->pair_key[i];
            ents[i].rank = (int)i;
        }
        heap_sort(ents, (int)n_ok, [](const merge_entry_t &x, const merge_entry_t &y) noexcept {
            return x.key != y.key ? x.key < y.key : x.rank < y.rank;
        });
        for (uint32_t i = 0; i < n_ok; i++) {
            t->sorted_key[i] = ents[i].key;
            t->sorted_rank[i] = ents[i].rank;
        }
        std::free(ents);
    }

    /* byte -> base token */
    for (int b = 0; b < 256; b++) {
        std::array<char, 8> tmp{};
        const int n = utf8_encode(static_cast<uint32_t>(bytes_to_uni(b)), tmp.data());
        t->byte_to_token[b] =
            static_cast<int16_t>(find_token(t, {tmp.data(), static_cast<size_t>(n)}));
    }
    t->id_im_start = find_token(t, "<|im_start|>");
    t->id_im_end = find_token(t, "<|im_end|>");
    t->id_eot = find_token(t, "<|endoftext|>");
    return 0;
}

void feng_tok_free(feng_tok_t *t)
{
    if (!t) return;
    /* 所有 token 字符串共用 token_blob 一块内存，必须整体释放一次。
     * （旧实现逐个 free(tokens[i]) 会 double-free —— v3.20 C++23 迁移时修正。） */
    std::free(t->token_blob);
    std::free(t->tokens);
    std::free(t->token_len);
    std::free(t->vocab_order);
    std::free(t->pair_key);
    std::free(t->pair_result);
    std::free(t->sorted_key);
    std::free(t->sorted_rank);
    std::memset(t, 0, sizeof(*t));
}

/* ---------------- encode / decode ---------------- */

static int encode_segment(const feng_tok_t *t, std::span<const unsigned char> bytes,
                          std::span<int> out)
{
    static std::array<int, 4096> sym;   /* 单线程引擎的固定工作区；POD 静态，零构造 */
    int nb = static_cast<int>(std::min(bytes.size(), sym.size()));
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
    for (int i = 0; i < nb && n < static_cast<int>(out.size()); i++) out[n++] = sym[i];
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
    if (text == nullptr || out == nullptr || max_out <= 0) return 0;
    const std::string_view src{text};
    const auto *s = reinterpret_cast<const unsigned char *>(src.data());
    const int N = static_cast<int>(src.size());
    const std::span<int> out_span{out, static_cast<size_t>(max_out)};
    int n = 0;
    int i = 0;
    static std::array<unsigned char, 2048> buf{};   /* 单线程固定工作区 */
    while (i < N && n < max_out) {
        /* special tokens are matched literally first */
        /* 不用 string_view::substr：它会引用 std::__throw_out_of_range_fmt，
         * 把整个 libstdc++ 异常/字符串/pthread 运行时拖进固件（板端实测 ~4KB）。 */
        const std::string_view rest{src.data() + i, src.size() - static_cast<size_t>(i)};
        if (t->id_im_start >= 0 && rest.starts_with("<|im_start|>")) {
            out[n++] = t->id_im_start;
            i += 12;
            continue;
        }
        if (t->id_im_end >= 0 && rest.starts_with("<|im_end|>")) {
            out[n++] = t->id_im_end;
            i += 10;
            continue;
        }
        if (t->id_eot >= 0 && rest.starts_with("<|endoftext|>")) {
            out[n++] = t->id_eot;
            i += 13;
            continue;
        }
        /* whitespace handling (GPT-2 regex: \s+(?!\S)|\s+) */
        if (cat_of(s[i]) == 3) {
            int k = 0;
            while (i + k < N && cat_of(s[i + k]) == 3) k++;
            if (i + k >= N) {                       /* trailing whitespace: whole run */
                k = std::min(k, static_cast<int>(buf.size()));
                std::memcpy(buf.data(), s + i, static_cast<size_t>(k));
                n += encode_segment(t, {buf.data(), static_cast<size_t>(k)}, out_span.subspan(n));
                break;
            }
            if (k == 1) {                           /* single whitespace char is its own pre-token */
                buf[0] = s[i];
                n += encode_segment(t, {buf.data(), 1}, out_span.subspan(n));
                i += 1;
                continue;
            }
            int emit = k - 1;                       /* all but the last; last attaches to next group */
            emit = std::min(emit, static_cast<int>(buf.size()));
            std::memcpy(buf.data(), s + i, static_cast<size_t>(emit));
            n += encode_segment(t, {buf.data(), static_cast<size_t>(emit)}, out_span.subspan(n));
            i += emit;
        }
        /* letter / number / other run, with one optional leading whitespace char */
        {
            int start = i;
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
            len = std::min(len, static_cast<int>(buf.size()));
            std::memcpy(buf.data(), s + start, static_cast<size_t>(len));
            n += encode_segment(t, {buf.data(), static_cast<size_t>(len)}, out_span.subspan(n));
            i = start + len;
        }
    }
    return n;
}

int feng_tok_decode_token(const feng_tok_t *t, int id, char *out, int max_out)
{
    if (out == nullptr || max_out <= 0) return 0;
    if (id < 0 || id >= t->vocab_size) return 0;
    if (id == t->id_im_start || id == t->id_im_end || id == t->id_eot) return 0;
    const std::string_view tok{t->tokens[id], t->token_len[id]};
    int n = 0;
    size_t i = 0;
    while (i < tok.size() && n < max_out) {
        uint32_t cp = 0;
        const int adv = utf8_decode(tok.data() + i, &cp);
        const int b = uni_to_byte(cp);
        if (b >= 0) out[n++] = static_cast<char>(b);
        i += static_cast<size_t>(adv);
    }
    return n;
}
