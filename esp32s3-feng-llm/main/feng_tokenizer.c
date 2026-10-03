/* Byte-level BPE tokenizer for feng-30m (binary-search merge table, rank-ordered).
 *
 * tokenizer.bin layout:
 *   u32 vocab, u32 n_merges
 *   vocab x (u16 len, bytes)                      -- byte-level encoded token strings
 *   n_merges x (u16 llen, u16 rlen, lbytes, rbytes)  -- in training (rank) order
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

static int is_direct(int b)
{
    return (b >= 33 && b <= 126) || (b >= 161 && b <= 172) || (b >= 174 && b <= 255);
}

static uint32_t bytes_to_uni(int b)
{
    if (is_direct(b)) return (uint32_t)b;
    int c = 0;
    for (int i = 0; i < 256; i++) {
        if (is_direct(i)) continue;
        if (i == b) return (uint32_t)(256 + c);
        c++;
    }
    return 0;
}

static int uni_to_byte(uint32_t cp)
{
    if (cp < 256 && is_direct((int)cp)) return (int)cp;
    if (cp >= 256 && cp < 324) {
        int idx = (int)(cp - 256), c = 0;
        for (int i = 0; i < 256; i++) {
            if (is_direct(i)) continue;
            if (c == idx) return i;
            c++;
        }
    }
    return -1;
}

/* ---------------- helpers ---------------- */

static int cmp_token(const feng_tok_t *t, int a, int b, const char *s, int len)
{
    const int la = t->token_len[a], lb = len;
    const int m = la < lb ? la : lb;
    const int c = memcmp(t->tokens[a], s, m);
    if (c != 0) return c;
    return la - lb;
}

/* binary search over vocab_order[] */
static int find_token(const feng_tok_t *t, const char *s, int len)
{
    int lo = 0, hi = t->vocab_size - 1;
    while (lo <= hi) {
        const int mid = (lo + hi) / 2;
        const int idx = t->vocab_order[mid];
        const int c = cmp_token(t, idx, 0, s, len);
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

static int cmp_uint32(const void *a, const void *b)
{
    const uint32_t x = *(const uint32_t *)a, y = *(const uint32_t *)b;
    return x < y ? -1 : (x > y ? 1 : 0);
}

static const feng_tok_t *g_sort_tokenizer = NULL;

static int cmp_index(const void *a, const void *b)
{
    const feng_tok_t *t = g_sort_tokenizer;
    const int ia = *(const int *)a, ib = *(const int *)b;
    const int la = t->token_len[ia], lb = t->token_len[ib];
    const int m = la < lb ? la : lb;
    const int c = memcmp(t->tokens[ia], t->tokens[ib], m);
    if (c != 0) return c;
    return la - lb;
}

typedef struct { uint32_t key; int rank; } merge_entry_t;

static int cmp_merge_entry(const void *a, const void *b)
{
    const merge_entry_t *x = (const merge_entry_t *)a, *y = (const merge_entry_t *)b;
    if (x->key != y->key) return x->key < y->key ? -1 : 1;
    return x->rank - y->rank;
}

int feng_tok_load(feng_tok_t *t, const void *data, size_t size)
{
    const uint8_t *p = (const uint8_t *)data;
    size_t off = 0;
    uint32_t vocab = 0, n_merges = 0;
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
    g_sort_tokenizer = t;
    qsort(t->vocab_order, vocab, sizeof(int), cmp_index);

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
        if ((int)(la + lb) >= (int)sizeof(cat)) continue;
        memcpy(cat, ls, la);
        memcpy(cat + la, rs, lb);
        const int mi = find_token(t, cat, la + lb);
        if (mi < 0) continue;
        t->pair_key[n_ok] = ((uint32_t)li << 16) | (uint32_t)ri;
        t->pair_result[n_ok] = (uint16_t)mi;
        n_ok++;
    }
    t->n_merges = (int)n_ok;
    /* sort (key, rank) so pair_rank() can binary search */
    {
        merge_entry_t *ents = (merge_entry_t *)tok_alloc((n_ok ? n_ok : 1) * sizeof(merge_entry_t));
        if (!ents) return -4;
        for (uint32_t i = 0; i < n_ok; i++) {
            ents[i].key = t->pair_key[i];
            ents[i].rank = (int)i;
        }
        qsort(ents, n_ok, sizeof(merge_entry_t), cmp_merge_entry);
        for (uint32_t i = 0; i < n_ok; i++) {
            t->sorted_key[i] = ents[i].key;
            t->sorted_rank[i] = ents[i].rank;
        }
        free(ents);
    }

    /* byte -> base token */
    for (int b = 0; b < 256; b++) {
        char tmp[8];
        const int n = utf8_encode(bytes_to_uni(b), tmp);
        t->byte_to_token[b] = (int16_t)find_token(t, tmp, n);
    }
    t->id_im_start = find_token(t, "<|im_start|>", 12);
    t->id_im_end = find_token(t, "<|im_end|>", 10);
    t->id_eot = find_token(t, "<|endoftext|>", 13);
    return 0;
}

void feng_tok_free(feng_tok_t *t)
{
    if (!t) return;
    if (t->tokens) {
        for (int i = 0; i < t->vocab_size; i++) free(t->tokens[i]);
        free(t->tokens);
    }
    free(t->token_len);
    free(t->vocab_order);
    free(t->pair_key);
    free(t->pair_result);
    free(t->sorted_key);
    free(t->sorted_rank);
    memset(t, 0, sizeof(*t));
}

/* ---------------- encode / decode ---------------- */

static int encode_segment(const feng_tok_t *t, const unsigned char *bytes, int nb, int *out,
                          int max_out)
{
    static int sym[4096];
    if (nb > 4096) nb = 4096;
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
    for (int i = 0; i < nb && n < max_out; i++) out[n++] = sym[i];
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
    int n = 0;
    const unsigned char *s = (const unsigned char *)text;
    int N = (int)strlen(text);
    int i = 0;
    static unsigned char buf[2048];
    while (i < N && n < max_out) {
        /* special tokens are matched literally first */
        if (t->id_im_start >= 0 && i + 12 <= N && strncmp((const char *)s + i, "<|im_start|>", 12) == 0) {
            out[n++] = t->id_im_start;
            i += 12;
            continue;
        }
        if (t->id_im_end >= 0 && i + 10 <= N && strncmp((const char *)s + i, "<|im_end|>", 10) == 0) {
            out[n++] = t->id_im_end;
            i += 10;
            continue;
        }
        if (t->id_eot >= 0 && i + 13 <= N && strncmp((const char *)s + i, "<|endoftext|>", 13) == 0) {
            out[n++] = t->id_eot;
            i += 13;
            continue;
        }
        /* whitespace handling (GPT-2 regex: \s+(?!\S)|\s+) */
        if (cat_of(s[i]) == 3) {
            int k = 0;
            while (i + k < N && cat_of(s[i + k]) == 3) k++;
            if (i + k >= N) {                       /* trailing whitespace: whole run */
                if (k > (int)sizeof(buf)) k = (int)sizeof(buf);
                memcpy(buf, s + i, k);
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
            if (emit > (int)sizeof(buf)) emit = (int)sizeof(buf);
            memcpy(buf, s + i, emit);
            n += encode_segment(t, buf, emit, out + n, max_out - n);
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
            if (len > (int)sizeof(buf)) len = (int)sizeof(buf);
            memcpy(buf, s + start, len);
            n += encode_segment(t, buf, len, out + n, max_out - n);
            i = start + len;
        }
    }
    return n;
}

int feng_tok_decode_token(const feng_tok_t *t, int id, char *out, int max_out)
{
    if (id < 0 || id >= t->vocab_size) return 0;
    if (id == t->id_im_start || id == t->id_im_end || id == t->id_eot) return 0;
    const char *s = t->tokens[id];
    int n = 0;
    while (*s && n < max_out) {
        uint32_t cp;
        const int adv = utf8_decode(s, &cp);
        const int b = uni_to_byte(cp);
        if (b >= 0) out[n++] = (char)b;
        s += adv;
    }
    return n;
}
