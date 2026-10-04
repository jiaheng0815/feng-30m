/* feng-30m chat on ESP32-S3 (N32R16: 32MB flash + 16MB octal PSRAM)
 *
 * Partition layout (see partitions.csv):
 *   model    model.bin     (Q4 weights, memory-mapped, executed straight from flash)
 *   tokdata  tokenizer.bin (vocab + merges; loaded into PSRAM)
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "esp_chip_info.h"
#include "esp_heap_caps.h"
#include "esp_log.h"
#include "esp_partition.h"
#include "esp_timer.h"
#include "driver/uart.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "sdkconfig.h"

#include "feng.h"
#include "feng_calc.h"
#include "feng_memory.h"
#include "feng_tools.h"
#include "feng_tokenizer.h"
#include <esp_timer.h>
#include "gbk.h"

static const char *TAG = "feng";
#define MODEL_PART "model"
#define TOK_PART "tokdata"
#define UART_PORT UART_NUM_0
/* KV cache is fp32 in PSRAM: n_layers*MAX_CTX*hidden*4 bytes per cache (K and V).
 * 11*512*448*4 = 10.1 MB each -> 20 MB total, more than the 16 MB PSRAM.
 * 11*256*448*4 =  5.05 MB each -> 10.1 MB total, fits comfortably. */
/* int8 KV cache is 4x smaller, which is what makes a 1024-token window fit the
 * 16 MB of PSRAM (fp32 only fits ~256). */
/* 测试用覆盖：-DFENG_MAX_CTX_OVERRIDE=256 可把上下文缩到很小，
 * 快速验证"写满自动开新对话"这条路径（正常固件不定义，保持下面的默认值）。 */
#ifndef FENG_MAX_CTX_OVERRIDE
#define FENG_MAX_CTX_OVERRIDE 0
#endif
#if FENG_MAX_CTX_OVERRIDE
#define MAX_CTX FENG_MAX_CTX_OVERRIDE
#elif FENG_KV_Q2
#define MAX_CTX 2048        /* q2 block8：448 B/token/层 -> 2048 ctx ≈ 10.1 MB PSRAM */
#elif FENG_KV_INT8
#define MAX_CTX 1024
#else
#define MAX_CTX 256
#endif
#define MAX_NEW 96

/* 注意力基准（-DFENG_BENCH_CTX=1 打开）：合成指定长度的 KV，测单次 forward 耗时，
 * 用来评估 KV 访问/注意力优化在长上下文下的收益。 */
#ifndef FENG_BENCH_CTX
#define FENG_BENCH_CTX 0
#endif

static feng_model_t s_model;
static feng_tok_t s_tok;
static feng_kv_t s_kv;
static feng_workspace_t s_ws;
static int s_hist[64];          /* recent tokens for repetition penalty */
static int s_nhist;
/* 网络时间：宿主发 \settime <epoch> 同步一次，之后用 esp_timer 走时 */
static long long s_epoch_base;
static long long s_time_base_us;
static int s_time_synced;

static long long fw_epoch_now(void)
{
    if (!s_time_synced) return 0;
    return s_epoch_base + (esp_timer_get_time() - s_time_base_us) / 1000000;
}

static long long fw_uptime_us(void)
{
    return esp_timer_get_time();
}

#if FENG_BENCH_CTX && FENG_KV_Q2
/* PIE 裸吞吐：判断"整行 PIE 内核"的理论上限（对比标量 GEMV 的 4.1 周期/权重，双核） */
/* 合成指定长度的 q2 KV，测单次 forward 的完整耗时（含权重 GEMV + 注意力）。
 * 每个长度跑两次：第 1 次是冷缓存，第 2 次是热缓存；对比不同内核版本用第 2 次。 */
static void bench_forward_ctx(void)
{
#if FENG_ATTN_PROF
    extern unsigned long long g_prof_cycles[3];
#endif
    const int h = s_model.hdr.hidden;
    const int n_layers = s_model.hdr.n_layers;
    const int nh = s_model.hdr.n_heads, hd = s_model.hdr.head_dim;
    const int qb = hd / 4, nb = hd / FENG_KV_Q2_BLOCK;
    const int ctxs[] = {256, 1024, 2048};
    const uint16_t one = feng_f32_to_f16(1.0f);
    for (int ci = 0; ci < (int)(sizeof(ctxs) / sizeof(ctxs[0])); ci++) {
        const int c = ctxs[ci];
        if (c > s_kv.ctx) break;
        for (int l = 0; l < n_layers; l++) {
            /* q2 布局 [layer][head][t]：逐 head 填该头的 c 个 token */
            for (int hh = 0; hh < nh; hh++) {
                const size_t row = ((size_t)l * nh + hh) * s_kv.ctx;
                uint8_t *kc = (uint8_t *)s_kv.k_cache + row * qb;
                uint8_t *vc = (uint8_t *)s_kv.v_cache + row * qb;
                uint16_t *ks = s_kv.k_scale + row * nb;
                uint16_t *vs = s_kv.v_scale + row * nb;
                for (size_t i = 0; i < (size_t)c * qb; i++) {
                    kc[i] = (uint8_t)(i * 37 + 11);
                    vc[i] = (uint8_t)(i * 53 + 7);
                }
                for (size_t i = 0; i < (size_t)c * nb; i++) { ks[i] = one; vs[i] = one; }
            }
        }
        int64_t dt[2] = {0, 0};
        for (int rep = 0; rep < 2; rep++) {
#if FENG_ATTN_PROF
            if (rep == 1) {
                g_prof_cycles[0] = g_prof_cycles[1] = g_prof_cycles[2] = 0;
            }
#endif
            const int64_t t0 = esp_timer_get_time();
            feng_forward(&s_model, &s_kv, &s_ws, 100, c - 1);
            dt[rep] = esp_timer_get_time() - t0;
        }
        ESP_LOGI(TAG, "attn bench ctx=%4d: cold %.0f ms/forward | warm %.0f ms/forward",
                 c, dt[0] / 1000.0, dt[1] / 1000.0);
#if FENG_ATTN_PROF
        ESP_LOGI(TAG, "  prof ctx=%4d: K=%.0f ms  softmax=%.0f ms  V=%.0f ms",
                 c, g_prof_cycles[0] / 240000.0, g_prof_cycles[1] / 240000.0,
                 g_prof_cycles[2] / 240000.0);
#endif
    }
    /* lm head（out_norm + tied head，7.34M 权重）的单次成本：pos=0 带/不带 logits 各一次 */
    int64_t h_with = 0, h_without = 0;
    s_kv.len = 0;
    {
        const int64_t t0 = esp_timer_get_time();
        feng_forward(&s_model, &s_kv, &s_ws, 100, 0);
        h_with = esp_timer_get_time() - t0;
    }
    s_kv.len = 0;
    {
        const int64_t t0 = esp_timer_get_time();
        feng_forward_ex(&s_model, &s_kv, &s_ws, 100, 0, 0);
        h_without = esp_timer_get_time() - t0;
    }
    ESP_LOGI(TAG, "lm head @pos0: with %.0f ms | without %.0f ms (prefill 每个中间 token 省 %.0f ms)",
             h_with / 1000.0, h_without / 1000.0, (h_with - h_without) / 1000.0);
    s_kv.len = 0;
}
#endif

#if FENG_BENCH_PIE
/* PIE 裸吞吐：判断"整行 PIE 内核"的理论上限（对比标量 GEMV 的 ~4 周期/权重，双核） */
extern void pie_mac_s16_bench(const int16_t *a, int n);
extern void pie_s8_zip_mac_bench(const int8_t *w, const int8_t *x, int n);
static void bench_pie(void)
{
    static int16_t b16[16];
    static int8_t b8[32];
    const int iters = 300000;
    for (int i = 0; i < 16; i++) b16[i] = (int16_t)(i + 1);
    for (int i = 0; i < 32; i++) b8[i] = (int8_t)(i - 8);
    const unsigned c0 = (unsigned)esp_cpu_get_cycle_count();
    pie_mac_s16_bench(b16, iters);
    const unsigned c1 = (unsigned)esp_cpu_get_cycle_count();
    pie_s8_zip_mac_bench(b8, b8 + 16, iters);
    const unsigned c2 = (unsigned)esp_cpu_get_cycle_count();
    ESP_LOGI(TAG, "PIE bench: s16 MAC 链 %.2f 周期/条 (%.3f 周期/MAC) | s8 流水 %.2f 周期/迭代 (%.3f 周期/MAC, 8MAC/迭代)",
             (double)(c1 - c0) / iters, (double)(c1 - c0) / iters / 8.0,
             (double)(c2 - c1) / iters, (double)(c2 - c1) / iters / 8.0);
}
#endif

/* ---- serial I/O: this firmware owns UART0 (console is disabled in sdkconfig) ---- */
/* write UTF-8 text out in whatever encoding the terminal speaks (GBK or UTF-8) */
static void write_text(const char *s, int n)
{
    if (n <= 0) return;
    if (g_out_gbk) {
        static char gb[2048];
        const int m = feng_utf8_to_gbk(s, n, gb, sizeof(gb));
        if (m > 0) uart_write_bytes(UART_PORT, gb, (size_t)m);
    } else {
        uart_write_bytes(UART_PORT, s, (size_t)n);
    }
}

static int uart_vprintf_impl(const char *fmt, va_list ap)
{
    static char buf[512];
    int n = vsnprintf(buf, sizeof(buf), fmt, ap);
    if (n > 0) {
        write_text(buf, n < (int)sizeof(buf) ? n : (int)sizeof(buf) - 1);
    }
    return n;
}

static void out_printf(const char *fmt, ...)
{
    va_list ap;
    va_start(ap, fmt);
    uart_vprintf_impl(fmt, ap);
    va_end(ap);
}

static void *ps_alloc(size_t n)
{
    void *p = heap_caps_malloc(n, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    if (!p) {
        ESP_LOGE(TAG, "PSRAM alloc failed: %u bytes", (unsigned)n);
        abort();
    }
    return p;
}

static float *ps_alloc_f32(size_t n)
{
    float *p = (float *)ps_alloc(n * sizeof(float));
    memset(p, 0, n * sizeof(float));
    return p;
}

static void log_mem(const char *stage)
{
    ESP_LOGI(TAG, "[mem] %-11s internal %7u B | psram %8u B (largest int %u)", stage,
             (unsigned)heap_caps_get_free_size(MALLOC_CAP_INTERNAL),
             (unsigned)heap_caps_get_free_size(MALLOC_CAP_SPIRAM),
             (unsigned)heap_caps_get_largest_free_block(MALLOC_CAP_INTERNAL));
}

static void bench_gemv(void);

static void setup_model(void)
{
    const esp_partition_t *mp = esp_partition_find_first(ESP_PARTITION_TYPE_DATA,
                                                        ESP_PARTITION_SUBTYPE_ANY, MODEL_PART);
    if (!mp) {
        ESP_LOGE(TAG, "model partition missing");
        abort();
    }
    const void *ptr = NULL;
    esp_partition_mmap_handle_t h;
    ESP_ERROR_CHECK(esp_partition_mmap(mp, 0, mp->size, ESP_PARTITION_MMAP_DATA, &ptr, &h));
    ESP_LOGI(TAG, "model mapped at %p (%u KB)", ptr, (unsigned)(mp->size / 1024));
    log_mem("mmap");

    /* quick streaming-read benchmark: how fast can we pull the weights out of flash? */
    {
        const uint8_t *p8 = (const uint8_t *)ptr;
        const size_t n = mp->size & ~(size_t)0x3F;
        volatile uint32_t acc = 0;
        const int64_t t0 = esp_timer_get_time();
        for (size_t i = 0; i < n; i += 64) acc += p8[i];
        const int64_t dt = esp_timer_get_time() - t0;
        ESP_LOGI(TAG, "mmap stream read: %u KB in %lld ms -> %.1f MB/s (checksum %u)",
                 (unsigned)(n / 1024), dt / 1000, (double)n / (double)dt * 1.0,
                 (unsigned)acc);
    }
    const int rc = feng_model_init(&s_model, ptr, mp->size);
    if (rc != 0) {
        ESP_LOGE(TAG, "model init failed rc=%d", rc);
        abort();
    }
    log_mem("model");
    ESP_LOGI(TAG, "model: %u layers, hidden %u, %u heads x %u, ffn %u, vocab %u, rope %.0f",
             s_model.hdr.n_layers, s_model.hdr.hidden, s_model.hdr.n_heads, s_model.hdr.head_dim,
             s_model.hdr.ffn, s_model.hdr.vocab, s_model.hdr.rope_theta);

    const esp_partition_t *tp = esp_partition_find_first(ESP_PARTITION_TYPE_DATA,
                                                        ESP_PARTITION_SUBTYPE_ANY, TOK_PART);
    if (!tp) {
        ESP_LOGE(TAG, "tokenizer partition missing");
        abort();
    }
    uint8_t *tokbuf = (uint8_t *)ps_alloc(tp->size);
    ESP_ERROR_CHECK(esp_partition_read(tp, 0, tokbuf, tp->size));
    if (feng_tok_load(&s_tok, tokbuf, tp->size) != 0) {
        ESP_LOGE(TAG, "tokenizer load failed");
        abort();
    }
    ESP_LOGI(TAG, "tokenizer: vocab %d, merges %d", s_tok.vocab_size, s_tok.n_merges);
    log_mem("tokenizer");
    /* boot self-test: tokenize a short string so tokenizer problems show up immediately */
    {
        static int test_ids[64];
        const int tn = feng_tok_encode(&s_tok, "你好", test_ids, 64);
        char dump[256];
        int off = 0;
        for (int i = 0; i < tn && off < 200; i++) {
            off += snprintf(dump + off, sizeof(dump) - off, "%d ", test_ids[i]);
        }
        dump[off] = 0;
        ESP_LOGI(TAG, "tokenizer self-test: %d ids -> %s", tn, dump);
        int back = 0;
        char text[128];
        int toff = 0;
        for (int i = 0; i < tn && toff < 120; i++) {
            toff += feng_tok_decode_token(&s_tok, test_ids[i], text + toff, 8);
        }
        text[toff] = 0;
        /* keep the boot log ASCII-only so any terminal encoding shows it correctly */
        ESP_LOGI(TAG, "tokenizer decode round-trip ok=%d (%d bytes)", strcmp(text, "你好") == 0, toff);
        back = toff;
        (void)back;
    }

    const int hidden = s_model.hdr.hidden;
    s_kv.ctx = MAX_CTX;
    s_kv.len = 0;
#if FENG_KV_INT8
    s_kv.k_cache = ps_alloc((size_t)s_model.hdr.n_layers * MAX_CTX * hidden);
    s_kv.v_cache = ps_alloc((size_t)s_model.hdr.n_layers * MAX_CTX * hidden);
#elif FENG_KV_Q2
    s_kv.k_cache = ps_alloc((size_t)s_model.hdr.n_layers * MAX_CTX * (hidden / 4));
    s_kv.v_cache = ps_alloc((size_t)s_model.hdr.n_layers * MAX_CTX * (hidden / 4));
#else
    s_kv.k_cache = ps_alloc_f32((size_t)s_model.hdr.n_layers * MAX_CTX * hidden);
    s_kv.v_cache = ps_alloc_f32((size_t)s_model.hdr.n_layers * MAX_CTX * hidden);
#endif
#if FENG_KV_INT8 || FENG_KV_Q2
    const size_t n_sc = feng_kv_scale_slots(&s_model, MAX_CTX);
    s_kv.k_scale = (uint16_t *)ps_alloc(n_sc * sizeof(uint16_t));
    s_kv.v_scale = (uint16_t *)ps_alloc(n_sc * sizeof(uint16_t));
#else
    s_kv.k_scale = NULL;
    s_kv.v_scale = NULL;
#endif
    s_ws.max_ctx = MAX_CTX;
    s_ws.x = ps_alloc_f32(hidden);
    s_ws.xn = ps_alloc_f32(hidden);
    s_ws.q = ps_alloc_f32(hidden);
    s_ws.k = ps_alloc_f32(hidden);
    s_ws.v = ps_alloc_f32(hidden);
    s_ws.attn = ps_alloc_f32(hidden);
    s_ws.proj = ps_alloc_f32(s_model.hdr.ffn);
    s_ws.gate = ps_alloc_f32(s_model.hdr.ffn);
    s_ws.up = ps_alloc_f32(s_model.hdr.ffn);
    s_ws.ffn = ps_alloc_f32(s_model.hdr.ffn);
    s_ws.logits = ps_alloc_f32(s_model.hdr.vocab);
    s_ws.scratch = ps_alloc_f32(s_model.hdr.ffn > MAX_CTX ? s_model.hdr.ffn : MAX_CTX);
#if FENG_KV_Q2
    const char *kv_mode = "q2/block" FENG_STR(FENG_KV_Q2_BLOCK);
#elif FENG_KV_INT8
    const char *kv_mode = "int8";
#else
    const char *kv_mode = "fp32";
#endif
    ESP_LOGI(TAG, "KV cache: %s, ctx %d, %.2f MB", kv_mode, MAX_CTX,
             (double)feng_kv_bytes(&s_model, MAX_CTX) / 1048576.0);
    ESP_LOGI(TAG, "PSRAM free after setup: %u KB",
             (unsigned)(heap_caps_get_free_size(MALLOC_CAP_SPIRAM) / 1024));
    log_mem("kv/ws");
    feng_smp_init();
    bench_gemv();
}

/* measure the raw GEMV throughput (single core vs both cores) once at boot */
static void bench_gemv(void)
{
    const int h = s_model.hdr.hidden, f = s_model.hdr.ffn;
    const feng_layer_t *L = &s_model.layers[0];
    const size_t wbytes = (size_t)f * (h / 64) * 34;   /* Q4 bytes read per call */
    const long iters = 20;
    static float xref[512];                            /* 未 prepare 的副本 -> 回退内核 */
    /* 激活缓冲在开机时是未初始化的；填成非零、有正有负的样值，
     * 否则"全零块跳过"会让基准测不到真正的权重解码 + 点积开销。 */
    for (int i = 0; i < h; i++) {
        s_ws.xn[i] = (float)((i % 7) - 3) * 0.13f;
        xref[i] = s_ws.xn[i];
    }
    int64_t t0 = esp_timer_get_time();
    for (long i = 0; i < iters; i++) {
        feng_gemv_range(L->gate, FENG_DT_Q4, s_ws.xn, s_ws.gate, 0, f, h);
    }
    int64_t dt1 = esp_timer_get_time() - t0;
#if FENG_GEMV_A8 || FENG_GEMV_PIE
    /* 准备激活 -> 启用 A8/PIE 内核（自检一次：整数点积必须一致） */
    feng_gemv_a8_prepare(s_ws.xn, h);
    {
        extern int feng_gemv_pie_selfcheck(const void *tensor, int n_in);
        const int chk = feng_gemv_pie_selfcheck(L->gate, h);
        ESP_LOGI(TAG, "PIE dot self-check: %s",
                 chk == 0 ? "MATCH" : (chk == 1 ? "MISMATCH" : "SKIP(no prepare)"));
    }
    t0 = esp_timer_get_time();
    for (long i = 0; i < iters; i++) {
        feng_gemv_range(L->gate, FENG_DT_Q4, s_ws.xn, s_ws.gate, 0, f, h);
    }
    int64_t dt1b = esp_timer_get_time() - t0;
#else
    const int64_t dt1b = dt1;
#endif
    (void)xref;
    t0 = esp_timer_get_time();
    for (long i = 0; i < iters; i++) {
        feng_gemv_par(L->gate, FENG_DT_Q4, s_ws.xn, s_ws.gate, f, h);
    }
    int64_t dt2 = esp_timer_get_time() - t0;
    ESP_LOGI(TAG, "gemv %dx%d: 1-core %lld us (%.0f MB/s) | 2-core %lld us (%.0f MB/s) | speedup %.2fx",
             f, h, dt1 / iters, (double)wbytes * iters / (double)dt1,
             dt2 / iters, (double)wbytes * iters / (double)dt2, (double)dt1 / (double)dt2);
#if FENG_GEMV_A8 || FENG_GEMV_PIE
    ESP_LOGI(TAG, "gemv 量化内核: 1-core %lld us（对比回退 %lld us，%.2fx）",
             dt1b / iters, dt1 / iters, (double)dt1 / (double)dt1b);
#endif
}

/* greedy sampling with repetition penalty over the recent window */
static int sample_next(float *logits, int vocab)
{
    for (int i = 0; i < s_nhist; i++) {
        const int t = s_hist[i];
        if (t >= 0 && t < vocab) {
            logits[t] = logits[t] > 0 ? logits[t] / 1.15f : logits[t] * 1.15f;
        }
    }
    return feng_argmax(logits, vocab);
}

static void push_hist(int t)
{
    if (s_nhist < (int)(sizeof(s_hist) / sizeof(s_hist[0]))) {
        s_hist[s_nhist++] = t;
    } else {
        memmove(s_hist, s_hist + 1, sizeof(s_hist) - sizeof(s_hist[0]));
        s_hist[sizeof(s_hist) / sizeof(s_hist[0]) - 1] = t;
    }
}

/* Streaming granularity: flush at a clause boundary or once the chunk is big
 * enough, so a terminal that stamps every received packet shows a timestamp per
 * clause instead of one per token. */
#define STREAM_CHUNK 30
static int s_stream_chunk = STREAM_CHUNK;   /* 0 = buffer the whole reply, flush once */

static int ends_clause(const char *s, int n)
{
    if (n <= 0) return 0;
    const char last = s[n - 1];
    if (last == '\n' || last == '!' || last == '?' || last == '.') return 1;
    if (n < 3) return 0;
    static const char *marks[] = {"\xE3\x80\x82", "\xEF\xBC\x81", "\xEF\xBC\x9F", "\xEF\xBC\x8C",
                                  "\xEF\xBC\x9B", "\xEF\xBC\x9A", "\xE3\x80\x81"};
    for (int i = 0; i < (int)(sizeof(marks) / sizeof(marks[0])); i++) {
        if (memcmp(s + n - 3, marks[i], 3) == 0) return 1;
    }
    return 0;
}

/* 换行 token（多轮续接时要补在 <|im_end|> 之后），首次用到时编码一次 */
static int newline_token(void)
{
    static int cached = -1;
    if (cached < 0) {
        int one = 0;
        cached = (feng_tok_encode(&s_tok, "\n", &one, 1) == 1) ? one : 0;
    }
    return cached;
}

/* keep=1：不动 KV，从已有上下文后面续写（多轮对话）；keep=0：清空重来 */
static int generate(const int *prompt, int np, char *out, int out_max, int keep)
{
    if (!keep) {
        s_kv.len = 0;
        s_nhist = 0;
        feng_mem_clear();          /* 写满=开新对话：引擎侧记忆一起清 */
    }
    int written = 0, pos = s_kv.len;
    const int vocab = s_model.hdr.vocab;
    const int64_t t0 = esp_timer_get_time();
    float *logits = NULL;
    ESP_LOGI(TAG, "prefill %d tokens ...", np);
    for (int i = 0; i < np; i++) {                      /* prefill（中间 token 不算 lm head） */
        logits = feng_forward_ex(&s_model, &s_kv, &s_ws, prompt[i], pos++, i + 1 == np);
    }
    ESP_LOGI(TAG, "prefill done");
    const int64_t t_prefill = esp_timer_get_time();
    int n_gen = 0;
    static char pend[512];
    int pn = 0;
    out_printf("<< ");
    fflush(stdout);
    for (int n = 0; n < MAX_NEW && written < out_max - 8; n++) {
        const int tok = sample_next(logits, vocab);
        if (tok == s_tok.id_im_end || tok == s_tok.id_eot) break;
        char tmp[32];
        const int nb = feng_tok_decode_token(&s_tok, tok, tmp, sizeof(tmp));
        if (nb > 0 && written + nb < out_max) {
            memcpy(out + written, tmp, nb);
            written += nb;
            out[written] = 0;
            if (pn + nb < (int)sizeof(pend)) {
                memcpy(pend + pn, tmp, nb);
                pn += nb;
            }
            if (s_stream_chunk > 0 && (pn >= s_stream_chunk || ends_clause(pend, pn))) {
                write_text(pend, pn);
                pn = 0;
            }
        }
        push_hist(tok);
        n_gen++;
        logits = feng_forward(&s_model, &s_kv, &s_ws, tok, pos++);
    }
    if (pn > 0) write_text(pend, pn);
    const int64_t dt = esp_timer_get_time() - t0;
    const float tps = (float)(pos) * 1e6f / (float)dt;
    if (keep) {
        /* 把模型自己吐出的 <|im_end|> 与换行补进 KV，下一轮才能无缝续接；
           若本轮是被长度截断的，也补一个 im_end 收尾，语义上等于结束这一轮。
           这两个 token 的 logits 没人用，跳过 lm head（每个省 ~124 ms） */
        logits = feng_forward_ex(&s_model, &s_kv, &s_ws, s_tok.id_im_end, pos++, 0);
        const int nl = newline_token();
        if (nl > 0) logits = feng_forward_ex(&s_model, &s_kv, &s_ws, nl, pos++, 0);
        (void)logits;
    }
    s_kv.len = pos;
    /* 和 pc_chat 同一口径：尾行给出上下文占用与速度，方便用户/脚本判断 */
    out_printf("\n>>END (ctx %d/%d, %.2f tok/s)\n", s_kv.len, MAX_CTX, tps);
    fflush(stdout);
    ESP_LOGI(TAG, "prompt %d | gen %d | prefill %lld ms | total %lld ms | %.2f tok/s (%.0f ms/token)",
             np, n_gen, (t_prefill - t0) / 1000, dt / 1000, tps,
             (float)dt / 1000.0f / (float)n_gen);
    return written;
}

static void chat_once(const char *user)
{
    /* 长文输入：正文按 token 预算安全截断（保 UTF-8 边界与模板收尾），
     * 2048 ctx 下大约能收 1500+ 个汉字，而不是原来的 ~340 个。 */
    static char prompt[4096];
    static int ids[2304];
    static char reply[1024];
    const int cap = MAX_CTX - MAX_NEW - 8;
    size_t ulen = strlen(user);
    if (ulen > sizeof(prompt) - 256) ulen = sizeof(prompt) - 256;
    int truncated = (ulen < strlen(user));
    int n = 0;
    for (;;) {
        while (ulen > 0 && ((unsigned char)user[ulen] & 0xC0) == 0x80) ulen--;
        snprintf(prompt, sizeof(prompt),
                 "<|im_start|>user\n%.*s<|im_end|>\n<|im_start|>assistant\n", (int)ulen, user);
        n = feng_tok_encode(&s_tok, prompt, ids, cap);
        if (n < cap) break;                    /* 收得下（含 <|im_end|> 收尾） */
        if (ulen <= 64) { out_printf("[输入太长，请分几次发]\n"); return; }
        ulen = ulen * 3 / 4;                   /* 还是太满：再截短正文 */
        truncated = 1;
    }
    if (truncated) out_printf("[输入过长，已截断到 %d 字节]\n", (int)ulen);
    /* 板端 prefill 成本实测 = 每个 token 的权重 GEMV ~0.38 s + 注意力的 O(n²)
     * （0.85 ms × n²/2）：491 tokens 实测 291 s（权重 ~187 s + 注意力 ~102 s）。 */
    if (n > 80) {
        out_printf("[长文输入 %d tokens：板端 prefill 约 %.0f 秒（%.0f 秒权重 + n² 注意力）]\n",
                   n, 0.38 * (double)n + 0.000425 * (double)n * (double)n,
                   0.38 * (double)n);
    }
    if (s_kv.len + n + MAX_NEW + 4 > MAX_CTX) {
        out_printf("[上下文已满 %d/%d，自动开始新对话；输入 \\reset 可手动清空]\n",
                   s_kv.len, MAX_CTX);
        s_kv.len = 0;
        s_nhist = 0;
    }
    ESP_LOGI(TAG, "in: %s (%d prompt tokens, ctx %d -> %d/%d)", user, n, s_kv.len,
             s_kv.len + n, MAX_CTX);
    reply[0] = 0;
    generate(ids, n, reply, sizeof(reply), 1);
}

void app_main(void)
{
    esp_chip_info_t info;

    log_mem("app_main");
    /* install the UART driver first, then route ESP_LOG and all chat I/O through it */
    ESP_ERROR_CHECK(uart_driver_install(UART_PORT, 2048, 2048, 0, NULL, 0));
    ESP_ERROR_CHECK(uart_set_baudrate(UART_PORT, 115200));
    esp_log_set_vprintf(uart_vprintf_impl);
    log_mem("uart");

    esp_chip_info(&info);
    ESP_LOGI(TAG, "feng-30m on ESP32-S3 (%d cores), flash 32MB / PSRAM 16MB", info.cores);
    setup_model();
#if FENG_BENCH_CTX && FENG_KV_Q2
    bench_forward_ctx();
#endif
#if FENG_BENCH_PIE
    bench_pie();
#endif
    feng_tools_set_time(fw_epoch_now);
    feng_tools_set_uptime(fw_uptime_us);

    static char line[4096];
    static char u8[4096];
    out_printf("\n=== feng-30m on ESP32-S3 (serial chat) ===\n");
    out_printf("one sentence per line; the reply streams between << and >>END\n");
    out_printf("多轮对话：自动保留上下文（2048 token），满了自动开新对话\n");
    out_printf("工具：算式（59+1）、现在几点、随机数、记忆（我叫X/我最喜欢Y…）都由板内 tool 直接回答\n");
    out_printf("commands: \\gbk  \\utf8  \\reset  \\stream N  \\help\n");
    out_printf("FENG_READY\n");
    fflush(stdout);
    while (1) {
        out_printf("you> ");
        fflush(stdout);
        size_t n = 0;
        int over = 0;
        for (;;) {                      /* read one line from UART0 */
            uint8_t ch = 0;
            const int r = uart_read_bytes(UART_NUM_0, &ch, 1, pdMS_TO_TICKS(200));
            if (r == 1) {
                if (ch == '\n' || ch == '\r') {
                    uart_write_bytes(UART_PORT, "\n", 1);
                    if (n > 0) break;
                    continue;
                }
                uart_write_bytes(UART_PORT, (const char *)&ch, 1);   /* echo so the user sees typing */
                if (n < sizeof(line) - 1) line[n++] = (char)ch;
                else over = 1;                                    /* 超出缓冲：丢弃并提示 */
            }
        }
        line[n] = 0;
        if (over) out_printf("[输入超过 %d 字节，多余部分已丢弃]\n", (int)sizeof(line) - 1);
        /* mirror the terminal's encoding: GBK terminals send GBK bytes */
        if (feng_has_high_byte(line, (int)n)) {
            g_out_gbk = !feng_utf8_valid(line, (int)n);
        }
        if (line[0] == '\\') {
            if (strncmp(line, "\\gbk", 4) == 0) {
                g_out_gbk = 1;
                out_printf("encoding: GBK\n");
            } else if (strncmp(line, "\\utf8", 5) == 0) {
                g_out_gbk = 0;
                out_printf("encoding: UTF-8\n");
            } else if (strncmp(line, "\\reset", 6) == 0) {
                s_kv.len = 0;
                s_nhist = 0;
                feng_mem_clear();
                out_printf("context cleared（记忆也一起清了，\\mem 可查看）\n");
            } else if (strncmp(line, "\\mem", 4) == 0) {
                if (strncmp(line, "\\mem clear", 10) == 0) {
                    feng_mem_clear();
                    out_printf("memory cleared\n");
                } else {
                    char snap[256];
                    feng_mem_snapshot(snap, sizeof(snap));
                    out_printf("memory: %s\n", snap);
                }
            } else if (strncmp(line, "\\settime", 8) == 0) {
                const long long e = atoll(line + 8);
                if (e > 1600000000LL) {
                    s_epoch_base = e;
                    s_time_base_us = esp_timer_get_time();
                    s_time_synced = 1;
                    out_printf("time synced: epoch %lld\n", e);
                } else {
                    out_printf("usage: \\settime <unix_epoch>\n");
                }
            } else if (strncmp(line, "\\help", 5) == 0) {
                out_printf("\\gbk   reply in GBK (for SuperCom/XCOM in ANSI mode)\n");
                out_printf("\\utf8  reply in UTF-8\n");
                out_printf("\\reset clear the conversation context (multi-turn is on by default)\n");
                out_printf("\\mem   查看引擎记住的事实（\\mem clear 清空）\n");
                out_printf("\\settime <unix秒> 宿主对时（脚本连接时会自动发）\n");
                out_printf("\\stream N  flush every N bytes (0 = whole reply at once, default 30)\n");
                out_printf("算式 / 现在几点 / 随机数 自动走板内 tool；本命令帮助不经过模型\n");
                out_printf("\\help  this text\n");
            } else if (strncmp(line, "\\stream", 7) == 0) {
                int v = atoi(line + 7);
                if (v < 0) v = 0;
                if (v > 256) v = 256;
                s_stream_chunk = v;
                out_printf("stream chunk = %d bytes%s\n", v,
                           v == 0 ? " (whole reply at once)" : "");
            } else {
                out_printf("unknown command, try \\help\n");
            }
            continue;
        }
        if (n == 0) continue;
        int m = (int)n;
        if (g_out_gbk) {
            m = feng_gbk_to_utf8(line, (int)n, u8, sizeof(u8) - 1);
            u8[m] = 0;
        } else {
            memcpy(u8, line, n);
            u8[n] = 0;
        }
        /* 计算 tool：纯算式直接由 SoC 运算器算，秒回、100% 准确，不占模型/上下文 */
        char calc_reply[256];
        (void)feng_mem_learn(u8);      /* 先记事实（不拦截：模型仍能看到这句，保持自己的多轮能力） */
        if (feng_calc_answer(u8, calc_reply, sizeof(calc_reply)) ||
            feng_mem_answer(u8, calc_reply, sizeof(calc_reply)) ||
            feng_time_answer(u8, calc_reply, sizeof(calc_reply)) ||
            feng_random_answer(u8, calc_reply, sizeof(calc_reply))) {
            ESP_LOGI(TAG, "tool: %s -> %s", u8, calc_reply);
            out_printf("<< %s\n>>END\n", calc_reply);
            fflush(stdout);
            continue;
        }
        chat_once(u8);
    }
}
