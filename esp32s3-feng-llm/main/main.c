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
#include "feng_tokenizer.h"
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
#if FENG_KV_INT8
#define MAX_CTX 1024
#else
#define MAX_CTX 256
#endif
#define MAX_NEW 96

static feng_model_t s_model;
static feng_tok_t s_tok;
static feng_kv_t s_kv;
static feng_workspace_t s_ws;
static int s_hist[64];          /* recent tokens for repetition penalty */
static int s_nhist;

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
    s_kv.k_scale = (uint16_t *)ps_alloc((size_t)s_model.hdr.n_layers * MAX_CTX * s_model.hdr.n_heads * sizeof(uint16_t));
    s_kv.v_scale = (uint16_t *)ps_alloc((size_t)s_model.hdr.n_layers * MAX_CTX * s_model.hdr.n_heads * sizeof(uint16_t));
#else
    s_kv.k_cache = ps_alloc_f32((size_t)s_model.hdr.n_layers * MAX_CTX * hidden);
    s_kv.v_cache = ps_alloc_f32((size_t)s_model.hdr.n_layers * MAX_CTX * hidden);
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
    ESP_LOGI(TAG, "KV cache: %s, ctx %d, %.2f MB",
             FENG_KV_INT8 ? "int8" : "fp32", MAX_CTX,
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
    int64_t t0 = esp_timer_get_time();
    for (long i = 0; i < iters; i++) {
        feng_gemv_range(L->gate, FENG_DT_Q4, s_ws.xn, s_ws.gate, 0, f, h);
    }
    int64_t dt1 = esp_timer_get_time() - t0;
    t0 = esp_timer_get_time();
    for (long i = 0; i < iters; i++) {
        feng_gemv_par(L->gate, FENG_DT_Q4, s_ws.xn, s_ws.gate, f, h);
    }
    int64_t dt2 = esp_timer_get_time() - t0;
    ESP_LOGI(TAG, "gemv %dx%d: 1-core %lld us (%.0f MB/s) | 2-core %lld us (%.0f MB/s) | speedup %.2fx",
             f, h, dt1 / iters, (double)wbytes * iters / (double)dt1,
             dt2 / iters, (double)wbytes * iters / (double)dt2, (double)dt1 / (double)dt2);
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

static int generate(const int *prompt, int np, char *out, int out_max)
{
    s_kv.len = 0;
    s_nhist = 0;
    int written = 0, pos = 0;
    const int vocab = s_model.hdr.vocab;
    const int64_t t0 = esp_timer_get_time();
    float *logits = NULL;
    ESP_LOGI(TAG, "prefill %d tokens ...", np);
    for (int i = 0; i < np; i++) {                      /* prefill */
        logits = feng_forward(&s_model, &s_kv, &s_ws, prompt[i], pos++);
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
    out_printf("\n>>END\n");
    fflush(stdout);
    const int64_t dt = esp_timer_get_time() - t0;
    const float tps = (float)(pos) * 1e6f / (float)dt;
    ESP_LOGI(TAG, "prompt %d | gen %d | prefill %lld ms | total %lld ms | %.2f tok/s (%.0f ms/token)",
             np, n_gen, (t_prefill - t0) / 1000, dt / 1000, tps,
             (float)dt / 1000.0f / (float)n_gen);
    return written;
}

static void chat_once(const char *user)
{
    static char prompt[2048];
    static int ids[1024];
    static char reply[1024];
    snprintf(prompt, sizeof(prompt), "<|im_start|>user\n%s<|im_end|>\n<|im_start|>assistant\n",
             user);
    const int n = feng_tok_encode(&s_tok, prompt, ids, 1024);
    ESP_LOGI(TAG, "in: %s (%d prompt tokens)", user, n);
    reply[0] = 0;
    generate(ids, n, reply, sizeof(reply));
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

    static char line[1024];
    static char u8[1024];
    out_printf("\n=== feng-30m on ESP32-S3 (serial chat) ===\n");
    out_printf("one sentence per line; the reply streams between << and >>END\n");
    out_printf("commands: \\gbk  \\utf8  \\help\n");
    out_printf("FENG_READY\n");
    fflush(stdout);
    while (1) {
        out_printf("you> ");
        fflush(stdout);
        size_t n = 0;
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
            }
        }
        line[n] = 0;
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
            } else if (strncmp(line, "\\help", 5) == 0) {
                out_printf("\\gbk   reply in GBK (for SuperCom/XCOM in ANSI mode)\n");
                out_printf("\\utf8  reply in UTF-8\n");
                out_printf("\\stream N  flush every N bytes (0 = whole reply at once, default 30)\n");
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
        chat_once(u8);
    }
}
