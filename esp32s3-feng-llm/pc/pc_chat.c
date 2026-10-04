/* PC 端聊天 CLI：和板端同一套 C 引擎 + 同一个计算 tool。
 *
 * 用法：pc_chat <model_export 目录> [max_new]
 *   - 直接输入算式（59+1、5.3+4.1）走板内计算器，秒回；
 *   - 其他输入走模型；默认多轮上下文（KV 跨轮累积），\reset 清空。
 * 编译（四种 KV 配置与 pc_kv_suite 相同）：
 *   gcc -O2 -DFENG_KV_Q2=1 -DFENG_KV_Q2_BLOCK=8 -o pc_chat.exe pc_chat.c \
 *       ../main/feng_model.c ../main/feng_llm.c ../main/feng_quant.c \
 *       ../main/feng_smp.c ../main/feng_tokenizer.c ../main/feng_calc.c -I../main -lm
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "feng.h"
#include "feng_calc.h"
#include "feng_tools.h"
#include "feng_tokenizer.h"
#include <time.h>

#if FENG_KV_Q2
#define KV_MODE "q2"
#else
#define KV_MODE "int8"
#endif

static void *xmalloc(size_t n)
{
    void *p = malloc(n);
    if (!p) { fprintf(stderr, "oom %zu\n", n); exit(1); }
    memset(p, 0, n);
    return p;
}

static unsigned char *read_file(const char *path, size_t *len)
{
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", path); exit(1); }
    fseek(f, 0, SEEK_END);
    const long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    unsigned char *buf = (unsigned char *)xmalloc((size_t)n + 1);
    if (fread(buf, 1, (size_t)n, f) != (size_t)n) { fprintf(stderr, "short read\n"); exit(1); }
    buf[n] = 0;
    fclose(f);
    *len = (size_t)n;
    return buf;
}

static long long pc_epoch_now(void)
{
    return (long long)time(NULL);                     /* PC 系统时钟（联网时即网络时间） */
}

static long long pc_uptime_us(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (long long)ts.tv_sec * 1000000 + ts.tv_nsec / 1000;
}

int main(int argc, char **argv)
{
    const char *dir = argc > 1 ? argv[1] : ".";
    const int max_new = argc > 2 ? atoi(argv[2]) : 96;
    const int ctx = 2048;
    char path[512];
    size_t mlen = 0, tlen = 0;

    snprintf(path, sizeof(path), "%s/model.bin", dir);
    unsigned char *mblob = read_file(path, &mlen);
    snprintf(path, sizeof(path), "%s/tokenizer.bin", dir);
    unsigned char *tblob = read_file(path, &tlen);

    feng_model_t m;
    if (feng_model_init(&m, mblob, mlen) != 0) { fprintf(stderr, "model init failed\n"); return 1; }
    feng_tok_t tok;
    if (feng_tok_load(&tok, tblob, tlen) != 0) { fprintf(stderr, "tokenizer load failed\n"); return 1; }

    feng_kv_t kv;
    memset(&kv, 0, sizeof(kv));
    kv.ctx = ctx;
    const size_t kv_bytes = feng_kv_bytes(&m, ctx);
    kv.k_cache = xmalloc(kv_bytes / 2 + 64);
    kv.v_cache = xmalloc(kv_bytes / 2 + 64);
#if FENG_KV_INT8 || FENG_KV_Q2
    const size_t n_sc = feng_kv_scale_slots(&m, ctx);
    kv.k_scale = (uint16_t *)xmalloc(n_sc * 2);
    kv.v_scale = (uint16_t *)xmalloc(n_sc * 2);
#endif

    feng_workspace_t ws;
    const int h = m.hdr.hidden, ff = m.hdr.ffn, v = m.hdr.vocab;
    ws.max_ctx = ctx;
    ws.x = (float *)xmalloc(sizeof(float) * h);
    ws.xn = (float *)xmalloc(sizeof(float) * h);
    ws.q = (float *)xmalloc(sizeof(float) * h);
    ws.k = (float *)xmalloc(sizeof(float) * h);
    ws.v = (float *)xmalloc(sizeof(float) * h);
    ws.attn = (float *)xmalloc(sizeof(float) * h);
    ws.proj = (float *)xmalloc(sizeof(float) * ff);
    ws.gate = (float *)xmalloc(sizeof(float) * ff);
    ws.up = (float *)xmalloc(sizeof(float) * ff);
    ws.ffn = (float *)xmalloc(sizeof(float) * ff);
    ws.logits = (float *)xmalloc(sizeof(float) * v);
    ws.scratch = (float *)xmalloc(sizeof(float) * (ctx > ff ? ctx : ff));

    int nl_id = 0;
    {
        int one = 0;
        if (feng_tok_encode(&tok, "\n", &one, 1) != 1) one = 0;
        nl_id = one;
    }

    feng_tools_set_time(pc_epoch_now);
    feng_tools_set_uptime(pc_uptime_us);

    printf("feng-30m PC chat ｜ model=%s ｜ KV=%s ctx=%d ｜ 算式/时间/随机数走 tool，\\reset 清空\n",
           dir, KV_MODE, ctx);
    char line[1024];
    while (1) {
        printf("you> ");
        fflush(stdout);
        if (!fgets(line, sizeof(line), stdin)) break;
        int n = (int)strlen(line);
        while (n > 0 && (line[n - 1] == '\n' || line[n - 1] == '\r')) line[--n] = 0;
        if (n == 0) continue;
        if (strcmp(line, "\\exit") == 0 || strcmp(line, "\\quit") == 0) break;
        if (strcmp(line, "\\reset") == 0) { kv.len = 0; printf("(context cleared)\n"); continue; }

        char calc_reply[256];
        if (feng_calc_answer(line, calc_reply, sizeof(calc_reply)) ||
            feng_time_answer(line, calc_reply, sizeof(calc_reply)) ||
            feng_random_answer(line, calc_reply, sizeof(calc_reply))) {
            printf("[tool] %s\n", calc_reply);
            continue;
        }

        char prompt[2048];
        snprintf(prompt, sizeof(prompt),
                 "<|im_start|>user\n%s<|im_end|>\n<|im_start|>assistant\n", line);
        int ids[1024];
        const int np = feng_tok_encode(&tok, prompt, ids, 1024);
        if (kv.len + np + max_new + 4 > ctx) {
            printf("[上下文已满 %d/%d，自动开始新对话]\n", kv.len, ctx);
            kv.len = 0;
        }
        if (np >= ctx - max_new) { printf("[输入太长]\n"); continue; }

        int pos = kv.len;
        float *logits = NULL;
        for (int i = 0; i < np; i++) {         /* prefill：中间 token 跳过 lm head */
            logits = feng_forward_ex(&m, &kv, &ws, ids[i], pos++, i + 1 == np);
        }
        printf("<< ");
        fflush(stdout);
        for (int step = 0; step < max_new; step++) {
            const int tk = feng_argmax(logits, v);
            if (tk == tok.id_im_end || tk == tok.id_eot) break;
            char b[16];
            const int nb = feng_tok_decode_token(&tok, tk, b, sizeof(b));
            fwrite(b, 1, (size_t)nb, stdout);
            fflush(stdout);
            logits = feng_forward(&m, &kv, &ws, tk, pos++);
        }
        logits = feng_forward(&m, &kv, &ws, tok.id_im_end, pos++);
        if (nl_id > 0) logits = feng_forward(&m, &kv, &ws, nl_id, pos++);
        (void)logits;
        kv.len = pos;
        printf(" >>END (ctx %d/%d)\n", kv.len, ctx);
    }
    return 0;
}
