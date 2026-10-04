/* Transformer forward pass for the feng-30m (Qwen3 architecture, MHA, tied head). */
#include "feng.h"

#include <math.h>
#include <stdio.h>
#include <string.h>

#if FENG_ATTN_PROF && defined(ESP_PLATFORM)
#include "esp_cpu.h"
unsigned long long g_prof_cycles[3];    /* K / softmax / V，单位 CPU 周期 */
#endif

/* fp16 -> f32 for norm weights etc. */
/* 与 feng_quant.c::f16_to_f32 完全同算法的本地内联版：注意力内层每个 block
 * 都要转一次 scale，走外部函数（~25 条指令/次、还有调用开销）在长上下文里
 * 被放大成主要开销；内联后只剩位运算。数值逐位一致。 */
static inline float f16_to_f32_local(uint16_t h)
{
    uint32_t sign = (uint32_t)(h >> 15) & 1u;
    uint32_t exp = (uint32_t)(h >> 10) & 0x1Fu;
    uint32_t man = (uint32_t)h & 0x3FFu;
    uint32_t f;
    if (exp == 0) {
        if (man == 0) {
            f = sign << 31;
        } else {
            exp = 127 - 15 + 1;
            while ((man & 0x400u) == 0) { man <<= 1; exp--; }
            man &= 0x3FFu;
            f = (sign << 31) | (exp << 23) | (man << 13);
        }
    } else if (exp == 31) {
        f = (sign << 31) | 0x7F800000u | (man << 13);
    } else {
        f = (sign << 31) | ((exp + 112) << 23) | (man << 13);
    }
    float out;
    memcpy(&out, &f, 4);
    return out;
}

static void f16_to_f32_vec(const uint16_t *src, float *dst, int n)
{
    for (int i = 0; i < n; i++) {
        uint32_t h = src[i];
        uint32_t sign = (h >> 15) & 1u, exp = (h >> 10) & 0x1Fu, man = h & 0x3FFu, f;
        if (exp == 0) {
            if (man == 0) {
                f = sign << 31;
            } else {
                exp = 127 - 15 + 1;
                while ((man & 0x400u) == 0) { man <<= 1; exp--; }
                man &= 0x3FFu;
                f = (sign << 31) | (exp << 23) | (man << 13);
            }
        } else if (exp == 31) {
            f = (sign << 31) | 0x7F800000u | (man << 13);
        } else {
            f = (sign << 31) | ((exp + 112) << 23) | (man << 13);
        }
        memcpy(&dst[i], &f, 4);
    }
}

/* scratch buffers live in .bss, not on the task stack (the S3's main stack is small) */
static float g_scores[4096];    /* one score per (query, past key) position */
static float g_tmp[512];
#if FENG_KV_Q2 && FENG_Q2_LUT
/* 字节 LUT：b -> { (b>>0&3)-1.5, (b>>2&3)-1.5, (b>>4&3)-1.5, (b>>6&3)-1.5 }
 * 表达式与原内层完全相同，只把移位/掩码/整数转浮点换成一次 L1 查表。 */
static float g_q2_lut[256][4];
static int g_q2_lut_ready;
static void q2_lut_init(void)
{
    if (g_q2_lut_ready) return;
    for (int b = 0; b < 256; b++) {
        for (int k = 0; k < 4; k++) {
            g_q2_lut[b][k] = (float)((b >> (2 * k)) & 3) - 1.5f;
        }
    }
    g_q2_lut_ready = 1;
}
#endif
/* RoPE inverse frequencies, computed once (powf per token per head is expensive) */
static float g_rope_inv[128];
static int g_rope_ready;

static void rmsnorm(float *out, const float *x, const void *w_fp16, int n, float eps)
{
    float ss = 0.f;
    for (int i = 0; i < n; i++) ss += x[i] * x[i];
    const float inv = 1.0f / sqrtf(ss / (float)n + eps);
    const uint16_t *w = (const uint16_t *)w_fp16;
    for (int i = 0; i < n; i++) {
        uint32_t h = w[i];
        uint32_t sign = (h >> 15) & 1u, exp = (h >> 10) & 0x1Fu, man = h & 0x3FFu, f;
        if (exp == 0) {
            if (man == 0) f = sign << 31;
            else {
                exp = 127 - 15 + 1;
                while ((man & 0x400u) == 0) { man <<= 1; exp--; }
                man &= 0x3FFu;
                f = (sign << 31) | (exp << 23) | (man << 13);
            }
        } else if (exp == 31) f = (sign << 31) | 0x7F800000u | (man << 13);
        else f = (sign << 31) | ((exp + 112) << 23) | (man << 13);
        float wf; memcpy(&wf, &f, 4);
        out[i] = x[i] * inv * wf;
    }
}

static void rope(float *vec, int n_heads, int head_dim, int pos, float theta)
{
    const int half = head_dim / 2;
    if (!g_rope_ready) {
        for (int i = 0; i < half && i < (int)(sizeof(g_rope_inv) / sizeof(g_rope_inv[0])); i++) {
            g_rope_inv[i] = powf(theta, -2.0f * (float)i / (float)head_dim);
        }
        g_rope_ready = 1;
    }
    for (int h = 0; h < n_heads; h++) {
        float *v = vec + h * head_dim;
        for (int i = 0; i < half; i++) {
            const float ang = (float)pos * g_rope_inv[i];
            const float c = cosf(ang), s = sinf(ang);
            const float x1 = v[i], x2 = v[i + half];
            v[i] = x1 * c - x2 * s;
            v[i + half] = x2 * c + x1 * s;
        }
    }
}

static void silu_mul(float *out, const float *g, const float *u, int n)
{
    for (int i = 0; i < n; i++) {
        const float x = g[i];
        out[i] = (x / (1.0f + expf(-x))) * u[i];
    }
}

size_t feng_kv_bytes(const feng_model_t *m, int ctx)
{
#if FENG_KV_INT8
    const size_t data = (size_t)m->hdr.n_layers * ctx * m->hdr.hidden * 2;
    const size_t scal = (size_t)m->hdr.n_layers * ctx * m->hdr.n_heads * 2 * 2;
    return data + scal;
#elif FENG_KV_Q2
    /* 2-bit 值：每 (层,位置) 的 K/V 各占 hidden/4 字节；
     * scale 按 FENG_KV_Q2_BLOCK 个值一块存 fp16 */
    const size_t data = (size_t)m->hdr.n_layers * ctx * (m->hdr.hidden / 4) * 2;
    const size_t scal = (size_t)m->hdr.n_layers * ctx * (m->hdr.hidden / FENG_KV_Q2_BLOCK) * 2 * 2;
    return data + scal;
#else
    return (size_t)m->hdr.n_layers * ctx * m->hdr.hidden * 2 * sizeof(float);
#endif
}

size_t feng_kv_scale_slots(const feng_model_t *m, int ctx)
{
    /* 单个 cache（K 或 V）需要的 fp16 scale 数量 */
#if FENG_KV_INT8
    return (size_t)m->hdr.n_layers * ctx * m->hdr.n_heads;
#elif FENG_KV_Q2
    return (size_t)m->hdr.n_layers * ctx * (m->hdr.hidden / FENG_KV_Q2_BLOCK);
#else
    (void)m; (void)ctx;
    return 0;
#endif
}

size_t feng_ws_bytes(const feng_model_t *m, int ctx)
{
    const int h = m->hdr.hidden, f = m->hdr.ffn, v = m->hdr.vocab;
    return (size_t)(h * 6 + f * 3 + v + ctx + 4096) * sizeof(float);
}

float *feng_forward(feng_model_t *m, feng_kv_t *kv, feng_workspace_t *ws, int token, int pos)
{
    return feng_forward_ex(m, kv, ws, token, pos, 1);
}

float *feng_forward_ex(feng_model_t *m, feng_kv_t *kv, feng_workspace_t *ws, int token, int pos,
                       int want_logits)
{
    const int h = m->hdr.hidden, nh = m->hdr.n_heads, hd = m->hdr.head_dim;
    const int f = m->hdr.ffn;
    float *x = ws->x, *xn = ws->xn, *q = ws->q, *k = ws->k, *v = ws->v;
    float *attn = ws->attn, *proj = ws->proj, *gate = ws->gate, *up = ws->up, *ffn = ws->ffn;
#if FENG_KV_Q2 && FENG_Q2_LUT
    q2_lut_init();
#endif

    /* embedding lookup (fp16) */
    if (m->tok_embd_dtype == FENG_DT_FP16) {
        const uint16_t *emb = (const uint16_t *)m->tok_embd;
        f16_to_f32_vec(emb + (size_t)token * h, x, h);
    } else {
        /* Q4 embedding row dequant (block 64) for a single row */
        const uint8_t *base = (const uint8_t *)m->tok_embd;
        const int n_blocks = h / 64;
        const size_t row_bytes = (size_t)n_blocks * (2 + 32);
        const uint8_t *row = base + (size_t)token * row_bytes;
        const uint8_t *scales = row;
        const uint8_t *packed = row + (size_t)n_blocks * 2;
        for (int b = 0; b < n_blocks; b++) {
            uint16_t hs;
            memcpy(&hs, scales + b * 2, 2);
            float sc;
            {
                uint32_t sign = (hs >> 15) & 1u, exp = (hs >> 10) & 0x1Fu, man = hs & 0x3FFu, f;
                if (exp == 0) {
                    if (man == 0) f = sign << 31;
                    else { exp = 113; while ((man & 0x400u) == 0) { man <<= 1; exp--; } man &= 0x3FFu;
                           f = (sign << 31) | (exp << 23) | (man << 13); }
                } else if (exp == 31) f = (sign << 31) | 0x7F800000u | (man << 13);
                else f = (sign << 31) | ((exp + 112) << 23) | (man << 13);
                memcpy(&sc, &f, 4);
            }
            const uint8_t *p = packed + (size_t)b * 32;
            for (int j = 0; j < 32; j++) {
                const int lo = (int)(p[j] & 0x0F) - 8;
                const int hi = (int)(p[j] >> 4) - 8;
                x[b * 64 + 2 * j] = (float)lo * sc;
                x[b * 64 + 2 * j + 1] = (float)hi * sc;
            }
        }
    }

    for (int l = 0; l < (int)m->hdr.n_layers; l++) {
        const feng_layer_t *L = &m->layers[l];
        rmsnorm(xn, x, L->attn_norm, h, m->hdr.rms_eps);
        feng_gemv_par(L->wq, FENG_DT_Q4, xn, q, h, h);
        feng_gemv_par(L->wk, FENG_DT_Q4, xn, k, h, h);
        feng_gemv_par(L->wv, FENG_DT_Q4, xn, v, h, h);

        /* per-head q/k RMSNorm (Qwen3) */
        {
            for (int hh = 0; hh < nh; hh++) {
                rmsnorm(g_tmp, q + hh * hd, L->q_norm, hd, m->hdr.rms_eps);
                memcpy(q + hh * hd, g_tmp, hd * sizeof(float));
                rmsnorm(g_tmp, k + hh * hd, L->k_norm, hd, m->hdr.rms_eps);
                memcpy(k + hh * hd, g_tmp, hd * sizeof(float));
            }
        }
        rope(q, nh, hd, pos, m->hdr.rope_theta);
        rope(k, nh, hd, pos, m->hdr.rope_theta);

        /* append to KV cache (int8 with per-head fp16 scale, or fp32) */
        const float scale = 1.0f / sqrtf((float)hd);
#if FENG_KV_INT8
        int8_t *kc = (int8_t *)kv->k_cache + (size_t)l * kv->ctx * h;
        int8_t *vc = (int8_t *)kv->v_cache + (size_t)l * kv->ctx * h;
        uint16_t *ksc = kv->k_scale + (size_t)l * kv->ctx * nh;
        uint16_t *vsc = kv->v_scale + (size_t)l * kv->ctx * nh;
        for (int hh = 0; hh < nh; hh++) {
            float ak = 1e-8f, av = 1e-8f;
            for (int d = 0; d < hd; d++) {
                const float a = fabsf(k[hh * hd + d]);
                const float b = fabsf(v[hh * hd + d]);
                if (a > ak) ak = a;
                if (b > av) av = b;
            }
            const float sk = ak / 127.0f, sv = av / 127.0f;
            ksc[(size_t)pos * nh + hh] = feng_f32_to_f16(sk);
            vsc[(size_t)pos * nh + hh] = feng_f32_to_f16(sv);
            int8_t *kd = kc + (size_t)pos * h + hh * hd;
            int8_t *vd = vc + (size_t)pos * h + hh * hd;
            for (int d = 0; d < hd; d++) {
                int qk = (int)lrintf(k[hh * hd + d] / sk);
                int qv = (int)lrintf(v[hh * hd + d] / sv);
                if (qk > 127) qk = 127; else if (qk < -127) qk = -127;
                if (qv > 127) qv = 127; else if (qv < -127) qv = -127;
                kd[d] = (int8_t)qk;
                vd[d] = (int8_t)qv;
            }
        }
#elif FENG_KV_Q2
        /* 2-bit 对称量化 + 每 FENG_KV_Q2_BLOCK 个值一个 fp16 scale（scale = 块内 max|·|/1.5，
         * q ∈ {0,1,2,3} → (q-1.5)*scale）；每字节打包 4 个值（低位在前）。
         * block=8 实测比 block=16 的续写保真高很多（0.33→0.73），代价是 scale 内存翻倍。 */
        const int qb = hd / 4;
        const int nb = hd / FENG_KV_Q2_BLOCK;         /* 每 head 的块数 */
        const int bw = FENG_KV_Q2_BLOCK / 4;          /* 每块的字节数 */
        /* q2 布局是 [layer][head][t]（与 int8/fp32 的 [layer][t][head] 不同）：
         * 逐 head 扫描时 t 连续，长上下文下是 PSRAM 顺序读；[layer][t][head] 每次
         * 只取 16B 却要占一条 32B 缓存行（利用率 50%），长文时被放大。 */
        for (int hh = 0; hh < nh; hh++) {
            const size_t row = ((size_t)l * nh + hh) * kv->ctx + pos;
            uint8_t *kd = (uint8_t *)kv->k_cache + row * qb;
            uint8_t *vd = (uint8_t *)kv->v_cache + row * qb;
            uint16_t *ksp = kv->k_scale + row * nb;
            uint16_t *vsp = kv->v_scale + row * nb;
            for (int blk = 0; blk < nb; blk++) {
                const int base = hh * hd + blk * FENG_KV_Q2_BLOCK;
                float ak = 1e-8f, av = 1e-8f;
                for (int d = 0; d < FENG_KV_Q2_BLOCK; d++) {
                    const float a = fabsf(k[base + d]);
                    const float b = fabsf(v[base + d]);
                    if (a > ak) ak = a;
                    if (b > av) av = b;
                }
                const float sk = ak / 1.5f, sv = av / 1.5f;
                ksp[blk] = feng_f32_to_f16(sk);
                vsp[blk] = feng_f32_to_f16(sv);
                for (int j = 0; j < bw; j++) {
                    uint8_t kb = 0, vb = 0;
                    for (int k4 = 0; k4 < 4; k4++) {
                        const int d = blk * FENG_KV_Q2_BLOCK + j * 4 + k4;
                        int qk = (int)lrintf(k[hh * hd + d] / sk + 1.5f);
                        int qv = (int)lrintf(v[hh * hd + d] / sv + 1.5f);
                        if (qk < 0) qk = 0; else if (qk > 3) qk = 3;
                        if (qv < 0) qv = 0; else if (qv > 3) qv = 3;
                        kb |= (uint8_t)(qk << (2 * k4));
                        vb |= (uint8_t)(qv << (2 * k4));
                    }
                    kd[blk * bw + j] = kb; vd[blk * bw + j] = vb;
                }
            }
        }
#else
        float *kc = (float *)kv->k_cache + (size_t)l * kv->ctx * h;
        float *vc = (float *)kv->v_cache + (size_t)l * kv->ctx * h;
        memcpy(kc + (size_t)pos * h, k, h * sizeof(float));
        memcpy(vc + (size_t)pos * h, v, h * sizeof(float));
#endif

        float *pv = ws->scratch;          /* scaled attention probabilities */
        memset(attn, 0, (size_t)nh * hd * sizeof(float));
        for (int hh = 0; hh < nh; hh++) {
            const float *qh = q + hh * hd;
            float *scores = g_scores;
            float maxs = -1e30f;
#if FENG_KV_INT8
            for (int t = 0; t <= pos; t++) {
                const int8_t *kh = kc + (size_t)t * h + hh * hd;
                float s = 0.f;
                for (int d = 0; d < hd; d++) s += qh[d] * (float)kh[d];
                s *= f16_to_f32_local(ksc[(size_t)t * nh + hh]) * scale;
                scores[t] = s;
                if (s > maxs) maxs = s;
            }
#elif FENG_KV_Q2
            const uint8_t *krow = (const uint8_t *)kv->k_cache
                                  + ((size_t)l * nh + hh) * kv->ctx * qb;
            const uint16_t *ksrow = kv->k_scale + ((size_t)l * nh + hh) * kv->ctx * nb;
#if FENG_ATTN_PROF
            const unsigned pc0 = (unsigned)esp_cpu_get_cycle_count();
#endif
#if FENG_Q2_LUT && FENG_Q2_PAIR
            /* 一次算 2 个上下文 token：两条独立的累加链填满 FPU 流水线，
             * qh 只加载一次、相邻 token 的 16B 数据同处一条 32B 缓存行。
             * 每个 score 的累加顺序与单 token 版完全一致 -> 逐位一致。 */
            {
                int t = 0;
                for (; t + 1 <= pos; t += 2) {
                    const uint8_t *kh0 = krow + (size_t)t * qb;
                    const uint8_t *kh1 = kh0 + qb;
                    const uint16_t *ks0 = ksrow + (size_t)t * nb;
                    const uint16_t *ks1 = ks0 + nb;
                    float s0 = 0.f, s1 = 0.f;
                    for (int blk = 0; blk < nb; blk++) {
                        const float sk0 = f16_to_f32_local(ks0[blk]);
                        const float sk1 = f16_to_f32_local(ks1[blk]);
                        const uint8_t *b0 = kh0 + blk * bw, *b1 = kh1 + blk * bw;
                        int d = blk * FENG_KV_Q2_BLOCK;
                        for (int j = 0; j < bw; j++) {
                            const float *l0 = g_q2_lut[b0[j]];
                            const float *l1 = g_q2_lut[b1[j]];
                            for (int k4 = 0; k4 < 4; k4++) {
                                const float qd = qh[d++];
                                s0 += qd * l0[k4] * sk0;
                                s1 += qd * l1[k4] * sk1;
                            }
                        }
                    }
                    s0 *= scale;
                    s1 *= scale;
                    scores[t] = s0;
                    scores[t + 1] = s1;
                    if (s0 > maxs) maxs = s0;
                    if (s1 > maxs) maxs = s1;
                }
                if (t <= pos) {                      /* 奇数长度：尾部单 token */
                    const uint8_t *kh = krow + (size_t)t * qb;
                    const uint16_t *ks = ksrow + (size_t)t * nb;
                    float s = 0.f;
                    for (int blk = 0; blk < nb; blk++) {
                        const float sk = f16_to_f32_local(ks[blk]);
                        const uint8_t *bb = kh + blk * bw;
                        int d = blk * FENG_KV_Q2_BLOCK;
                        for (int j = 0; j < bw; j++) {
                            const float *lu = g_q2_lut[bb[j]];
                            for (int k4 = 0; k4 < 4; k4++) s += qh[d++] * lu[k4] * sk;
                        }
                    }
                    s *= scale;
                    scores[t] = s;
                    if (s > maxs) maxs = s;
                }
            }
#else
            for (int t = 0; t <= pos; t++) {
                const uint8_t *kh = krow + (size_t)t * qb;
                const uint16_t *ks = ksrow + (size_t)t * nb;
                float s = 0.f;
#if FENG_Q2_LUT
                for (int blk = 0; blk < nb; blk++) {
                    const float sk = f16_to_f32_local(ks[blk]);
                    const uint8_t *bb = kh + blk * bw;
                    int d = blk * FENG_KV_Q2_BLOCK;
                    for (int j = 0; j < bw; j++) {
                        const float *lu = g_q2_lut[bb[j]];
                        for (int k4 = 0; k4 < 4; k4++) {
                            s += qh[d++] * lu[k4] * sk;
                        }
                    }
                }
#else
                for (int blk = 0; blk < nb; blk++) {
                    const float sk = f16_to_f32_local(ks[blk]);
                    for (int j = 0; j < bw; j++) {
                        const uint8_t b = kh[blk * bw + j];
                        for (int k4 = 0; k4 < 4; k4++) {
                            const int qq = (b >> (2 * k4)) & 3;
                            s += qh[blk * FENG_KV_Q2_BLOCK + j * 4 + k4]
                                 * ((float)qq - 1.5f) * sk;
                        }
                    }
                }
#endif
                s *= scale;
                scores[t] = s;
                if (s > maxs) maxs = s;
            }
#endif
#if FENG_ATTN_PROF
            {
                const unsigned pc1 = (unsigned)esp_cpu_get_cycle_count();
                g_prof_cycles[0] += (unsigned long long)(pc1 - pc0);
            }
#endif
#else
            for (int t = 0; t <= pos; t++) {
                const float *kh = kc + (size_t)t * h + hh * hd;
                float s = 0.f;
                for (int d = 0; d < hd; d++) s += qh[d] * kh[d];
                s *= scale;
                scores[t] = s;
                if (s > maxs) maxs = s;
            }
#endif
            float sum = 0.f;
#if FENG_ATTN_PROF
            const unsigned ps0 = (unsigned)esp_cpu_get_cycle_count();
#endif
            for (int t = 0; t <= pos; t++) {
                scores[t] = expf(scores[t] - maxs);
                sum += scores[t];
            }
            const float inv = 1.0f / sum;
#if FENG_ATTN_PROF
            g_prof_cycles[1] += (unsigned long long)((unsigned)esp_cpu_get_cycle_count() - ps0);
#endif
#if FENG_KV_INT8
            for (int t = 0; t <= pos; t++) {
                pv[t] = scores[t] * f16_to_f32_local(vsc[(size_t)t * nh + hh]);
            }
            for (int d = 0; d < hd; d++) {
                float acc = 0.f;
                for (int t = 0; t <= pos; t++) {
                    acc += pv[t] * (float)vc[(size_t)t * h + hh * hd + d];
                }
                attn[hh * hd + d] = acc * inv;
            }
#elif FENG_KV_Q2
            /* 连续访问版：原实现对每个输出维度 d 都按 112B 跨步去 PSRAM 抓 1 个字节
             * （缓存行利用率 1/32，长上下文时被放大成几十秒/轮）。改成 t 外层、
             * 每个 (t,head) 的 16 个打包字节顺序读完再累加；每个 d 对 t 的求和顺序
             * 与乘法分组 (scores[t]*bit)*sv 保持不变，结果逐位一致。 */
            {
                float vacc[64];
                for (int d = 0; d < hd; d++) vacc[d] = 0.f;
#if FENG_ATTN_PROF
                const unsigned pv0 = (unsigned)esp_cpu_get_cycle_count();
#endif
                const uint8_t *vrow = (const uint8_t *)kv->v_cache
                                      + ((size_t)l * nh + hh) * kv->ctx * qb;
                const uint16_t *vsrow = kv->v_scale + ((size_t)l * nh + hh) * kv->ctx * nb;
#if FENG_Q2_LUT && FENG_Q2_PAIR
                {
                    int t = 0;
                    for (; t + 1 <= pos; t += 2) {
                        const uint8_t *vb0 = vrow + (size_t)t * qb;
                        const uint8_t *vb1 = vb0 + qb;
                        const uint16_t *vs0 = vsrow + (size_t)t * nb;
                        const uint16_t *vs1 = vs0 + nb;
                        const float st0 = scores[t], st1 = scores[t + 1];
                        for (int blk = 0; blk < nb; blk++) {
                            const float sv0 = f16_to_f32_local(vs0[blk]);
                            const float sv1 = f16_to_f32_local(vs1[blk]);
                            const uint8_t *b0 = vb0 + blk * bw, *b1 = vb1 + blk * bw;
                            int d = blk * FENG_KV_Q2_BLOCK;
                            for (int j = 0; j < bw; j++) {
                                const float *l0 = g_q2_lut[b0[j]];
                                const float *l1 = g_q2_lut[b1[j]];
                                for (int k4 = 0; k4 < 4; k4++) {
                                    vacc[d] += (st0 * l0[k4]) * sv0;
                                    vacc[d] += (st1 * l1[k4]) * sv1;
                                    d++;
                                }
                            }
                        }
                    }
                    if (t <= pos) {                  /* 奇数长度：尾部单 token */
                        const uint8_t *vb = vrow + (size_t)t * qb;
                        const uint16_t *vs = vsrow + (size_t)t * nb;
                        const float st = scores[t];
                        for (int blk = 0; blk < nb; blk++) {
                            const float sv = f16_to_f32_local(vs[blk]);
                            const uint8_t *bb = vb + blk * bw;
                            int d = blk * FENG_KV_Q2_BLOCK;
                            for (int j = 0; j < bw; j++) {
                                const float *lu = g_q2_lut[bb[j]];
                                for (int k4 = 0; k4 < 4; k4++) {
                                    vacc[d++] += (st * lu[k4]) * sv;
                                }
                            }
                        }
                    }
                }
#else
                for (int t = 0; t <= pos; t++) {
                    const uint8_t *vb = vrow + (size_t)t * qb;
                    const uint16_t *vs = vsrow + (size_t)t * nb;
                    const float st = scores[t];
#if FENG_Q2_LUT
                    for (int blk = 0; blk < nb; blk++) {
                        const float sv = f16_to_f32_local(vs[blk]);
                        const uint8_t *bb = vb + blk * bw;
                        int d = blk * FENG_KV_Q2_BLOCK;
                        for (int j = 0; j < bw; j++) {
                            const float *lu = g_q2_lut[bb[j]];
                            for (int k4 = 0; k4 < 4; k4++) {
                                vacc[d++] += (st * lu[k4]) * sv;
                            }
                        }
                    }
#else
                    for (int blk = 0; blk < nb; blk++) {
                        const float sv = f16_to_f32_local(vs[blk]);
                        for (int j = 0; j < bw; j++) {
                            const uint8_t b = vb[blk * bw + j];
                            for (int k4 = 0; k4 < 4; k4++) {
                                const int d = blk * FENG_KV_Q2_BLOCK + j * 4 + k4;
                                vacc[d] += (st * ((float)((b >> (2 * k4)) & 3) - 1.5f)) * sv;
                            }
                        }
                    }
#endif
                }
#endif
                for (int d = 0; d < hd; d++) attn[hh * hd + d] = vacc[d] * inv;
#if FENG_ATTN_PROF
                g_prof_cycles[2] += (unsigned long long)((unsigned)esp_cpu_get_cycle_count() - pv0);
#endif
            }
#else
            for (int d = 0; d < hd; d++) {
                float acc = 0.f;
                for (int t = 0; t <= pos; t++) {
                    acc += scores[t] * vc[(size_t)t * h + hh * hd + d];
                }
                attn[hh * hd + d] = acc * inv;
            }
#endif
        }
        feng_gemv_par(L->wo, FENG_DT_Q4, attn, proj, h, h);
        for (int i = 0; i < h; i++) x[i] += proj[i];

        rmsnorm(xn, x, L->ffn_norm, h, m->hdr.rms_eps);
        feng_gemv_par(L->gate, FENG_DT_Q4, xn, gate, f, h);
        feng_gemv_par(L->up, FENG_DT_Q4, xn, up, f, h);
        silu_mul(ffn, gate, up, f);
        feng_gemv_par(L->down, FENG_DT_Q4, ffn, proj, h, f);
        for (int i = 0; i < h; i++) x[i] += proj[i];
    }
    if (want_logits) {
        rmsnorm(xn, x, m->out_norm, h, m->hdr.rms_eps);
        /* tied lm head: logits = xn @ tok_embd^T (fp16 weights) */
        feng_gemv_par(m->tok_embd, m->tok_embd_dtype, xn, ws->logits, m->hdr.vocab, h);
    }
    kv->len = pos + 1;
    return ws->logits;
}
