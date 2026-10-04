/* Transformer forward pass for the feng-30m (Qwen3 architecture, MHA, tied head). */
#include "feng.h"

#include <math.h>
#include <stdio.h>
#include <string.h>

/* fp16 -> f32 for norm weights etc. */
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
    const int h = m->hdr.hidden, nh = m->hdr.n_heads, hd = m->hdr.head_dim;
    const int f = m->hdr.ffn;
    float *x = ws->x, *xn = ws->xn, *q = ws->q, *k = ws->k, *v = ws->v;
    float *attn = ws->attn, *proj = ws->proj, *gate = ws->gate, *up = ws->up, *ffn = ws->ffn;

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
        uint8_t *kc = (uint8_t *)kv->k_cache + (size_t)l * kv->ctx * (h / 4);
        uint8_t *vc = (uint8_t *)kv->v_cache + (size_t)l * kv->ctx * (h / 4);
        uint16_t *ksc = kv->k_scale + (size_t)l * kv->ctx * (h / FENG_KV_Q2_BLOCK);
        uint16_t *vsc = kv->v_scale + (size_t)l * kv->ctx * (h / FENG_KV_Q2_BLOCK);
        uint16_t *ksp = ksc + (size_t)pos * (h / FENG_KV_Q2_BLOCK);
        uint16_t *vsp = vsc + (size_t)pos * (h / FENG_KV_Q2_BLOCK);
        for (int hh = 0; hh < nh; hh++) {
            uint8_t *kd = kc + (size_t)pos * (h / 4) + hh * qb;
            uint8_t *vd = vc + (size_t)pos * (h / 4) + hh * qb;
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
                ksp[hh * nb + blk] = feng_f32_to_f16(sk);
                vsp[hh * nb + blk] = feng_f32_to_f16(sv);
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
                s *= feng_f16_to_f32(ksc[(size_t)t * nh + hh]) * scale;
                scores[t] = s;
                if (s > maxs) maxs = s;
            }
#elif FENG_KV_Q2
            for (int t = 0; t <= pos; t++) {
                const uint8_t *kh = kc + (size_t)t * (h / 4) + hh * qb;
                const uint16_t *ks = ksc + (size_t)t * (h / FENG_KV_Q2_BLOCK)
                                     + hh * nb;
                float s = 0.f;
                for (int blk = 0; blk < nb; blk++) {
                    const float sk = feng_f16_to_f32(ks[blk]);
                    for (int j = 0; j < bw; j++) {
                        const uint8_t b = kh[blk * bw + j];
                        for (int k4 = 0; k4 < 4; k4++) {
                            const int qq = (b >> (2 * k4)) & 3;
                            s += qh[blk * FENG_KV_Q2_BLOCK + j * 4 + k4]
                                 * ((float)qq - 1.5f) * sk;
                        }
                    }
                }
                s *= scale;
                scores[t] = s;
                if (s > maxs) maxs = s;
            }
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
            for (int t = 0; t <= pos; t++) {
                scores[t] = expf(scores[t] - maxs);
                sum += scores[t];
            }
            const float inv = 1.0f / sum;
#if FENG_KV_INT8
            for (int t = 0; t <= pos; t++) {
                pv[t] = scores[t] * feng_f16_to_f32(vsc[(size_t)t * nh + hh]);
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
                for (int t = 0; t <= pos; t++) {
                    const uint8_t *vb = vc + (size_t)t * (h / 4) + hh * qb;
                    const uint16_t *vs = vsc + (size_t)t * (h / FENG_KV_Q2_BLOCK) + hh * nb;
                    const float st = scores[t];
                    for (int blk = 0; blk < nb; blk++) {
                        const float sv = feng_f16_to_f32(vs[blk]);
                        for (int j = 0; j < bw; j++) {
                            const uint8_t b = vb[blk * bw + j];
                            for (int k4 = 0; k4 < 4; k4++) {
                                const int d = blk * FENG_KV_Q2_BLOCK + j * 4 + k4;
                                vacc[d] += (st * ((float)((b >> (2 * k4)) & 3) - 1.5f)) * sv;
                            }
                        }
                    }
                }
                for (int d = 0; d < hd; d++) attn[hh * hd + d] = vacc[d] * inv;
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
    rmsnorm(xn, x, m->out_norm, h, m->hdr.rms_eps);
    /* tied lm head: logits = xn @ tok_embd^T (fp16 weights) */
    feng_gemv_par(m->tok_embd, m->tok_embd_dtype, xn, ws->logits, m->hdr.vocab, h);
    kv->len = pos + 1;
    return ws->logits;
}
