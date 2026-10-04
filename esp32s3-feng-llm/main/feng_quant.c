/* Q4 (block 64, fp16 scale) + fp16 GEMV.  Layout per output row:
 *   [scale fp16] x n_blocks   followed by   [packed nibbles] x n_blocks x 32 bytes
 * nibble k of a block holds weight index 2k (low) and 2k+1 (high), value = (n-8)*scale
 */
#include "feng.h"

#include <math.h>
#include <string.h>

#define QK 64

/* byte -> (low nibble - 8, high nibble - 8) as floats, built once into internal RAM.
 * Replaces 2 shifts + 2 subtracts + 2 int->float converts per packed byte with two
 * 32-bit loads, which is the dominant per-weight cost of the Q4 kernel. */
static float s_q4_lut[256][2];
static int s_q4_lut_ready;

#if FENG_GEMV_A8 || FENG_GEMV_PIE
/* A8/PIE 的激活量化缓冲区：每个 GEMV 调用准备一次（prepare），行循环只读。
 * 推理是单线程的，两个 SMP worker 都在 prepare 之后并发读同一份数据。 */
#define A8_MAX_IN 1024
static int8_t s_a8_xq[A8_MAX_IN];
static float s_a8_xscale[A8_MAX_IN / QK];
static const float *s_a8_x;
static int s_a8_nin;
#if FENG_GEMV_PIE
static int16_t s_x16[A8_MAX_IN] __attribute__((aligned(16)));
/* 256 项 pair LUT：打包字节 -> 两个 int16 权重 (lo-8, hi-8)，按 u32 packed */
static uint32_t s_q4_pair32[256];
static int s_q4_pair32_ready;
extern int32_t pie_dot64(const int16_t *w, const int16_t *x);
static void q4_pair32_init(void)
{
    if (s_q4_pair32_ready) return;
    for (int b = 0; b < 256; b++) {
        const int16_t lo = (int16_t)((b & 0x0F) - 8);
        const int16_t hi = (int16_t)(((b >> 4) & 0x0F) - 8);
        s_q4_pair32[b] = (uint32_t)(uint16_t)lo | ((uint32_t)(uint16_t)hi << 16);
    }
    s_q4_pair32_ready = 1;
}

/* 自检：同一行、同一份 int8 激活，PIE 点积 vs 纯标量整数点积。
 * 两边都是整数运算、同样的操作数，结果必须逐位一致（返回 0 = MATCH）。 */
int feng_gemv_pie_selfcheck(const void *tensor, int n_in)
{
    if (!(s_a8_x && s_a8_nin == n_in)) return -1;      /* 需要先 prepare */
    q4_pair32_init();
    const uint8_t *row = (const uint8_t *)tensor;
    const int n_blocks = n_in / QK;
    const uint8_t *packed = row + (size_t)n_blocks * 2;
    int32_t pie_sum = 0, ref_sum = 0;
    for (int b = 0; b < n_blocks; b++) {
        const uint8_t *p = packed + (size_t)b * 32;
        int16_t w16[QK] __attribute__((aligned(16)));
        uint32_t *st = (uint32_t *)w16;
        for (int j = 0; j < 32; j++) st[j] = s_q4_pair32[p[j]];
        pie_sum += pie_dot64(w16, s_x16 + (size_t)b * QK);
        int32_t r = 0;
        for (int j = 0; j < 32; j++) {
            const int lo = (int)(p[j] & 0x0F) - 8;
            const int hi = (int)(p[j] >> 4) - 8;
            r += lo * (int)s_a8_xq[(size_t)b * QK + 2 * j]
                 + hi * (int)s_a8_xq[(size_t)b * QK + 2 * j + 1];
        }
        ref_sum += r;
    }
    return pie_sum == ref_sum ? 0 : 1;
}
#endif

void feng_gemv_a8_prepare(const float *x, int n_in)
{
    if (n_in > A8_MAX_IN || (n_in % QK) != 0) { s_a8_x = 0; s_a8_nin = 0; return; }
    const int n_blocks = n_in / QK;
    for (int b = 0; b < n_blocks; b++) {
        const float *xb = x + (size_t)b * QK;
        float amax = 0.f;
        for (int i = 0; i < QK; i++) {
            const float v = xb[i] < 0 ? -xb[i] : xb[i];
            if (v > amax) amax = v;
        }
        int8_t *xq = s_a8_xq + (size_t)b * QK;
        if (amax < 1e-8f) {            /* 与原逐行实现一致：整块跳过 */
            s_a8_xscale[b] = 0.f;
            memset(xq, 0, QK);
            continue;
        }
        const float scale_x = amax / 127.f;
        s_a8_xscale[b] = scale_x;
        for (int i = 0; i < QK; i++) {
            int q = (int)(xb[i] / scale_x + (xb[i] >= 0 ? 0.5f : -0.5f));
            if (q > 127) q = 127;
            if (q < -127) q = -127;
            xq[i] = (int8_t)q;
#if FENG_GEMV_PIE
            s_x16[(size_t)b * QK + i] = (int16_t)q;
#endif
        }
    }
    s_a8_x = x;
    s_a8_nin = n_in;
}
#endif


static void q4_lut_init(void)
{
    if (s_q4_lut_ready) return;
    for (int b = 0; b < 256; b++) {
        s_q4_lut[b][0] = (float)((b & 0x0F) - 8);
        s_q4_lut[b][1] = (float)((b >> 4) - 8);
    }
    s_q4_lut_ready = 1;
}

static inline float f16_to_f32(uint16_t h)
{
    /* IEEE-754 half -> float */
    uint32_t sign = (uint32_t)(h >> 15) & 1u;
    uint32_t exp = (uint32_t)(h >> 10) & 0x1Fu;
    uint32_t man = (uint32_t)h & 0x3FFu;
    uint32_t f;
    if (exp == 0) {
        if (man == 0) {
            f = sign << 31;
        } else {
            exp = 127 - 15 + 1;
            while ((man & 0x400u) == 0) {
                man <<= 1;
                exp--;
            }
            man &= 0x3FFu;
            f = (sign << 31) | (exp << 23) | (man << 13);
        }
    } else if (exp == 31) {
        f = (sign << 31) | 0x7F800000u | (man << 13);
    } else {
        f = (sign << 31) | ((exp + 127 - 15) << 23) | (man << 13);
    }
    float out;
    memcpy(&out, &f, 4);
    return out;
}

float feng_f16_to_f32(uint16_t h)
{
    return f16_to_f32(h);
}

uint16_t feng_f32_to_f16(float f)
{
    uint32_t x;
    memcpy(&x, &f, 4);
    const uint32_t sign = (x >> 31) & 1u;
    int32_t exp = (int32_t)((x >> 23) & 0xFFu) - 127;
    uint32_t man = x & 0x7FFFFFu;
    if (exp > 15) {                       /* overflow -> +-inf */
        return (uint16_t)((sign << 15) | 0x7C00u);
    }
    if (exp < -24) return (uint16_t)(sign << 15);
    if (exp < -14) {                      /* subnormal */
        const uint32_t shift = (uint32_t)(-14 - exp);
        man = (man | 0x800000u) >> (shift + 13);
        return (uint16_t)((sign << 15) | (man & 0x3FFu));
    }
    return (uint16_t)((sign << 15) | ((uint32_t)(exp + 15) << 10) | (man >> 13));
}

FENG_HOT void feng_gemv_range(const void *tensor, uint32_t dtype, const float *x, float *y,
                     int r0, int r1, int n_in)
{
    const uint8_t *base = (const uint8_t *)tensor;
    if (dtype == FENG_DT_FP16) {
        const uint16_t *w = (const uint16_t *)tensor;
        for (int o = r0; o < r1; o++) {
            const uint16_t *row = w + (size_t)o * n_in;
            float acc = 0.f;
            for (int i = 0; i < n_in; i++) {
                acc += f16_to_f32(row[i]) * x[i];
            }
            y[o] = acc;
        }
        return;
    }
    const int n_blocks = n_in / QK;
    const size_t row_bytes = (size_t)n_blocks * (2 + 32);
    q4_lut_init();
#if FENG_GEMV_PIE
    /* PIE 内核：激活 int16（prepare 阶段已算好）+ 权重 LUT 展开 + vmulas.s16。
     * 整数点积与 A8 路径完全同源（同样的 xq/scale），结果应逐位一致。 */
    if (s_a8_x == x && s_a8_nin == n_in) {
        q4_pair32_init();
        for (int o = r0; o < r1; o++) {
            const uint8_t *row = base + (size_t)o * row_bytes;
            const uint8_t *scales = row;
            const uint8_t *packed = row + (size_t)n_blocks * 2;
            float acc = 0.f;
            for (int b = 0; b < n_blocks; b++) {
                const float sx = s_a8_xscale[b];
                if (sx == 0.f) continue;
                uint16_t hs;
                memcpy(&hs, scales + b * 2, 2);
                const float sw = f16_to_f32(hs);
                const uint8_t *p = packed + (size_t)b * 32;
                int16_t w16[QK] __attribute__((aligned(16)));
                uint32_t *st = (uint32_t *)w16;
                for (int j = 0; j < 32; j++) st[j] = s_q4_pair32[p[j]];
                const int32_t dot = pie_dot64(w16, s_x16 + (size_t)b * QK);
                acc += (float)dot * (sw * sx);
            }
            y[o] = acc;
        }
        return;
    }
#endif
#if FENG_GEMV_A8
    /* A8 模式：激活按 64 值一块量化成 int8（scale_x = max|x|/127），
     * 权重仍是 Q4（-8..7），点积走整数，再乘 scale_w*scale_x。
     * 量化原本每个输出行都重做一遍（O(n_out×n_in) 的额外开销），现在改成
     * 每个 GEMV 调用一次（feng_gemv_a8_prepare）——数值与逐行版逐位一致。 */
    const int a8_prepared = (s_a8_x == x && s_a8_nin == n_in);
    int8_t xq_local[QK];
    for (int o = r0; o < r1; o++) {
        const uint8_t *row = base + (size_t)o * row_bytes;
        const uint8_t *scales = row;
        const uint8_t *packed = row + (size_t)n_blocks * 2;
        float acc = 0.f;
        for (int b = 0; b < n_blocks; b++) {
            uint16_t hs;
            memcpy(&hs, scales + b * 2, 2);
            const float scale_w = f16_to_f32(hs);
            const int8_t *xq;
            float scale_x;
            if (a8_prepared) {
                scale_x = s_a8_xscale[b];
                if (scale_x == 0.f) continue;
                xq = s_a8_xq + (size_t)b * QK;
            } else {
                const float *xb = x + (size_t)b * QK;
                float amax = 0.f;
                for (int i = 0; i < QK; i++) {
                    const float v = xb[i] < 0 ? -xb[i] : xb[i];
                    if (v > amax) amax = v;
                }
                if (amax < 1e-8f) continue;
                scale_x = amax / 127.f;
                for (int i = 0; i < QK; i++) {
                    int q = (int)(xb[i] / scale_x + (xb[i] >= 0 ? 0.5f : -0.5f));
                    if (q > 127) q = 127;
                    if (q < -127) q = -127;
                    xq_local[i] = (int8_t)q;
                }
                xq = xq_local;
            }
            int32_t dot = 0;
            const uint8_t *p = packed + (size_t)b * 32;
            for (int j = 0; j < 32; j++) {
                const int lo = (int)(p[j] & 0x0F) - 8;
                const int hi = (int)(p[j] >> 4) - 8;
                dot += lo * (int)xq[2 * j] + hi * (int)xq[2 * j + 1];
            }
            acc += (float)dot * (scale_w * scale_x);
        }
        y[o] = acc;
    }
    return;
#endif
    for (int o = r0; o < r1; o++) {
        const uint8_t *row = base + (size_t)o * row_bytes;
        const uint8_t *scales = row;
        const uint8_t *packed = row + (size_t)n_blocks * 2;
        /* several independent accumulators: a single one serialises the FPU pipeline
         * on the dependency chain (add latency x 32 madds per block) */
        float a0 = 0.f, a1 = 0.f, a2 = 0.f, a3 = 0.f;
        for (int b = 0; b < n_blocks; b++) {
            uint16_t hs;
            memcpy(&hs, scales + b * 2, 2);
            const float scale = f16_to_f32(hs);
            const uint8_t *p = packed + (size_t)b * 32;
            const float *xb = x + (size_t)b * QK;
            float s0 = 0.f, s1 = 0.f, s2 = 0.f, s3 = 0.f;
            for (int j = 0; j < 32; j += 4) {
                const uint8_t b0 = p[j], b1 = p[j + 1], b2 = p[j + 2], b3 = p[j + 3];
#if FENG_GEMV_MADD
                /* 纯 madd 链：每 2 个权重 2 个 FP 运算（对比下面的 3 个） */
                s0 += s_q4_lut[b0][0] * xb[2 * j];
                s0 += s_q4_lut[b0][1] * xb[2 * j + 1];
                s1 += s_q4_lut[b1][0] * xb[2 * j + 2];
                s1 += s_q4_lut[b1][1] * xb[2 * j + 3];
                s2 += s_q4_lut[b2][0] * xb[2 * j + 4];
                s2 += s_q4_lut[b2][1] * xb[2 * j + 5];
                s3 += s_q4_lut[b3][0] * xb[2 * j + 6];
                s3 += s_q4_lut[b3][1] * xb[2 * j + 7];
#else
                s0 += s_q4_lut[b0][0] * xb[2 * j] + s_q4_lut[b0][1] * xb[2 * j + 1];
                s1 += s_q4_lut[b1][0] * xb[2 * j + 2] + s_q4_lut[b1][1] * xb[2 * j + 3];
                s2 += s_q4_lut[b2][0] * xb[2 * j + 4] + s_q4_lut[b2][1] * xb[2 * j + 5];
                s3 += s_q4_lut[b3][0] * xb[2 * j + 6] + s_q4_lut[b3][1] * xb[2 * j + 7];
#endif
            }
            a0 += s0 * scale;
            a1 += s1 * scale;
            a2 += s2 * scale;
            a3 += s3 * scale;
        }
        y[o] = (a0 + a1) + (a2 + a3);
    }
}

void feng_gemv(const void *tensor, uint32_t dtype, const float *x, float *y,
               int n_out, int n_in, float *scratch)
{
    (void)scratch;
    feng_gemv_range(tensor, dtype, x, y, 0, n_out, n_in);
}

int feng_argmax(const float *logits, int n)
{
    int best = 0;
    float bv = logits[0];
    for (int i = 1; i < n; i++) {
        if (logits[i] > bv) {
            bv = logits[i];
            best = i;
        }
    }
    return best;
}
