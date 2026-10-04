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
#if FENG_GEMV_A8
    /* A8 模式：激活按 64 值一块量化成 int8（scale_x = max|x|/127），
     * 权重仍是 Q4（-8..7），点积走整数，再乘 scale_w*scale_x。
     * 这是给 PIE（ee.vmulas.s16.accx）铺路的算法，先做成可移植版本量质量代价。 */
    int8_t xq[QK];
    for (int o = r0; o < r1; o++) {
        const uint8_t *row = base + (size_t)o * row_bytes;
        const uint8_t *scales = row;
        const uint8_t *packed = row + (size_t)n_blocks * 2;
        float acc = 0.f;
        for (int b = 0; b < n_blocks; b++) {
            uint16_t hs;
            memcpy(&hs, scales + b * 2, 2);
            const float scale_w = f16_to_f32(hs);
            const float *xb = x + (size_t)b * QK;
            float amax = 0.f;
            for (int i = 0; i < QK; i++) {
                const float v = xb[i] < 0 ? -xb[i] : xb[i];
                if (v > amax) amax = v;
            }
            if (amax < 1e-8f) continue;
            const float scale_x = amax / 127.f;
            for (int i = 0; i < QK; i++) {
                int q = (int)(xb[i] / scale_x + (xb[i] >= 0 ? 0.5f : -0.5f));
                if (q > 127) q = 127;
                if (q < -127) q = -127;
                xq[i] = (int8_t)q;
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
                s0 += s_q4_lut[b0][0] * xb[2 * j] + s_q4_lut[b0][1] * xb[2 * j + 1];
                s1 += s_q4_lut[b1][0] * xb[2 * j + 2] + s_q4_lut[b1][1] * xb[2 * j + 3];
                s2 += s_q4_lut[b2][0] * xb[2 * j + 4] + s_q4_lut[b2][1] * xb[2 * j + 5];
                s3 += s_q4_lut[b3][0] * xb[2 * j + 6] + s_q4_lut[b3][1] * xb[2 * j + 7];
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
