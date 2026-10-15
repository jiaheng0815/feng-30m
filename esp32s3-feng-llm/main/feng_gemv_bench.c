/* Q4 GEMV 内层循环的**周期级**剖析（-DFENG_BENCH_GEMV=1 编译进去）。
 *
 * 为什么需要它：板端实测 GEMV 是每核 **7.52 周期/权重**（6282 µs / 401408 权重 × 240 MHz），
 * 而内层每 8 个权重只有约 31 条指令（4 次字节 load + 8 次 LUT load + 8 次激活 load +
 * 8 次 madd + 循环开销）≈ 3.9 条/权重。IPC 只有 ~0.52 —— **说明瓶颈不是指令条数，
 * 而是流水线停顿**。但停顿来自哪儿（LUT 的 load-use 延迟？激活加载？flash 流式读的
 * cache miss？FPU 的 madd 延迟链？）在板下是看不出来的，只能上板测。
 *
 * 做法：对同一个真实张量（第 0 层 gate，896x448 Q4）跑 6 个变体，每个变体拿掉/替换
 * 一个变量，差值就把停顿归因到具体来源。全部单核、同样的输出行区间，可直接相减。
 *
 *   V_full       现役内层循环（基线）
 *   V_nolut      LUT 换成常量浮点   -> 隔离 LUT 的 load-use 延迟
 *   V_nox        激活换成常量浮点   -> 隔离激活加载延迟
 *   V_now        权重字节换成常量   -> 隔离 flash / cache 流式读
 *   V_maddonly   纯 madd 链、零加载 -> FPU 吞吐上限
 *   V_blockcopy  先把 32B 块拷进栈上缓冲再算 -> 把 flash 流式读与计算解耦
 */
#include "feng.h"

#include <stdio.h>
#include <string.h>

#if defined(ESP_PLATFORM)
#include "esp_cpu.h"
#include "esp_timer.h"
#endif


/* 与本文件同款布局的 256 项浮点 LUT（byte -> 两个 dequant 值）。
 * 自建一份而不是借用 feng_quant.c 里的静态表：① 那张表是 static，本文件看不到；
 * ② 基准要的是"LUT 在内部 SRAM 里"这个条件，自建同样满足，且不引入耦合。 */
static float g_lut[256][2];
static int g_lut_ready;

static void gmb_lut_init(void)
{
    if (g_lut_ready) return;
    for (int b = 0; b < 256; b++) {
        g_lut[b][0] = (float)((b & 0x0F) - 8);
        g_lut[b][1] = (float)((b >> 4) - 8);
    }
    g_lut_ready = 1;
}

#define QK_B 64
#define GMB_ITERS 20

/* 防止编译器把没有可见副作用的循环整个消掉 */
static volatile float g_sink;

static inline int64_t cyc_now(void)
{
#if defined(ESP_PLATFORM)
    return (int64_t)esp_cpu_get_cycle_count();
#else
    return 0;
#endif
}

/* 各变体共用的行首地址计算，保证布局解读与 gemv_range_cpu 完全一致 */
#define ROW_SETUP                                                                        \
    const uint8_t *row = base + (size_t)o * row_bytes;                                   \
    const uint8_t *scales = row;                                                         \
    const uint8_t *packed = row + (size_t)n_blocks * 2;

/* ---------------- V_full：现役实现（照抄 gemv_range_cpu 的内层） ---------------- */
static float v_full(const uint8_t *base, int row_bytes, int n_blocks, const float *x,
                    int r0, int r1)
{
    float sink = 0.f;
    for (int o = r0; o < r1; o++) {
        ROW_SETUP
        float a0 = 0.f, a1 = 0.f, a2 = 0.f, a3 = 0.f;
        for (int b = 0; b < n_blocks; b++) {
            uint16_t hs;
            memcpy(&hs, scales + b * 2, 2);
            const float scale = feng_f16_to_f32(hs);
            const uint8_t *p = packed + (size_t)b * 32;
            const float *xb = x + (size_t)b * QK_B;
            float s0 = 0.f, s1 = 0.f, s2 = 0.f, s3 = 0.f;
            for (int j = 0; j < 32; j += 4) {
                const uint8_t b0 = p[j], b1 = p[j + 1], b2 = p[j + 2], b3 = p[j + 3];
                s0 += g_lut[b0][0] * xb[2 * j];
                s0 += g_lut[b0][1] * xb[2 * j + 1];
                s1 += g_lut[b1][0] * xb[2 * j + 2];
                s1 += g_lut[b1][1] * xb[2 * j + 3];
                s2 += g_lut[b2][0] * xb[2 * j + 4];
                s2 += g_lut[b2][1] * xb[2 * j + 5];
                s3 += g_lut[b3][0] * xb[2 * j + 6];
                s3 += g_lut[b3][1] * xb[2 * j + 7];
            }
            a0 += s0 * scale; a1 += s1 * scale; a2 += s2 * scale; a3 += s3 * scale;
        }
        sink += (a0 + a1) + (a2 + a3);
    }
    return sink;
}

/* ---------------- V_nolut：LUT 访问换成常量（保留权重/激活加载） ---------------- */
static float v_nolut(const uint8_t *base, int row_bytes, int n_blocks, const float *x,
                     int r0, int r1)
{
    float sink = 0.f;
    for (int o = r0; o < r1; o++) {
        ROW_SETUP
        float a0 = 0.f, a1 = 0.f, a2 = 0.f, a3 = 0.f;
        for (int b = 0; b < n_blocks; b++) {
            uint16_t hs;
            memcpy(&hs, scales + b * 2, 2);
            const float scale = feng_f16_to_f32(hs);
            const uint8_t *p = packed + (size_t)b * 32;
            const float *xb = x + (size_t)b * QK_B;
            float s0 = 0.f, s1 = 0.f, s2 = 0.f, s3 = 0.f;
            for (int j = 0; j < 32; j += 4) {
                const uint8_t b0 = p[j], b1 = p[j + 1], b2 = p[j + 2], b3 = p[j + 3];
                /* 仍然读字节，但用它们做整数运算而不是当 LUT 下标，避免查表 */
                s0 += (float)(b0 & 15) * xb[2 * j];
                s0 += (float)(b0 >> 4) * xb[2 * j + 1];
                s1 += (float)(b1 & 15) * xb[2 * j + 2];
                s1 += (float)(b1 >> 4) * xb[2 * j + 3];
                s2 += (float)(b2 & 15) * xb[2 * j + 4];
                s2 += (float)(b2 >> 4) * xb[2 * j + 5];
                s3 += (float)(b3 & 15) * xb[2 * j + 6];
                s3 += (float)(b3 >> 4) * xb[2 * j + 7];
            }
            a0 += s0 * scale; a1 += s1 * scale; a2 += s2 * scale; a3 += s3 * scale;
        }
        sink += (a0 + a1) + (a2 + a3);
    }
    return sink;
}

/* ---------------- V_nox：激活换成常量（保留权重加载 + LUT） ---------------- */
static float v_nox(const uint8_t *base, int row_bytes, int n_blocks, const float *x,
                   int r0, int r1)
{
    const float xc = x[0];                 /* 常量，但取值来自真实激活，避免全零被优化 */
    float sink = 0.f;
    for (int o = r0; o < r1; o++) {
        ROW_SETUP
        float a0 = 0.f, a1 = 0.f, a2 = 0.f, a3 = 0.f;
        for (int b = 0; b < n_blocks; b++) {
            uint16_t hs;
            memcpy(&hs, scales + b * 2, 2);
            const float scale = feng_f16_to_f32(hs);
            const uint8_t *p = packed + (size_t)b * 32;
            float s0 = 0.f, s1 = 0.f, s2 = 0.f, s3 = 0.f;
            for (int j = 0; j < 32; j += 4) {
                const uint8_t b0 = p[j], b1 = p[j + 1], b2 = p[j + 2], b3 = p[j + 3];
                s0 += g_lut[b0][0] * xc;
                s0 += g_lut[b0][1] * xc;
                s1 += g_lut[b1][0] * xc;
                s1 += g_lut[b1][1] * xc;
                s2 += g_lut[b2][0] * xc;
                s2 += g_lut[b2][1] * xc;
                s3 += g_lut[b3][0] * xc;
                s3 += g_lut[b3][1] * xc;
            }
            a0 += s0 * scale; a1 += s1 * scale; a2 += s2 * scale; a3 += s3 * scale;
        }
        sink += (a0 + a1) + (a2 + a3);
    }
    return sink;
}

/* ---------------- V_now：权重字节换成常量（保留 LUT + 激活加载） ----------------
 * 注意：这条**不能**用来判断 flash 带宽，因为权重不再被读；它的作用是给出
 * "零权重读取 + 全 LUT + 全激活加载"的耗时，与 V_full 相减得到权重读取的代价。 */
static float v_now(const uint8_t *base, int row_bytes, int n_blocks, const float *x,
                   int r0, int r1)
{
    (void)base; (void)row_bytes;
    float sink = 0.f;
    for (int o = r0; o < r1; o++) {
        const uint8_t wb = (uint8_t)(o * 37);      /* 每行换一个，防 CSE */
        for (int b = 0; b < n_blocks; b++) {
            const float scale = 0.03125f;
            const float *xb = x + (size_t)b * QK_B;
            float s0 = 0.f, s1 = 0.f, s2 = 0.f, s3 = 0.f;
            for (int j = 0; j < 32; j += 4) {
                const uint8_t b0 = wb, b1 = wb + 1, b2 = wb + 2, b3 = wb + 3;
                s0 += g_lut[b0][0] * xb[2 * j];
                s0 += g_lut[b0][1] * xb[2 * j + 1];
                s1 += g_lut[b1][0] * xb[2 * j + 2];
                s1 += g_lut[b1][1] * xb[2 * j + 3];
                s2 += g_lut[b2][0] * xb[2 * j + 4];
                s2 += g_lut[b2][1] * xb[2 * j + 5];
                s3 += g_lut[b3][0] * xb[2 * j + 6];
                s3 += g_lut[b3][1] * xb[2 * j + 7];
            }
            s0 *= scale;
            sink += s0 + s1 + s2 + s3;
        }
    }
    return sink;
}

/* ---------------- V_maddonly：纯 madd 链、零加载 -> FPU 吞吐上限 ---------------- */
static float v_maddonly(const uint8_t *base, int row_bytes, int n_blocks, const float *x,
                        int r0, int r1)
{
    (void)base; (void)row_bytes;
    const float c = x[0];
    float sink = 0.f;
    for (int o = r0; o < r1; o++) {
        float s0 = 0.f, s1 = 0.f, s2 = 0.f, s3 = 0.f;
        for (int b = 0; b < n_blocks; b++) {
            for (int j = 0; j < 32; j += 4) {
                s0 += c * c; s0 += c * c;
                s1 += c * c; s1 += c * c;
                s2 += c * c; s2 += c * c;
                s3 += c * c; s3 += c * c;
            }
        }
        sink += (s0 + s1) + (s2 + s3);
    }
    return sink;
}

/* ---------------- V_blockcopy：先把整块 32B + scale 拷到栈上再算 ----------------
 * 把 flash 流式读与计算解耦：拷贝是顺序的、可预取的，内层循环则完全在栈上跑。 */
static float v_blockcopy(const uint8_t *base, int row_bytes, int n_blocks, const float *x,
                         int r0, int r1)
{
    float sink = 0.f;
    for (int o = r0; o < r1; o++) {
        ROW_SETUP
        float a0 = 0.f, a1 = 0.f, a2 = 0.f, a3 = 0.f;
        for (int b = 0; b < n_blocks; b++) {
            uint8_t blk[32];
            uint16_t hs;
            memcpy(&hs, scales + b * 2, 2);
            const float scale = feng_f16_to_f32(hs);
            /* 32 字节 = 8 次字 load + 8 次字 store，摊到 64 个权重上只要 0.25 条/权重 */
            memcpy(blk, packed + (size_t)b * 32, 32);
            const float *xb = x + (size_t)b * QK_B;
            float s0 = 0.f, s1 = 0.f, s2 = 0.f, s3 = 0.f;
            for (int j = 0; j < 32; j += 4) {
                const uint8_t b0 = blk[j], b1 = blk[j + 1], b2 = blk[j + 2], b3 = blk[j + 3];
                s0 += g_lut[b0][0] * xb[2 * j];
                s0 += g_lut[b0][1] * xb[2 * j + 1];
                s1 += g_lut[b1][0] * xb[2 * j + 2];
                s1 += g_lut[b1][1] * xb[2 * j + 3];
                s2 += g_lut[b2][0] * xb[2 * j + 4];
                s2 += g_lut[b2][1] * xb[2 * j + 5];
                s3 += g_lut[b3][0] * xb[2 * j + 6];
                s3 += g_lut[b3][1] * xb[2 * j + 7];
            }
            a0 += s0 * scale; a1 += s1 * scale; a2 += s2 * scale; a3 += s3 * scale;
        }
        sink += (a0 + a1) + (a2 + a3);
    }
    return sink;
}


/* ================= 针对"内存延迟未重叠"的候选优化 =================
 * 基线结论：权重流式读占 4.69/7.31 周期（64%），有效带宽仅 27 MB/s，
 * 而裸流式读是 108 MB/s —— 4 倍余量卡在"每次 miss 都原地等"。
 * 下面 4 个变体都在试同一件事：**把内存并行度做大**。
 * 全部保持与 V_full 完全相同的运算顺序，结果校验值必须等于 52.209。
 */

/* P_word：一次读入 4 个 32 位字（16 字节），覆盖 8 个权重的 decode 面更宽。
 * 字节 load 变字 load 后，未完成的访存请求从 4 个窄请求变成 4 个宽请求。 */
static float p_word(const uint8_t *base, int row_bytes, int n_blocks, const float *x,
                    int r0, int r1)
{
    float sink = 0.f;
    for (int o = r0; o < r1; o++) {
        ROW_SETUP
        float a0 = 0.f, a1 = 0.f, a2 = 0.f, a3 = 0.f;
        for (int b = 0; b < n_blocks; b++) {
            uint16_t hs;
            memcpy(&hs, scales + b * 2, 2);
            const float scale = feng_f16_to_f32(hs);
            const uint32_t *pw = (const uint32_t *)(packed + (size_t)b * 32);
            const float *xb = x + (size_t)b * QK_B;
            float s0 = 0.f, s1 = 0.f, s2 = 0.f, s3 = 0.f;
            for (int j = 0; j < 8; j++) {
                const uint32_t w = pw[j];                 /* 1 次字 load = 4 个权重 */
                const uint8_t b0 = (uint8_t)(w & 0xFF), b1 = (uint8_t)((w >> 8) & 0xFF);
                const uint8_t b2 = (uint8_t)((w >> 16) & 0xFF), b3 = (uint8_t)(w >> 24);
                const float *xw = xb + 8 * j;
                s0 += g_lut[b0][0] * xw[0];
                s0 += g_lut[b0][1] * xw[1];
                s1 += g_lut[b1][0] * xw[2];
                s1 += g_lut[b1][1] * xw[3];
                s2 += g_lut[b2][0] * xw[4];
                s2 += g_lut[b2][1] * xw[5];
                s3 += g_lut[b3][0] * xw[6];
                s3 += g_lut[b3][1] * xw[7];
            }
            a0 += s0 * scale; a1 += s1 * scale; a2 += s2 * scale; a3 += s3 * scale;
        }
        sink += (a0 + a1) + (a2 + a3);
    }
    return sink;
}

/* P_row2：两行同时算。两条独立权重流 -> 同时挂起 2 个 cache miss，
 * 内存并行度翻倍。accumulator 也翻倍（8 路），FPU 依赖链更宽。 */
static float p_row2(const uint8_t *base, int row_bytes, int n_blocks, const float *x,
                    int r0, int r1)
{
    float sink = 0.f;
    int o = r0;
    for (; o + 1 < r1; o += 2) {
        const uint8_t *rowA = base + (size_t)o * row_bytes;
        const uint8_t *rowB = base + (size_t)(o + 1) * row_bytes;
        const uint8_t *scA = rowA, *pkA = rowA + (size_t)n_blocks * 2;
        const uint8_t *scB = rowB, *pkB = rowB + (size_t)n_blocks * 2;
        float A0 = 0.f, A1 = 0.f, A2 = 0.f, A3 = 0.f;
        float B0 = 0.f, B1 = 0.f, B2 = 0.f, B3 = 0.f;
        for (int b = 0; b < n_blocks; b++) {
            uint16_t ha, hb;
            memcpy(&ha, scA + b * 2, 2);
            memcpy(&hb, scB + b * 2, 2);
            const float sa = feng_f16_to_f32(ha), sb = feng_f16_to_f32(hb);
            const uint8_t *pa = pkA + (size_t)b * 32;
            const uint8_t *pb = pkB + (size_t)b * 32;
            const float *xb = x + (size_t)b * QK_B;
            float s0 = 0.f, s1 = 0.f, s2 = 0.f, s3 = 0.f;
            float t0 = 0.f, t1 = 0.f, t2 = 0.f, t3 = 0.f;
            for (int j = 0; j < 32; j += 4) {
                /* 先把两行的字节都取出来：8 个独立 load 全部提前发出 */
                const uint8_t a0 = pa[j], a1 = pa[j + 1], a2 = pa[j + 2], a3 = pa[j + 3];
                const uint8_t c0 = pb[j], c1 = pb[j + 1], c2 = pb[j + 2], c3 = pb[j + 3];
                s0 += g_lut[a0][0] * xb[2 * j];
                s0 += g_lut[a0][1] * xb[2 * j + 1];
                s1 += g_lut[a1][0] * xb[2 * j + 2];
                s1 += g_lut[a1][1] * xb[2 * j + 3];
                s2 += g_lut[a2][0] * xb[2 * j + 4];
                s2 += g_lut[a2][1] * xb[2 * j + 5];
                s3 += g_lut[a3][0] * xb[2 * j + 6];
                s3 += g_lut[a3][1] * xb[2 * j + 7];
                t0 += g_lut[c0][0] * xb[2 * j];
                t0 += g_lut[c0][1] * xb[2 * j + 1];
                t1 += g_lut[c1][0] * xb[2 * j + 2];
                t1 += g_lut[c1][1] * xb[2 * j + 3];
                t2 += g_lut[c2][0] * xb[2 * j + 4];
                t2 += g_lut[c2][1] * xb[2 * j + 5];
                t3 += g_lut[c3][0] * xb[2 * j + 6];
                t3 += g_lut[c3][1] * xb[2 * j + 7];
            }
            A0 += s0 * sa; A1 += s1 * sa; A2 += s2 * sa; A3 += s3 * sa;
            B0 += t0 * sb; B1 += t1 * sb; B2 += t2 * sb; B3 += t3 * sb;
        }
        sink += ((A0 + A1) + (A2 + A3)) + ((B0 + B1) + (B2 + B3));
    }
    for (; o < r1; o++) {                       /* 奇数行收尾 */
        const uint8_t *row = base + (size_t)o * row_bytes;
        const uint8_t *pk = row + (size_t)n_blocks * 2;
        float a = 0.f;
        for (int b = 0; b < n_blocks; b++) {
            uint16_t hs; memcpy(&hs, row + b * 2, 2);
            const float sc = feng_f16_to_f32(hs);
            const uint8_t *p = pk + (size_t)b * 32;
            const float *xb = x + (size_t)b * QK_B;
            float s = 0.f;
            for (int j = 0; j < 32; j++) s += g_lut[p[j]][0] * xb[j];
            a += s * sc;
        }
        sink += a;
    }
    return sink;
}

/* P_prefetch：算当前行之前，先"碰"一下后两行的每个 cache line，
 * 把行填充提前发起（Xtensa 没有 prefetch 指令，普通 load 就能把行拉进 cache）。 */
static float p_prefetch(const uint8_t *base, int row_bytes, int n_blocks, const float *x,
                        int r0, int r1)
{
    volatile uint8_t touch = 0;
    float sink = 0.f;
    for (int o = r0; o < r1; o++) {
        /* 后 2 行，每 64 字节碰一个字节 */
        for (int ahead = 1; ahead <= 2; ahead++) {
            const int oo = o + ahead;
            if (oo >= r1) break;
            const uint8_t *pr = base + (size_t)oo * row_bytes;
            for (int k = 0; k < row_bytes; k += 64) touch += pr[k];
        }
        ROW_SETUP
        float a0 = 0.f, a1 = 0.f, a2 = 0.f, a3 = 0.f;
        for (int b = 0; b < n_blocks; b++) {
            uint16_t hs;
            memcpy(&hs, scales + b * 2, 2);
            const float scale = feng_f16_to_f32(hs);
            const uint8_t *p = packed + (size_t)b * 32;
            const float *xb = x + (size_t)b * QK_B;
            float s0 = 0.f, s1 = 0.f, s2 = 0.f, s3 = 0.f;
            for (int j = 0; j < 32; j += 4) {
                const uint8_t b0 = p[j], b1 = p[j + 1], b2 = p[j + 2], b3 = p[j + 3];
                s0 += g_lut[b0][0] * xb[2 * j];
                s0 += g_lut[b0][1] * xb[2 * j + 1];
                s1 += g_lut[b1][0] * xb[2 * j + 2];
                s1 += g_lut[b1][1] * xb[2 * j + 3];
                s2 += g_lut[b2][0] * xb[2 * j + 4];
                s2 += g_lut[b2][1] * xb[2 * j + 5];
                s3 += g_lut[b3][0] * xb[2 * j + 6];
                s3 += g_lut[b3][1] * xb[2 * j + 7];
            }
            a0 += s0 * scale; a1 += s1 * scale; a2 += s2 * scale; a3 += s3 * scale;
        }
        sink += (a0 + a1) + (a2 + a3);
    }
    g_sink = (float)touch;
    return sink;
}

typedef float (*gmb_fn)(const uint8_t *, int, int, const float *, int, int);

void feng_gemv_bench(const void *tensor, int n_out, int n_in, const float *x)
{
#if defined(ESP_PLATFORM)
    gmb_lut_init();
    const int n_blocks = n_in / QK_B;
    const int row_bytes = n_blocks * 34;
    const uint8_t *base = (const uint8_t *)tensor;
    const int64_t nw = (int64_t)n_out * n_in;            /* 总权重数 */

    const struct { const char *name; gmb_fn fn; } V[] = {
        {"V_full     ", v_full},
        {"V_nolut    ", v_nolut},
        {"V_nox      ", v_nox},
        {"V_now      ", v_now},
        {"V_maddonly ", v_maddonly},
        {"V_blockcopy", v_blockcopy},
        {"P_word     ", p_word},
        {"P_row2     ", p_row2},
        {"P_prefetch ", p_prefetch},
    };

    printf("  [gemv 周期剖析] %d x %d, %d 权重/次\n", n_out, n_in, (int)nw);
    float ref = 0.f;
    for (unsigned k = 0; k < sizeof(V) / sizeof(V[0]); k++) {
        /* 预热一次（把 LUT / 激活读进 cache，并让分支预测稳定） */
        float s = V[k].fn(base, row_bytes, n_blocks, x, 0, n_out);
        int64_t c0 = cyc_now();
        for (int i = 0; i < GMB_ITERS; i++) {
            s += V[k].fn(base, row_bytes, n_blocks, x, 0, n_out);
        }
        int64_t c1 = cyc_now();
        g_sink = s;
        if (k == 0) ref = s;
        const double cyc = (double)(c1 - c0) / GMB_ITERS;
        printf("    %s  %10.0f cyc/次  %7.3f cyc/权重  (结果校验 %.3f)\n",
               V[k].name, cyc, cyc / (double)nw, (double)s);
    }
    (void)ref;
    printf("    （注意 V_now 会把 LUT 提升出内层循环，它省下的不只是权重读取，\n"
           "      所以下面的「缓存常驻」对照才是干净的归因实验）\n");

    /* --- 干净的归因实验：同样代码，只跑一个小到能常驻 L1D 的行区间 ---
     * L1D = 64 KB，每行 238 B；96 行 = 22.8 KB，加激活与 LUT 也远小于 64 KB。
     * 反复跑同一个小区间 -> 命中率接近 100% -> 测出来的是**纯计算 + 依赖链**。
     * 与上面的全量（213 KB，必然 miss）相减，才是 flash miss 的净代价。 */
    {
        const int small_rows = 96;
        printf("  [缓存常驻对照] 只跑前 %d 行 (%.1f KB, L1D 常驻)\n",
               small_rows, (double)small_rows * row_bytes / 1024.0);
        const int64_t nw_small = (int64_t)small_rows * n_in;
        for (unsigned k = 0; k < sizeof(V) / sizeof(V[0]); k++) {
            float s2 = V[k].fn(base, row_bytes, n_blocks, x, 0, small_rows);
            const int64_t d0 = cyc_now();
            for (int i = 0; i < GMB_ITERS; i++) {
                s2 += V[k].fn(base, row_bytes, n_blocks, x, 0, small_rows);
            }
            const int64_t d1 = cyc_now();
            g_sink = s2;
            const double cyc2 = (double)(d1 - d0) / GMB_ITERS;
            printf("    %s  %10.0f cyc/次  %7.3f cyc/权重\n",
                   V[k].name, cyc2, cyc2 / (double)nw_small);
        }
    }
#else
    (void)tensor; (void)n_out; (void)n_in; (void)x;
    printf("  [gemv 周期剖析] 仅板端可用\n");
#endif
}
