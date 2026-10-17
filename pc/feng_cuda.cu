/* CUDA GEMV 后端，说明见 feng_cuda.h。
 *
 * 数值约定：kernel 与 CPU（feng_quant.cpp 的 feng_gemv_range）使用完全相同的浮点
 * 运算顺序；本文件以 -fmad=false 编译，避免 mul+add 被收缩成 FMA 而改变末位。 */
#include "feng_cuda.h"
#include "feng.h"          /* FENG_DT_FP16 / FENG_DT_Q4（单一事实源） */

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace {

constexpr int kQK = 64;            /* 与 feng_quant.cpp 的 QK 一致 */
constexpr int kMaxTensors = 512;

/* 每 token 的运行时参数（device 侧读取，图捕获一次即可复用） */
struct FwdParams {
    int token;
    int pos;
};

struct DevTensor {
    const void *host;
    void *dev;
    size_t bytes;
};

DevTensor g_tab[kMaxTensors];
int g_n = 0;
int g_state = -1;                  /* -1 = 未探测；0 = 不可用；1 = 可用 */
size_t g_uploaded = 0;

bool init_once()
{
    if (g_state >= 0) return g_state == 1;
    /* 调试开关：FENG_CUDA=0 时强制走 CPU 路径（同一二进制即可对比两条路径）。 */
    const char *env = std::getenv("FENG_CUDA");
    if (env && std::strcmp(env, "0") == 0) {
        g_state = 0;
        return false;
    }
    int ndev = 0;
    if (cudaGetDeviceCount(&ndev) != cudaSuccess || ndev <= 0) {
        g_state = 0;
        return false;
    }
    if (cudaSetDevice(0) != cudaSuccess) {
        g_state = 0;
        return false;
    }
    g_state = 1;
    return true;
}

/* 同一 host 指针只上传一次；后续以更大尺寸命中时重新分配。 */
void *device_tensor(const void *host, size_t bytes)
{
    for (int i = 0; i < g_n; ++i) {
        if (g_tab[i].host != host) continue;
        if (g_tab[i].bytes >= bytes) return g_tab[i].dev;
        cudaFree(g_tab[i].dev);
        g_uploaded -= g_tab[i].bytes;
        g_tab[i].dev = nullptr;
        g_tab[i].bytes = 0;
        break;
    }
    void *dev = nullptr;
    if (cudaMalloc(&dev, bytes) != cudaSuccess) return nullptr;
    if (cudaMemcpy(dev, host, bytes, cudaMemcpyHostToDevice) != cudaSuccess) {
        cudaFree(dev);
        return nullptr;
    }
    if (g_n < kMaxTensors) {
        g_tab[g_n].host = host;
        g_tab[g_n].dev = dev;
        g_tab[g_n].bytes = bytes;
        ++g_n;
    }
    g_uploaded += bytes;
    return dev;
}

/* Q4（block64，每行 [scales: nb×fp16][packed: nb×32B]）：每行一个线程，
 * 运算顺序与 CPU 版逐条一致。 */
__global__ void q4_gemv_kernel(const unsigned char *__restrict__ w,
                               const float *__restrict__ x,
                               float *__restrict__ out, int r0, int r1, int n_in)
{
    const int local = blockIdx.x * blockDim.x + threadIdx.x;
    const int o = r0 + local;
    if (o >= r1) return;              /* 行数不是 blockDim 整数倍时的边界线程 */
    const int nb = n_in >> 6;
    const unsigned char *row = w + (size_t)o * ((size_t)nb * 34u);
    const unsigned char *scales = row;
    const unsigned char *packed = row + (size_t)nb * 2u;
    float a0 = 0.f, a1 = 0.f, a2 = 0.f, a3 = 0.f;
    for (int b = 0; b < nb; ++b) {
        const float scale = __half2float(__ushort_as_half(
            *reinterpret_cast<const unsigned short *>(scales + (size_t)b * 2u)));
        const unsigned char *p = packed + (size_t)b * 32u;
        const float *xb = x + (size_t)b * 64u;
        float s0 = 0.f, s1 = 0.f, s2 = 0.f, s3 = 0.f;
#pragma unroll 4
        for (int j = 0; j < 32; j += 4) {
            s0 += (float)((int)(p[j] & 0x0Fu) - 8) * xb[2 * j];
            s0 += (float)((int)(p[j] >> 4) - 8) * xb[2 * j + 1];
            s1 += (float)((int)(p[j + 1] & 0x0Fu) - 8) * xb[2 * j + 2];
            s1 += (float)((int)(p[j + 1] >> 4) - 8) * xb[2 * j + 3];
            s2 += (float)((int)(p[j + 2] & 0x0Fu) - 8) * xb[2 * j + 4];
            s2 += (float)((int)(p[j + 2] >> 4) - 8) * xb[2 * j + 5];
            s3 += (float)((int)(p[j + 3] & 0x0Fu) - 8) * xb[2 * j + 6];
            s3 += (float)((int)(p[j + 3] >> 4) - 8) * xb[2 * j + 7];
        }
        a0 += s0 * scale;
        a1 += s1 * scale;
        a2 += s2 * scale;
        a3 += s3 * scale;
    }
    out[local] = (a0 + a1) + (a2 + a3);
}

/* fp16 权重：CPU 版是顺序累加，这里同样每行一个线程顺序累加。 */
/* 每行一个 warp（32 线程协作 + shuffle 归约）的 Q4 GEMV：并行度比"每行一线程"
 * 高 32 倍，可隐藏显存延迟。注意：累加顺序与 CPU 串行版不同（logits 存在 ~1e-6
 * 量级差异），完整 GPU forward 的验收口径是"32 题矩阵输出与 CPU 版逐字一致"。 */
__global__ void q4_gemv_warp_kernel(const unsigned char *__restrict__ w,
                                    const float *__restrict__ x,
                                    float *__restrict__ out, int r0, int r1, int n_in)
{
    const int lane = threadIdx.x & 31;
    const int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int o = r0 + warp;
    if (o >= r1) return;
    const int nb = n_in >> 6;
    const unsigned char *row = w + (size_t)o * ((size_t)nb * 34u);
    const unsigned char *scales = row;
    const unsigned char *packed = row + (size_t)nb * 2u;
    float acc = 0.f;
    for (int b = 0; b < nb; ++b) {
        const float sc = __half2float(__ushort_as_half(
            *reinterpret_cast<const unsigned short *>(scales + (size_t)b * 2u)));
        const unsigned char *p = packed + (size_t)b * 32u;
        const float *xb = x + (size_t)b * 64u;
        for (int k = lane; k < 64; k += 32) {
            const unsigned char byte = p[k >> 1];
            const int q = (k & 1) ? (byte >> 4) : (byte & 0x0Fu);
            acc += ((float)(q - 8) * sc) * xb[k];
        }
    }
    for (int off = 16; off > 0; off >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, off);
    if (lane == 0) out[warp] = acc;
}

__global__ void fp16_gemv_warp_kernel(const unsigned short *__restrict__ w,
                                      const float *__restrict__ x,
                                      float *__restrict__ out, int r0, int r1, int n_in)
{
    const int lane = threadIdx.x & 31;
    const int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int o = r0 + warp;
    if (o >= r1) return;
    const __half *row = reinterpret_cast<const __half *>(w) + (size_t)o * (size_t)n_in;
    float acc = 0.f;
    for (int i = lane; i < n_in; i += 32) acc += __half2float(row[i]) * x[i];
    for (int off = 16; off > 0; off >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, off);
    if (lane == 0) out[warp] = acc;
}

__global__ void fp16_gemv_kernel(const unsigned short *__restrict__ w,
                                 const float *__restrict__ x,
                                 float *__restrict__ out, int r0, int r1, int n_in)
{
    const int local = blockIdx.x * blockDim.x + threadIdx.x;
    const int o = r0 + local;
    if (o >= r1) return;
    const __half *row = reinterpret_cast<const __half *>(w) + (size_t)o * (size_t)n_in;
    float acc = 0.f;
    for (int i = 0; i < n_in; ++i) acc += __half2float(row[i]) * x[i];
    out[local] = acc;
}

float *g_dx = nullptr;
float *g_dy = nullptr;
size_t g_dx_cap = 0;
size_t g_dy_cap = 0;

bool reserve(float **buf, size_t *cap, size_t need)
{
    if (*cap >= need) return true;
    if (*buf) {
        cudaFree(*buf);
        *buf = nullptr;
        *cap = 0;
    }
    if (cudaMalloc(buf, need) != cudaSuccess) return false;
    *cap = need;
    return true;
}

}  // namespace

extern "C" int feng_cuda_available(void)
{
    return init_once() ? 1 : 0;
}

extern "C" void feng_cuda_gemv_range(const void *tensor, uint32_t dtype, const float *x,
                                     float *y, int r0, int r1, int n_in)
{
    if (!init_once() || r0 >= r1 || n_in <= 0) return;
    const int rows = r1 - r0;
    size_t wbytes = 0;
    if (dtype == FENG_DT_FP16) {
        wbytes = (size_t)r1 * (size_t)n_in * 2u;
    } else {
        const int nb = n_in / kQK;
        if (nb <= 0 || (n_in % kQK) != 0) return;
        wbytes = (size_t)r1 * ((size_t)nb * 34u);
    }
    void *dw = device_tensor(tensor, wbytes);
    if (!dw) return;
    const size_t xb = (size_t)n_in * sizeof(float);
    const size_t yb = (size_t)rows * sizeof(float);
    if (!reserve(&g_dx, &g_dx_cap, xb) || !reserve(&g_dy, &g_dy_cap, yb)) return;
    /* 实测（RTX 5060 Ti / Windows WDDM）：同步 pageable 拷贝比 pinned+async 快一倍
     * （约 330 vs 150 tok/s）——WDDM 下每次 async 提交的固定开销超过 pinned 的收益，
     * 因此这里固定用同步拷贝。 */
    if (cudaMemcpy(g_dx, x, xb, cudaMemcpyHostToDevice) != cudaSuccess) return;
    const int threads = 256;
    const int blocks = (rows + threads - 1) / threads;
    if (dtype == FENG_DT_FP16) {
        fp16_gemv_kernel<<<blocks, threads>>>(
            reinterpret_cast<const unsigned short *>(dw), g_dx, g_dy, r0, r1, n_in);
    } else {
        q4_gemv_kernel<<<blocks, threads>>>(
            reinterpret_cast<const unsigned char *>(dw), g_dx, g_dy, r0, r1, n_in);
    }
    /* 写回调用方缓冲的 [r0, r1) 行（与 CPU 版语义一致） */
    if (cudaMemcpy(y + r0, g_dy, yb, cudaMemcpyDeviceToHost) != cudaSuccess) return;
    (void)cudaGetLastError();   /* 清除可能的 launch 错误标记 */
}

/* 共享输入 x 的多 GEMV 合并：x 只传一次，然后逐个权重 launch + 回拷。
 * 与 feng_gemv3 语义一致（w_i == NULL 跳过）。 */
extern "C" void feng_cuda_gemv3(const void *w0, const void *w1, const void *w2, uint32_t dtype,
                                const float *x, float *y0, float *y1, float *y2,
                                int n_out0, int n_out1, int n_out2, int n_in)
{
    if (!init_once() || n_in <= 0) return;
    const void *ws[3] = { w0, w1, w2 };
    float *ys[3] = { y0, y1, y2 };
    const int outs[3] = { n_out0, n_out1, n_out2 };
    const int nb = n_in / kQK;
    if (dtype != FENG_DT_FP16 && (nb <= 0 || (n_in % kQK) != 0)) return;
    const size_t xb = (size_t)n_in * sizeof(float);
    if (!reserve(&g_dx, &g_dx_cap, xb)) return;
    if (cudaMemcpy(g_dx, x, xb, cudaMemcpyHostToDevice) != cudaSuccess) return;   /* x 只传一次 */
    const int threads = 256;
    for (int i = 0; i < 3; ++i) {
        if (!ws[i] || outs[i] <= 0) continue;
        const size_t yb = (size_t)outs[i] * sizeof(float);
        const size_t wbytes = (dtype == FENG_DT_FP16)
            ? (size_t)outs[i] * (size_t)n_in * 2u
            : (size_t)outs[i] * ((size_t)nb * 34u);
        void *dw = device_tensor(ws[i], wbytes);
        if (!dw) return;
        if (!reserve(&g_dy, &g_dy_cap, yb)) return;
        const int blocks = (outs[i] + threads - 1) / threads;
        if (dtype == FENG_DT_FP16) {
            fp16_gemv_kernel<<<blocks, threads>>>(
                reinterpret_cast<const unsigned short *>(dw), g_dx, g_dy, 0, outs[i], n_in);
        } else {
            q4_gemv_kernel<<<blocks, threads>>>(
                reinterpret_cast<const unsigned char *>(dw), g_dx, g_dy, 0, outs[i], n_in);
        }
        if (cudaMemcpy(ys[i], g_dy, yb, cudaMemcpyDeviceToHost) != cudaSuccess) return;
    }
    (void)cudaGetLastError();
}

extern "C" void feng_cuda_info(char *buf, int buf_sz)
{
    if (!buf || buf_sz <= 0) return;
    if (!init_once()) {
        std::snprintf(buf, (size_t)buf_sz, "CUDA unavailable");
        return;
    }
    cudaDeviceProp prop{};
    cudaGetDeviceProperties(&prop, 0);
    std::snprintf(buf, (size_t)buf_sz, "CUDA %s sm_%d%d | uploaded %d tensors / %.1f MB",
                  prop.name, prop.major, prop.minor, g_n,
                  (double)g_uploaded / (1024.0 * 1024.0));
}

extern "C" void feng_cuda_shutdown(void)
{
    for (int i = 0; i < g_n; ++i) cudaFree(g_tab[i].dev);
    g_n = 0;
    g_uploaded = 0;
    if (g_dx) { cudaFree(g_dx); g_dx = nullptr; g_dx_cap = 0; }
    if (g_dy) { cudaFree(g_dy); g_dy = nullptr; g_dy_cap = 0; }
}

/* ============================================================================
 * 完整 GPU forward（优化目标：单 token 一次提交，避免每 GEMV 的 PCIe 往返）
 *
 * 数值策略：每个 kernel 逐条复刻 CPU（feng_llm.cpp）的运算顺序——
 *   - RMSNorm / softmax / attention 的归约用「单线程串行」，和 CPU 完全同序；
 *   - GEMV 沿用 q4_gemv_kernel（已与 CPU 位一致）；
 *   - RoPE 的 cos/sin 用 host 的 libm 预算成表上传（避开 GPU 三角函数的精度差）；
 *   - f32→f16 的 KV scale 用 __float2half_rz（CPU 实现是截断尾数）；
 *   - fast_expf 多项式原样复刻。
 * 仅支持 FENG_KV_Q2 的 q2/block8 KV 布局（与板端发布配置一致）。
 * ========================================================================== */

#include <cmath>

namespace {

__device__ __forceinline__ float fast_expf_dev(float x)
{
    if (x > 88.722839f) return INFINITY;
    if (x < -87.336548f) return 0.f;
    const float log2e = 1.4426950408889634f;
    const float ln2 = 0.6931471805599453f;
    const float kf = x * log2e;
    const int ki = (int)(kf + (kf >= 0.f ? 0.5f : -0.5f));
    const float r = x - (float)ki * ln2;
    const float p = 1.0f + r * (1.0f + r * (0.5f + r * (0.16666667f + r *
                    (0.041666668f + r * (0.008333334f + r * 0.0013888889f)))));
    return p * __uint_as_float((uint32_t)(ki + 127) << 23);
}

/* embedding：Q4 行反量化（每元素一个线程，与 CPU 同式） */
__global__ void fwd_embed_kernel(const unsigned char *__restrict__ tok_embd, uint32_t dtype,
                                 const FwdParams *__restrict__ prm, float *__restrict__ x, int h)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= h) return;
    const int token = prm->token;
    if (dtype == FENG_DT_FP16) {
        const unsigned short *emb = reinterpret_cast<const unsigned short *>(tok_embd);
        x[i] = __half2float(__ushort_as_half(emb[(size_t)token * h + i]));
        return;
    }
    const int nb = h / 64;
    const unsigned char *row = tok_embd + (size_t)token * ((size_t)nb * 34u);
    const unsigned char *scales = row;
    const unsigned char *packed = row + (size_t)nb * 2u;
    const int b = i >> 6, k = i & 63;
    const float sc = __half2float(__ushort_as_half(
        *reinterpret_cast<const unsigned short *>(scales + (size_t)b * 2u)));
    const unsigned char byte = packed[(size_t)b * 32u + (unsigned)(k >> 1)];
    const int q = (k & 1) ? (byte >> 4) : (byte & 0x0Fu);
    x[i] = (float)(q - 8) * sc;
}

/* RMSNorm：thread 0 串行求和（与 CPU 同序），然后逐元素 x*inv*w */
__global__ void fwd_rmsnorm_kernel(const float *__restrict__ x,
                                   const unsigned short *__restrict__ w,
                                   float *__restrict__ out, int n, float eps)
{
    __shared__ float s_inv;
    if (threadIdx.x == 0) {
        float ss = 0.f;
        for (int i = 0; i < n; ++i) ss += x[i] * x[i];
        s_inv = 1.0f / sqrtf(ss / (float)n + eps);
    }
    __syncthreads();
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] = x[i] * s_inv * __half2float(__ushort_as_half(w[i]));
}

/* per-head QK-norm + RoPE（一个 block 处理一个 head） */
__global__ void fwd_qk_rope_kernel(float *__restrict__ vec,
                                   const unsigned short *__restrict__ w,
                                   const float *__restrict__ rope_cos,
                                   const float *__restrict__ rope_sin,
                                   const FwdParams *__restrict__ prm,
                                   int hd, float eps)
{
    float *v = vec + (size_t)blockIdx.x * hd;
    const int half = hd / 2;
    const float *cos_t = rope_cos + (size_t)prm->pos * half;
    const float *sin_t = rope_sin + (size_t)prm->pos * half;
    __shared__ float s_inv;
    if (threadIdx.x == 0) {
        float ss = 0.f;
        for (int i = 0; i < hd; ++i) ss += v[i] * v[i];
        s_inv = 1.0f / sqrtf(ss / (float)hd + eps);
    }
    __syncthreads();
    for (int i = threadIdx.x; i < hd; i += blockDim.x) {
        v[i] = v[i] * s_inv * __half2float(__ushort_as_half(w[i]));
    }
    __syncthreads();
    for (int i = threadIdx.x; i < half; i += blockDim.x) {
        const float c = cos_t[i], s = sin_t[i];
        const float x1 = v[i], x2 = v[i + half];
        v[i] = x1 * c - x2 * s;
        v[i + half] = x2 * c + x1 * s;
    }
}

/* KV 写入：q2 block 量化（每 (head,block) 一个线程串行 8 值，与 CPU 同序） */
__global__ void fwd_kv_quant_kernel(const float *__restrict__ k,
                                    const float *__restrict__ v,
                                    unsigned char *__restrict__ kcache,
                                    unsigned char *__restrict__ vcache,
                                    unsigned short *__restrict__ ksc,
                                    unsigned short *__restrict__ vsc,
                                    int layer, int nh, int hd, int ctx,
                                    const FwdParams *__restrict__ prm, int blk_sz)
{
    const int pos = prm->pos;
    const int nb = hd / blk_sz;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nh * nb) return;
    const int hh = idx / nb, blk = idx % nb;
    const int base = hh * hd + blk * blk_sz;
    float ak = 1e-8f, av = 1e-8f;
    for (int d = 0; d < blk_sz; ++d) {
        const float a = fabsf(k[base + d]);
        const float b = fabsf(v[base + d]);
        if (a > ak) ak = a;
        if (b > av) av = b;
    }
    const float sk = ak / 1.5f, sv = av / 1.5f;
    const size_t row = ((size_t)layer * nh + hh) * ctx + pos;
    const int qb = hd / 4, bw = blk_sz / 4;
    /* 必须取 half 的位模式：直接赋 __half 给 unsigned short 会走 float→int 转换，
     * 把 0.02 这类 scale 截断成 0（曾导致 attention 全 0、输出乱码）。 */
    ksc[row * nb + blk] = __half_as_ushort(__float2half_rz(sk));
    vsc[row * nb + blk] = __half_as_ushort(__float2half_rz(sv));
    unsigned char *kd = kcache + row * qb + (size_t)blk * bw;
    unsigned char *vd = vcache + row * qb + (size_t)blk * bw;
    for (int j = 0; j < bw; ++j) {
        unsigned char kb = 0, vb = 0;
        for (int k4 = 0; k4 < 4; ++k4) {
            /* base 已含 blk*blk_sz，这里只加块内偏移（曾经重复叠加导致 block≥1 全错） */
            const int off = j * 4 + k4;
            int qk = (int)rintf(k[base + off] / sk + 1.5f);
            int qv = (int)rintf(v[base + off] / sv + 1.5f);
            qk = qk < 0 ? 0 : (qk > 3 ? 3 : qk);
            qv = qv < 0 ? 0 : (qv > 3 ? 3 : qv);
            kb |= (unsigned char)(qk << (2 * k4));
            vb |= (unsigned char)(qv << (2 * k4));
        }
        kd[j] = kb;
        vd[j] = vb;
    }
}

/* attention scores：每 (head,t) 一个线程串行 hd 维（与 CPU 的累加顺序一致） */
__global__ void fwd_attn_scores_kernel(const float *__restrict__ q,
                                       const unsigned char *__restrict__ kcache,
                                       const unsigned short *__restrict__ ksc,
                                       float *__restrict__ scores,
                                       int layer, int nh, int hd, int ctx,
                                       const FwdParams *__restrict__ prm, int blk_sz, float scale)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int hh = idx / ctx, t = idx % ctx;      /* grid 按最大 ctx 铺满 */
    const int npos = prm->pos + 1;
    if (hh >= nh || t >= npos) return;
    const float *qh = q + (size_t)hh * hd;
    const int qb = hd / 4, nb = hd / blk_sz, bw = blk_sz / 4;
    const size_t row = ((size_t)layer * nh + hh) * ctx + t;
    const unsigned char *kh = kcache + row * qb;
    const unsigned short *ks = ksc + row * nb;
    float s = 0.f;
    for (int blk = 0; blk < nb; ++blk) {
        const float sk = __half2float(__ushort_as_half(ks[blk]));
        const unsigned char *bb = kh + (size_t)blk * bw;
        int d = blk * blk_sz;
        for (int j = 0; j < bw; ++j) {
            const unsigned char byte = bb[j];
            for (int k4 = 0; k4 < 4; ++k4) {
                const float l = (float)((int)((byte >> (2 * k4)) & 3u)) - 1.5f;
                s += qh[d++] * l * sk;
            }
        }
    }
    scores[(size_t)hh * ctx + t] = s * scale;
}

/* softmax：每 head 一个线程（串行 max / exp / sum，与 CPU 同序） */
__global__ void fwd_softmax_kernel(float *__restrict__ scores, int ctx, int nh,
                                   const FwdParams *__restrict__ prm,
                                   float *__restrict__ inv_out)
{
    const int hh = blockIdx.x * blockDim.x + threadIdx.x;
    if (hh >= nh) return;
    const int npos = prm->pos + 1;
    float *s = scores + (size_t)hh * ctx;
    float maxs = -1e30f;
    for (int t = 0; t < npos; ++t) if (s[t] > maxs) maxs = s[t];
    float sum = 0.f;
    for (int t = 0; t < npos; ++t) {
        s[t] = fast_expf_dev(s[t] - maxs);
        sum += s[t];
    }
    inv_out[hh] = 1.0f / sum;
}

/* V 加权：每 (head,d) 一个线程串行 t（复刻 VFOLD 的 stv=st*sv 顺序） */
__global__ void fwd_attn_v_kernel(const float *__restrict__ scores,
                                  const float *__restrict__ inv,
                                  const unsigned char *__restrict__ vcache,
                                  const unsigned short *__restrict__ vsc,
                                  float *__restrict__ attn,
                                  int layer, int nh, int hd, int ctx,
                                  const FwdParams *__restrict__ prm, int blk_sz)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nh * hd) return;
    const int npos = prm->pos + 1;
    const int hh = idx / hd, d = idx % hd;
    const int qb = hd / 4, nb = hd / blk_sz, bw = blk_sz / 4;
    const int blk = d / blk_sz, off = d % blk_sz, j = off / 4, k4 = off % 4;
    const float *sc = scores + (size_t)hh * ctx;
    float acc = 0.f;
    for (int t = 0; t < npos; ++t) {
        const size_t row = ((size_t)layer * nh + hh) * ctx + t;
        const float sv = __half2float(__ushort_as_half(vsc[row * nb + blk]));
        const unsigned char byte = vcache[row * qb + (size_t)blk * bw + j];
        const float l = (float)((int)((byte >> (2 * k4)) & 3u)) - 1.5f;
        acc += (sc[t] * sv) * l;
    }
    attn[idx] = acc * inv[hh];
}

__global__ void fwd_silu_mul_kernel(float *__restrict__ out, const float *__restrict__ g,
                                    const float *__restrict__ u, int n)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float x = g[i];
    out[i] = (x / (1.0f + fast_expf_dev(-x))) * u[i];
}

__global__ void fwd_add_kernel(float *__restrict__ x, const float *__restrict__ p, int n)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] += p[i];
}

/* ---- 每 (kv 指针) 一份 device 状态 ---- */
struct FwdState {
    const void *kv_key;
    int ctx;
    int h, nh, hd, ffn, vocab;
    float *x, *xn, *q, *k, *v, *attn, *proj, *gate, *up, *ffn_buf, *logits, *scores, *inv;
    float *rope_cos, *rope_sin;             /* [ctx][hd/2] */
    unsigned char *k_cache, *v_cache;
    unsigned short *k_scale, *v_scale;
    FwdParams *d_params;                    /* token/pos 打包成 device 标量，便于整图捕获 */
    float *host_logits;                     /* 图内 D2H 的目标（=调用方的 ws->logits） */
    cudaStream_t stream;                    /* 非默认流（capture 不能用默认流） */
    cudaGraphExec_t graph_exec;
    int graph_ready;
};

constexpr int kMaxFwdStates = 4;
FwdState g_fwd[kMaxFwdStates];
int g_fwd_n = 0;
bool g_fwd_failed = false;      /* 一次性失败：调用方回退 CPU 路径 */

bool fwd_alloc_f(float **p, size_t n)
{
    return cudaMalloc(reinterpret_cast<void **>(p), n * sizeof(float)) == cudaSuccess;
}

FwdState *fwd_state(feng_model_t *m, feng_kv_t *kv)
{
    for (int i = 0; i < g_fwd_n; ++i) {
        if (g_fwd[i].kv_key == kv) return &g_fwd[i];
    }
    if (g_fwd_n >= kMaxFwdStates) return nullptr;
    FwdState *s = &g_fwd[g_fwd_n];
    std::memset(s, 0, sizeof(*s));
    s->kv_key = kv;
    s->ctx = kv->ctx;
    s->h = (int)m->hdr.hidden;
    s->nh = (int)m->hdr.n_heads;
    s->hd = (int)m->hdr.head_dim;
    s->ffn = (int)m->hdr.ffn;
    s->vocab = (int)m->hdr.vocab;
    const int h = s->h, nh = s->nh, hd = s->hd, f = s->ffn, ctx = s->ctx;
    bool ok = fwd_alloc_f(&s->x, h) && fwd_alloc_f(&s->xn, h) && fwd_alloc_f(&s->q, h) &&
              fwd_alloc_f(&s->k, h) && fwd_alloc_f(&s->v, h) && fwd_alloc_f(&s->attn, h) &&
              fwd_alloc_f(&s->proj, h) && fwd_alloc_f(&s->gate, f) && fwd_alloc_f(&s->up, f) &&
              fwd_alloc_f(&s->ffn_buf, f) && fwd_alloc_f(&s->logits, s->vocab) &&
              fwd_alloc_f(&s->scores, (size_t)nh * ctx) && fwd_alloc_f(&s->inv, nh);
    const int qb = hd / 4, nb = hd / FENG_KV_Q2_BLOCK, half = hd / 2;
    ok = ok && cudaMalloc(reinterpret_cast<void **>(&s->k_cache),
                          (size_t)m->hdr.n_layers * nh * ctx * qb) == cudaSuccess &&
         cudaMalloc(reinterpret_cast<void **>(&s->v_cache),
                    (size_t)m->hdr.n_layers * nh * ctx * qb) == cudaSuccess &&
         cudaMalloc(reinterpret_cast<void **>(&s->k_scale),
                    (size_t)m->hdr.n_layers * nh * ctx * nb * 2) == cudaSuccess &&
         cudaMalloc(reinterpret_cast<void **>(&s->v_scale),
                    (size_t)m->hdr.n_layers * nh * ctx * nb * 2) == cudaSuccess &&
         fwd_alloc_f(&s->rope_cos, (size_t)ctx * half) &&
         fwd_alloc_f(&s->rope_sin, (size_t)ctx * half);
    ok = ok && cudaStreamCreate(&s->stream) == cudaSuccess &&
         cudaMalloc(reinterpret_cast<void **>(&s->d_params), sizeof(FwdParams)) == cudaSuccess;
    if (!ok) return nullptr;
    /* 预上传全部张量：之后 device_tensor 只命中缓存，forward 路径不再分配/拷贝权重 */
    for (uint32_t i = 0; i < m->hdr.n_tensors; ++i) {
        const feng_tensor_t *t = &m->dir[i];
        if (!device_tensor(m->blob + t->offset, (size_t)t->nbytes)) return nullptr;
    }
    /* RoPE 表：host libm 与 CPU 引擎同源（同一工具链 CRT），逐位一致 */
    {
        std::vector<float> hc((size_t)ctx * half), hs((size_t)ctx * half);
        std::vector<float> inv((size_t)half);
        for (int i = 0; i < half; ++i) {
            inv[i] = powf(m->hdr.rope_theta, -2.0f * (float)i / (float)hd);
        }
        for (int p = 0; p < ctx; ++p) {
            for (int i = 0; i < half; ++i) {
                const float ang = (float)p * inv[i];
                hc[(size_t)p * half + i] = cosf(ang);
                hs[(size_t)p * half + i] = sinf(ang);
            }
        }
        if (cudaMemcpy(s->rope_cos, hc.data(), hc.size() * sizeof(float),
                       cudaMemcpyHostToDevice) != cudaSuccess ||
            cudaMemcpy(s->rope_sin, hs.data(), hs.size() * sizeof(float),
                       cudaMemcpyHostToDevice) != cudaSuccess) {
            return nullptr;
        }
    }
    ++g_fwd_n;
    return s;
}

/* Q4 GEMV 的行范围版（device 指针，一次调用只算 [0, n_out)） */
void fwd_gemv_device(const void *host_tensor, size_t bytes, uint32_t dtype,
                     const float *dx, float *dy, int n_out, int n_in, cudaStream_t stream)
{
    void *dw = device_tensor(host_tensor, bytes);
    if (!dw) return;
    /* 每行一个 warp：并行度 ×32，隐藏显存延迟（完整 forward 专用；验收口径是
     * 32 题矩阵输出与 CPU 版逐字一致，而非 logits 位精确）。 */
    const int threads = 128;                       /* 4 warps/block */
    const int blocks = (n_out + 3) / 4;             /* 每 block 4 行（4 个 warp），不是 128 行 */
    if (dtype == FENG_DT_FP16) {
        fp16_gemv_warp_kernel<<<blocks, threads, 0, stream>>>(
            reinterpret_cast<const unsigned short *>(dw), dx, dy, 0, n_out, n_in);
    } else {
        q4_gemv_warp_kernel<<<blocks, threads, 0, stream>>>(
            reinterpret_cast<const unsigned char *>(dw), dx, dy, 0, n_out, n_in);
    }
}

size_t tensor_bytes(uint32_t dtype, int n_out, int n_in)
{
    if (dtype == FENG_DT_FP16) return (size_t)n_out * (size_t)n_in * 2u;
    const int nb = n_in / kQK;
    return (size_t)n_out * ((size_t)nb * 34u);
}

}  // namespace

extern "C" int feng_cuda_forward_ready(void)
{
    if (g_fwd_failed) return 0;
    if (!init_once()) return 0;
    const char *env = std::getenv("FENG_CUDA");
    if (env && std::strcmp(env, "gemv") == 0) return 0;   /* 回退：只加速 GEMV */
    return 1;
}

extern "C" int feng_cuda_forward_supported(const feng_model_t *m, const feng_kv_t *kv)
{
    (void)m;
    (void)kv;
#if !FENG_KV_Q2
    return 0;      /* 完整 GPU forward 只实现了 q2 KV 布局 */
#else
    if (FENG_KV_Q2_BLOCK != 8) return 0;
    return 1;
#endif
}

extern "C" float *feng_cuda_forward(feng_model_t *m, feng_kv_t *kv, feng_workspace_t *ws,
                                    int token, int pos, int want_logits)
{
    FwdState *s = fwd_state(m, kv);
    if (!s) {
        std::printf("[cuda fwd] state alloc failed: %s\n", cudaGetErrorString(cudaGetLastError()));
        std::fflush(stdout);
        g_fwd_failed = true;
        return nullptr;
    }
    cudaStream_t stream = s->stream;
    const int h = s->h, nh = s->nh, hd = s->hd, f = s->ffn;
    const int half = hd / 2;
    const float eps = m->hdr.rms_eps;
    const int blk_sz = FENG_KV_Q2_BLOCK;
    const float attn_scale = 1.0f / sqrtf((float)hd);
    const int ctx = s->ctx;

    /* 输入上传（图外，不等待）：token/pos 打包 8 字节；同流顺序保证图在其后执行。
     * D2H 目标（调用方 logits 缓冲）在图捕获时固定。 */
    (void)want_logits;      /* 图内总是回拷 logits（64KB / ~20µs，简化图结构） */
    s->host_logits = ws->logits;
    const FwdParams hp{token, pos};
    if (cudaMemcpyAsync(s->d_params, &hp, sizeof(hp), cudaMemcpyHostToDevice, stream) != cudaSuccess) {
        g_fwd_failed = true;
        return nullptr;
    }
    if (!s->graph_ready) {
        if (cudaStreamBeginCapture(stream, cudaStreamCaptureModeRelaxed) != cudaSuccess) {
            std::printf("[cuda fwd] BeginCapture failed: %s\n", cudaGetErrorString(cudaGetLastError()));
            std::fflush(stdout);
            g_fwd_failed = true;
            return nullptr;
        }

    fwd_embed_kernel<<<(h + 255) / 256, 256, 0, stream>>>(
        reinterpret_cast<const unsigned char *>(device_tensor(m->tok_embd,
            tensor_bytes(m->tok_embd_dtype, s->vocab, h))),
        m->tok_embd_dtype, s->d_params, s->x, h);
    for (int l = 0; l < (int)m->hdr.n_layers; ++l) {
        const feng_layer_t *L = &m->layers[l];
        /* attn_norm（fp16 权重 448） */
        fwd_rmsnorm_kernel<<<1, 1024, 0, stream>>>(
            s->x, reinterpret_cast<const unsigned short *>(
                device_tensor(L->attn_norm, (size_t)h * 2)),
            s->xn, h, eps);
        fwd_gemv_device(L->wq, tensor_bytes(FENG_DT_Q4, h, h), FENG_DT_Q4, s->xn, s->q, h, h, stream);
        fwd_gemv_device(L->wk, tensor_bytes(FENG_DT_Q4, h, h), FENG_DT_Q4, s->xn, s->k, h, h, stream);
        fwd_gemv_device(L->wv, tensor_bytes(FENG_DT_Q4, h, h), FENG_DT_Q4, s->xn, s->v, h, h, stream);
        /* QK-norm + RoPE（每 head 一个 block；q 与 k 各一次） */
        fwd_qk_rope_kernel<<<nh, 64, 0, stream>>>(
            s->q, reinterpret_cast<const unsigned short *>(device_tensor(L->q_norm, (size_t)hd * 2)),
            s->rope_cos, s->rope_sin, s->d_params, hd, eps);
        fwd_qk_rope_kernel<<<nh, 64, 0, stream>>>(
            s->k, reinterpret_cast<const unsigned short *>(device_tensor(L->k_norm, (size_t)hd * 2)),
            s->rope_cos, s->rope_sin, s->d_params, hd, eps);
        /* KV 写入 */
        fwd_kv_quant_kernel<<<(nh * (hd / blk_sz) + 127) / 128, 128, 0, stream>>>(
            s->k, s->v, s->k_cache, s->v_cache, s->k_scale, s->v_scale,
            l, nh, hd, s->ctx, s->d_params, blk_sz);
        /* attention */
        fwd_attn_scores_kernel<<<(nh * ctx + 255) / 256, 256, 0, stream>>>(
            s->q, s->k_cache, s->k_scale, s->scores, l, nh, hd, s->ctx, s->d_params, blk_sz, attn_scale);
        fwd_softmax_kernel<<<1, 32, 0, stream>>>(s->scores, s->ctx, nh, s->d_params, s->inv);
        fwd_attn_v_kernel<<<(nh * hd + 255) / 256, 256, 0, stream>>>(
            s->scores, s->inv, s->v_cache, s->v_scale, s->attn, l, nh, hd, s->ctx, s->d_params, blk_sz);
        fwd_gemv_device(L->wo, tensor_bytes(FENG_DT_Q4, h, h), FENG_DT_Q4, s->attn, s->proj, h, h, stream);
        fwd_add_kernel<<<(h + 255) / 256, 256, 0, stream>>>(s->x, s->proj, h);
        /* FFN */
        fwd_rmsnorm_kernel<<<1, 1024, 0, stream>>>(
            s->x, reinterpret_cast<const unsigned short *>(
                device_tensor(L->ffn_norm, (size_t)h * 2)),
            s->xn, h, eps);
        fwd_gemv_device(L->gate, tensor_bytes(FENG_DT_Q4, f, h), FENG_DT_Q4, s->xn, s->gate, f, h, stream);
        fwd_gemv_device(L->up, tensor_bytes(FENG_DT_Q4, f, h), FENG_DT_Q4, s->xn, s->up, f, h, stream);
        fwd_silu_mul_kernel<<<(f + 255) / 256, 256, 0, stream>>>(s->ffn_buf, s->gate, s->up, f);
        fwd_gemv_device(L->down, tensor_bytes(FENG_DT_Q4, h, f), FENG_DT_Q4, s->ffn_buf, s->proj, h, f, stream);
        fwd_add_kernel<<<(h + 255) / 256, 256, 0, stream>>>(s->x, s->proj, h);
    }
    /* 输出投影：图内总是执行（GPU 上仅 ~16µs；want_logits=0 只是不回拷） */
    fwd_rmsnorm_kernel<<<1, 1024, 0, stream>>>(
        s->x, reinterpret_cast<const unsigned short *>(
            device_tensor(m->out_norm, (size_t)h * 2)),
        s->xn, h, eps);
    fwd_gemv_device(m->tok_embd, tensor_bytes(m->tok_embd_dtype, s->vocab, h),
                    m->tok_embd_dtype, s->xn, s->logits, s->vocab, h, stream);
        /* 图内回拷 logits：把 D2H 也收进图，每 token 只剩 1 次同步 */
        (void)cudaMemcpyAsync(s->host_logits, s->logits, (size_t)s->vocab * sizeof(float),
                              cudaMemcpyDeviceToHost, stream);
        cudaGraph_t graph = nullptr;
        if (cudaStreamEndCapture(stream, &graph) != cudaSuccess ||
            cudaGraphInstantiate(&s->graph_exec, graph, 0) != cudaSuccess) {
            std::printf("[cuda fwd] EndCapture/Instantiate failed: %s\n",
                        cudaGetErrorString(cudaGetLastError()));
            std::fflush(stdout);
            if (graph) cudaGraphDestroy(graph);
            g_fwd_failed = true;
            return nullptr;
        }
        cudaGraphDestroy(graph);
        s->graph_ready = 1;
    }
    /* FENG_CUDA_PROF=1 时打印每 32 个 token 的图执行耗时（调试用） */
    static int s_prof = -1;
    static cudaEvent_t s_ev0, s_ev1;
    static int s_prof_cnt = 0;
    if (s_prof < 0) {
        const char *e = std::getenv("FENG_CUDA_PROF");
        s_prof = (e && std::strcmp(e, "0") != 0) ? 1 : 0;
        if (s_prof) {
            cudaEventCreate(&s_ev0);
            cudaEventCreate(&s_ev1);
        }
    }
    if (s_prof) cudaEventRecord(s_ev0, stream);
    if (cudaGraphLaunch(s->graph_exec, stream) != cudaSuccess) {
        g_fwd_failed = true;
        return nullptr;
    }
    /* 图内已含 D2H。同步用忙等：Windows/WDDM 下 cudaStreamSynchronize 的线程
     * 唤醒延迟可达 ~1ms，而 GPU 实际只需 ~20µs（实测 1.3ms/token 全在这里）。 */
    cudaError_t q = cudaErrorNotReady;
    while ((q = cudaStreamQuery(stream)) == cudaErrorNotReady) {
        /* spin */
    }
    if (q != cudaSuccess) {
        g_fwd_failed = true;
        return nullptr;
    }
    if (s_prof) {
        cudaEventRecord(s_ev1, stream);
        cudaEventSynchronize(s_ev1);
        float ms = 0.f;
        cudaEventElapsedTime(&ms, s_ev0, s_ev1);
        if ((s_prof_cnt++ % 32) == 0) {
            std::printf("[cuda fwd] graph GPU time: %.3f ms\n", (double)ms);
            std::fflush(stdout);
        }
    }
    /* FENG_CUDA_DUMP=1：图外回读关键中间值（不受 graph printf 限制） */
    {
        static int s_dump = -1;
        if (s_dump < 0) {
            const char *e = std::getenv("FENG_CUDA_DUMP");
            s_dump = (e && std::strcmp(e, "0") != 0) ? 1 : 0;
        }
        if (s_dump) {
            float fdbg[2] = {0.f, 0.f};
            unsigned short udbg[2] = {0, 0};
            unsigned char bdbg[4] = {0, 0, 0, 0};
            cudaMemcpy(fdbg, s->scores, sizeof(fdbg), cudaMemcpyDeviceToHost);
            cudaMemcpy(udbg, s->k_scale, sizeof(udbg), cudaMemcpyDeviceToHost);
            cudaMemcpy(bdbg, s->k_cache, sizeof(bdbg), cudaMemcpyDeviceToHost);
            unsigned char vdbg[4] = {0, 0, 0, 0};
            unsigned short vsdbg[2] = {0, 0};
            cudaMemcpy(vdbg, s->v_cache, sizeof(vdbg), cudaMemcpyDeviceToHost);
            cudaMemcpy(vsdbg, s->v_scale, sizeof(vsdbg), cudaMemcpyDeviceToHost);
            std::printf("[fwd-dump] kbytes=%02x %02x %02x %02x vbytes=%02x %02x %02x %02x kscale=%04x %04x vscale=%04x %04x\n",
                        (unsigned)bdbg[0], (unsigned)bdbg[1], (unsigned)bdbg[2], (unsigned)bdbg[3],
                        (unsigned)vdbg[0], (unsigned)vdbg[1], (unsigned)vdbg[2], (unsigned)vdbg[3],
                        (unsigned)udbg[0], (unsigned)udbg[1],
                        (unsigned)vsdbg[0], (unsigned)vsdbg[1]);
            cudaMemcpy(fdbg + 1, s->attn, sizeof(float), cudaMemcpyDeviceToHost);
            float adbg[2] = {0.f, 0.f};
            float pdbg[2] = {0.f, 0.f};
            cudaMemcpy(adbg, s->attn + 4, sizeof(adbg), cudaMemcpyDeviceToHost);
            cudaMemcpy(pdbg, s->proj, sizeof(pdbg), cudaMemcpyDeviceToHost);
            std::printf("[fwd-dump] attn0=%.6f attn4=%.6f proj0=%.6f proj4=%.6f k_scale0=0x%04x 0x%04x kbyte=%02x %02x\n",
                        (double)fdbg[1], (double)adbg[0], (double)pdbg[0], (double)pdbg[1],
                        (unsigned)udbg[0], (unsigned)udbg[1],
                        (unsigned)bdbg[0], (unsigned)bdbg[1]);
            float abig[16];
            if (cudaMemcpy(abig, s->attn, sizeof(abig), cudaMemcpyDeviceToHost) == cudaSuccess) {
                std::printf("[fwd-dump] attn16:");
                for (int i = 0; i < 16; ++i) std::printf(" %.6f", (double)abig[i]);
                std::printf("\n");
            }
            std::fflush(stdout);
        }
    }
    kv->len = pos + 1;
    return ws->logits;      /* 成功：总是返回 logits 指针（与 CPU 版语义一致） */
}
