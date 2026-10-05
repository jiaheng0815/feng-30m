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

namespace {

constexpr int kQK = 64;            /* 与 feng_quant.cpp 的 QK 一致 */
constexpr int kMaxTensors = 512;

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
