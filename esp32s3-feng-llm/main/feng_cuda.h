/* PC-only CUDA 后端（仅当 FENG_USE_CUDA=1 编译时编入/链接；ESP32 固件不包含本文件）。
 *
 * 只加速 GEMV（Q4 block64 / fp16），其余算子仍在 CPU：
 *   - 权重张量在首次使用时整块上传显存并常驻（host 指针 -> device 指针查找表）；
 *   - 每次 GEMV 只拷贝激活 x 与结果 y（KB 级）；
 *   - kernel 与 CPU 的运算顺序逐条对齐（4 累加器 + 块内 4 路 + (a0+a1)+(a2+a3)），
 *     并以 -fmad=false 编译，保证数值与 CPU 版一致。
 * 接口是 C 风格 POD，可在 C++23 主机代码里直接调用。 */
#ifndef FENG_CUDA_H
#define FENG_CUDA_H

#include <stdint.h>

#include "feng.h"       /* feng_model_t / feng_kv_t / feng_workspace_t */

#ifdef __cplusplus
extern "C" {
#endif

/* 1 = CUDA 可用（首次调用探测并缓存；0 = 无可用设备） */
int feng_cuda_available(void);

/* 与 feng_gemv_range 同签名：计算 y[r0..r1) = W[r0..r1) × x */
void feng_cuda_gemv_range(const void *tensor, uint32_t dtype, const float *x, float *y,
                          int r0, int r1, int n_in);

/* 共享输入 x 的 2~3 个 GEMV 合并提交（与 feng_gemv3 对应） */
void feng_cuda_gemv3(const void *w0, const void *w1, const void *w2, uint32_t dtype,
                     const float *x, float *y0, float *y1, float *y2,
                     int n_out0, int n_out1, int n_out2, int n_in);

/* 设备名 / 已上传张量数与显存占用，用于启动打印 */
void feng_cuda_info(char *buf, int buf_sz);

/* 完整 GPU forward（仅 q2 KV 布局）：1 = 可用。 */
int feng_cuda_forward_ready(void);
int feng_cuda_forward_supported(const feng_model_t *m, const feng_kv_t *kv);
/* 在 GPU 上跑完整个 token 前向（权重/KV/激活常驻显存），结果写回 ws->logits。
 * 成功返回 ws->logits；不可用/失败返回 NULL（调用方回退 CPU 路径）。 */
float *feng_cuda_forward(feng_model_t *m, feng_kv_t *kv, feng_workspace_t *ws,
                         int token, int pos, int want_logits);

/* 释放全部设备内存 */
void feng_cuda_shutdown(void);

#ifdef __cplusplus
}
#endif

#endif /* FENG_CUDA_H */
