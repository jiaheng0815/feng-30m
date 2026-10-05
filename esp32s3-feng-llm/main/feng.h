/* feng-30m on ESP32-S3: model container + inference core (portable C++23).
 *
 * C++23 约定（见 AGENTS.md）：
 *  - 零堆分配：本头文件与核心模块不使用 new/delete/malloc/free；
 *    唯一的例外是 tokenizer 的启动期 PSRAM 加载（一次性、固定大小）；
 *  - 编译期 -fno-exceptions -fno-rtti，函数不抛异常；
 *  - 不使用需要运行时构造的全局/静态对象（.init_array 为空）；
 *  - 只使用无堆的标准库组件（<array>/<span>/<string_view>/<bit>/<type_traits> 等）。 */
#ifndef FENG_H
#define FENG_H

#include <cstddef>
#include <cstdint>
#include <span>

/* KV cache mode: fp32 / int8 / q2, selected at compile time.
 *   FENG_KV_INT8=1 : int8 values + fp16 scale per (layer, position, head) -> 4x less PSRAM than fp32
 *   FENG_KV_Q2=1   : 2-bit values (4 packed per byte) + one fp16 scale per FENG_KV_Q2_BLOCK values.
 *                    block16 = 336 B/token/层（精度差）；block8 = 448 B/token/层（约省 int8 一半，
 *                    实测长文召回与续写都可用，见 scripts/kv_quant_experiment.py）。
 * 两种量化模式都用 kv->k_scale / v_scale 存 fp16 scale。 */
#ifndef FENG_KV_INT8
#define FENG_KV_INT8 0
#endif
#ifndef FENG_KV_Q2
#define FENG_KV_Q2 0
#endif
#ifndef FENG_KV_Q2_BLOCK
#define FENG_KV_Q2_BLOCK 16      /* 每多少个值共享一个 fp16 scale（须整除 head_dim、是 4 的倍数） */
#endif
/* q2 注意力内层用字节 LUT 取代移位/掩码/整数转浮点（数值与原式完全一致，结果逐位不变）。
 * 默认开；-DFENG_Q2_LUT=0 可切回原实现做 A/B。 */
#ifndef FENG_Q2_LUT
#define FENG_Q2_LUT 1
#endif
/* q2 注意力一次算 2 个上下文 token（两条独立累加链填 FPU 流水线、共享加载）。
 * 数值逐位不变；-DFENG_Q2_PAIR=0 可切回单 token 版做 A/B。 */
#ifndef FENG_Q2_PAIR
#define FENG_Q2_PAIR 1
#endif
/* 性能剖析（调试用，默认关）：累计 q2 注意力的 K / softmax / V 三段的 CPU 周期，
 * 由 FENG_BENCH_CTX 基准打印，用来定位长上下文的时间去向。 */
#ifndef FENG_ATTN_PROF
#define FENG_ATTN_PROF 0
#endif

/* 实验开关：激活 int8 量化的 GEMV（给 PIE 铺路；默认 0 = 原来的 LUT/fp32 路径） */
#ifndef FENG_GEMV_A8
#define FENG_GEMV_A8 0
#endif
/* Q4 GEMV 累加形式：1 = 纯 madd 链（2 个 FP 运算/2 权重，实测更快：ctx-256 -5%、
 * lm head -7.5%，见 CHANGELOG 附录）；0 = 两两求和再累加（3 个 FP 运算/2 权重）。
 * 两者 logits 差 ≤ 3.8e-6（Q4 量化误差是 2.94），32 题矩阵输出逐字相同；
 * 想要与旧版逐位一致时用 -DFENG_GEMV_MADD=0。 */
#ifndef FENG_GEMV_MADD
#define FENG_GEMV_MADD 1
#endif
/* 快速 exp（多项式 + 2^k 缩放，~20 条指令）替换 softmax / silu 里的 newlib expf
 * （实测 ~257 周期/次）。板端 softmax 从 169ms 降到 48ms（2048 ctx），
 * logits 差 ≤3.8e-6、32 题矩阵输出逐字相同；-DFENG_FAST_EXP=0 可切回 newlib。 */
#ifndef FENG_FAST_EXP
#define FENG_FAST_EXP 1
#endif
/* q2 注意力的 V 段：1 = 把 (scores[t] * vscale[blk]) 先乘好，内层每值只剩 1 个 madd
 * （原来是 (st*val)*sv 的 3 个 FP 运算）。板端 V 段 912→832 ms（2048 ctx）、
 * 32 题输出逐字相同；三重数值优化累计 logits 差 4.8e-6（见 CHANGELOG 附录）。 */
#ifndef FENG_Q2_VFOLD
#define FENG_Q2_VFOLD 1
#endif


#define FENG_STR2(x) #x
#define FENG_STR(x) FENG_STR2(x)

#if defined(ESP_PLATFORM)
#include "esp_attr.h"
#define FENG_HOT IRAM_ATTR
#else
#define FENG_HOT
#endif

inline constexpr uint32_t FENG_MAGIC = 0x46574E31u;   /* "FWN1" */
inline constexpr uint32_t FENG_MAX_LAYERS = 24;
/* 11 tensors/layer * layers + tok_embd + output_norm (+margin) */
inline constexpr uint32_t FENG_MAX_TENSORS = 168;

typedef enum { FENG_DT_FP16 = 0, FENG_DT_Q4 = 1 } feng_dtype_t;

typedef struct {
    uint32_t magic, version, n_tensors;
    uint32_t n_layers, hidden, n_heads, head_dim, ffn, vocab;
    float rms_eps, rope_theta;
} feng_header_t;

typedef struct {
    char name[32];
    uint32_t dtype;
    uint32_t shape[4];
    uint64_t offset, nbytes;
} feng_tensor_t;

typedef struct {
    const void *attn_norm, *ffn_norm;
    const void *wq, *wk, *wv, *wo;
    const void *q_norm, *k_norm;
    const void *gate, *up, *down;
} feng_layer_t;

typedef struct {
    feng_header_t hdr;
    feng_tensor_t dir_storage[FENG_MAX_TENSORS];   /* parsed, packed-format records */
    const feng_tensor_t *dir;
    const uint8_t *blob;            /* base pointer of the model file */
    const void *tok_embd;
    uint32_t tok_embd_dtype;
    const void *out_norm;
    feng_layer_t layers[FENG_MAX_LAYERS];
} feng_model_t;

/* parse a memory image of model.bin; returns 0 on success */
[[nodiscard]] int feng_model_init(feng_model_t *m, std::span<const std::byte> data) noexcept;

/* Q4 / fp16 GEMV: y[n_out] = W[n_out x n_in] * x[n_in] (+ optional bias=NULL) */
void feng_gemv(const void *tensor, uint32_t dtype, const float *x, float *y,
               int n_out, int n_in, float *scratch);
float feng_f16_to_f32(uint16_t h);
uint16_t feng_f32_to_f16(float f);

/* same, but only output rows [r0, r1) -- used to split work across cores */
void feng_gemv_range(const void *tensor, uint32_t dtype, const float *x, float *y,
                     int r0, int r1, int n_in);

/* two-core version of feng_gemv (falls back to single core if init failed) */
void feng_gemv_par(const void *tensor, uint32_t dtype, const float *x, float *y,
                   int n_out, int n_in);
/* 共享同一输入 x 的 2~3 个 GEMV 合并提交（q/k/v 或 gate/up 常成对出现）：
 *   y_i = W_i * x；w_i == NULL 表示跳过。CPU/板端 = 顺序调用 feng_gemv_par；
 * PC CUDA 后端 = 一次 H2D + 多次 kernel，减少每 token 的固定调度开销。 */
void feng_gemv3(const void *w0, const void *w1, const void *w2, uint32_t dtype,
                const float *x, float *y0, float *y1, float *y2,
                int n_out0, int n_out1, int n_out2, int n_in);
/* FENG_GEMV_A8=1 时：每个 GEMV 调用前准备一次 int8 激活（两核并发只读） */
void feng_gemv_a8_prepare(const float *x, int n_in);

/* start the worker on core 1; call once before using feng_gemv_par */
void feng_smp_init(void);

/* transformer state (allocated by caller in PSRAM) */
typedef struct {
    void *k_cache;       /* fp32 / int8 / q2: [n_layers][ctx][n_heads*head_dim]（q2 时为 1/4 字节数） */
    void *v_cache;
    uint16_t *k_scale;   /* int8 / q2 modes: [n_layers][ctx][n_heads] fp16 */
    uint16_t *v_scale;
    int ctx, len;        /* filled length */
} feng_kv_t;

typedef struct {
    float *x, *xn, *q, *k, *v, *attn, *proj, *gate, *up, *ffn, *logits;
    float *scratch;
    int max_ctx;
} feng_workspace_t;

size_t feng_kv_bytes(const feng_model_t *m, int ctx);
/* fp16 scale slots for ONE cache (K or V) in the current KV mode; 0 for fp32 */
size_t feng_kv_scale_slots(const feng_model_t *m, int ctx);
size_t feng_ws_bytes(const feng_model_t *m, int ctx);

/* run one token through the model, returns logits pointer (vocab floats) */
float *feng_forward(feng_model_t *m, feng_kv_t *kv, feng_workspace_t *ws, int token, int pos);
/* want_logits=0：跳过 out_norm + tied lm head（7.34M 权重、板端 ~125 ms/次）。
 * 只用于 prefill 的中间 token——KV 与 hidden 状态和 want_logits=1 完全相同，
 * 只是不算那份马上会被丢掉的 logits。 */
float *feng_forward_ex(feng_model_t *m, feng_kv_t *kv, feng_workspace_t *ws, int token, int pos,
                       int want_logits);

/* greedy-sample helpers */
int feng_argmax(const float *logits, int n);
/* 共享贪心采样：对 hist 里的 token 施加重复惩罚；no_repeat_n>=2 时禁止补全已出现过的
 * n-gram（压复读循环）。板端与 PC 引擎同口径。 */
[[nodiscard]] int feng_sample_greedy(std::span<float> logits, std::span<const int> hist,
                                     float penalty, int no_repeat_n) noexcept;

#endif /* FENG_H */
