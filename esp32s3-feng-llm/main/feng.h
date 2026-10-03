/* feng-30m on ESP32-S3: model container + inference core (portable C11) */
#ifndef FENG_H
#define FENG_H

#include <stdint.h>
#include <stddef.h>

/* KV cache mode: fp32 / int8 / q2, selected at compile time.
 *   FENG_KV_INT8=1 : int8 values + fp16 scale per (layer, position, head) -> 4x less PSRAM than fp32
 *   FENG_KV_Q2=1   : 2-bit values (4 packed per byte) + the same fp16 scales -> 16x less than fp32,
 *                    i.e. 4x the context of int8 in the same PSRAM budget.
 * 两种量化模式都用 kv->k_scale / v_scale 存 fp16 scale。 */
#ifndef FENG_KV_INT8
#define FENG_KV_INT8 0
#endif
#ifndef FENG_KV_Q2
#define FENG_KV_Q2 0
#endif

#if defined(ESP_PLATFORM)
#include "esp_attr.h"
#define FENG_HOT IRAM_ATTR
#else
#define FENG_HOT
#endif

#define FENG_MAGIC 0x46574E31u
#define FENG_MAX_LAYERS 24
#define FENG_MAX_TENSORS 168   /* 11 tensors/layer * layers + tok_embd + output_norm (+margin) */

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
int feng_model_init(feng_model_t *m, const void *data, size_t size);

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
size_t feng_ws_bytes(const feng_model_t *m, int ctx);

/* run one token through the model, returns logits pointer (vocab floats) */
float *feng_forward(feng_model_t *m, feng_kv_t *kv, feng_workspace_t *ws, int token, int pos);

/* greedy-sample helpers */
int feng_argmax(const float *logits, int n);

#endif /* FENG_H */
