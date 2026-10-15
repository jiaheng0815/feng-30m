/* phone/engine 派生副本：自 esp32s3-feng-llm/main/feng_model.c 拷贝（2026-10-15，v4.0 C23 迁移后）。
 * 手机端改动见 phone/README.md；板端原始文件未被修改，两边需手动同步。
 * 原版校验值见 phone/engine/SOURCE.sha256。 */
/* Parse the model.bin image and resolve tensor pointers.
 *
 * 解析保持逐字段 memcpy（对齐安全，与导出端字节精确一致）。
 * v4.0 起是纯 C23：原来的 std::span 表达内存镜像、std::string_view 查名字、
 * std::array 表示定长表全部换成了裸指针 + 长度 + C 字符串——
 * 这套数据结构本来就是 POD，C++ 只污染了函数签名的语法糖。
 * 仍然零分配、零全局构造。 */
#include "feng.h"

#include <stdio.h>
#include <string.h>

#define K_HEADER_BYTES   44u    /* 9 x u32 + 2 x f32 */
#define K_DIRENT_BYTES   68u    /* name[32] + dtype + shape[4] + offset + nbytes */

static const feng_tensor_t *find_tensor(const feng_model_t *m, const char *name)
{
    for (uint32_t i = 0; i < m->hdr.n_tensors; i++) {
        /* 目录项在第 70 行左右被强制 NUL 结尾，所以 strcmp 与原来的
         * string_view 相等比较语义一致 */
        if (strcmp(m->dir[i].name, name) == 0) return &m->dir[i];
    }
    return NULL;
}

static const void *ptr_of(const feng_model_t *m, const feng_tensor_t *t)
{
    return m->blob + t->offset;
}

/* blk.<layer>.<tensor>，顺序与 feng_layer_t 字段一致 */
static const char *const kLayerTensorNames[11] = {
    "attn_norm.weight", "ffn_norm.weight", "attn_q.weight", "attn_k.weight",
    "attn_v.weight",    "attn_output.weight", "attn_q_norm.weight", "attn_k_norm.weight",
    "ffn_gate.weight",  "ffn_up.weight",    "ffn_down.weight",
};

[[nodiscard]] int feng_model_init(feng_model_t *m, const void *data, size_t size)
{
    if (m == NULL || data == NULL || size < K_HEADER_BYTES) return -1;
    const uint8_t *p = (const uint8_t *)data;
    /* header: 9 x u32 + 2 x f32 = 44 bytes (packed, parsed field by field) */
    uint32_t v[9] = {0};
    float fv[2] = {0};
    memcpy(v, p, sizeof(v));
    memcpy(fv, p + 36, sizeof(fv));
    m->hdr.magic = v[0];
    m->hdr.version = v[1];
    m->hdr.n_tensors = v[2];
    m->hdr.n_layers = v[3];
    m->hdr.hidden = v[4];
    m->hdr.n_heads = v[5];
    m->hdr.head_dim = v[6];
    m->hdr.ffn = v[7];
    m->hdr.vocab = v[8];
    m->hdr.rms_eps = fv[0];
    m->hdr.rope_theta = fv[1];
    if (m->hdr.magic != (uint32_t)FENG_MAGIC) return -2;
    if (m->hdr.n_layers > FENG_MAX_LAYERS) return -3;
    if (m->hdr.n_tensors > FENG_MAX_TENSORS) return -8;
    if (size < K_HEADER_BYTES + (size_t)m->hdr.n_tensors * K_DIRENT_BYTES) return -4;
    /* directory entries are packed at 68 bytes each: name[32], dtype, shape[4], offset, nbytes */
    for (uint32_t i = 0; i < m->hdr.n_tensors; i++) {
        const uint8_t *e = p + K_HEADER_BYTES + (size_t)i * K_DIRENT_BYTES;
        feng_tensor_t *d = &m->dir_storage[i];
        memcpy(d->name, e, 32);
        d->name[31] = 0;
        memcpy(&d->dtype, e + 32, 4);
        memcpy(d->shape, e + 36, 16);
        memcpy(&d->offset, e + 52, 8);
        memcpy(&d->nbytes, e + 60, 8);
    }
    m->dir = m->dir_storage;
    m->blob = p;

    const feng_tensor_t *t = find_tensor(m, "tok_embd.weight");
    if (t == NULL) return -5;
    m->tok_embd = ptr_of(m, t);
    m->tok_embd_dtype = t->dtype;
    t = find_tensor(m, "output_norm.weight");
    if (t == NULL) return -6;
    m->out_norm = ptr_of(m, t);

    for (uint32_t l = 0; l < m->hdr.n_layers; l++) {
        const void *slots[11] = {0};
        for (int k = 0; k < 11; k++) {
            char buf[64];
            /* 名字已 NUL 结尾，不需要原来的 %.*s + string_view.size() */
            snprintf(buf, sizeof(buf), "blk.%u.%s", (unsigned)l, kLayerTensorNames[k]);
            const feng_tensor_t *tt = find_tensor(m, buf);
            if (tt == NULL) {
                printf("missing tensor %s\n", buf);
                return -7;
            }
            slots[k] = ptr_of(m, tt);
        }
        m->layers[l].attn_norm = slots[0];
        m->layers[l].ffn_norm = slots[1];
        m->layers[l].wq = slots[2];
        m->layers[l].wk = slots[3];
        m->layers[l].wv = slots[4];
        m->layers[l].wo = slots[5];
        m->layers[l].q_norm = slots[6];
        m->layers[l].k_norm = slots[7];
        m->layers[l].gate = slots[8];
        m->layers[l].up = slots[9];
        m->layers[l].down = slots[10];
    }
    return 0;
}
