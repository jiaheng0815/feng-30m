/* Parse the model.bin image and resolve tensor pointers.
 *
 * 解析保持逐字段 memcpy（对齐安全，与导出端字节精确一致）；
 * C++23 只改表达方式：std::span 表示内存镜像、std::string_view 查名字、
 * std::array 表示定长表——零分配、零异常、零运行时初始化。 */
#include "feng.h"

#include <array>
#include <cstdio>
#include <cstring>
#include <string_view>

namespace {

constexpr size_t kHeaderBytes = 44;     /* 9 x u32 + 2 x f32 */
constexpr size_t kDirEntryBytes = 68;   /* name[32] + dtype + shape[4] + offset + nbytes */

[[nodiscard]] const feng_tensor_t *find_tensor(const feng_model_t *m, std::string_view name)
{
    for (uint32_t i = 0; i < m->hdr.n_tensors; i++) {
        if (std::string_view{m->dir[i].name} == name) return &m->dir[i];
    }
    return nullptr;
}

[[nodiscard]] const void *ptr_of(const feng_model_t *m, const feng_tensor_t *t)
{
    return m->blob + t->offset;
}

/* blk.<layer>.<tensor>，顺序与 feng_layer_t 字段一致 */
constexpr std::array<std::string_view, 11> kLayerTensorNames{
    "attn_norm.weight", "ffn_norm.weight", "attn_q.weight", "attn_k.weight",
    "attn_v.weight",    "attn_output.weight", "attn_q_norm.weight", "attn_k_norm.weight",
    "ffn_gate.weight",  "ffn_up.weight",    "ffn_down.weight",
};

}  // namespace

[[nodiscard]] int feng_model_init(feng_model_t *m, std::span<const std::byte> data) noexcept
{
    if (m == nullptr || data.size() < kHeaderBytes) return -1;
    const auto *p = reinterpret_cast<const uint8_t *>(data.data());
    const size_t size = data.size();
    /* header: 9 x u32 + 2 x f32 = 44 bytes (packed, parsed field by field) */
    std::array<uint32_t, 9> v{};
    std::array<float, 2> fv{};
    std::memcpy(v.data(), p, sizeof(v));
    std::memcpy(fv.data(), p + 36, sizeof(fv));
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
    if (m->hdr.magic != FENG_MAGIC) return -2;
    if (m->hdr.n_layers > FENG_MAX_LAYERS) return -3;
    if (m->hdr.n_tensors > FENG_MAX_TENSORS) return -8;
    if (size < kHeaderBytes + (size_t)m->hdr.n_tensors * kDirEntryBytes) return -4;
    /* directory entries are packed at 68 bytes each: name[32], dtype, shape[4], offset, nbytes */
    for (uint32_t i = 0; i < m->hdr.n_tensors; i++) {
        const uint8_t *e = p + kHeaderBytes + (size_t)i * kDirEntryBytes;
        feng_tensor_t *d = &m->dir_storage[i];
        std::memcpy(d->name, e, 32);
        d->name[31] = 0;
        std::memcpy(&d->dtype, e + 32, 4);
        std::memcpy(d->shape, e + 36, 16);
        std::memcpy(&d->offset, e + 52, 8);
        std::memcpy(&d->nbytes, e + 60, 8);
    }
    m->dir = m->dir_storage;
    m->blob = p;

    const feng_tensor_t *t = find_tensor(m, "tok_embd.weight");
    if (!t) return -5;
    m->tok_embd = ptr_of(m, t);
    m->tok_embd_dtype = t->dtype;
    t = find_tensor(m, "output_norm.weight");
    if (!t) return -6;
    m->out_norm = ptr_of(m, t);

    for (uint32_t l = 0; l < m->hdr.n_layers; l++) {
        std::array<const void *, 11> slots{};
        for (int k = 0; k < 11; k++) {
            std::array<char, 64> buf{};
            std::snprintf(buf.data(), buf.size(), "blk.%u.%.*s", (unsigned)l,
                          (int)kLayerTensorNames[k].size(), kLayerTensorNames[k].data());
            const feng_tensor_t *tt = find_tensor(m, buf.data());
            if (!tt) {
                std::printf("missing tensor %s\n", buf.data());
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
