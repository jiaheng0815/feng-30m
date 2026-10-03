/* Parse the model.bin image and resolve tensor pointers. */
#include "feng.h"

#include <stdio.h>
#include <string.h>

static const feng_tensor_t *find(const feng_model_t *m, const char *name)
{
    for (uint32_t i = 0; i < m->hdr.n_tensors; i++) {
        if (strncmp(m->dir[i].name, name, 31) == 0) return &m->dir[i];
    }
    return NULL;
}

static const void *ptr_of(const feng_model_t *m, const feng_tensor_t *t)
{
    return m->blob + t->offset;
}

int feng_model_init(feng_model_t *m, const void *data, size_t size)
{
    const uint8_t *p = (const uint8_t *)data;
    if (size < 44) return -1;
    /* header: 9 x u32 + 2 x f32 = 44 bytes (packed, parsed field by field) */
    uint32_t v[9];
    float fv[2];
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
    if (m->hdr.magic != FENG_MAGIC) return -2;
    if (m->hdr.n_layers > FENG_MAX_LAYERS) return -3;
    if (m->hdr.n_tensors > FENG_MAX_TENSORS) return -8;
    if (size < 44 + (size_t)m->hdr.n_tensors * 68) return -4;
    /* directory entries are packed at 68 bytes each: name[32], dtype, shape[4], offset, nbytes */
    for (uint32_t i = 0; i < m->hdr.n_tensors; i++) {
        const uint8_t *e = p + 44 + (size_t)i * 68;
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

    const feng_tensor_t *t = find(m, "tok_embd.weight");
    if (!t) return -5;
    m->tok_embd = ptr_of(m, t);
    m->tok_embd_dtype = t->dtype;
    t = find(m, "output_norm.weight");
    if (!t) return -6;
    m->out_norm = ptr_of(m, t);

    static const char *names[] = {"attn_norm.weight", "ffn_norm.weight", "attn_q.weight",
                                  "attn_k.weight", "attn_v.weight", "attn_output.weight",
                                  "attn_q_norm.weight", "attn_k_norm.weight",
                                  "ffn_gate.weight", "ffn_up.weight", "ffn_down.weight"};
    for (uint32_t l = 0; l < m->hdr.n_layers; l++) {
        const void *slots[11];
        for (int k = 0; k < 11; k++) {
            char buf[64];
            snprintf(buf, sizeof(buf), "blk.%u.%s", (unsigned)l, names[k]);
            const feng_tensor_t *tt = find(m, buf);
            if (!tt) {
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
