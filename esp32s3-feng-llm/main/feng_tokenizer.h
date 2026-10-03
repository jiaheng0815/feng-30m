#ifndef FENG_TOKENIZER_H
#define FENG_TOKENIZER_H

#include <stddef.h>
#include <stdint.h>

typedef struct {
    char **tokens;          /* vocab strings (byte-level encoded), index = token id */
    uint16_t *token_len;
    int vocab_size;
    uint32_t *merge_key;    /* sorted keys: (left << 15) | right */
    uint16_t *merge_val;    /* resulting token id */
    int n_merges;
    int *vocab_order;       /* vocab indices sorted by (len, bytes) */
    uint32_t *pair_key;     /* rank-order merge pair keys */
    uint16_t *pair_result;  /* rank-order merged token id */
    uint32_t *sorted_key;   /* merges sorted by key */
    int *sorted_rank;       /* rank of the key above */
    int16_t byte_to_token[256];
    int id_im_start, id_im_end, id_eot;
} feng_tok_t;

int feng_tok_load(feng_tok_t *t, const void *data, size_t size);
void feng_tok_free(feng_tok_t *t);

/* encode text (ASCII/UTF-8) -> token ids; returns number of ids (<= max_out) */
int feng_tok_encode(const feng_tok_t *t, const char *text, int *out, int max_out);

/* decode one token to raw bytes (byte-level -> original bytes); returns byte count */
int feng_tok_decode_token(const feng_tok_t *t, int id, char *out, int max_out);

#endif
