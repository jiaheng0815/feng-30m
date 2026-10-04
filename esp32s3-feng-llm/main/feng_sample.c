/* 共享贪心采样器：重复惩罚 + no-repeat n-gram。
 * 板端固件与 PC 引擎（pc_chat/pc_kv_suite/pc_mt_suite）统一走这里，
 * 避免"PC 端纯 argmax、板端有惩罚"的口径不一致，并用 n-gram 约束压住复读循环。 */
#include "feng.h"

int feng_sample_greedy(float *logits, int vocab, const int *hist, int nhist,
                       float penalty, int no_repeat_n)
{
    if (penalty > 0.0f && penalty != 1.0f) {
        for (int i = 0; i < nhist; i++) {
            const int t = hist[i];
            if (t >= 0 && t < vocab) {
                logits[t] = logits[t] > 0 ? logits[t] / penalty : logits[t] * penalty;
            }
        }
    }
    if (no_repeat_n >= 2 && nhist >= no_repeat_n - 1) {
        const int m = no_repeat_n - 1;                  /* 需要匹配的前缀长度 */
        const int *tail = hist + (nhist - m);
        for (int h = 0; h + m < nhist; h++) {           /* 候选 = hist[h+m] */
            int match = 1;
            for (int k = 0; k < m; k++) {
                if (hist[h + k] != tail[k]) { match = 0; break; }
            }
            if (match) {
                const int cand = hist[h + m];
                if (cand >= 0 && cand < vocab) logits[cand] = -1e30f;
            }
        }
        /* 注：vocab 16384、禁的是"补全已出现的 n-gram"，不可能全禁，无需回退分支 */
    }
    return feng_argmax(logits, vocab);
}
