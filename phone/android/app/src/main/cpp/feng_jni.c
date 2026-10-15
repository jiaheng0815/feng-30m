/* Android JNI 薄壳：只做 JVM <-> C 引擎的搬运，推理逻辑一行都不在这里。
 *
 * 设计要点：
 *  - 引擎是 C23 且**单实例、非重入**（大量 .bss 静态缓冲），所以这里用一把互斥锁
 *    把 nativeAsk 串行化；Kotlin 侧同时只会有一个推理在跑。
 *  - 推理在**调用 JNI 的那个线程**上跑（Kotlin 的 Dispatchers.Default 协程线程），
 *    worker 线程只做 GEMV、不碰 JNI，因此不需要 AttachCurrentThread。
 *  - 每个 token 回调一次 Java 方法；每次 NewStringUTF 产生的 local ref 立刻删掉，
 *    否则 512 个 local ref 的表会被撑爆（长回答时会踩到）。 */
#include <jni.h>
#include <android/log.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "feng.h"
#include "feng_calc.h"
#include "feng_memory.h"
#include "feng_phone.h"
#include "feng_tokenizer.h"
#include "feng_tools.h"

#define TAG "feng_jni"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, TAG, __VA_ARGS__)

typedef struct {
    feng_model_t m;
    feng_tok_t tok;
    feng_kv_t kv;
    feng_workspace_t ws;
    const uint8_t *mblob;
    const uint8_t *tblob;
    size_t mlen, tlen;
    int ctx;
    int nl_id;
    int abort_flag;
    pthread_mutex_t lock;
} jni_session;

static void *xmalloc(size_t n)
{
    void *p = malloc(n);
    if (p) memset(p, 0, n);
    return p;
}

/* 把一段以 NUL 结尾的 token/回答交给 Java，并立刻释放 local ref——
 * 一次回答可能回调上百次，不释放会把 512 条 local ref 的表撑爆。 */
static void call_string(JNIEnv *env, jobject cb, jmethodID mid, const char *s)
{
    if (!mid || !s) return;
    jstring js = (*env)->NewStringUTF(env, s);
    if (!js) {
        /* 非法 UTF-8（半截多字节字符）时 NewStringUTF 会返回 NULL：跳过这个 token，
         * 总比让整轮崩掉好。这里极少发生——tokenizer 是按字符边界切词的。 */
        (*env)->ExceptionClear(env);
        return;
    }
    (*env)->CallVoidMethod(env, cb, mid, js);
    (*env)->DeleteLocalRef(env, js);
}

/* 一轮问答：tool 优先（算式 → 记忆 → 时间 → 随机数，与固件/PC 同序），否则走模型。
 * 返回生成的 token 数；-1 表示参数错误。 */
JNIEXPORT jint JNICALL
Java_com_feng_feng30m_FengNative_nativeAsk(JNIEnv *env, jclass cls, jlong handle,
                                           jstring jtext, jint maxNew, jobject callback)
{
    (void)cls;
    jni_session *s = (jni_session *)(intptr_t)handle;
    if (!s || !jtext) return -1;

    const char *text = (*env)->GetStringUTFChars(env, jtext, NULL);
    if (!text) return -1;

    jclass cbCls = (*env)->GetObjectClass(env, callback);
    jmethodID midToken = cbCls ? (*env)->GetMethodID(env, cbCls, "onToken", "(Ljava/lang/String;)V") : NULL;
    jmethodID midDone = cbCls ? (*env)->GetMethodID(env, cbCls, "onDone", "(IJ)V") : NULL;
    if (cbCls) (*env)->DeleteLocalRef(env, cbCls);

    pthread_mutex_lock(&s->lock);
    s->abort_flag = 0;

    char reply[512];
    int gen = 0;
    double pre_ms = 0, dec_ms = 0;

    (void)feng_mem_learn(text);
    if (feng_calc_answer(text, reply, sizeof(reply)) ||
        feng_mem_answer(text, reply, sizeof(reply)) ||
        feng_time_answer(text, reply, sizeof(reply)) ||
        feng_random_answer(text, reply, sizeof(reply))) {
        /* tool 秒回：一次回调交完，gen 记 0 让界面知道这不是模型生成的 */
        call_string(env, callback, midToken, reply);
        if (midDone) (*env)->CallVoidMethod(env, callback, midDone, (jint)0, (jlong)0);
        pthread_mutex_unlock(&s->lock);
        (*env)->ReleaseStringUTFChars(env, jtext, text);
        return 0;
    }

    const int v = s->m.hdr.vocab;
    char prompt[4096];
    snprintf(prompt, sizeof(prompt),
             "<|im_start|>user\n%s<|im_end|>\n<|im_start|>assistant\n", text);

    const int cap = (int)strlen(text) + 64;
    int *ids = (int *)xmalloc(sizeof(int) * (size_t)cap);
    const int n = ids ? feng_tok_encode(&s->tok, prompt, ids, cap) : 0;
    if (!ids || n <= 0 || n >= s->ctx - maxNew) {
        free(ids);
        pthread_mutex_unlock(&s->lock);
        (*env)->ReleaseStringUTFChars(env, jtext, text);
        return -1;
    }
    if (s->kv.len > 0 && s->kv.len + n + maxNew + 4 > s->ctx) {
        s->kv.len = 0;                 /* 上下文满：自动开新对话（与 CLI 同口径） */
        feng_mem_clear();
    }

    int pos = s->kv.len;
    float *logits = NULL;
    const double t0 = (double)feng_phone_uptime_us();
    for (int i = 0; i < n; i++) {
        logits = feng_forward_ex(&s->m, &s->kv, &s->ws, ids[i], pos++, i + 1 == n);
    }
    const double t1 = (double)feng_phone_uptime_us();
    free(ids);

    int hist[64];
    int nhist = 0;
    for (int step = 0; step < maxNew; step++) {
        if (__atomic_load_n(&s->abort_flag, __ATOMIC_RELAXED)) break;
        const int tk = feng_sample_greedy(logits, v, hist, nhist, 1.15f, 3);
        if (tk == s->tok.id_im_end || tk == s->tok.id_eot) break;
        char b[16];
        const int nb = feng_tok_decode_token(&s->tok, tk, b, sizeof(b));
        if (nb > 0 && nb < (int)sizeof(b)) {
            b[nb] = 0;
            call_string(env, callback, midToken, b);   /* 文本由 Kotlin 侧累积 */
        }
        if (nhist < 64) hist[nhist++] = tk;
        else { memmove(hist, hist + 1, sizeof(hist) - sizeof(hist[0])); hist[63] = tk; }
        logits = feng_forward(&s->m, &s->kv, &s->ws, tk, pos++);
        gen++;
    }
    const double t2 = (double)feng_phone_uptime_us();
    pre_ms = (t1 - t0) / 1000.0;
    dec_ms = (t2 - t1) / 1000.0;

    /* 把 im_end / 换行也补进 KV，下一轮才有正确上下文 */
    (void)feng_forward_ex(&s->m, &s->kv, &s->ws, s->tok.id_im_end, pos++, 0);
    if (s->nl_id > 0) (void)feng_forward_ex(&s->m, &s->kv, &s->ws, s->nl_id, pos++, 0);
    s->kv.len = pos;

    LOGI("ask: prompt %d tok, gen %d tok, prefill %.0f ms, decode %.0f ms (%.1f tok/s)",
         n, gen, pre_ms, dec_ms, dec_ms > 0 ? gen * 1000.0 / dec_ms : 0.0);

    if (midDone) (*env)->CallVoidMethod(env, callback, midDone, (jint)gen, (jlong)(dec_ms + 0.5));

    pthread_mutex_unlock(&s->lock);
    (*env)->ReleaseStringUTFChars(env, jtext, text);
    return (jint)gen;
}

JNIEXPORT void JNICALL
Java_com_feng_feng30m_FengNative_nativeAbort(JNIEnv *env, jclass cls, jlong handle)
{
    (void)env; (void)cls;
    jni_session *s = (jni_session *)(intptr_t)handle;
    if (s) __atomic_store_n(&s->abort_flag, 1, __ATOMIC_RELAXED);   /* decode 循环每步检查一次 */
}

JNIEXPORT void JNICALL
Java_com_feng_feng30m_FengNative_nativeReset(JNIEnv *env, jclass cls, jlong handle)
{
    (void)env; (void)cls;
    jni_session *s = (jni_session *)(intptr_t)handle;
    if (!s) return;
    pthread_mutex_lock(&s->lock);
    s->kv.len = 0;
    feng_mem_clear();
    pthread_mutex_unlock(&s->lock);
}

JNIEXPORT jstring JNICALL
Java_com_feng_feng30m_FengNative_nativeMemory(JNIEnv *env, jclass cls, jlong handle)
{
    (void)cls;
    jni_session *s = (jni_session *)(intptr_t)handle;
    if (!s) return NULL;
    char snap[256];
    pthread_mutex_lock(&s->lock);
    feng_mem_snapshot(snap, sizeof(snap));
    pthread_mutex_unlock(&s->lock);
    return (*env)->NewStringUTF(env, snap);
}

JNIEXPORT jstring JNICALL
Java_com_feng_feng30m_FengNative_nativeStats(JNIEnv *env, jclass cls, jlong handle)
{
    (void)cls;
    jni_session *s = (jni_session *)(intptr_t)handle;
    if (!s) return NULL;
    char buf[256];
    snprintf(buf, sizeof(buf),
             "ctx %d/%d ｜ 线程 %d/%d ｜ NEON %s ｜ KV %.2f MB",
             s->kv.len, s->ctx, feng_phone_threads(), feng_phone_cpus(),
             feng_phone_neon_on() ? "开" : "关",
             (double)feng_kv_bytes(&s->m, s->ctx) / 1048576.0);
    return (*env)->NewStringUTF(env, buf);
}

JNIEXPORT jstring JNICALL
Java_com_feng_feng30m_FengNative_nativeBackend(JNIEnv *env, jclass cls, jlong handle)
{
    (void)cls; (void)handle;
    char buf[128];
    snprintf(buf, sizeof(buf), "NEON %s ｜ 线程 %d/%d",
             feng_phone_neon_on() ? "开" : "关", feng_phone_threads(), feng_phone_cpus());
    return (*env)->NewStringUTF(env, buf);
}

JNIEXPORT void JNICALL
Java_com_feng_feng30m_FengNative_nativeFree(JNIEnv *env, jclass cls, jlong handle)
{
    (void)env; (void)cls;
    jni_session *s = (jni_session *)(intptr_t)handle;
    if (!s) return;
    feng_phone_unload(s->mblob, s->mlen);
    feng_phone_unload(s->tblob, s->tlen);
    free(s->kv.k_cache);
    free(s->kv.v_cache);
    free(s->kv.k_scale);
    free(s->kv.v_scale);
    free(s->ws.x); free(s->ws.xn); free(s->ws.q); free(s->ws.k); free(s->ws.v);
    free(s->ws.attn); free(s->ws.proj); free(s->ws.gate); free(s->ws.up);
    free(s->ws.ffn); free(s->ws.logits); free(s->ws.scratch);
    feng_tok_free(&s->tok);
    pthread_mutex_destroy(&s->lock);
    free(s);
}

/* 初始化：加载模型 + tokenizer、分配 KV/workspace、拉起线程池。
 * 返回会话句柄；失败返回 0（Kotlin 侧抛出并显示原因）。 */
JNIEXPORT jlong JNICALL
Java_com_feng_feng30m_FengNative_nativeInit(JNIEnv *env, jclass cls, jstring jmodel,
                                            jstring jtok, jint ctx, jint threads)
{
    (void)cls;
    const char *mp = (*env)->GetStringUTFChars(env, jmodel, NULL);
    const char *tp = (*env)->GetStringUTFChars(env, jtok, NULL);
    if (!mp || !tp) return 0;

    if (threads > 0) feng_phone_set_threads(threads);
    feng_phone_init();
    feng_tools_set_time(feng_phone_epoch);
    feng_tools_set_uptime(feng_phone_uptime_us);
    feng_smp_init();

    jni_session *s = (jni_session *)xmalloc(sizeof(jni_session));
    if (!s) goto fail;

    if (ctx < 256) ctx = 2048;
    s->ctx = ctx;

    if (feng_phone_load(mp, &s->mblob, &s->mlen) != 0) {
        LOGE("打不开 %s", mp); goto fail;
    }
    if (feng_phone_load(tp, &s->tblob, &s->tlen) != 0) {
        LOGE("打不开 %s", tp); goto fail;
    }
    if (feng_model_init(&s->m, s->mblob, s->mlen) != 0) {
        LOGE("model.bin 解析失败"); goto fail;
    }
    if (feng_tok_load(&s->tok, s->tblob, s->tlen) != 0) {
        LOGE("tokenizer.bin 解析失败"); goto fail;
    }

    const size_t kv_bytes = feng_kv_bytes(&s->m, ctx);
    s->kv.ctx = ctx;
    s->kv.len = 0;
    s->kv.k_cache = xmalloc(kv_bytes / 2 + 64);
    s->kv.v_cache = xmalloc(kv_bytes / 2 + 64);
#if FENG_KV_INT8 || FENG_KV_Q2
    const size_t n_sc = feng_kv_scale_slots(&s->m, ctx);
    s->kv.k_scale = (uint16_t *)xmalloc(n_sc * 2);
    s->kv.v_scale = (uint16_t *)xmalloc(n_sc * 2);
#endif

    feng_workspace_t *ws = &s->ws;
    const int h = s->m.hdr.hidden, ff = s->m.hdr.ffn, v = s->m.hdr.vocab;
    ws->max_ctx = ctx;
    ws->x = (float *)xmalloc(sizeof(float) * h);
    ws->xn = (float *)xmalloc(sizeof(float) * h);
    ws->q = (float *)xmalloc(sizeof(float) * h);
    ws->k = (float *)xmalloc(sizeof(float) * h);
    ws->v = (float *)xmalloc(sizeof(float) * h);
    ws->attn = (float *)xmalloc(sizeof(float) * h);
    ws->proj = (float *)xmalloc(sizeof(float) * ff);
    ws->gate = (float *)xmalloc(sizeof(float) * ff);
    ws->up = (float *)xmalloc(sizeof(float) * ff);
    ws->ffn = (float *)xmalloc(sizeof(float) * ff);
    ws->logits = (float *)xmalloc(sizeof(float) * v);
    ws->scratch = (float *)xmalloc(sizeof(float) * (ctx > ff ? ctx : ff));

    if (!s->kv.k_cache || !s->kv.v_cache || !s->kv.k_scale || !s->kv.v_scale ||
        !ws->x || !ws->logits || !ws->scratch) {
        LOGE("内存不足"); goto fail;
    }

    { int one = 0; s->nl_id = (feng_tok_encode(&s->tok, "\n", &one, 1) == 1) ? one : 0; }
    pthread_mutex_init(&s->lock, NULL);

    LOGI("初始化完成：%u 层 / h=%u / vocab=%u ｜ ctx=%d ｜ KV %.2f MB ｜ 线程 %d ｜ NEON %s",
         s->m.hdr.n_layers, s->m.hdr.hidden, s->m.hdr.vocab, ctx,
         kv_bytes / 1048576.0, feng_phone_threads(), feng_phone_neon_on() ? "on" : "off");

    (*env)->ReleaseStringUTFChars(env, jmodel, mp);
    (*env)->ReleaseStringUTFChars(env, jtok, tp);
    return (jlong)(intptr_t)s;

fail:
    if (s) free(s);
    (*env)->ReleaseStringUTFChars(env, jmodel, mp);
    (*env)->ReleaseStringUTFChars(env, jtok, tp);
    return 0;
}
