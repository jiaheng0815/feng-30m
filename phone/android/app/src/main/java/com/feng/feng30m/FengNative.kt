package com.feng.feng30m

/**
 * JNI 声明：与 app/src/main/cpp/feng_jni.c 一一对应。
 *
 * 引擎是单实例、非重入的，nativeInit 返回的句柄内部带一把互斥锁，
 * 调用方只需保证「一个会话同一时间只有一次 ask」。
 */
object FengNative {

    init {
        System.loadLibrary("feng")
    }

    /** 流式回调：每个 token 一次 onToken，整轮结束时一次 onDone。 */
    interface StreamCallback {
        fun onToken(token: String)

        /** genTokens = 本轮生成/回答的 token 数（tool 秒回时为 0）；decodeMs 只含 decode 阶段 */
        fun onDone(genTokens: Int, decodeMs: Long)
    }

    /**
     * 加载模型并建立会话。失败返回 0。
     * @param ctx 上下文长度（KV 容量），默认 2048
     * @param threads 线程数，<=0 表示用引擎默认（min(4, CPU 数)）
     */
    @JvmStatic
    external fun nativeInit(modelPath: String, tokPath: String, ctx: Int, threads: Int): Long

    @JvmStatic
    external fun nativeFree(handle: Long)

    /** 跑一轮：tool 命中就直接回答案，否则走模型流式生成。返回生成 token 数，-1 = 参数/长度错误。 */
    @JvmStatic
    external fun nativeAsk(handle: Long, text: String, maxNew: Int, cb: StreamCallback): Int

    /** 请求中止当前生成（下一个 token 边界生效）。 */
    @JvmStatic
    external fun nativeAbort(handle: Long)

    /** 清空 KV 上下文与记忆。 */
    @JvmStatic
    external fun nativeReset(handle: Long)

    /** 记忆 tool 的快照文本（\mem）。 */
    @JvmStatic
    external fun nativeMemory(handle: Long): String?

    /** 运行时状态：ctx 占用 / 线程 / NEON / KV 占用。 */
    @JvmStatic
    external fun nativeStats(handle: Long): String?

    /** 后端一行摘要（是否走 NEON、线程数）。 */
    @JvmStatic
    external fun nativeBackend(handle: Long): String?
}
