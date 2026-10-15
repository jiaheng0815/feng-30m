package com.feng.feng30m

import android.content.Context
import java.io.File

/**
 * 会话管理：把 assets 里的模型解包到 filesDir，再交给 native 侧 mmap 加载。
 *
 * 为什么先解包而不是直接读 assets：
 * assets 在 APK 里是压缩包内的一段，拿不到「一个可以 mmap 的连续文件」；
 * 解包一次之后，engine 用 mmap 只读映射，页面由内核按需调入 page cache，
 * 不占 Java 堆，也不会因为文件大而 OOM。
 */
class ChatEngine(private val context: Context) {

    private var handle = 0L

    val isReady: Boolean get() = handle != 0L

    companion object {
        const val CTX = 2048
        const val MAX_NEW = 96

        /** 引擎默认线程数：min(4, 可用核数)，与 CLI 口径一致 */
        fun defaultThreads(): Int =
            Runtime.getRuntime().availableProcessors().coerceIn(1, 4)
    }

    private fun modelFile(name: String): File = File(context.filesDir, name)

    /** asset 的未压缩长度；拿不到返回 -1。
     *  注意 openFd 只对**未压缩**的 asset 有效（build.gradle.kts 里已经把它加进
     *  noCompress，但这里仍然兜底，免得换构建配置就崩）。 */
    private fun assetSize(name: String): Long =
        try {
            context.assets.openFd(name).use { it.length }
        } catch (e: Exception) {
            -1L
        }

    /**
     * 确保 model.bin / tokenizer.bin 已解包到 filesDir。
     * @param onProgress 0f..1f
     */
    fun extractAssets(onProgress: (Float) -> Unit) {
        val names = listOf("model.bin", "tokenizer.bin")
        val sizes = names.map { assetSize(it) }
        val total = sizes.filter { it > 0 }.sum().takeIf { it > 0 } ?: -1L
        var done = 0L

        names.forEachIndexed { i, name ->
            val out = modelFile(name)
            val want = sizes[i]
            if (out.exists() && (want <= 0 || out.length() == want)) {
                done += if (want > 0) want else out.length()      // 上次已解包好，跳过
            } else {
                context.assets.open(name).use { input ->          // 压缩与否都能读
                    out.outputStream().use { output ->
                        val buf = ByteArray(1 shl 16)
                        while (true) {
                            val n = input.read(buf)
                            if (n <= 0) break
                            output.write(buf, 0, n)
                            done += n
                            onProgress(
                                if (total > 0) (done.toFloat() / total).coerceIn(0f, 1f)
                                else (i + 0.5f) / names.size
                            )
                        }
                    }
                }
            }
            onProgress(if (total > 0) (done.toFloat() / total).coerceIn(0f, 1f) else (i + 1f) / names.size)
        }
    }

    /** 模型是否已随 APK 打包（构建脚本会把 model_export 目录拷进 assets） */
    fun hasBundledModel(): Boolean =
        try {
            context.assets.open("model.bin").close()
            true
        } catch (e: Exception) {
            false
        }

    fun open(ctx: Int = CTX, threads: Int = defaultThreads()): Boolean {
        if (handle != 0L) return true
        handle = FengNative.nativeInit(
            modelFile("model.bin").absolutePath,
            modelFile("tokenizer.bin").absolutePath,
            ctx,
            threads
        )
        return handle != 0L
    }

    fun ask(text: String, maxNew: Int = MAX_NEW, cb: FengNative.StreamCallback): Int {
        if (handle == 0L) return -1
        return FengNative.nativeAsk(handle, text, maxNew, cb)
    }

    fun abort() {
        if (handle != 0L) FengNative.nativeAbort(handle)
    }

    fun reset() {
        if (handle != 0L) FengNative.nativeReset(handle)
    }

    fun memory(): String = if (handle != 0L) FengNative.nativeMemory(handle) ?: "（空）" else "（未初始化）"

    fun stats(): String = if (handle != 0L) FengNative.nativeStats(handle) ?: "" else ""

    fun close() {
        if (handle != 0L) {
            FengNative.nativeFree(handle)
            handle = 0L
        }
    }
}
