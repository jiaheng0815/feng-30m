/* 手机端平台层（Termux / Android）：文件加载、时间源、通用线程池、NEON 开关。
 *
 * 这是 phone/engine 这份引擎副本**新增**的文件，板端（esp32s3-feng-llm/main）
 * 不编译它；板端对应能力由 main.c + feng_smp.c 的 FreeRTOS 分支提供。
 *
 * 与引擎主体一样是纯 C23（v4.0 起引擎全面 C23，见 AGENTS.md）：
 *  - 不使用需要运行时构造的全局/静态对象（全部是 POD 静态 + 显式初始化）；
 *  - 错误处理靠返回码；不抛异常（C 本来就没有）；
 *  - 线程池在 feng_phone_init() 里一次性创建，推理期间只做信号量握手，不建线程。 */
#ifndef FENG_PHONE_H
#define FENG_PHONE_H

#include <stddef.h>
#include <stdint.h>

/* 上下文上限（编译期）。板端 MAX_CTX=2048（q2）；手机默认放到 32768，
 * 与底座模型的训练上限一致。注意这是**容量上限，不是默认工作点**：
 *   - q2 KV @32768 = 154 MB（@2048 只要 9.62 MB），初始化时才分配；
 *   - 板端 v3.19-embed 的 QAT 工作点是 2048，32k 属于偏离训练分布；
 *   - CLI 仍默认 --ctx 2048；APK 默认 32768，失败时回退 2048。
 * feng_llm.c 里的 g_scores 会按这个值放大（32768 个 float ≈128 KB .bss）。 */
#ifndef FENG_PHONE_MAX_CTX
#define FENG_PHONE_MAX_CTX 32768
#endif

/* 线程数硬上限（worker 线程 + 调用者）。 */
#ifndef FENG_PHONE_MAX_THREADS
#define FENG_PHONE_MAX_THREADS 8
#endif

/* aarch64 上默认启用 NEON 内核；其它架构恒为 0（feng_neon.c 整体为空实现）。 */
#if !defined(FENG_PHONE_NEON)
#if defined(__aarch64__) || defined(_M_ARM64)
#define FENG_PHONE_NEON 1
#else
#define FENG_PHONE_NEON 0
#endif
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* 启动期调用一次：解析环境变量、拉起 worker 线程。
 *   FENG_PHONE_THREADS=1..8   总线程数（含调用者）；默认 min(4, 在线 CPU 数)
 *   FENG_PHONE_NEON=0/1       0 = 强制标量路径（A/B 与回归用）
 * 返回实际线程数（>=1）。可重复调用（幂等）。 */
int feng_phone_init(void);

/* 运行期查询 */
int feng_phone_neon_on(void);    /* 1 = 走 NEON 内核（且 FENG_PHONE_NEON=1） */
int feng_phone_threads(void);    /* 总线程数，含调用者；>=1 */
int feng_phone_cpus(void);       /* 在线 CPU 数（受 FENG_PHONE_MAX_THREADS 截断） */
/* 覆盖线程数（1..FENG_PHONE_MAX_THREADS）；返回生效值。线程池只在 init 时建立。 */
int feng_phone_set_threads(int n);
void feng_phone_set_neon(int on);

/* 读文件到内存：优先 mmap(MAP_PRIVATE|PROT_READ)（零拷贝、不占常驻匿名内存），
 * 失败回退 malloc+fread。blob 直到 feng_phone_unload() 前一直有效。
 * 返回 0 成功；-1 打不开；-2 读失败；-3 OOM。 */
int feng_phone_load(const char *path, const uint8_t **blob, size_t *len);
void feng_phone_unload(const uint8_t *blob, size_t len);

/* 通用并行：把 [0, n) 的索引分给 feng_phone_threads() 个线程（含调用者），
 * 动态领任务（原子计数器），因此各行耗时不均时也能自动均衡。
 * 约定：fn(ctx, i) 只能读共享输入、写各自 id 独立的输出；不得嵌套调用本函数。
 * n < 8 或只有 1 个线程时退化为串行循环（逐位结果与串行完全一致）。 */
void feng_phone_parallel_for(int n, void (*fn)(void *, int), void *ctx);

/* 注入给 feng_tools 的时间源 */
long long feng_phone_epoch(void);      /* UTC 秒 */
long long feng_phone_uptime_us(void);  /* 单调微秒 */

#ifdef __cplusplus
}
#endif

#endif /* FENG_PHONE_H */
