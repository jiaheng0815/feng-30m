/* 手机端多核 GEMV：把每个 GEMV 的输出行切成若干块，交给常驻 worker 线程池并行算。
 *
 * 派生自 core/feng_smp.c（2026-10-15 拷贝）。板端那份是双核
 * FreeRTOS 版（core0 + core1，信号量握手，最多 2 路）；手机有 4~8 个核，
 * 这里改成基于 feng_phone.c 的通用 pthread 线程池，路数由 FENG_PHONE_THREADS 决定。
 *
 * 为什么按「输出行」切：
 *  - 每一行的计算只读 x、只写 y[o]，行与行完全独立；
 *  - 因此并行结果与单线程**逐位相同**（不像按输入维切分那样会改变累加顺序）；
 *  - 线程池里用原子计数器动态领任务，行数少或耗时不均时也能自动均衡。
 */
#include "feng.h"
#include "feng_phone.h"

/* 每次领任务算多少行。
 * 一行只有几百字节权重（不到 1 微秒的活），如果一行一个任务，8 个线程会为了
 * 同一个计数器反复争抢同一条缓存行——lm_head 有 16384 行，光原子操作就能吃掉
 * 可观的时间。按块领（32 行）把这个开销摊掉，同时块足够小、动态领取仍然能均衡。 */
#define GEMV_ROW_CHUNK 32

static int n_chunks(int n_out)
{
    return (n_out + GEMV_ROW_CHUNK - 1) / GEMV_ROW_CHUNK;
}

typedef struct {
    const void *tensor;
    uint32_t dtype;
    const float *x;
    float *y;
    int n_in;
    int n_out;
} gemv_job_t;

static void gemv_row_chunk(void *ctx, int i)
{
    const gemv_job_t *j = (const gemv_job_t *)ctx;
    const int r0 = i * GEMV_ROW_CHUNK;
    int r1 = r0 + GEMV_ROW_CHUNK;
    if (r1 > j->n_out) r1 = j->n_out;
    feng_gemv_range(j->tensor, j->dtype, j->x, j->y, r0, r1, j->n_in);
}

void feng_smp_init(void)
{
    /* 线程池由 feng_phone_init() 建立；这里只做幂等确认，保持与板端同名 API。 */
    (void)feng_phone_init();
}

void feng_gemv_par(const void *tensor, uint32_t dtype, const float *x, float *y,
                   int n_out, int n_in)
{
#if FENG_GEMV_A8 || FENG_GEMV_PIE || FENG_PHONE_NEON
    feng_gemv_a8_prepare(x, n_in);
#endif
    if (n_out <= 0) return;
    /* 行数太少的调用（norm 之类的小张量）并行反而更贵，直接串行整段。
     * 阈值与板端一致（n_out < 32 退回单核）。 */
    if (n_out < 32 || feng_phone_threads() <= 1) {
        feng_gemv_range(tensor, dtype, x, y, 0, n_out, n_in);
        return;
    }
    gemv_job_t job = { tensor, dtype, x, y, n_in, n_out };
    feng_phone_parallel_for(n_chunks(n_out), gemv_row_chunk, &job);
}

/* ---------------- feng_gemv3 的合并并行版 ---------------- */

typedef struct {
    const void *t[3];
    float *y[3];
    int n_out[3];
    int nc[3];          /* 每个矩阵切成几块 */
    int n_mat;
    uint32_t dtype;
    const float *x;
    int n_in;
} gemv3_job_t;

static void gemv3_row_chunk(void *ctx, int i)
{
    /* 把一维块号 i 映射回「第几个矩阵的第几块」——三个矩阵共处一个并行区间，
     * 每 token 少两次信号量往返（q/k/v 与 gate/up 各一次）。 */
    const gemv3_job_t *j = (const gemv3_job_t *)ctx;
    int k = 0;
    while (k < j->n_mat - 1 && i >= j->nc[k]) {
        i -= j->nc[k];
        k++;
    }
    const int r0 = i * GEMV_ROW_CHUNK;
    int r1 = r0 + GEMV_ROW_CHUNK;
    if (r1 > j->n_out[k]) r1 = j->n_out[k];
    feng_gemv_range(j->t[k], j->dtype, j->x, j->y[k], r0, r1, j->n_in);
}

void feng_gemv3_par(const void *w0, const void *w1, const void *w2, uint32_t dtype,
                    const float *x, float *y0, float *y1, float *y2,
                    int n_out0, int n_out1, int n_out2, int n_in)
{
#if FENG_GEMV_A8 || FENG_GEMV_PIE || FENG_PHONE_NEON
    feng_gemv_a8_prepare(x, n_in);
#endif
    gemv3_job_t job;
    int n = 0;
    if (w0) { job.t[n] = w0; job.y[n] = y0; job.n_out[n] = n_out0; n++; }
    if (w1) { job.t[n] = w1; job.y[n] = y1; job.n_out[n] = n_out1; n++; }
    if (w2) { job.t[n] = w2; job.y[n] = y2; job.n_out[n] = n_out2; n++; }
    job.n_mat = n;
    if (n == 0) return;
    job.dtype = dtype;
    job.x = x;
    job.n_in = n_in;

    int total_rows = 0, total_chunks = 0;
    for (int i = 0; i < n; i++) {
        job.nc[i] = n_chunks(job.n_out[i]);
        total_rows += job.n_out[i];
        total_chunks += job.nc[i];
    }
    if (total_rows < 32 || feng_phone_threads() <= 1) {
        for (int i = 0; i < n; i++) {
            feng_gemv_range(job.t[i], dtype, x, job.y[i], 0, job.n_out[i], n_in);
        }
        return;
    }
    feng_phone_parallel_for(total_chunks, gemv3_row_chunk, &job);
}
