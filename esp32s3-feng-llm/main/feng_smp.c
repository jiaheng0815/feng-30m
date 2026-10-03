/* Split each GEMV across the two ESP32-S3 cores.
 *
 * Two worker tasks (one pinned to core 0, one to core 1) each compute half of the
 * output rows.  The caller posts the job and blocks on the completion semaphores,
 * so a GEMV really runs on both cores at once.
 */
#include "feng.h"

#if defined(ESP_PLATFORM)

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/semphr.h"
#include "esp_log.h"
#include "esp_heap_caps.h"
#include "esp_attr.h"
#include "esp_timer.h"

#define SMP_NPARTS 2
#define SMP_TASK_STACK 4096
#define SMP_TASK_PRIO 5

typedef struct {
    SemaphoreHandle_t start;
    SemaphoreHandle_t done;
    const void *tensor;
    uint32_t dtype;
    const float *x;
    float *y;
    int n_in, r0, r1;
    /* diagnostics */
    int core, runs;
    int64_t us_last;
} smp_part_t;

static smp_part_t s_parts[SMP_NPARTS];
static StaticSemaphore_t s_start_buf[SMP_NPARTS];
static StaticSemaphore_t s_done_buf[SMP_NPARTS];
static int s_smp_ok;
static const char *TAG = "feng_smp";

static void smp_worker(void *arg)
{
    smp_part_t *p = (smp_part_t *)arg;
    for (;;) {
        if (xSemaphoreTake(p->start, portMAX_DELAY) != pdTRUE) continue;
        p->core = xPortGetCoreID();
        const int64_t t0 = esp_timer_get_time();
        feng_gemv_range(p->tensor, p->dtype, p->x, p->y, p->r0, p->r1, p->n_in);
        p->us_last = esp_timer_get_time() - t0;
        p->runs++;
        xSemaphoreGive(p->done);
    }
}

void feng_smp_init(void)
{
    if (s_smp_ok) return;
    for (int i = 0; i < SMP_NPARTS; i++) {
        s_parts[i].start = xSemaphoreCreateBinaryStatic(&s_start_buf[i]);
        s_parts[i].done = xSemaphoreCreateBinaryStatic(&s_done_buf[i]);
        if (!s_parts[i].start || !s_parts[i].done) {
            ESP_LOGE(TAG, "semaphore init failed (internal free %u)",
                     (unsigned)heap_caps_get_free_size(MALLOC_CAP_INTERNAL));
            return;
        }
    }
    ESP_LOGI(TAG, "heap before workers: internal %u B (largest %u), psram %u B",
             (unsigned)heap_caps_get_free_size(MALLOC_CAP_INTERNAL),
             (unsigned)heap_caps_get_largest_free_block(MALLOC_CAP_INTERNAL),
             (unsigned)heap_caps_get_free_size(MALLOC_CAP_SPIRAM));
    for (int i = 0; i < SMP_NPARTS; i++) {
        TaskHandle_t h = NULL;
        BaseType_t ok = xTaskCreatePinnedToCoreWithCaps(smp_worker, "feng_smp", SMP_TASK_STACK,
                                                        &s_parts[i], SMP_TASK_PRIO, &h, i,
                                                        MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
        if (ok != pdPASS) {
            ESP_LOGW(TAG, "internal stack failed on core %d, retrying in PSRAM", i);
            ok = xTaskCreatePinnedToCoreWithCaps(smp_worker, "feng_smp", SMP_TASK_STACK,
                                                 &s_parts[i], SMP_TASK_PRIO, &h, i,
                                                 MALLOC_CAP_SPIRAM);
        }
        if (ok != pdPASS) {
            ESP_LOGE(TAG, "task create failed on core %d", i);
            return;
        }
    }
    s_smp_ok = 1;
    ESP_LOGI(TAG, "GEMV workers up on core 0 + core 1");
}

void feng_gemv_par(const void *tensor, uint32_t dtype, const float *x, float *y,
                   int n_out, int n_in)
{
    if (!s_smp_ok || n_out < 32) {
        feng_gemv_range(tensor, dtype, x, y, 0, n_out, n_in);
        return;
    }
    const int mid = n_out / 2;
    for (int i = 0; i < SMP_NPARTS; i++) {
        smp_part_t *p = &s_parts[i];
        p->tensor = tensor;
        p->dtype = dtype;
        p->x = x;
        p->y = y;
        p->n_in = n_in;
        p->r0 = (i == 0) ? 0 : mid;
        p->r1 = (i == 0) ? mid : n_out;
    }
    /* Both jobs must be posted before either worker can preempt us: otherwise the
     * worker on our own core starts immediately and the second job is only posted
     * once it finishes (that serialised the two halves completely). */
    const UBaseType_t old_prio = uxTaskPriorityGet(NULL);
    if (old_prio <= SMP_TASK_PRIO) vTaskPrioritySet(NULL, SMP_TASK_PRIO + 1);
    xSemaphoreGive(s_parts[0].start);
    xSemaphoreGive(s_parts[1].start);
    if (old_prio <= SMP_TASK_PRIO) vTaskPrioritySet(NULL, old_prio);
    xSemaphoreTake(s_parts[0].done, portMAX_DELAY);
    xSemaphoreTake(s_parts[1].done, portMAX_DELAY);
    if (s_parts[0].runs <= 3) {
        ESP_LOGI(TAG, "par n_out=%d: part0 core%d rows[%d,%d) %lldus | part1 core%d rows[%d,%d) %lldus",
                 n_out, s_parts[0].core, s_parts[0].r0, s_parts[0].r1, s_parts[0].us_last,
                 s_parts[1].core, s_parts[1].r0, s_parts[1].r1, s_parts[1].us_last);
    }
}

#else /* host build (tools/pc check): single core, no RTOS */

void feng_smp_init(void)
{
}

void feng_gemv_par(const void *tensor, uint32_t dtype, const float *x, float *y,
                   int n_out, int n_in)
{
    feng_gemv_range(tensor, dtype, x, y, 0, n_out, n_in);
}

#endif
