/* phone/engine 派生副本：自 core/feng_tools.h 拷贝（2026-10-15，v4.0 C23 迁移后）。
 * 手机端改动见 phone/README.md；板端原始文件未被修改，两边需手动同步。
 * 原版校验值见 phone/engine/SOURCE.sha256。 */
/* 运行时 tool：时间（UTC+8）与随机数。
 *
 * 时间：
 *   - 板端没有 RTC 电池、也放不下 WiFi 协议栈（app+model 受 16MB mmap 窗口限制），
 *     所以由宿主在上电后发 `\settime <unix秒>`（宿主自己走 NTP/系统时钟）把
 *     网络时间戳推进来；固件用 esp_timer 走时，误差只来自晶振漂移。
 *   - PC C 引擎直接用系统时钟。
 * 随机数：
 *   seed = 当前运行时间(秒) × 1.54 × 1000；取第 2 个随机数（第 1 个按需求丢弃）。
 */
#ifndef FENG_TOOLS_H
#define FENG_TOOLS_H

#ifdef __cplusplus
extern "C" {
#endif

/* epoch 秒提供者；返回 0 表示"还没对时" */
typedef long long (*feng_epoch_fn)(void);
/* 运行时间提供者（微秒，单调递增） */
typedef long long (*feng_uptime_us_fn)(void);

void feng_tools_set_time(feng_epoch_fn fn);
void feng_tools_set_uptime(feng_uptime_us_fn fn);

/* 用户问时间/日期/时间戳 -> 写 answer 并返回 1；否则返回 0 */
int feng_time_answer(const char *user, char *answer, int answer_sz);
/* 用户要随机数 -> 写 answer 并返回 1；否则返回 0 */
int feng_random_answer(const char *user, char *answer, int answer_sz);

/* 工具函数（PC 单测用） */
void feng_time_format_utc8(long long epoch, char *buf, int buf_sz);
/* 只输出日期（用于"N天后是几号"） */
void feng_time_format_date_utc8(long long epoch, char *buf, int buf_sz);
long long feng_rand_range(unsigned long long seed, long long lo, long long hi);

#ifdef __cplusplus
}
#endif

#endif
