/* 时间/随机数 tool 实现：不依赖 libc 时区数据库，手算 UTC+8 日历。 */
#include "feng_tools.h"

#include <stdio.h>
#include <string.h>

static feng_epoch_fn s_epoch_fn;
static feng_uptime_us_fn s_uptime_fn;

void feng_tools_set_time(feng_epoch_fn fn) { s_epoch_fn = fn; }
void feng_tools_set_uptime(feng_uptime_us_fn fn) { s_uptime_fn = fn; }

/* ---- UTC+8 日历（Howard Hinnant 的 civil_from_days 算法） ---- */
static void civil_from_days(long long z, int *y, int *m, int *d)
{
    z += 719468;
    const long long era = (z >= 0 ? z : z - 146096) / 146097;
    const unsigned long long doe = (unsigned long long)(z - era * 146097);
    const unsigned long long yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    const long long yy = (long long)yoe + era * 400;
    const unsigned long long doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    const unsigned long long mp = (5 * doy + 2) / 153;
    const unsigned long long dd = doy - (153 * mp + 2) / 5 + 1;
    const unsigned long long mm = mp < 10 ? mp + 3 : mp - 9;
    *y = (int)(yy + (mm <= 2));
    *m = (int)mm;
    *d = (int)dd;
}

static const char *k_week[] = {"日", "一", "二", "三", "四", "五", "六"};

void feng_time_format_utc8(long long epoch, char *buf, int buf_sz)
{
    long long t = epoch + 8 * 3600;                  /* UTC+8 */
    long long days = t / 86400;
    long long rem = t % 86400;
    if (rem < 0) { rem += 86400; days -= 1; }
    int y, m, d;
    civil_from_days(days, &y, &m, &d);
    const int hh = (int)(rem / 3600), mm = (int)((rem % 3600) / 60), ss = (int)(rem % 60);
    const int wd = (int)((days % 7 + 11) % 7);       /* 1970-01-01 是周四 */
    snprintf(buf, buf_sz, "%04d年%02d月%02d日 %02d:%02d:%02d（周%s，UTC+8）",
             y, m, d, hh, mm, ss, k_week[wd]);
}

static int contains(const char *s, const char *kw)
{
    return strstr(s, kw) != NULL;
}

int feng_time_answer(const char *user, char *answer, int answer_sz)
{
    static const char *kw[] = {"几点", "现在时间", "现在的时间", "当前时间", "时间是多少",
                               "什么时间", "今天几号", "今天几月", "日期", "星期几", "时间戳"};
    int hit = 0;
    for (int i = 0; i < (int)(sizeof(kw) / sizeof(kw[0])); i++) {
        if (contains(user, kw[i])) { hit = 1; break; }
    }
    if (!hit) return 0;
    const long long now = s_epoch_fn ? s_epoch_fn() : 0;
    if (now <= 0) {
        snprintf(answer, answer_sz, "我还没对上网络时间（宿主连接后会自动发 \\settime）。");
        return 1;
    }
    char t[128];
    feng_time_format_utc8(now, t, sizeof(t));
    snprintf(answer, answer_sz, "现在是 %s。", t);
    return 1;
}

/* xorshift64*：跨平台一致，不依赖 libc rand */
static unsigned long long xs64(unsigned long long *s)
{
    unsigned long long x = *s;
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    *s = x;
    return x * 2685821657736338717ULL;
}

long long feng_rand_range(unsigned long long seed, long long lo, long long hi)
{
    if (seed == 0) seed = 88172645463325252ULL;
    unsigned long long st = seed;
    (void)xs64(&st);                                  /* 第一个随机数：按需求丢弃 */
    const unsigned long long r = xs64(&st);
    if (hi < lo) { const long long t = hi; hi = lo; lo = t; }
    const unsigned long long span = (unsigned long long)(hi - lo + 1);
    return lo + (long long)(r % span);
}

int feng_random_answer(const char *user, char *answer, int answer_sz)
{
    if (!contains(user, "随机")) return 0;
    /* 解析可选范围："1到100" / "1-100" / "0~9"；默认 1..100 */
    long long lo = 1, hi = 100;
    const char *p = user;
    long long nums[2] = {0, 0};
    int nnum = 0;
    while (*p && nnum < 2) {
        if (*p >= '0' && *p <= '9') {
            long long v = 0;
            while (*p >= '0' && *p <= '9') { v = v * 10 + (*p - '0'); p++; }
            nums[nnum++] = v;
        } else {
            p++;                                      /* 逐字节扫 ASCII 数字 */
        }
    }
    if (nnum == 2 && nums[0] != nums[1]) { lo = nums[0]; hi = nums[1]; }
    const long long up_us = s_uptime_fn ? s_uptime_fn() : 0;
    const double sec = (double)up_us / 1e6;
    const unsigned long long seed = (unsigned long long)(sec * 1.54 * 1000.0);
    const long long v = feng_rand_range(seed, lo, hi);
    snprintf(answer, answer_sz, "随机数（%lld~%lld）：%lld。", lo, hi, v);
    return 1;
}
