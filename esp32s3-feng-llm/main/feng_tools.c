/* 时间/随机数 tool 实现：不依赖 libc 时区数据库，手算 UTC+8 日历。 */
#include "feng_tools.h"

#include <stdio.h>
#include <stdlib.h>
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

static int contains(const char *s, const char *kw);

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

void feng_time_format_date_utc8(long long epoch, char *buf, int buf_sz)
{
    long long t = epoch + 8 * 3600;
    long long days = t / 86400;
    int y, m, d;
    civil_from_days(days, &y, &m, &d);
    const int wd = (int)((days % 7 + 11) % 7);
    snprintf(buf, buf_sz, "%04d年%02d月%02d日（周%s）", y, m, d, k_week[wd]);
}

/* 从 "3天后" 里取出 3；返回 0 表示没有这种模式 */
static int days_offset(const char *user, int *days_out)
{
    static const struct { const char *kw; int delta; } REL[] = {
        {"明天", 1}, {"后天", 2}, {"昨天", -1}, {"前天", -2},
    };
    for (int i = 0; i < (int)(sizeof(REL) / sizeof(REL[0])); i++) {
        if (contains(user, REL[i].kw)) { *days_out = REL[i].delta; return 1; }
    }
    static const char *kws[] = {"天后", "天前", "天之后", "天之前"};
    for (int i = 0; i < (int)(sizeof(kws) / sizeof(kws[0])); i++) {
        const char *hit = strstr(user, kws[i]);
        if (!hit) continue;
        /* 往前找数字 */
        const char *p = hit;
        while (p > user && p[-1] >= '0' && p[-1] <= '9') p--;
        if (p == hit) continue;
        int v = atoi(p);
        *days_out = (i == 0 || i == 2) ? v : -v;
        return 1;
    }
    return 0;
}

static int contains(const char *s, const char *kw)
{
    return strstr(s, kw) != NULL;
}

/* 时钟推算："现在7点，再过3小时是几点？" / "3小时后是几点？"（有"现在X点"用 X，否则用板端当前时间）
 * 返回 1 = 已作答。 */
static int time_math_answer(const char *user, char *answer, int answer_sz)
{
    const char *rel = strstr(user, "小时");
    if (!rel) return 0;
    const char *tail = rel + 6;                    /* "小时" 的 UTF-8 长度 */
    if (strncmp(tail, "候", 3) == 0) return 0;     /* "小时候" 不是时钟推算 */
    const int sign = (strncmp(tail, "前", 3) == 0) ? -1 : 1;
    /* "小时"前面紧邻的数字串（支持"再多3小时"/"3 小时"） */
    const char *p = rel;
    while (p > user && (p[-1] == ' ' || p[-1] == '\t')) p--;
    const char *num_end = p;
    while (p > user && p[-1] >= '0' && p[-1] <= '9') p--;
    if (p == num_end) return 0;                   /* 没有数字，不当作时钟推算 */
    const int delta = atoi(p) * sign;
    /* 起点小时：优先"现在X点"，否则用板端当前时间 */
    int base_hour = -1;
    const char *cur = strstr(user, "现在");
    if (cur) {
        const char *q = cur + 6;                   /* "现在" UTF-8 长度 */
        int guard = 0;
        while (*q && (q[0] < '0' || q[0] > '9') && guard++ < 12) q++;
        if (q[0] >= '0' && q[0] <= '9') {
            const int h = atoi(q);
            if (h >= 0 && h <= 23) base_hour = h;
        }
    }
    if (base_hour < 0) {
        const long long now = s_epoch_fn ? s_epoch_fn() : 0;
        if (now <= 0) return 0;
        base_hour = (int)(((now + 8 * 3600) / 3600) % 24);
    }
    int total = base_hour + delta;
    int day = 0;
    while (total < 0) { total += 24; day--; }
    while (total >= 24) { total -= 24; day++; }
    if (day == 0) snprintf(answer, answer_sz, "再过 %d 小时是 %d 点。", delta, total);
    else if (day == 1) snprintf(answer, answer_sz, "再过 %d 小时是明天 %d 点。", delta, total);
    else if (day == -1) snprintf(answer, answer_sz, "%d 小时前是昨天 %d 点。", -delta, total);
    else if (day == 2) snprintf(answer, answer_sz, "再过 %d 小时是后天 %d 点。", delta, total);
    else snprintf(answer, answer_sz, "再过 %d 小时是 %d 天后 %d 点。", delta, day, total);
    return 1;
}

int feng_time_answer(const char *user, char *answer, int answer_sz)
{
    if (time_math_answer(user, answer, answer_sz)) return 1;
    static const char *kw[] = {"几点", "现在时间", "现在的时间", "当前时间", "时间是多少",
                               "什么时间", "今天几号", "今天几月", "几号", "几月", "日期",
                               "星期几", "时间戳", "明天", "后天", "昨天", "前天"};
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
    int days = 0;
    if (days_offset(user, &days)) {
        char d[96], t[128];
        feng_time_format_date_utc8(now + (long long)days * 86400, d, sizeof(d));
        feng_time_format_utc8(now, t, sizeof(t));
        if (days > 0) snprintf(answer, answer_sz, "%d 天后是 %s。", days, d);
        else if (days < 0) snprintf(answer, answer_sz, "%d 天前是 %s。", -days, d);
        else snprintf(answer, answer_sz, "今天是 %s。", d);
        (void)t;
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
    const int coin = contains(user, "硬币") || contains(user, "正反面");
    const int dice = contains(user, "骰子") || contains(user, "色子");
    if (!coin && !dice && !contains(user, "随机")) return 0;
    const long long up_us0 = s_uptime_fn ? s_uptime_fn() : 0;
    const double sec0 = (double)up_us0 / 1e6;
    const unsigned long long seed0 = (unsigned long long)(sec0 * 1.54 * 1000.0);
    if (coin) {
        const long long v = feng_rand_range(seed0, 0, 1);
        snprintf(answer, answer_sz, "抛硬币：%s。", v ? "正面" : "反面");
        return 1;
    }
    /* 解析可选范围："1到100" / "1-100" / "0~9"；默认 1..100（骰子默认 1..6） */
    long long lo = 1, hi = dice ? 6 : 100;
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
    const long long v = feng_rand_range(seed0, lo, hi);
    if (dice) snprintf(answer, answer_sz, "掷骰子：%lld 点。", v);
    else snprintf(answer, answer_sz, "随机数（%lld~%lld）：%lld。", lo, hi, v);
    return 1;
}
