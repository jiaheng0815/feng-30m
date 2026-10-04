/* 时间/随机数 tool 单测：
 *  - UTC+8 日历转换对照 Python datetime 的结果（含闰日、跨年、世纪边界）
 *  - 问句路由（几点/几号/N 天后…）与"还没对时"提示
 *  - 随机数语义：seed = 运行时间(秒)×1.54×1000，xs64* 第 1 个丢弃、第 2 个使用
 *
 * 编译：
 *   gcc -O2 -o pc/pc_tools_test.exe pc/pc_tools_test.c ../main/feng_tools.c -I../main -lm
 */
#include <stdio.h>
#include <string.h>

#include "feng_tools.h"

static int fails;
static long long g_epoch;
static long long g_uptime_us;

static long long fake_epoch(void) { return g_epoch; }
static long long fake_uptime(void) { return g_uptime_us; }

static void check_str(const char *what, const char *got, const char *want)
{
    const int ok = got && strcmp(got, want) == 0;
    if (!ok) fails++;
    printf("[%s] %-22s -> %s\n", ok ? "PASS" : "FAIL", what, got ? got : "(null)");
    if (!ok) printf("     %-25s 期望: %s\n", "", want);
}

static void check_true(const char *what, int ok, const char *detail)
{
    if (!ok) fails++;
    printf("[%s] %-22s %s\n", ok ? "PASS" : "FAIL", what, detail ? detail : "");
}

/* 独立复刻 xorshift64*（纯函数，不含丢弃逻辑），用来验证"第 2 个随机数" */
static unsigned long long ref_xs64(unsigned long long *s)
{
    unsigned long long x = *s;
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    *s = x;
    return x * 2685821657736338717ULL;
}

static unsigned long long ref_first(unsigned long long seed)
{
    unsigned long long st = seed ? seed : 88172645463325252ULL;
    return ref_xs64(&st);
}

static unsigned long long ref_second(unsigned long long seed)
{
    unsigned long long st = seed ? seed : 88172645463325252ULL;
    (void)ref_xs64(&st);
    return ref_xs64(&st);
}

int main(void)
{
    char buf[192], want[192], detail[160];
    feng_tools_set_time(fake_epoch);
    feng_tools_set_uptime(fake_uptime);

    /* --- UTC+8 日历（期望值来自 Python 3 datetime + timezone(+8)） --- */
    feng_time_format_utc8(0LL, buf, sizeof(buf));
    check_str("epoch 0", buf, "1970年01月01日 08:00:00（周四，UTC+8）");
    feng_time_format_utc8(1709222399LL, buf, sizeof(buf));
    check_str("闰日 2024-02-29", buf, "2024年02月29日 23:59:59（周四，UTC+8）");
    feng_time_format_utc8(1735675200LL, buf, sizeof(buf));
    check_str("跨年 UTC→+8", buf, "2025年01月01日 04:00:00（周三，UTC+8）");
    feng_time_format_utc8(946656001LL, buf, sizeof(buf));
    check_str("世纪边界 1999→2000", buf, "2000年01月01日 00:00:01（周六，UTC+8）");
    feng_time_format_date_utc8(1791095400LL, buf, sizeof(buf));
    check_str("日期（N 天后用）", buf, "2026年10月04日（周日）");

    /* --- 问句路由 --- */
    g_epoch = 0;
    check_true("未对时提示", feng_time_answer("现在几点？", buf, sizeof(buf)) == 1 &&
               strcmp(buf, "我还没对上网络时间（宿主连接后会自动发 \\settime）。") == 0, buf);
    g_epoch = 1791095400LL;                  /* 2026-10-04 06:30 UTC = 14:30 UTC+8 */
    check_true("现在几点", feng_time_answer("现在几点？", buf, sizeof(buf)) == 1 &&
               strcmp(buf, "现在是 2026年10月04日 14:30:00（周日，UTC+8）。") == 0, buf);
    check_true("3天后是几号", feng_time_answer("3天后是几号", buf, sizeof(buf)) == 1 &&
               strcmp(buf, "3 天后是 2026年10月07日（周三）。") == 0, buf);
    check_true("明天是几号", feng_time_answer("明天是几号", buf, sizeof(buf)) == 1 &&
               strcmp(buf, "1 天后是 2026年10月05日（周一）。") == 0, buf);
    check_true("昨天是几号", feng_time_answer("昨天是几号", buf, sizeof(buf)) == 1 &&
               strcmp(buf, "1 天前是 2026年10月03日（周六）。") == 0, buf);
    check_true("非时间问句不拦", feng_time_answer("讲个笑话", buf, sizeof(buf)) == 0, "");

    /* --- 随机数：确定性 + 范围 + 第 2 个抽取 --- */
    {
        static const unsigned long long seeds[] = {1ULL, 1540ULL, 88172645463325252ULL,
                                                   1791095400123ULL, 0xDEADBEEFULL};
        static const long long ranges[][2] = {{1, 100}, {0, 9}, {1, 6}, {-5, 5}, {7, 7}, {100, 1}};
        for (unsigned i = 0; i < sizeof(seeds) / sizeof(seeds[0]); i++) {
            for (unsigned j = 0; j < sizeof(ranges) / sizeof(ranges[0]); j++) {
                const long long lo = ranges[j][0], hi = ranges[j][1];
                const long long wlo = lo < hi ? lo : hi, whi = lo < hi ? hi : lo;
                const long long got = feng_rand_range(seeds[i], lo, hi);
                const long long again = feng_rand_range(seeds[i], lo, hi);
                const long long exp = wlo + (long long)(ref_second(seeds[i]) %
                                                        (unsigned long long)(whi - wlo + 1));
                snprintf(detail, sizeof(detail), "seed=%llu 范围[%lld,%lld] -> %lld（期望 %lld）",
                         seeds[i], lo, hi, got, exp);
                check_true("第2抽取+范围", got == again && got == exp, detail);
            }
        }
        /* 找到一个"第 1 个与第 2 个结果不同"的 seed，确认实现真的丢弃了第一个 */
        unsigned long long found = 0;
        for (unsigned long long s = 1; s < 100000ULL; s++) {
            if (ref_first(s) % 100ULL != ref_second(s) % 100ULL) { found = s; break; }
        }
        snprintf(detail, sizeof(detail),
                 "seed=%llu：第1个%%100=%llu 第2个%%100=%llu", found,
                 ref_first(found) % 100ULL, ref_second(found) % 100ULL);
        check_true("第一个确实被丢弃", found != 0 &&
                   feng_rand_range(found, 1, 100) == 1 + (long long)(ref_second(found) % 100ULL) &&
                   feng_rand_range(found, 1, 100) != 1 + (long long)(ref_first(found) % 100ULL),
                   detail);
    }

    /* --- 随机数 tool 的 seed 口径：运行时间(秒) × 1.54 × 1000 --- */
    g_uptime_us = 1000000LL;                 /* 1.000 s -> seed 1540 */
    feng_random_answer("给我个1到100的随机数", buf, sizeof(buf));
    snprintf(want, sizeof(want), "随机数（1~100）：%lld。",
             1 + (long long)(ref_second(1540ULL) % 100ULL));
    check_str("随机数(1~100)@1s", buf, want);
    feng_random_answer("随机 0-9", buf, sizeof(buf));
    snprintf(want, sizeof(want), "随机数（0~9）：%lld。",
             (long long)(ref_second(1540ULL) % 10ULL));
    check_str("随机数(0~9)@1s", buf, want);
    feng_random_answer("掷骰子", buf, sizeof(buf));
    snprintf(want, sizeof(want), "掷骰子：%lld 点。",
             1 + (long long)(ref_second(1540ULL) % 6ULL));
    check_str("掷骰子@1s", buf, want);
    feng_random_answer("抛硬币", buf, sizeof(buf));
    snprintf(want, sizeof(want), "抛硬币：%s。",
             (ref_second(1540ULL) % 2ULL) ? "正面" : "反面");
    check_str("抛硬币@1s", buf, want);
    g_uptime_us = 2500000LL;                 /* 2.500 s -> seed 3850 */
    feng_random_answer("给我个1到100的随机数", buf, sizeof(buf));
    snprintf(want, sizeof(want), "随机数（1~100）：%lld。",
             1 + (long long)(ref_second(3850ULL) % 100ULL));
    check_str("随机数@2.5s(seed 变)", buf, want);
    check_true("非随机问句不拦", feng_random_answer("讲个笑话", buf, sizeof(buf)) == 0, "");

    printf("\n%s（%d 个失败）\n", fails ? "有失败" : "全部通过", fails);
    return fails ? 1 : 0;
}
