/* 计算 tool 单测：识别 + 求值 + 中文句式。 */
#include <stdio.h>
#include <string.h>

#include "feng_calc.h"

static int fails;

static void check(const char *q, const char *want)
{
    char got[256];
    const int hit = feng_calc_answer(q, got, sizeof(got));
    const int ok = want ? (hit == 1 && strcmp(got, want) == 0) : (hit == 0);
    if (!ok) fails++;
    printf("[%s] %-28s -> %s%s%s\n", ok ? "PASS" : "FAIL", q,
           hit ? got : "(不是算式)",
           want && !ok ? "   期望: " : "",
           want && !ok ? want : "");
}

int main(void)
{
    /* 用户实测过的用例 */
    check("59+1", "59 加 1 等于 60。");
    check("445+15", "445 加 15 等于 460。");
    check("84+6", "84 加 6 等于 90。");
    check("10+4.", "10 加 4 等于 14。");
    check("5.3+4.1", "5.3 加 4.1 等于 9.4。");
    check("59+1等于几？", "59 加 1 等于 60。");
    check("7减7等于几？", "7 减 7 等于 0。");
    check("1减4等于几？", "1 减 4 等于 -3。");
    check("12×8", "12 乘 8 等于 96。");
    check("100 ÷ 4", "100 除以 4 等于 25。");
    check("(3+4)*2", "结果是 14。");
    check("帮我算一下 99+1", "99 加 1 等于 100。");
    check("计算 36/6", "36 除以 6 等于 6。");
    check("1/0", "这个算式里除数是 0，算不出来。");
    /* 中文数字 / 百分号（v3.14 扩展） */
    check("五十九加一", "59 加 1 等于 60。");
    check("一百零五加二十", "105 加 20 等于 125。");
    check("两千五百减三百", "2500 减 300 等于 2200。");
    check("十五乘以四", "15 乘 4 等于 60。");
    check("一百*15%", "100 乘 15% 等于 15。");
    check("15%+5%", "15% 加 5% 等于 0.2。");
    check("100的15%", "100 乘 15% 等于 15。");
    check("12的平方", "12 乘 12 等于 144。");
    check("三的立方", "结果是 27。");
    check("根号16", "结果是 4。");
    check("100的15%是多少钱", "100 乘 15% 等于 15。");
    check("把59+1算一下", "59 加 1 等于 60。");
    check("麻烦算一下 445+15 是多少", "445 加 15 等于 460。");
    /* 不该被当成算式 */
    check("你好", NULL);
    check("我2-3点有空", NULL);
    check("今天25度", NULL);
    check("加起来", NULL);
    check("", NULL);
    printf("\n%s（%d 个失败）\n", fails ? "有失败" : "全部通过", fails);
    return fails ? 1 : 0;
}
