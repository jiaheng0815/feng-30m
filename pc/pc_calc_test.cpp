/* 计算 tool 单测：识别 + 求值 + 中文句式。 */
#include <stdio.h>
#include <string.h>

#include "feng_calc.h"

static int fails;

static void check_true(const char *what, int ok, const char *detail)
{
    if (!ok) fails++;
    printf("[%s] %-28s %s\n", ok ? "PASS" : "FAIL", what, detail ? detail : "");
}

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
    check("(3+4)*2", "(3+4) 乘 2 等于 14。");
    check("(1+2)", "结果是 3。");
    check("帮我算一下 99+1", "99 加 1 等于 100。");
    check("计算 36/6", "36 除以 6 等于 6。");
    check("1/0", "这个算式里除数是 0，算不出来。");
    /* 中文数字 / 百分号（v3.14 扩展） */
    check("五十九加一", "59 加 1 等于 60。");
    check("一百零五加二十", "105 加 20 等于 125。");
    check("两千五百减三百", "2500 减 300 等于 2200。");
    check("两千三加一百", "2300 加 100 等于 2400。");
    check("二零二六减一", "2026 减 1 等于 2025。");
    check("十五乘以四", "15 乘 4 等于 60。");
    check("一百*15%", "100 乘 15% 等于 15。");
    check("15%+5%", "15% 加 5% 等于 0.2。");
    check("100的15%", "100 乘 15% 等于 15。");
    check("12的平方", "12 乘 12 等于 144。");
    check("三的立方", "结果是 27。");
    check("根号16", "结果是 4。");
    /* 前导空格 + 平方：旧实现按 sizeof(work) 而不是剩余长度写 snprintf，
     * 会越过 work[] 写穿栈（实测 40 字节）。这里同时验证"算得对"和"不越界"。 */
    check("   12的平方", "12 乘 12 等于 144。");
    check("\t5的立方", "结果是 125。");
    {
        /* 放不下（前导空格 + 长算式 + 的平方）：必须整条拒绝，不能输出半截算式，
         * 更不能越界写。旧实现这条会破坏栈上相邻变量。 */
        char big[256];
        int p = 0;
        for (int i = 0; i < 40; i++) big[p++] = ' ';
        for (int i = 0; i < 90; i++) { big[p++] = '1'; big[p++] = '+'; }
        big[p++] = '1';
        memcpy(big + p, "的平方", strlen("的平方") + 1);
        check(big, NULL);
    }
    /* 极大值格式化：|v| ≥ 2^63 时旧实现 (long long)v 是 UB */
    {
        char num[64];
        feng_calc_fmt(1e20, num, sizeof(num));
        check_true("fmt(1e20)", strcmp(num, "100000000000000000000") == 0, num);
        feng_calc_fmt(-1e20, num, sizeof(num));
        check_true("fmt(-1e20)", strcmp(num, "-100000000000000000000") == 0, num);
        feng_calc_fmt(1234.5, num, sizeof(num));
        check_true("fmt(1234.5)", strcmp(num, "1234.5") == 0, num);
        feng_calc_fmt(-0.0, num, sizeof(num));
        check_true("fmt(-0)", strcmp(num, "0") == 0, num);
    }
    check("100的15%是多少钱", "100 乘 15% 等于 15。");
    check("把59+1算一下", "59 加 1 等于 60。");
    check("麻烦算一下 445+15 是多少", "445 加 15 等于 460。");
    /* 序列数数（v3.20 扩展）：倒数/正数交给引擎，模型不再学 */
    check("把 1 到 5 倒着数一遍。", "5、4、3、2、1。");
    check("把 1 到 5 倒着数一遍", "5、4、3、2、1。");
    check("从 3 数到 8。", "3、4、5、6、7、8。");
    check("把十到十五倒着数。", "15、14、13、12、11、10。");
    check("从3数到8", "3、4、5、6、7、8。");
    check("把 8 到 3 倒着数", "8、7、6、5、4、3。");
    check("把 1 到 100 倒着数。", "范围有点大（1 到 100），给我 50 个以内的区间吧。");
    check("从999999900数到999999950", "范围有点大（999999900 到 999999950），输出放不下，换个小一点的区间吧。");
    /* 日期/电话/超长数字不能当算式 */
    check("2026-10-18", NULL);
    check("138-1234-5678", NULL);
    check("从999999999数到1000000049", NULL);
    /* 嵌套过深按语法错误处理，不能打穿板端 8 KB 栈 */
    {
        char deep[256];
        int p = 0;
        deep[p++] = '1'; deep[p++] = '+';
        for (int i = 0; i < 70; i++) deep[p++] = '(';
        deep[p++] = '1';
        for (int i = 0; i < 70; i++) deep[p++] = ')';
        deep[p] = 0;
        check(deep, "这个算式我没看懂。");
    }
    /* 不该被当成算式 */
    check("你好", NULL);
    check("我2-3点有空", NULL);
    check("今天25度", NULL);
    check("加起来", NULL);
    check("我从1数到100也数不完", NULL);      /* 普通陈述不拦 */
    check("数到 8", NULL);                    /* 没给起点，交给模型 */
    check("", NULL);
    /* 中文数字病态输入：累加溢出（UB）与回写越界的回归。
     * 85 个「九」：纯数字串路径 val 溢出 long long；「万」连写：单位展开
     * 后 out 比 s 长，回写会越过 work 末尾。两者都必须"安全地不当算式"。 */
    {
        char huge[512];
        int p = 0;
        for (int i = 0; i < 85; i++) { memcpy(huge + p, "九", 3); p += 3; }
        huge[p] = 0;
        check(huge, NULL);                    /* 纯数字串：没有运算符，拒 */
        p = 0;
        for (int i = 0; i < 84; i++) { memcpy(huge + p, "万", 3); p += 3; }
        huge[p] = 0;
        check(huge, NULL);                    /* 万连写：溢出封顶后是纯数字，拒 */
        p = 0;
        for (int i = 0; i < 3; i++) huge[p++] = ' ';
        for (int i = 0; i < 84; i++) { memcpy(huge + p, "万", 3); p += 3; }
        huge[p] = 0;
        check(huge, NULL);                    /* 前导空格 + 万连写：回写越界回归 */
        /* 正常的中文大数仍然要认（展开很短，放得下） */
        check("三万五千加一万", "35000 加 10000 等于 45000。");
        check("一万万", NULL);                /* 1 万万 = 1e8，纯数字无运算符，拒 */
    }
    /* 截断的 UTF-8 尾字节：不能读过 NUL，也不能崩 */
    {
        char bad[16];
        memcpy(bad, "1+1", 3);
        bad[3] = (char)0xE4;                  /* 三字节首字节后直接结束 */
        bad[4] = 0;
        check(bad, NULL);
        char bad2[16];
        memcpy(bad2, "1+1", 3);
        bad2[3] = (char)0xE4; bad2[4] = (char)0xB8;   /* 两字节就断 */
        bad2[5] = 0;
        check(bad2, NULL);
    }
    printf("\n%s（%d 个失败）\n", fails ? "有失败" : "全部通过", fails);
    return fails ? 1 : 0;
}
