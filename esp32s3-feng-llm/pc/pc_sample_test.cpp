/* 共享采样器单测：重复惩罚 + no-repeat 3-gram（板端与 PC 引擎同口径）。 */
#include <stdio.h>
#include <string.h>

#include "feng.h"

static int fails;

static void expect_tok(const char *name, int got, int want)
{
    const int ok = got == want;
    if (!ok) fails++;
    printf("[%s] %-34s got %d want %d\n", ok ? "PASS" : "FAIL", name, got, want);
}

int main(void)
{
    float lg[16];

    /* 1) no-repeat 3-gram：历史 [1,2,3,1,2,3,1,2]，tail=[1,2]，
     *    补全 token=3 已出现过 -> 应被禁，选次高 7。 */
    memset(lg, 0, sizeof(lg));
    lg[3] = 5.0f;
    lg[7] = 4.0f;
    const int hist1[8] = {1, 2, 3, 1, 2, 3, 1, 2};
    expect_tok("3-gram 循环被禁", feng_sample_greedy({lg, 16}, {hist1, 8}, 1.15f, 3), 7);

    /* 2) 没有历史时不受影响：还是选最高 */
    memset(lg, 0, sizeof(lg));
    lg[3] = 5.0f;
    lg[7] = 4.0f;
    expect_tok("无历史不干预", feng_sample_greedy({lg, 16}, {}, 1.15f, 3), 3);

    /* 3) 重复惩罚：token 5 被历史用过一次，1.0/1.15=0.8696 < 0.9 -> 选 6 */
    memset(lg, 0, sizeof(lg));
    lg[5] = 1.0f;
    lg[6] = 0.9f;
    const int hist2[1] = {5};
    expect_tok("重复惩罚生效", feng_sample_greedy({lg, 16}, {hist2, 1}, 1.15f, 3), 6);

    /* 4) no_repeat_n=0 时只做惩罚，不禁 n-gram */
    memset(lg, 0, sizeof(lg));
    lg[3] = 5.0f;
    lg[7] = 4.0f;
    const int hist3[6] = {1, 2, 3, 1, 2, 3};
    expect_tok("关闭 n-gram 约束", feng_sample_greedy({lg, 16}, {hist3, 6}, 1.0f, 0), 3);

    printf("\n%s（%d 个失败）\n", fails ? "有失败" : "全部通过", fails);
    return fails ? 1 : 0;
}
