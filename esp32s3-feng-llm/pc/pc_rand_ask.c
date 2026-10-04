/* 读 stdin 的 "seed lo hi" 三元组，输出 feng_rand_range 结果（每行一个）。
 * 用于 tools/check_rand_parity.py 对比 C/Python 随机数算法是否逐值一致。 */
#include <stdio.h>

#include "feng_tools.h"

int main(void)
{
    unsigned long long seed;
    long long lo, hi;
    while (scanf("%llu %lld %lld", &seed, &lo, &hi) == 3) {
        printf("%lld\n", feng_rand_range(seed, lo, hi));
    }
    return 0;
}
