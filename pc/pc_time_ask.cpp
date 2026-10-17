/* 逐行读 stdin（UTF-8）：
 *   @<epoch>  -> 设定板端"当前时间"；其它行 -> feng_time_answer 输出回答或 NONE。
 * 用于 tools/check_time_parity.py 对比 C/Python 时间 tool 的日期/推算输出。 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "feng_tools.h"

static long long g_epoch;
static long long epoch_fn(void) { return g_epoch; }

int main(void)
{
    char line[512];
    char out[256];
    feng_tools_set_time(epoch_fn);
    while (fgets(line, sizeof(line), stdin)) {
        size_t n = strlen(line);
        while (n > 0 && (line[n - 1] == '\n' || line[n - 1] == '\r')) line[--n] = 0;
        if (n == 0) continue;
        if (line[0] == '@') {
            g_epoch = atoll(line + 1);
            printf("EPOCH\n");
            continue;
        }
        if (feng_time_answer(line, out, sizeof(out))) {
            printf("%s\n", out);
        } else {
            printf("NONE\n");
        }
    }
    return 0;
}
