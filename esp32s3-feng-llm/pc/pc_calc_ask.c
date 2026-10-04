/* 逐行读 stdin（UTF-8），对每行调用 feng_calc_answer：
 * 命中输出回答，否则输出 NONE。用于 tools/check_tool_parity.py 做 C/Python 一致性对比。 */
#include <stdio.h>
#include <string.h>

#include "feng_calc.h"

int main(void)
{
    char line[512];
    char out[256];
    while (fgets(line, sizeof(line), stdin)) {
        size_t n = strlen(line);
        while (n > 0 && (line[n - 1] == '\n' || line[n - 1] == '\r')) line[--n] = 0;
        if (n == 0) continue;
        if (feng_calc_answer(line, out, sizeof(out))) {
            printf("%s\n", out);
        } else {
            printf("NONE\n");
        }
    }
    return 0;
}
