/* 逐行读 stdin（UTF-8），复刻固件的记忆 tool 处理顺序：
 *   \clear -> 清空；\mem -> 快照；其他行先 learn 再 answer（未命中输出 NONE）。
 * 用于 tools/check_mem_parity.py 做 C/Python 记忆 tool 一致性对比。 */
#include <stdio.h>
#include <string.h>

#include "feng_memory.h"

int main(void)
{
    char line[1024];
    char out[512];
    while (fgets(line, sizeof(line), stdin)) {
        size_t n = strlen(line);
        while (n > 0 && (line[n - 1] == '\n' || line[n - 1] == '\r')) line[--n] = 0;
        if (n == 0) continue;
        if (strcmp(line, "\\clear") == 0) {
            feng_mem_clear();
            printf("CLEARED\n");
            continue;
        }
        if (strcmp(line, "\\mem") == 0) {
            feng_mem_snapshot(out, sizeof(out));
            printf("%s\n", out);
            continue;
        }
        feng_mem_learn(line);
        if (feng_mem_answer(line, out, sizeof(out))) {
            printf("%s\n", out);
        } else {
            printf("NONE\n");
        }
    }
    return 0;
}
