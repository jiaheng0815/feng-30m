/* Host check of the serial encoding helpers (main/gbk.c + main/gbk_table.c). */
#include <stdio.h>
#include <string.h>

#include "gbk.h"

static void hex(const char *s, int n)
{
    for (int i = 0; i < n; i++) printf("%02X ", (unsigned char)s[i]);
}

int main(void)
{
    const char *zh = "你是谁？good";
    const char *expect = "\xC4\xE3\xCA\xC7\xCB\xAD\xA3\xBF" "good";   /* 你是谁？ in GBK */
    char gb[64], back[128];
    const int gl = feng_utf8_to_gbk(zh, (int)strlen(zh), gb, sizeof(gb));
    gb[gl] = 0;
    printf("utf8 -> gbk : %s  [%s]\n", gb, memcmp(gb, expect, 9) == 0 ? "MATCH" : "MISMATCH");
    hex(gb, gl);
    printf("\nexpected    : ");
    hex(expect, 9);
    printf("\n");
    const int bl = feng_gbk_to_utf8(gb, gl, back, sizeof(back) - 1);
    back[bl] = 0;
    printf("gbk  -> utf8: %s  [%s]\n", back, strcmp(back, zh) == 0 ? "ROUNDTRIP OK" : "ROUNDTRIP FAIL");
    printf("utf8_valid(utf8)=%d  utf8_valid(gbk)=%d  has_high=%d\n",
           feng_utf8_valid(zh, (int)strlen(zh)), feng_utf8_valid(gb, gl),
           feng_has_high_byte(gb, gl));
    return memcmp(gb, expect, 9) == 0 && strcmp(back, zh) == 0 ? 0 : 1;
}
