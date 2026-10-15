/* feng_bonus.c —— 藏在固件里的一个小彩蛋。
 *
 * 设计意图是「有味道但找不到」：
 *   - 明文 **既不在源码里、也不在编译出的二进制里**，所以 `grep -r` 和 `strings`
 *     都搜不到它（用 XOR 混淆存着，只在命中时才在栈上解出来）；
 *   - `\help` **不列**这条命令；
 *   - 唯一的线索在 README 的小字里——那里提到有个 `nobonus` 编译选项，
 *     于是你知道"有东西"，但得自己把命令猜出来。
 *
 * 关掉它：`idf.py -DFENG_NO_BONUS=ON build`（那时连这段混淆数据都不会进二进制）。
 *
 * C23 约定照旧：零堆分配、零全局构造、纯栈上解码，不引入任何运行时初始化。
 * 打印交给调用方（走 main.c 的 out_printf），这样 GBK/UTF-8 自动跟随仍然生效。
 */
#include "feng_bonus.h"

#include <string.h>

#if !defined(FENG_NO_BONUS)

/* 18 字节 UTF-8，每字节 XOR 0x5A。明文不出现在这里。 */
static const unsigned char kObf[18] = {
    0xBF, 0xD7, 0xD9, 0xBC, 0xDB, 0xD1, 0xBE, 0xE2, 0xDD,
    0xB2, 0xD0, 0xEB, 0xBC, 0xC6, 0xDA, 0xBF, 0xE6, 0xE0,
};
#define FENG_BONUS_KEY 0x5A
#define FENG_BONUS_LEN 18

/* 触发词同样不明写：比对前先就地 XOR，省得 `strings` 里直接出现 "\bonus"。 */
static int cmd_is_bonus(const char *line)
{
    static const unsigned char kCmd[6] = {
        '\\' ^ 0x33, 'b' ^ 0x33, 'o' ^ 0x33, 'n' ^ 0x33, 'u' ^ 0x33, 's' ^ 0x33
    };
    if (line == NULL || line[0] != '\\') return 0;
    for (int i = 0; i < 6; i++) {
        if (line[i] == 0) return 0;
        if ((unsigned char)(line[i] ^ 0x33) != kCmd[i]) return 0;
    }
    /* 命令后只允许空白或结束 */
    const char c = line[6];
    return c == 0 || c == ' ' || c == '\n' || c == '\r';
}

int feng_bonus_decode(const char *line, char *out, int cap)
{
    if (out == NULL || cap <= FENG_BONUS_LEN || !cmd_is_bonus(line)) return 0;
    /* 只有命中这一条命令时才会算出这段文字——平时它在二进制里只是一串无意义的字节 */
    for (int i = 0; i < FENG_BONUS_LEN; i++) out[i] = (char)(kObf[i] ^ FENG_BONUS_KEY);
    out[FENG_BONUS_LEN] = 0;
    return 1;
}

#else /* FENG_NO_BONUS */

int feng_bonus_decode(const char *line, char *out, int cap)
{
    (void)line; (void)out; (void)cap;
    return 0;
}

#endif /* !FENG_NO_BONUS */
