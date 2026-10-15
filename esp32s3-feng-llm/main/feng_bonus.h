/* feng_bonus.h —— 一个藏起来的小彩蛋（见 feng_bonus.c 文件头）。
 *
 * 默认编译进来；`idf.py -DFENG_NO_BONUS=ON build` 可以关掉。
 * README 的小字里提了这件事，但没有说触发方式。 */
#ifndef FENG_BONUS_H
#define FENG_BONUS_H

#ifdef __cplusplus
extern "C" {
#endif

/* 命中隐藏命令就把明文解进 out（容量至少 19 字节）并返回 1；
 * 否则返回 0，交回调用方的命令链。打印由调用方做，以复用它的编码处理。 */
int feng_bonus_decode(const char *line, char *out, int cap);

#ifdef __cplusplus
}
#endif

#endif /* FENG_BONUS_H */
