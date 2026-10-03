#ifndef FENG_GBK_TABLE_H
#define FENG_GBK_TABLE_H

#include <stdint.h>

extern const int g_gbk_count;
extern const uint32_t g_gbk_u2g_cp[];   /* unicode codepoints, ascending */
extern const uint16_t g_gbk_u2g_val[];  /* matching GBK codes */
extern const uint16_t g_gbk_g2u_val[];  /* GBK codes, ascending */
extern const uint32_t g_gbk_g2u_cp[];   /* matching unicode codepoints */

#endif
