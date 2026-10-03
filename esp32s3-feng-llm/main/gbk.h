#ifndef FENG_GBK_H
#define FENG_GBK_H

/* Serial text encoding helpers.
 * The board mirrors the encoding of whatever the user types: GBK terminals
 * (SuperCom/XCOM in ANSI mode) send GBK bytes, UTF-8 terminals send UTF-8.
 */
extern int g_out_gbk;   /* 1 = reply in GBK, 0 = reply in UTF-8 */

int feng_utf8_valid(const char *s, int n);
int feng_utf8_to_gbk(const char *in, int n, char *out, int out_max);
int feng_gbk_to_utf8(const char *in, int n, char *out, int out_max);
int feng_has_high_byte(const char *s, int n);

#endif
