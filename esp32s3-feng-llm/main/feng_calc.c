/* 计算 tool 实现：纯 C、无动态内存，ESP32-S3 的 FPU 直接算。 */
#include "feng_calc.h"

#include <math.h>
#include <stdio.h>
#include <string.h>

#define CALC_MAX 256

static int is_space(char c)
{
    return c == ' ' || c == '\t' || c == '\r' || c == '\n';
}

static int is_digit(char c)
{
    return c >= '0' && c <= '9';
}

/* 把 UTF-8 的中文运算符/前后缀替换成 ASCII 形式；返回写出的长度 */
static int normalize_ops(const char *in, char *out, int sz)
{
    int n = 0;
    const unsigned char *p = (const unsigned char *)in;
    while (*p && n < sz - 1) {
        if (p[0] == 0xE5 && p[1] == 0x8A && p[2] == 0xA0) {          /* 加 */
            out[n++] = '+'; p += 3; continue;
        }
        if (p[0] == 0xE5 && p[1] == 0x87 && p[2] == 0x8F) {          /* 减 */
            out[n++] = '-'; p += 3; continue;
        }
        if (p[0] == 0xE4 && p[1] == 0xB9 && p[2] == 0x98) {          /* 乘 */
            out[n++] = '*'; p += 3; continue;
        }
        if (p[0] == 0xE9 && p[1] == 0x99 && p[2] == 0xA4) {          /* 除 */
            out[n++] = '/'; p += 3; continue;
        }
        if (p[0] == 0xC3 && p[1] == 0x97) { out[n++] = '*'; p += 2; continue; }   /* × */
        if (p[0] == 0xC3 && p[1] == 0xB7) { out[n++] = '/'; p += 2; continue; }   /* ÷ */
        if (p[0] == 0xEF && p[1] == 0xBC && p[2] == 0x88) { out[n++] = '('; p += 3; continue; } /* （ */
        if (p[0] == 0xEF && p[1] == 0xBC && p[2] == 0x89) { out[n++] = ')'; p += 3; continue; } /* ） */
        out[n++] = (char)*p++;
    }
    out[n] = 0;
    return n;
}

/* 后缀/前缀外壳：返回 1 表示剥掉了一层 */
static int strip_suffix(char *s)
{
    static const char *suf[] = {"等于几", "等于多少", "是多少", "多少", "=?", "＝?", "?", "？",
                                "。", ".", "!", "！", "=", "＝"};
    const int n = (int)strlen(s);
    for (int i = 0; i < (int)(sizeof(suf) / sizeof(suf[0])); i++) {
        const int L = (int)strlen(suf[i]);
        if (n >= L && strcmp(s + n - L, suf[i]) == 0) {
            s[n - L] = 0;
            return 1;
        }
    }
    return 0;
}

static int strip_prefix(char *s)
{
    static const char *pre[] = {"帮我算一下", "帮我计算", "帮我算", "计算一下", "计算",
                                "算一下", "算算", "请问一下", "请问", "求"};
    for (int i = 0; i < (int)(sizeof(pre) / sizeof(pre[0])); i++) {
        const int L = (int)strlen(pre[i]);
        if (strncmp(s, pre[i], L) == 0) {
            memmove(s, s + L, strlen(s + L) + 1);
            return 1;
        }
    }
    return 0;
}

int feng_calc_extract(const char *user, char *expr, int expr_sz)
{
    char work[CALC_MAX];
    normalize_ops(user, work, sizeof(work));
    /* 去掉首尾空白 */
    char *s = work;
    while (is_space(*s)) s++;
    int n = (int)strlen(s);
    while (n > 0 && is_space(s[n - 1])) s[--n] = 0;
    /* 剥外壳（可能叠加：先前后缀各来一轮） */
    for (int round = 0; round < 4; round++) {
        int changed = 0;
        while (n > 0 && is_space(s[n - 1])) s[--n] = 0;
        if (strip_suffix(s)) changed = 1;
        n = (int)strlen(s);
        while (n > 0 && is_space(s[n - 1])) s[--n] = 0;
        if (strip_prefix(s)) changed = 1;
        n = (int)strlen(s);
        while (n > 0 && is_space(s[n - 1])) s[--n] = 0;
        if (!changed) break;
    }
    /* 校验：只允许数字/小数点/运算符/括号/空格，且至少一个数字与一个运算符 */
    int has_digit = 0, has_op = 0;
    for (const char *p = s; *p; p++) {
        if (is_digit(*p)) has_digit = 1;
        else if (strchr("+-*/()", *p)) has_op = (*p == '+' || *p == '-' || *p == '*' || *p == '/');
        else if (*p == '.' || is_space(*p)) continue;
        else return 0;
    }
    if (!has_digit || !has_op) return 0;
    /* 去掉空格，输出紧凑算式 */
    int m = 0;
    for (const char *p = s; *p && m < expr_sz - 1; p++) {
        if (!is_space(*p)) expr[m++] = *p;
    }
    expr[m] = 0;
    return m > 0;
}

/* ---- 递归下降求值 ---- */
typedef struct {
    const char *p;
    int err;                        /* 0 ok, -1 div0, -2 syntax */
} calc_ctx;

static double parse_expr(calc_ctx *c);

static void skip_ws(calc_ctx *c)
{
    while (is_space(*c->p)) c->p++;
}

static double parse_number(calc_ctx *c)
{
    skip_ws(c);
    double v = 0;
    int any = 0;
    while (is_digit(*c->p)) { v = v * 10 + (*c->p - '0'); c->p++; any = 1; }
    if (*c->p == '.') {
        c->p++;
        double scale = 0.1;
        while (is_digit(*c->p)) { v += (*c->p - '0') * scale; scale *= 0.1; c->p++; any = 1; }
    }
    if (!any) c->err = -2;
    return v;
}

static double parse_factor(calc_ctx *c)
{
    skip_ws(c);
    if (*c->p == '-') { c->p++; return -parse_factor(c); }
    if (*c->p == '+') { c->p++; return parse_factor(c); }
    if (*c->p == '(') {
        c->p++;
        double v = parse_expr(c);
        skip_ws(c);
        if (*c->p == ')') c->p++;
        else c->err = -2;
        return v;
    }
    return parse_number(c);
}

static double parse_term(calc_ctx *c)
{
    double v = parse_factor(c);
    for (;;) {
        skip_ws(c);
        const char op = *c->p;
        if (op != '*' && op != '/') break;
        c->p++;
        const double r = parse_factor(c);
        if (c->err) return v;
        if (op == '*') {
            v *= r;
        } else {
            if (r == 0.0) { c->err = -1; return v; }
            v /= r;
        }
    }
    return v;
}

static double parse_expr(calc_ctx *c)
{
    double v = parse_term(c);
    for (;;) {
        skip_ws(c);
        const char op = *c->p;
        if (op != '+' && op != '-') break;
        c->p++;
        const double r = parse_term(c);
        if (c->err) return v;
        v = (op == '+') ? v + r : v - r;
    }
    return v;
}

int feng_calc_eval(const char *expr, double *out)
{
    calc_ctx c = {expr, 0};
    const double v = parse_expr(&c);
    skip_ws(&c);
    if (c.err) return c.err;
    if (*c.p) return -2;
    *out = v;
    return 1;
}

void feng_calc_fmt(double v, char *buf, int buf_sz)
{
    if (v == 0) v = 0;                                   /* 去掉 -0 */
    if (fabs(v - (double)(long long)v) < 1e-9 && fabs(v) < 1e15) {
        snprintf(buf, buf_sz, "%lld", (long long)v);
        return;
    }
    snprintf(buf, buf_sz, "%.4f", v);
    int n = (int)strlen(buf);
    while (n > 0 && buf[n - 1] == '0') buf[--n] = 0;
    if (n > 0 && buf[n - 1] == '.') buf[--n] = 0;
}

/* 把 "59+1" 拆成 a、op、b（用于生成 "59 加 1 等于 60。" 句式） */
static int split_binary(const char *expr, char *a, int asz, char *op, char *b, int bsz)
{
    int depth = 0;
    for (int i = 1; expr[i]; i++) {
        if (expr[i] == '(') depth++;
        else if (expr[i] == ')') depth--;
        else if (depth == 0 && (expr[i] == '+' || expr[i] == '-' || expr[i] == '*' || expr[i] == '/')) {
            /* 只接受"最外层唯一运算符"的最简形式 */
            if (strchr(expr + i + 1, '+') || strchr(expr + i + 1, '-') ||
                strchr(expr + i + 1, '*') || strchr(expr + i + 1, '/')) return 0;
            if (i >= asz || (int)strlen(expr + i + 1) >= bsz) return 0;
            memcpy(a, expr, (size_t)i); a[i] = 0;
            *op = expr[i];
            strcpy(b, expr + i + 1);
            return 1;
        }
    }
    return 0;
}

int feng_calc_answer(const char *user, char *answer, int answer_sz)
{
    char expr[CALC_MAX];
    if (!feng_calc_extract(user, expr, sizeof(expr))) return 0;
    double v = 0;
    const int rc = feng_calc_eval(expr, &v);
    if (rc < 0) {
        snprintf(answer, answer_sz, "%s", rc == -1 ? "这个算式里除数是 0，算不出来。" : "这个算式我没看懂。");
        return 1;
    }
    char a[64], b[64], op = 0;
    char num[CALC_MAX];
    feng_calc_fmt(v, num, sizeof(num));
    if (split_binary(expr, a, sizeof(a), &op, b, sizeof(b))) {
        const char *sym = op == '+' ? "加" : op == '-' ? "减" : op == '*' ? "乘" : "除以";
        snprintf(answer, answer_sz, "%s %s %s 等于 %s。", a, sym, b, num);
    } else {
        snprintf(answer, answer_sz, "结果是 %s。", num);
    }
    return 1;
}
