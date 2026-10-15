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
        /* 双字算符优先（乘以 / 除以 / 加上 / 减去） */
        if (p[0] == 0xE4 && p[1] == 0xB9 && p[2] == 0x98 && p[3] == 0xE4 && p[4] == 0xBB && p[5] == 0xA5) { out[n++] = '*'; p += 6; continue; } /* 乘以 */
        if (p[0] == 0xE9 && p[1] == 0x99 && p[2] == 0xA4 && p[3] == 0xE4 && p[4] == 0xBB && p[5] == 0xA5) { out[n++] = '/'; p += 6; continue; } /* 除以 */
        if (p[0] == 0xE5 && p[1] == 0x8A && p[2] == 0xA0 && p[3] == 0xE4 && p[4] == 0xB8 && p[5] == 0x8A) { out[n++] = '+'; p += 6; continue; } /* 加上 */
        if (p[0] == 0xE5 && p[1] == 0x87 && p[2] == 0x8F && p[3] == 0xE5 && p[4] == 0x8E && p[5] == 0xBB) { out[n++] = '-'; p += 6; continue; } /* 减去 */
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
        if (p[0] == 0xEF && p[1] == 0xBC && p[2] == 0x85) { out[n++] = '%'; p += 3; continue; } /* ％ */
        out[n++] = (char)*p++;
    }
    out[n] = 0;
    return n;
}

/* ---- 中文数字（五十九、一百零五、两千三）转成阿拉伯数字 ---- */
static int utf8_cp(const unsigned char *p, unsigned *cp)
{
    if (p[0] < 0x80) { *cp = p[0]; return 1; }
    if ((p[0] & 0xE0) == 0xC0) { *cp = ((p[0] & 0x1F) << 6) | (p[1] & 0x3F); return 2; }
    if ((p[0] & 0xF0) == 0xE0) {
        *cp = ((p[0] & 0x0F) << 12) | ((p[1] & 0x3F) << 6) | (p[2] & 0x3F);
        return 3;
    }
    *cp = p[0];
    return 1;
}

/* 返回：0-9 数字；10/100/1000/10000 单位；-1 不是中文数字 */
static int cn_map(unsigned cp)
{
    switch (cp) {
        case 0x96F6: case 0x3007: return 0;          /* 零 〇 */
        case 0x4E00: return 1;                        /* 一 */
        case 0x4E8C: case 0x4E24: return 2;           /* 二 两 */
        case 0x4E09: return 3;
        case 0x56DB: return 4;
        case 0x4E94: return 5;
        case 0x516D: return 6;
        case 0x4E03: return 7;
        case 0x4E5D: return 9;
        case 0x516B: return 8;
        case 0x5341: return 10;                       /* 十 */
        case 0x767E: return 100;                      /* 百 */
        case 0x5343: return 1000;                     /* 千 */
        case 0x4E07: return 10000;                    /* 万 */
        default: return -1;
    }
}

/* 把 s 里的中文数字段替换成阿拉伯数字（就地重写，可能变短） */
static void cn_to_ascii(char *s)
{
    char out[CALC_MAX];
    int n = 0;
    const unsigned char *p = (const unsigned char *)s;
    while (*p && n < (int)sizeof(out) - 24) {
        unsigned cp;
        const int len = utf8_cp(p, &cp);
        if (cp < 0x80 || cn_map(cp) < 0) {
            for (int i = 0; i < len && n < (int)sizeof(out) - 1; i++) out[n++] = (char)p[i];
            p += len;
            continue;
        }
        /* 一个连续中文数字段 */
        long long total = 0, section = 0, cur = 0;
        while (*p) {
            unsigned c2;
            const int l2 = utf8_cp(p, &c2);
            const int v = cn_map(c2);
            if (v < 0) break;
            if (v <= 9) {
                cur = v;
            } else if (v < 10000) {
                section += (cur ? cur : 1) * v;
                cur = 0;
            } else {
                total += (section + cur) * 10000;
                section = 0;
                cur = 0;
            }
            p += l2;
        }
        n += snprintf(out + n, sizeof(out) - (size_t)n, "%lld", total + section + cur);
    }
    out[n] = 0;
    strcpy(s, out);
}

/* 后缀/前缀外壳：返回 1 表示剥掉了一层 */
static int strip_suffix(char *s)
{
    static const char *suf[] = {"是多少钱", "是多少元", "多少钱", "多少元", "是多少呢", "是多少呀",
                                "算一下", "算算", "算下", "计算一下",
                                "等于几", "等于多少", "是多少", "多少", "=?", "＝?", "?", "？",
                                "呢", "呀", "。", ".", "!", "！", "=", "＝"};
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
    static const char *pre[] = {"帮我算一下", "帮我计算", "帮我算算", "帮我算", "麻烦算一下",
                                "麻烦算算", "计算一下", "计算", "算一下", "算算", "请问一下",
                                "请问", "求解", "求", "把"};
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
    for (int round = 0; round < 6; round++) {
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
    /* 中文数字放在外壳剥离之后转换（否则"帮我算一下"里的"一"会被改成 1） */
    cn_to_ascii(s);
    /* "100的15%" -> "100*15%"：只有一个"的"且换成 * 后是合法算式时才认 */
    {
        char *de = strstr(s, "的");
        if (de && !strstr(de + 3, "的")) {
            const size_t pos = (size_t)(de - s);
            memmove(de + 1, de + 3, strlen(de + 3) + 1);
            s[pos] = '*';
        }
    }
    /* "根号16" -> "V16"（V = 平方根，解析器里处理） */
    if (strncmp(s, "根号", 6) == 0) {
        memmove(s + 1, s + 6, strlen(s + 6) + 1);
        s[0] = 'V';
    }
    /* "X的平方/立方" 已在上一步变成 "X*平方/立方"？——中文单位先转成重复乘法 */
    {
        char tmp[CALC_MAX];
        strcpy(tmp, s);
        char *p = strstr(tmp, "*平方");
        if (p) { *p = 0; snprintf(s, sizeof(tmp), "%s*%s", tmp, tmp); }
        else {
            char *q = strstr(tmp, "*立方");
            if (q) { *q = 0; snprintf(s, sizeof(tmp), "%s*%s*%s", tmp, tmp, tmp); }
        }
    }
    /* 校验：只允许数字/小数点/运算符/括号/空格，且至少一个数字与一个运算符 */
    int has_digit = 0, has_op = 0;
    for (const char *p = s; *p; p++) {
        if (is_digit(*p)) has_digit = 1;
        else if (strchr("+-*/()", *p)) has_op = (*p == '+' || *p == '-' || *p == '*' || *p == '/');
        else if (*p == 'V' || *p == 'v') has_op = 1;
        else if (*p == '.' || *p == '%' || is_space(*p)) continue;
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
    if (*c->p == '%') { v /= 100.0; c->p++; }         /* 15% -> 0.15 */
    return v;
}

static double parse_factor(calc_ctx *c)
{
    skip_ws(c);
    if (*c->p == 'V' || *c->p == 'v') {           /* 平方根 */
        c->p++;
        const double v = parse_factor(c);
        if (v < 0) { c->err = -2; return 0; }
        return sqrt(v);
    }
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

/* 序列数数："把 1 到 5 倒着数一遍" / "从 3 数到 8"。
 * 只在明确的祈使句上触发（倒着数/倒序，或以"从/把"开头且含"数到/数一遍"），
 * 避免"我从1数到100也数不完"这类普通陈述被截走。返回 1 = 已作答。 */
static int feng_calc_seq_answer(const char *user, char *answer, int answer_sz)
{
    const int rev = (strstr(user, "倒着数") != NULL) || (strstr(user, "倒序") != NULL) ||
                    (strstr(user, "倒过来数") != NULL);
    int fwd = 0;
    if (!rev) {
        const int starts = (strncmp(user, "从", 3) == 0) || (strncmp(user, "把", 3) == 0) ||
                           strstr(user, "请从") != NULL;
        if (strstr(user, "数到"))
            fwd = starts || strstr(user, "数一遍") != NULL || strstr(user, "数一下") != NULL;
        else if (strstr(user, "数一遍") != NULL || strstr(user, "数一下") != NULL)
            fwd = starts;
    }
    if (!rev && !fwd) return 0;

    char work[CALC_MAX];
    strncpy(work, user, sizeof(work) - 1);
    work[sizeof(work) - 1] = 0;
    cn_to_ascii(work);
    int found[8];
    int nf = 0;
    for (const char *p = work; *p && nf < 8; ) {
        if (is_digit(*p)) {
            int v = 0;
            while (is_digit(*p)) { v = v * 10 + (*p - '0'); p++; }
            found[nf++] = v;
        } else {
            p++;
        }
    }
    if (nf < 2) return 0;                    /* 只有一个数（如"数到8"）交给模型 */
    int a = found[0], b = found[1];
    if (!rev && a > b) return 0;             /* "从8数到3"这类交给模型 */
    if (a > b) { const int t = a; a = b; b = t; }
    if (b - a > 50) {
        snprintf(answer, answer_sz, "范围有点大（%d 到 %d），给我 50 个以内的区间吧。", a, b);
        return 1;
    }
    int n = 0;
    if (rev) {
        for (int v = b; v >= a; v--)
            n += snprintf(answer + n, (size_t)(answer_sz - n), (v == b) ? "%d" : "、%d", v);
    } else {
        for (int v = a; v <= b; v++)
            n += snprintf(answer + n, (size_t)(answer_sz - n), (v == a) ? "%d" : "、%d", v);
    }
    snprintf(answer + n, (size_t)(answer_sz - n), "。");
    return 1;
}

int feng_calc_answer(const char *user, char *answer, int answer_sz)
{
    /* 序列任务（"把1到5倒着数一遍"/"从3数到8"）也走引擎，确定性输出 */
    if (feng_calc_seq_answer(user, answer, answer_sz)) return 1;
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
