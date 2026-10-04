/* 引擎侧记忆 tool 的实现：纯 C、无动态内存，板端/PC 共用。 */
#include "feng_memory.h"

#include <stdio.h>
#include <string.h>

#define VAL_CAP 32

typedef struct {
    char name[VAL_CAP];   /* 用户的名字 */
    char color[VAL_CAP];
    char sport[VAL_CAP];
    char city[VAL_CAP];
    char pet[VAL_CAP];
    char food[VAL_CAP];
    int has_name, has_color, has_sport, has_city, has_pet, has_food;
} mem_t;

static mem_t g;

void feng_mem_clear(void) { memset(&g, 0, sizeof(g)); }

/* 值里出现这些前缀说明用户在提问，不是陈述 */
static int looks_like_question(const char *v)
{
    static const char *kw[] = {"什么", "啥", "哪", "几", "谁", "多少", "吗", "呢"};
    for (int i = 0; i < (int)(sizeof(kw) / sizeof(kw[0])); i++) {
        if (strncmp(v, kw[i], strlen(kw[i])) == 0) return 1;
    }
    return 0;
}

/* UTF-8 字符长度（本模块只做字节级处理，够用） */
static int u8len(const char *c)
{
    const unsigned char b = (unsigned char)*c;
    if (b < 0x80) return 1;
    if ((b >> 5) == 0x6) return 2;
    if ((b >> 4) == 0xE) return 3;
    if ((b >> 3) == 0x1E) return 4;
    return 1;
}

static int starts_with(const char *s, const char *pfx)
{
    return strncmp(s, pfx, strlen(pfx)) == 0;
}

/* 句读/终止符 */
static int is_term(const char *p)
{
    static const char *t[] = {"。", "，", "、", "！", "？", "；", ",", ".", "!", "?", ";",
                              "\n", "\r"};
    for (unsigned i = 0; i < sizeof(t) / sizeof(t[0]); i++) {
        if (starts_with(p, t[i])) return 1;
    }
    return 0;
}

/* 去掉尾部虚词（改了/了/呀…/，请记住） */
static void trim_tail(char *out)
{
    static const char *t[] = {"请记住", "一下", "了", "的", "呀", "啊", "哦", "吧", "嘛", "呢"};
    int changed = 1;
    while (changed && *out) {
        changed = 0;
        for (unsigned i = 0; i < sizeof(t) / sizeof(t[0]); i++) {
            const size_t lt = strlen(t[i]);
            const size_t lo = strlen(out);
            if (lo >= lt && strcmp(out + lo - lt, t[i]) == 0) {
                out[lo - lt] = 0;
                changed = 1;
            }
        }
    }
}

/* 取 kw 之后的值：到句读/换行或串尾；再去掉尾部虚词 */
static int take_value(const char *p, char *out, int cap)
{
    while (*p == ' ' || *p == '\t' || starts_with(p, "：") || *p == ':') {
        p++;
    }
    int n = 0;
    while (*p && n < cap - 1) {
        if (is_term(p)) break;
        const int l = u8len(p);
        if (n + l >= cap) break;
        memcpy(out + n, p, (size_t)l);
        n += l;
        p += l;
    }
    out[n] = 0;
    trim_tail(out);
    while (*out == ' ') memmove(out, out + 1, strlen(out));
    return out[0] != 0;
}

/* 在 user 里找 kw，取跟随的值写进 dst（带 has 标记）；返回 1 = 命中 */
static int learn_kw(const char *user, const char *kw, char *dst, int *has)
{
    const char *p = strstr(user, kw);
    if (!p) return 0;
    char v[VAL_CAP];
    if (!take_value(p + strlen(kw), v, sizeof(v))) return 0;
    if (looks_like_question(v)) return 0;
    if (strcmp(v, dst) == 0 && *has) return 0;          /* 没有变化 */
    snprintf(dst, VAL_CAP, "%s", v);
    *has = 1;
    return 1;
}

int feng_mem_learn(const char *user)
{
    int changed = 0;
    /* 每类只取"最具体"的一个说法（先匹配到的优先，避免通用句式覆盖更精确的提取） */
    /* 名字：我的名字是X / 我叫X（"我是X" 见下） */
    if (learn_kw(user, "我的名字是", g.name, &g.has_name)) changed = 1;
    else if (learn_kw(user, "我叫", g.name, &g.has_name)) changed = 1;
    if (!g.has_name) {
        const char *p = strstr(user, "我是");
        if (p) {
            char v[VAL_CAP];
            if (take_value(p + 6, v, sizeof(v)) && !looks_like_question(v) &&
                strlen(v) <= 12 && !strstr(v, "学生") && !strstr(v, "模型")) {
                snprintf(g.name, VAL_CAP, "%s", v);
                g.has_name = 1;
                changed = 1;
            }
        }
    }
    /* 颜色 / 运动（"改成X了" 的说法也覆盖） */
    if (learn_kw(user, "颜色是", g.color, &g.has_color)) changed = 1;
    else if (learn_kw(user, "颜色改成", g.color, &g.has_color)) changed = 1;
    if (learn_kw(user, "运动是", g.sport, &g.has_sport)) changed = 1;
    else if (learn_kw(user, "运动换成", g.sport, &g.has_sport)) changed = 1;
    /* 城市 / 宠物 */
    if (learn_kw(user, "住在", g.city, &g.has_city)) changed = 1;
    else if (learn_kw(user, "搬到", g.city, &g.has_city)) changed = 1;
    if (learn_kw(user, "养了一只", g.pet, &g.has_pet)) changed = 1;
    else if (learn_kw(user, "养的是", g.pet, &g.has_pet)) changed = 1;
    else if (learn_kw(user, "养了", g.pet, &g.has_pet)) changed = 1;
    /* 食物：没有类目关键词的 "最喜欢X"（有"颜色/运动"时上面已经吃掉） */
    if (!strstr(user, "颜色") && !strstr(user, "运动") && !strstr(user, "城市")) {
        if (learn_kw(user, "最喜欢吃", g.food, &g.has_food)) changed = 1;
        else if (learn_kw(user, "最喜欢", g.food, &g.has_food)) changed = 1;
    }
    return changed;
}

static int answer_fmt(char *out, int cap, const char *fmt, const char *v)
{
    snprintf(out, cap, fmt, v);
    return 1;
}

int feng_mem_answer(const char *user, char *answer, int answer_sz)
{
    /* 身份：固定口径，确定性回答（这也是"不许串成用户名字"的最终保证） */
    static const char *id_q[] = {"你叫什么名字", "你叫什么", "你是谁", "你的名字是什么",
                                 "你是谁开发的", "谁训练了你"};
    for (int i = 0; i < (int)(sizeof(id_q) / sizeof(id_q[0])); i++) {
        if (strstr(user, id_q[i])) {
            if (strstr(user, "谁训练") || strstr(user, "谁开发"))
                return answer_fmt(answer, answer_sz, "%s",
                                  "个人开发者 jiaheng 训练了我，我叫 feng。");
            return answer_fmt(answer, answer_sz, "%s",
                              "我叫 feng，由个人开发者 jiaheng 开发训练。");
        }
    }
    /* 用户名字 */
    if (strstr(user, "我叫什么") || strstr(user, "我叫啥") ||
        strstr(user, "记得我叫什么")) {
        if (g.has_name) return answer_fmt(answer, answer_sz, "你叫%s。", g.name);
        return 0;
    }
    /* 颜色 / 运动 / 城市 / 宠物 / 食物 */
    if (strstr(user, "什么颜色") || strstr(user, "颜色是什么")) {
        if (g.has_color) return answer_fmt(answer, answer_sz, "你最喜欢%s。", g.color);
        return 0;
    }
    if (strstr(user, "什么运动")) {
        if (g.has_sport) return answer_fmt(answer, answer_sz, "你最喜欢%s。", g.sport);
        return 0;
    }
    if (strstr(user, "住在哪")) {
        if (g.has_city) return answer_fmt(answer, answer_sz, "你住在%s。", g.city);
        return 0;
    }
    if (strstr(user, "养了什么") || strstr(user, "养了啥")) {
        if (g.has_pet) return answer_fmt(answer, answer_sz, "你养了%s。", g.pet);
        return 0;
    }
    if ((strstr(user, "我最喜欢什么") || strstr(user, "我喜欢什么") ||
         strstr(user, "喜欢吃什么")) && !strstr(user, "颜色") && !strstr(user, "运动")) {
        if (g.has_food) return answer_fmt(answer, answer_sz, "你最喜欢%s。", g.food);
        return 0;
    }
    return 0;
}

void feng_mem_snapshot(char *buf, int buf_sz)
{
    int n = 0;
    n += snprintf(buf + n, (size_t)(buf_sz - n), "名字=%s", g.has_name ? g.name : "-");
    n += snprintf(buf + n, (size_t)(buf_sz - n), " 颜色=%s", g.has_color ? g.color : "-");
    n += snprintf(buf + n, (size_t)(buf_sz - n), " 运动=%s", g.has_sport ? g.sport : "-");
    n += snprintf(buf + n, (size_t)(buf_sz - n), " 城市=%s", g.has_city ? g.city : "-");
    n += snprintf(buf + n, (size_t)(buf_sz - n), " 宠物=%s", g.has_pet ? g.pet : "-");
    snprintf(buf + n, (size_t)(buf_sz - n), " 食物=%s", g.has_food ? g.food : "-");
}
