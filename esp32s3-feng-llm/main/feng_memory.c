/* 引擎侧记忆 tool 的实现：纯 C、无动态内存，板端/PC/Python 同口径。
 *
 * 存储分三块：
 *   - 专用槽：名字（我叫X）、城市（住在/搬到X）、宠物（养了X）——它们有固定的追问句式
 *   - 通用槽：键值表（"最喜欢的<键>是<值>"、"我的<键>是<值>"）——覆盖
 *     颜色/运动/食物/书/电影/生日/职业…等任意短键
 */
#include "feng_memory.h"

#include <stdio.h>
#include <string.h>

#define VAL_CAP 32
#define KEY_CAP 24
#define MAX_SLOTS 8

typedef struct {
    char key[KEY_CAP];
    char val[VAL_CAP];
    int used;
} slot_t;

static char g_name[VAL_CAP];
static char g_city[VAL_CAP];
static char g_pet[VAL_CAP];
static int g_has_name, g_has_city, g_has_pet;
static slot_t g_slots[MAX_SLOTS];
static int g_slot_next;
/* 墓碑：用户明确说"忘掉"的键。引擎忘掉后，该键的追问由引擎答"不记得了"，
 * 避免模型凭对话上下文又把这条说出来（重新学习会解除墓碑）。 */
static char g_forgot[MAX_SLOTS][KEY_CAP];
static int g_nforgot;

static void mark_forgot(const char *key)
{
    if (!key[0]) return;
    for (int i = 0; i < g_nforgot; i++) if (strcmp(g_forgot[i], key) == 0) return;
    if (g_nforgot < MAX_SLOTS) snprintf(g_forgot[g_nforgot++], KEY_CAP, "%s", key);
}

static int is_forgot(const char *key)
{
    for (int i = 0; i < g_nforgot; i++) {
        if (strcmp(g_forgot[i], key) == 0) return 1;
    }
    return 0;
}

static void clear_forgot(const char *key)
{
    for (int i = 0; i < g_nforgot; i++) {
        if (strcmp(g_forgot[i], key) != 0) continue;
        for (int j = i; j + 1 < g_nforgot; j++) memcpy(g_forgot[j], g_forgot[j + 1], KEY_CAP);
        g_nforgot--;
        return;
    }
}

void feng_mem_clear(void)
{
    g_name[0] = g_city[0] = g_pet[0] = 0;
    g_has_name = g_has_city = g_has_pet = 0;
    memset(g_slots, 0, sizeof(g_slots));
    g_slot_next = 0;
    g_nforgot = 0;
}

/* ---- 通用键值表 ---- */
static void slot_put(const char *key, const char *val)
{
    if (!val[0]) return;                               /* 空键 = 无名偏好槽（"我最喜欢X"） */
    for (int i = 0; i < MAX_SLOTS; i++) {              /* 同键覆盖 */
        if (g_slots[i].used && strcmp(g_slots[i].key, key) == 0) {
            snprintf(g_slots[i].val, VAL_CAP, "%s", val);
            return;
        }
    }
    for (int i = 0; i < MAX_SLOTS; i++) {              /* 空位 */
        if (!g_slots[i].used) {
            snprintf(g_slots[i].key, KEY_CAP, "%s", key);
            snprintf(g_slots[i].val, VAL_CAP, "%s", val);
            g_slots[i].used = 1;
            return;
        }
    }
    slot_t *s = &g_slots[g_slot_next++ % MAX_SLOTS];   /* 满了就轮换覆盖最旧的 */
    snprintf(s->key, KEY_CAP, "%s", key);
    snprintf(s->val, VAL_CAP, "%s", val);
}

static const char *slot_get(const char *key)
{
    for (int i = 0; i < MAX_SLOTS; i++) {
        if (g_slots[i].used && strcmp(g_slots[i].key, key) == 0) return g_slots[i].val;
    }
    return NULL;
}

/* 清掉某一项（名字/城市/宠物/任意键槽）；返回 1 = 确实清掉了 */
static void trim_tail(char *out);      /* 定义在下方 */
static void norm_forget_key(char *key)
{
    trim_tail(key);
    static const char *pfx[] = {"我的", "你记的", "你记住的", "关于"};
    for (unsigned i = 0; i < sizeof(pfx) / sizeof(pfx[0]); i++) {
        const size_t lp = strlen(pfx[i]);
        if (strncmp(key, pfx[i], lp) == 0) {
            memmove(key, key + lp, strlen(key + lp) + 1);
            break;
        }
    }
    trim_tail(key);
}

static int mem_forget_key(const char *key)
{
    if (strcmp(key, "名字") == 0) {
        const int had = g_has_name;
        g_has_name = 0;
        mark_forgot("名字");
        return had;
    }
    if (strcmp(key, "城市") == 0 || strcmp(key, "住的地方") == 0) {
        const int had = g_has_city;
        g_has_city = 0;
        mark_forgot("城市");
        return had;
    }
    if (strcmp(key, "宠物") == 0) {
        const int had = g_has_pet;
        g_has_pet = 0;
        mark_forgot("宠物");
        return had;
    }
    for (int i = 0; i < MAX_SLOTS; i++) {
        if (g_slots[i].used && strcmp(g_slots[i].key, key) == 0) {
            g_slots[i].used = 0;
            mark_forgot(key);
            return 1;
        }
    }
    mark_forgot(key);              /* 没记过也记墓碑：之后追问统一答"不记得" */
    return 0;
}

/* 把已记事实列成一句话；返回 0 = 什么都没记 */
static int mem_list(char *out, int cap)
{
    int n = 0, cnt = 0;
    n += snprintf(out + n, (size_t)(cap - n), "我记得：");
    if (g_has_name && n < cap - 24) { n += snprintf(out + n, (size_t)(cap - n), "你叫%s；", g_name); cnt++; }
    if (g_has_city && n < cap - 24) { n += snprintf(out + n, (size_t)(cap - n), "你住在%s；", g_city); cnt++; }
    if (g_has_pet && n < cap - 24) { n += snprintf(out + n, (size_t)(cap - n), "你养了%s；", g_pet); cnt++; }
    for (int i = 0; i < MAX_SLOTS && cnt < 6; i++) {
        if (!g_slots[i].used) continue;
        if (n >= cap - 32) { n += snprintf(out + n, (size_t)(cap - n), "…"); break; }
        if (g_slots[i].key[0])
            n += snprintf(out + n, (size_t)(cap - n), "你的%s是%s；", g_slots[i].key, g_slots[i].val);
        else
            n += snprintf(out + n, (size_t)(cap - n), "你最喜欢%s；", g_slots[i].val);
        cnt++;
    }
    if (cnt == 0) return 0;
    n = (int)strlen(out);
    if (n >= 3 && strcmp(out + n - 3, "；") == 0) { out[n - 3] = 0; n -= 3; }
    strncat(out + n, "。", (size_t)(cap - n - 1));
    return 1;
}

/* ---- 文本工具 ---- */
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

static int is_term(const char *p)
{
    static const char *t[] = {"。", "，", "、", "！", "？", "；", ",", ".", "!", "?", ";",
                              "\n", "\r", "的"};
    for (unsigned i = 0; i < sizeof(t) / sizeof(t[0]); i++) {
        if (starts_with(p, t[i])) return 1;
    }
    return 0;
}

static int looks_like_question(const char *v)
{
    static const char *kw[] = {"什么", "啥", "哪", "几", "谁", "多少", "吗", "呢", "怎么"};
    for (unsigned i = 0; i < sizeof(kw) / sizeof(kw[0]); i++) {
        if (starts_with(v, kw[i])) return 1;
    }
    return 0;
}

static void trim_tail(char *out)
{
    static const char *t[] = {"请记住", "一下", "了", "呀", "啊", "哦", "吧", "嘛", "呢"};
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

/* 取一段值：到句读/换行或串尾；去尾部虚词；问句返回失败 */
static int take_value(const char *p, char *out, int cap)
{
    while (*p == ' ' || *p == '\t' || starts_with(p, "：") || *p == ':') p++;
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
    if (!out[0] || looks_like_question(out)) return 0;
    return 1;
}

/* 取一段"键"：到 stop 串或句读为止，允许含"的"（调用方自行 trim） */
static int take_key_to(const char *p, const char *stop, char *out, int cap)
{
    int n = 0;
    while (*p && n < cap - 1) {
        if (stop && starts_with(p, stop)) break;
        if (starts_with(p, "。") || starts_with(p, "？") || starts_with(p, "，") ||
            starts_with(p, "！") || starts_with(p, "、") || starts_with(p, "？") ||
            *p == '?' || *p == '!' || *p == '\n' || *p == '\r' || *p == ',') break;
        const int l = u8len(p);
        if (n + l >= cap) break;
        memcpy(out + n, p, (size_t)l);
        n += l;
        p += l;
    }
    out[n] = 0;
    while (out[0] && out[0] == ' ') memmove(out, out + 1, strlen(out));
    while (out[0]) {                                   /* 去掉开头的"的"和尾部空格 */
        const size_t lo = strlen(out);
        if (lo >= 3 && strcmp(out, "的") == 0) { out[0] = 0; break; }
        if (lo >= 3 && starts_with(out, "的")) { memmove(out, out + 3, lo - 3 + 1); continue; }
        if (out[lo - 1] == ' ') { out[lo - 1] = 0; continue; }
        break;
    }
    return out[0] != 0;
}

static int take_key_before(const char *start, const char *stop_kw, char *out, int cap)
{
    const char *stop = strstr(start, stop_kw);
    if (!stop) return 0;
    char tmp[KEY_CAP * 2];
    const size_t len = (size_t)(stop - start);
    if (len == 0 || len >= sizeof(tmp)) return 0;
    memcpy(tmp, start, len);
    tmp[len] = 0;
    return take_key_to(tmp, NULL, out, cap);           /* 复用清理逻辑 */
}

/* 明确的 [start, end) 区间版本（学习路径用：end 是"是/改成/换成"的指针） */
static int take_key_range(const char *start, const char *end, char *out, int cap)
{
    const size_t len = (size_t)(end - start);
    char tmp[KEY_CAP * 2];
    if (len == 0 || len >= sizeof(tmp)) return 0;
    memcpy(tmp, start, len);
    tmp[len] = 0;
    return take_key_to(tmp, NULL, out, cap);
}

/* ---- 学习 ---- */
int feng_mem_learn(const char *user)
{
    int changed = 0;
    char v[VAL_CAP], key[KEY_CAP];

    /* 名字（专用）：我的名字是X / 我叫X / 我是X（短名字） */
    if (take_value(strstr(user, "我的名字是") ? strstr(user, "我的名字是") + strlen("我的名字是") : "", v, sizeof(v))) {
        snprintf(g_name, VAL_CAP, "%s", v); g_has_name = 1; clear_forgot("名字"); changed = 1;
    } else if (strstr(user, "我叫") && take_value(strstr(user, "我叫") + strlen("我叫"), v, sizeof(v))) {
        snprintf(g_name, VAL_CAP, "%s", v); g_has_name = 1; clear_forgot("名字"); changed = 1;
    }
    if (!g_has_name) {
        const char *p = strstr(user, "我是");
        if (p && take_value(p + strlen("我是"), v, sizeof(v)) &&
            strlen(v) <= 12 && !strstr(v, "学生") && !strstr(v, "模型") && !strstr(v, "AI")) {
            snprintf(g_name, VAL_CAP, "%s", v); g_has_name = 1; changed = 1;
            clear_forgot("名字");
        }
    }

    /* 城市 / 宠物（专用，问法固定） */
    const char *p;
    if ((p = strstr(user, "住在")) && take_value(p + strlen("住在"), v, sizeof(v))) {
        snprintf(g_city, VAL_CAP, "%s", v); g_has_city = 1; clear_forgot("城市"); changed = 1;
    } else if ((p = strstr(user, "搬到")) && take_value(p + strlen("搬到"), v, sizeof(v))) {
        snprintf(g_city, VAL_CAP, "%s", v); g_has_city = 1; clear_forgot("城市"); changed = 1;
    }
    if ((p = strstr(user, "养了一只")) && take_value(p + strlen("养了一只"), v, sizeof(v))) {
        snprintf(g_pet, VAL_CAP, "%s", v); g_has_pet = 1; clear_forgot("宠物"); changed = 1;
    } else if ((p = strstr(user, "养的是")) && take_value(p + strlen("养的是"), v, sizeof(v))) {
        snprintf(g_pet, VAL_CAP, "%s", v); g_has_pet = 1; clear_forgot("宠物"); changed = 1;
    } else if ((p = strstr(user, "养了")) && take_value(p + strlen("养了"), v, sizeof(v))) {
        snprintf(g_pet, VAL_CAP, "%s", v); g_has_pet = 1; clear_forgot("宠物"); changed = 1;
    }

    /* 通用槽 1：最喜欢的<键>是/改成/换成<值>（键可为空 -> 无名偏好，如"我最喜欢蛋糕"） */
    if ((p = strstr(user, "最喜欢的")) != NULL) {
        static const char *marks[] = {"是", "改成", "换成"};
        const char *m = NULL;
        size_t mlen = 0;
        for (unsigned i = 0; i < sizeof(marks) / sizeof(marks[0]); i++) {
            const char *q = strstr(p + strlen("最喜欢的"), marks[i]);
            if (q && (!m || q < m)) { m = q; mlen = strlen(marks[i]); }
        }
        const char *k0 = p + strlen("最喜欢的");
        if (m && m - k0 <= KEY_CAP * 2 && take_key_range(k0, m, key, sizeof(key)) &&
            take_value(m + mlen, v, sizeof(v))) {
            slot_put(key, v);
            clear_forgot(key);
            changed = 1;
        }
    } else if ((p = strstr(user, "最喜欢吃")) != NULL || (p = strstr(user, "喜欢吃")) != NULL ||
               (p = strstr(user, "最喜欢")) != NULL) {
        const char *v0;
        if (starts_with(p, "最喜欢吃")) v0 = p + strlen("最喜欢吃");
        else if (starts_with(p, "喜欢吃")) v0 = p + strlen("喜欢吃");
        else v0 = p + strlen("最喜欢");
        if (take_value(v0, v, sizeof(v))) {            /* 无名偏好槽（键 = ""） */
            slot_put("", v);
            clear_forgot("");
            changed = 1;
        }
    }

    /* 通用槽 2：我的<键>是<值>（生日/职业/家乡…；"名字"已在上面处理） */
    if ((p = strstr(user, "我的")) != NULL) {
        const char *k0 = p + strlen("我的");
        static const char *marks2[] = {"是", "叫"};
        const char *m = NULL;
        size_t mlen = 0;
        for (unsigned i = 0; i < sizeof(marks2) / sizeof(marks2[0]); i++) {
            const char *q = strstr(k0, marks2[i]);
            if (q && (!m || q < m)) { m = q; mlen = strlen(marks2[i]); }
        }
        if (m && take_key_range(k0, m, key, sizeof(key)) && strcmp(key, "名字") != 0 &&
            take_value(m + mlen, v, sizeof(v))) {
            if (strcmp(key, "宠物") == 0) {            /* 我的宠物是猫 -> 专用宠物槽 */
                snprintf(g_pet, VAL_CAP, "%s", v);
                g_has_pet = 1;
                clear_forgot("宠物");
            } else {
                slot_put(key, v);
                clear_forgot(key);
            }
            changed = 1;
        }
    }
    return changed;
}

static int answer_fmt(char *out, int cap, const char *fmt, const char *a, const char *b)
{
    snprintf(out, cap, fmt, a, b);
    return 1;
}

/* ---- 回答 ---- */
int feng_mem_answer(const char *user, char *answer, int answer_sz)
{
    /* 身份（固定口径） */
    static const char *id_q[] = {"你叫什么名字", "你叫什么", "你是谁", "你的名字是什么",
                                 "你是谁开发的", "谁训练了你"};
    for (unsigned i = 0; i < sizeof(id_q) / sizeof(id_q[0]); i++) {
        if (strstr(user, id_q[i])) {
            if (strstr(user, "谁训练") || strstr(user, "谁开发"))
                return answer_fmt(answer, answer_sz, "%s", "个人开发者 jiaheng 训练了我，我叫 feng。", "");
            return answer_fmt(answer, answer_sz, "%s", "我叫 feng，由个人开发者 jiaheng 开发训练。", "");
        }
    }
    /* 遗忘优先于其它追问：忘掉我的生日 / 别记我的名字了 / 把记住的都忘掉 */
    {
        static const char *fv[] = {"忘掉", "忘记", "别记", "删掉", "不要记"};
        const char *f = NULL;
        size_t flen = 0;
        for (unsigned i = 0; i < sizeof(fv) / sizeof(fv[0]); i++) {
            const char *r = strstr(user, fv[i]);
            if (r && (!f || r < f)) { f = r; flen = strlen(fv[i]); }
        }
        if (f) {
            char key[KEY_CAP];
            take_key_to(f + flen, NULL, key, sizeof(key));
            norm_forget_key(key);
            /* "都/全部/一切" 视作全清 */
            if (!key[0] || strstr(key, "都") || strstr(key, "全部") || strstr(key, "一切") ||
                strstr(key, "所有")) {
                feng_mem_clear();
                return answer_fmt(answer, answer_sz, "%s", "好，我把记住的这些都忘掉了。", "");
            }
            if (mem_forget_key(key))
                return answer_fmt(answer, answer_sz, "好，我忘掉了你的%s。", key, "");
            return answer_fmt(answer, answer_sz, "我没有记过你的%s。", key, "");
        }
    }
    /* 列出已记事实 */
    if (strstr(user, "记得什么") || strstr(user, "记住什么") || strstr(user, "记住哪些") ||
        strstr(user, "记住了什么") || strstr(user, "记得哪些") || strstr(user, "记忆里有什么") ||
        strstr(user, "都记住了") || strstr(user, "记得的东西")) {
        if (mem_list(answer, answer_sz)) return 1;
        return answer_fmt(answer, answer_sz, "%s", "我还没有记住你的信息。", "");
    }
    /* 用户名字 */
    if (strstr(user, "我叫什么") || strstr(user, "我叫啥") || strstr(user, "记得我叫什么") ||
        strstr(user, "我的名字是什么") || strstr(user, "我的名字是啥")) {
        if (g_has_name) return answer_fmt(answer, answer_sz, "你叫%s。", g_name, "");
        if (is_forgot("名字")) return answer_fmt(answer, answer_sz, "%s", "我不记得你的名字了。", "");
        return 0;
    }
    /* 专用槽 */
    if (strstr(user, "住在哪") || strstr(user, "哪个城市") || strstr(user, "什么地方住")) {
        if (g_has_city) return answer_fmt(answer, answer_sz, "你住在%s。", g_city, "");
        if (is_forgot("城市")) return answer_fmt(answer, answer_sz, "%s", "我不记得你住在哪里了。", "");
        return 0;
    }
    if (strstr(user, "养了什么") || strstr(user, "养了啥") || strstr(user, "我的宠物") ||
        strstr(user, "养的什么宠物")) {
        if (g_has_pet) return answer_fmt(answer, answer_sz, "你养了%s。", g_pet, "");
        if (is_forgot("宠物")) return answer_fmt(answer, answer_sz, "%s", "我不记得你养了什么了。", "");
        return 0;
    }
    /* 通用槽：最喜欢<什么键>？ / 最喜欢的<键>是什么？ */
    char key[KEY_CAP] = "";
    if (!strstr(user, "我")) return 0;      /* "你最喜欢什么颜色"问的是助手，不是用户记忆 */
    const char *q = strstr(user, "喜欢什么");      /* 覆盖"最喜欢什么X"和"喜欢什么X" */
    if (q) {
        take_key_to(q + strlen("喜欢什么"), NULL, key, sizeof(key));
    } else if ((q = strstr(user, "喜欢啥")) != NULL) {
        take_key_to(q + strlen("喜欢啥"), NULL, key, sizeof(key));
    } else if ((q = strstr(user, "最喜欢的")) != NULL) {
        if (!take_key_before(q + strlen("最喜欢的"), "是什么", key, sizeof(key)) &&
            !take_key_before(q + strlen("最喜欢的"), "是啥", key, sizeof(key))) {
            key[0] = 0;
        }
    } else if (strstr(user, "我最喜欢什么") || strstr(user, "我喜欢什么") ||
               strstr(user, "喜欢吃什么")) {
        key[0] = 0;                                     /* 无名偏好槽 */
    }
    if (q || key[0] || strstr(user, "我最喜欢什么") || strstr(user, "我喜欢什么") ||
        strstr(user, "喜欢吃什么")) {
        const char *v = slot_get(key);
        if (v) return answer_fmt(answer, answer_sz, "你最喜欢%s。", v, "");
        if (is_forgot(key)) {
            if (key[0]) return answer_fmt(answer, answer_sz, "我不记得你的%s了。", key, "");
            return answer_fmt(answer, answer_sz, "%s", "我不记得你最喜欢什么了。", "");
        }
        return 0;
    }
    /* 通用槽：我的<键>是什么/是多少/是几号… */
    if ((q = strstr(user, "我的")) != NULL) {
        static const char *marks[] = {"是什么", "是啥", "是多少", "是几号", "是哪个", "是哪里", "是几"};
        const char *m = NULL;
        for (unsigned i = 0; i < sizeof(marks) / sizeof(marks[0]); i++) {
            const char *r = strstr(q + strlen("我的"), marks[i]);
            if (r && (!m || r < m)) { m = r; }
        }
        if (m && take_key_before(q + strlen("我的"), m, key, sizeof(key)) &&
            strcmp(key, "名字") != 0) {
            const char *v = slot_get(key);
            if (v) return answer_fmt(answer, answer_sz, "你的%s是%s。", key, v);
            if (is_forgot(key)) return answer_fmt(answer, answer_sz, "我不记得你的%s了。", key, "");
        }
    }
    return 0;
}

void feng_mem_snapshot(char *buf, int buf_sz)
{
    int n = 0;
    n += snprintf(buf + n, (size_t)(buf_sz - n), "名字=%s", g_has_name ? g_name : "-");
    n += snprintf(buf + n, (size_t)(buf_sz - n), " 城市=%s", g_has_city ? g_city : "-");
    n += snprintf(buf + n, (size_t)(buf_sz - n), " 宠物=%s", g_has_pet ? g_pet : "-");
    for (int i = 0; i < MAX_SLOTS && n < buf_sz - 8; i++) {
        if (!g_slots[i].used) continue;
        n += snprintf(buf + n, (size_t)(buf_sz - n), " %s=%s",
                      g_slots[i].key[0] ? g_slots[i].key : "(偏好)", g_slots[i].val);
    }
}
