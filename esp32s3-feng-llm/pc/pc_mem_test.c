/* 引擎侧记忆 tool 单测：学习/追问/身份/更新/提问不误存。
 * 编译： gcc -O2 -o pc/pc_mem_test.exe pc/pc_mem_test.c ../main/feng_memory.c -I../main
 */
#include <stdio.h>
#include <string.h>

#include "feng_memory.h"

static int fails;

static void answer_is(const char *q, const char *want)
{
    char got[128];
    const int hit = feng_mem_answer(q, got, sizeof(got));
    const int ok = want ? (hit == 1 && strcmp(got, want) == 0) : (hit == 0);
    if (!ok) fails++;
    printf("[%s] %-26s -> %s%s%s\n", ok ? "PASS" : "FAIL", q,
           hit ? got : "(交给模型)",
           want && !ok ? "   期望: " : "", want && !ok ? want : "");
}

static void learn_is(const char *s, int want)
{
    const int got = feng_mem_learn(s);
    const int ok = (got == want);
    if (!ok) fails++;
    printf("[%s] learn %-22s -> %d（期望 %d）\n", ok ? "PASS" : "FAIL", s, got, want);
}

int main(void)
{
    feng_mem_clear();

    /* 记忆套件的 12 个原始用例（说事实 → 追问） */
    learn_is("我叫小雨，请记住。", 1);
    answer_is("我叫什么名字？", "你叫小雨。");
    learn_is("我最喜欢的颜色是黄色。", 1);
    answer_is("我最喜欢什么颜色？", "你最喜欢黄色。");
    learn_is("我住在武汉。", 1);
    answer_is("我住在哪里？", "你住在武汉。");
    learn_is("我养了一只兔子。", 1);
    answer_is("我养了什么？", "你养了兔子。");
    learn_is("我最喜欢蛋糕。", 1);
    answer_is("我最喜欢什么？", "你最喜欢蛋糕。");
    learn_is("我最喜欢的运动是羽毛球。", 1);
    answer_is("我最喜欢什么运动？", "你最喜欢羽毛球。");

    /* 更新类说法：取最新 */
    learn_is("我现在最喜欢的颜色改成蓝色了。", 1);
    answer_is("我最喜欢什么颜色？", "你最喜欢蓝色。");
    learn_is("我搬到成都了。", 1);
    answer_is("我住在哪里？", "你住在成都。");
    learn_is("我现在养的是乌龟。", 1);
    answer_is("我养了什么？", "你养了乌龟。");

    /* 身份：确定性口径（这正是板端"串名字"残余的最终保证） */
    answer_is("你叫什么名字？", "我叫 feng，由个人开发者 jiaheng 开发训练。");
    answer_is("你是谁？", "我叫 feng，由个人开发者 jiaheng 开发训练。");
    answer_is("谁训练了你？", "个人开发者 jiaheng 训练了我，我叫 feng。");
    answer_is("你是谁开发的？", "个人开发者 jiaheng 训练了我，我叫 feng。");

    /* 提问不能被当成陈述存进去 */
    feng_mem_clear();
    learn_is("我叫什么名字？", 0);
    answer_is("我叫什么名字？", NULL);                 /* 没存过 -> 交回模型 */
    learn_is("我最喜欢的颜色是什么？", 0);
    answer_is("我最喜欢什么颜色？", NULL);

    /* 没学过的类目 -> 交回模型 */
    feng_mem_clear();
    answer_is("我最喜欢什么运动？", NULL);
    answer_is("我住在哪里？", NULL);
    answer_is("今天天气怎么样？", NULL);

    /* "我是X" 只在像名字时才记 */
    learn_is("我是一个学生。", 0);
    learn_is("我是阿强。", 1);
    answer_is("我叫什么名字？", "你叫阿强。");

    /* 通用键值槽（v3.17 扩展）：任意短键都能记/问 */
    feng_mem_clear();
    learn_is("我最喜欢的书是《小王子》。", 1);
    answer_is("我最喜欢什么书？", "你最喜欢《小王子》。");
    answer_is("我最喜欢的书是什么？", "你最喜欢《小王子》。");
    learn_is("我的生日是3月5日。", 1);
    answer_is("我的生日是什么？", "你的生日是3月5日。");
    answer_is("我的生日是几号？", "你的生日是3月5日。");
    learn_is("我的家乡是武汉。", 1);
    answer_is("我的家乡是哪里？", "你的家乡是武汉。");
    learn_is("我的职业是程序员。", 1);
    answer_is("我的职业是什么？", "你的职业是程序员。");
    /* 更新与覆盖 */
    learn_is("我最喜欢的书是《活着》。", 1);
    answer_is("我最喜欢什么书？", "你最喜欢《活着》。");
    /* 问助手自己的偏好 -> 不该拿用户记忆回答 */
    answer_is("你最喜欢什么颜色？", NULL);
    /* 没记过的键 -> 交回模型 */
    answer_is("我最喜欢什么电影？", NULL);
    /* 少一个"最"的追问也要命中同一个键（回归：曾经落到无名槽答错） */
    learn_is("我最喜欢的颜色是蓝色。", 1);
    answer_is("我喜欢什么颜色？", "你最喜欢蓝色。");
    /* 专用槽的其它问法 */
    learn_is("我的宠物是猫。", 1);
    answer_is("我的宠物是什么？", "你养了猫。");
    learn_is("我叫小雨。", 1);
    answer_is("我的名字是什么？", "你叫小雨。");
    learn_is("我住在杭州。", 1);
    answer_is("我住在哪个城市？", "你住在杭州。");

    /* 列出已记事实（v3.17 扩展） */
    feng_mem_clear();
    answer_is("你还记得什么？", "我还没有记住你的信息。");
    learn_is("我叫小雨，请记住。", 1);
    learn_is("我最喜欢的颜色是蓝色。", 1);
    learn_is("我养了一只乌龟。", 1);
    answer_is("你还记得什么？", "我记得：你叫小雨；你养了乌龟；你的颜色是蓝色。");
    /* 遗忘单项与全部 */
    answer_is("忘掉我的颜色", "好，我忘掉了你的颜色。");
    answer_is("我最喜欢什么颜色？", "我不记得你的颜色了。");
    answer_is("忘掉我的生日", "我没有记过你的生日。");
    learn_is("我的生日是3月5日。", 1);
    answer_is("别记我的生日了", "好，我忘掉了你的生日。");
    answer_is("我的生日是什么？", "我不记得你的生日了。");    /* 墓碑：不再让模型凭上下文答出来 */
    learn_is("我的生日是6月1日。", 1);                       /* 重新学习解除墓碑 */
    answer_is("我的生日是什么？", "你的生日是6月1日。");
    answer_is("把记住的都忘掉", "好，我把记住的这些都忘掉了。");
    answer_is("你还记得什么？", "我还没有记住你的信息。");

    printf("\n%s（%d 个失败）\n", fails ? "有失败" : "全部通过", fails);
    return fails ? 1 : 0;
}
