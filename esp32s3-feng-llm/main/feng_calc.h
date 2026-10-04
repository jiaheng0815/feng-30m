/* 计算 tool：用 SoC 自己的整数/浮点运算器算算术，模型不再背算术。
 *
 * 约定：
 *   feng_calc_extract()  从用户输入里剥出纯算式（去掉"等于几/？/计算"等外壳）
 *   feng_calc_eval()     递归下降求值（+ - * / × ÷ 括号，支持小数）
 *   feng_calc_answer()   一步到位：识别 -> 计算 -> 生成中文回答
 * 返回 1 表示"这是算术请求且已给出回答"，0 表示不是算术请求。
 */
#ifndef FENG_CALC_H
#define FENG_CALC_H

/* 从 user 抽算式到 expr；返回 1 表示确实是算术请求 */
int feng_calc_extract(const char *user, char *expr, int expr_sz);
/* 求值：1 成功；-1 除零；-2 语法错误 */
int feng_calc_eval(const char *expr, double *out);
/* 一步到位：算术请求则写 answer（含中文句式）并返回 1，否则 0 */
int feng_calc_answer(const char *user, char *answer, int answer_sz);
/* 数字转字符串（整数不带小数点，小数最多 4 位并去尾零） */
void feng_calc_fmt(double v, char *buf, int buf_sz);

#endif
