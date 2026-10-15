/* phone/engine 派生副本：自 esp32s3-feng-llm/main/feng_memory.h 拷贝（2026-10-15，v4.0 C23 迁移后）。
 * 手机端改动见 phone/README.md；板端原始文件未被修改，两边需手动同步。
 * 原版校验值见 phone/engine/SOURCE.sha256。 */
/* 引擎侧记忆 tool：把"多轮记事实"从模型容量问题变成确定性的工程问题。
 *
 * 与算式/时间/随机数同一哲学：可枚举的句式交给 C 引擎直接处理，模型负责开放聊天。
 *   - learn：从用户陈述里抽取事实（我叫X / 最喜欢的颜色是Y / 住在Z / 养了一只W …）
 *   - answer：命中记忆/身份追问时直接作答（"我叫什么名字？" → "你叫X。"；
 *     "你叫什么名字？" → 固定身份口径），查不到就返回 0 交回模型
 *   - 只覆盖可枚举句式（见 pc/pc_mem_test.c 的用例）；其他说法仍走模型自己的多轮能力
 */
#ifndef FENG_MEMORY_H
#define FENG_MEMORY_H

#ifdef __cplusplus
extern "C" {
#endif

/* 清空记忆（\reset 或 \mem clear 时调用） */
void feng_mem_clear(void);
/* 从用户输入里学习事实；返回 1 = 新记录/更新了某条事实 */
int feng_mem_learn(const char *user);
/* 命中记忆/身份追问 → 写 answer 返回 1；否则返回 0（交回模型） */
int feng_mem_answer(const char *user, char *answer, int answer_sz);
/* 供 \mem 命令展示：把当前已知事实写进 buf（没有事实时写"（空）"） */
void feng_mem_snapshot(char *buf, int buf_sz);

#ifdef __cplusplus
}
#endif

#endif
