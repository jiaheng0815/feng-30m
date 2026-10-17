/* Q4 GEMV 的 PIE 内核（64 个 int16 MAC 一个调用点）。
 *
 * 调用方（feng_quant.c）先用 256 项 LUT 把 32 个打包字节展开成 64 个 int16
 * 权重（1 load + 1 store / 字节），激活也已量化成 int16；这里做 8 次
 * vld.128 + 8 次 vmulas.s16.accx，最后 srs.accx 取出水平和（=点积）。
 *
 * 微基准（同芯片实测）：纯 MAC 链 0.63 周期/MAC、含 load/zip 的流水 1.38 周期/MAC，
 * 标量 FPU 路径是 ~8.2 周期/权重（单核）。
 */
    .text
    .align  4

/* int32_t pie_dot64(const int16_t *w, const int16_t *x); */
    .align  4
    .global pie_dot64
    .type   pie_dot64, @function
pie_dot64:
    entry   a1, 16
    ee.zero.accx
    ee.vld.128.ip  q0, a2, 16
    ee.vld.128.ip  q1, a3, 16
    ee.vmulas.s16.accx q0, q1
    ee.vld.128.ip  q2, a2, 16
    ee.vld.128.ip  q3, a3, 16
    ee.vmulas.s16.accx q2, q3
    ee.vld.128.ip  q4, a2, 16
    ee.vld.128.ip  q5, a3, 16
    ee.vmulas.s16.accx q4, q5
    ee.vld.128.ip  q6, a2, 16
    ee.vld.128.ip  q7, a3, 16
    ee.vmulas.s16.accx q6, q7
    /* 剩下的 4 个 16B 用 q0..q3 复用（前 4 个已经算完） */
    ee.vld.128.ip  q0, a2, 16
    ee.vld.128.ip  q1, a3, 16
    ee.vmulas.s16.accx q0, q1
    ee.vld.128.ip  q2, a2, 16
    ee.vld.128.ip  q3, a3, 16
    ee.vmulas.s16.accx q2, q3
    ee.vld.128.ip  q4, a2, 16
    ee.vld.128.ip  q5, a3, 16
    ee.vmulas.s16.accx q4, q5
    ee.vld.128.ip  q6, a2, 16
    ee.vld.128.ip  q7, a3, 16
    ee.vmulas.s16.accx q6, q7
    movi    a4, 0
    ee.srs.accx a5, a4, 0
    mov.n   a2, a5
    retw.n
    .size   pie_dot64, . - pie_dot64
