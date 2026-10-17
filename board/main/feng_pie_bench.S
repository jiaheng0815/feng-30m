/* PIE（128 位 SIMD）裸吞吐微基准：先量清楚指令代价，再决定要不要写整行 Q4 内核。
 *
 *   pie_mac_s16_bench(a, n) : n 次 ee.vmulas.s16.accx（8 个 int16 MAC/次，共享 ACCX 的依赖链）
 *   pie_s8_zip_mac_bench(w, x, n) : 每次迭代 = 2×ld128 + 2×vcmp + 2×vzip + 1×vmulas.s16
 *                                  （esp-nn 的 int8→int16 符号扩展 + MAC 模式，8 MAC/次）
 */
    .text
    .align  4

    .align  4
    .global pie_mac_s16_bench
    .type   pie_mac_s16_bench, @function
pie_mac_s16_bench:
    entry   a1, 16
    ee.vld.128.ip  q0, a2, 0
    ee.vld.128.ip  q1, a2, 0
    ee.zero.accx
.Lpm_loop:
    ee.vmulas.s16.accx q0, q1
    addi    a3, a3, -1
    bnez    a3, .Lpm_loop
    ee.srs.accx a2, a3, 0        /* 消费累加器，防优化 */
    retw.n
    .size   pie_mac_s16_bench, . - pie_mac_s16_bench

    .align  4
    .global pie_s8_zip_mac_bench
    .type   pie_s8_zip_mac_bench, @function
pie_s8_zip_mac_bench:
    entry   a1, 16
    ee.zero.q   q7
    ee.zero.accx
.Lps_loop:
    ee.vld.128.ip  q0, a2, 0
    ee.vld.128.ip  q1, a3, 0
    ee.vcmp.lt.s8 q2, q0, q7
    ee.vcmp.lt.s8 q3, q1, q7
    ee.vzip.8   q0, q2
    ee.vzip.8   q1, q3
    ee.vmulas.s16.accx q0, q1
    addi    a4, a4, -1
    bnez    a4, .Lps_loop
    ee.srs.accx a2, a3, 0
    retw.n
    .size   pie_s8_zip_mac_bench, . - pie_s8_zip_mac_bench
