"""检查 phone/engine 这份派生副本与板端 esp32s3-feng-llm/main/ 是否已经分叉。

phone/engine 的每个文件头部都记了来源，原版校验值存在 phone/engine/SOURCE.sha256。
板端改了引擎之后跑一下这个脚本，它会告诉你：

  [未变]  板端与拷贝时一致 —— 手机端那份可以放心继续用
  [已改]  板端改了、手机端没跟 —— 需要人工判断这个改动要不要搬过来
          （手机端只改了 feng_smp.c / feng_quant.c / feng_llm.c / feng.h 四处，
           见 phone/README.md 的「分叉清单」）
  [新增]  板端多了新文件 —— 手机端可能也要加进构建脚本
  [缺失]  板端删了文件

用法：
    python phone/tools/check_drift.py            # 只检查
    python phone/tools/check_drift.py --write    # 同步完成之后重建校验基准
"""
import hashlib
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[2]
BOARD = ROOT / "esp32s3-feng-llm" / "main"
PHONE = ROOT / "phone" / "engine"
STAMP = PHONE / "SOURCE.sha256"

# 手机端有意改过的文件：板端变了也**不会**自动同步，必须人工看
FORKED = {"feng_smp.c", "feng_quant.c", "feng_llm.c", "feng.h"}


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    h.update(p.read_bytes())
    return h.hexdigest()


# 基准里记录的文件 = 手机端抄过来的那些（不含 main.c / gbk / PIE 汇编等手机不编的）。
# feng_smp.c 也在内：它属于「抄过来之后有意重写」的分叉文件，基准照样记板端版本，
# 这样板端一改就会报「已改·分叉文件」而不是被误当成新增。
STAMP_FILES = ["feng.h", "feng_model.c", "feng_llm.c", "feng_quant.c", "feng_smp.c",
               "feng_tokenizer.c", "feng_tokenizer.h", "feng_sample.c", "feng_calc.c",
               "feng_calc.h", "feng_tools.c", "feng_tools.h", "feng_memory.c",
               "feng_memory.h"]


def read_stamp() -> dict:
    if not STAMP.exists():
        print(f"找不到 {STAMP.relative_to(ROOT)}，无法判断漂移")
        sys.exit(2)
    out = {}
    for line in STAMP.read_text(encoding="utf-8").splitlines():
        parts = line.split(None, 1)
        if len(parts) == 2:
            # GNU sha256sum 二进制模式会在文件名前加 '*'（Windows 上的 Git Bash 默认如此）
            out[parts[1].strip().lstrip("*")] = parts[0]
    return out


def write_stamp() -> int:
    """按**板端当前内容**重建基准：同步完手机端副本之后调用。"""
    lines = []
    for name in STAMP_FILES:
        src = BOARD / name
        if not src.exists():
            print(f"板端缺少 {name}，终止", file=sys.stderr)
            return 2
        lines.append(f"{sha256(src)}  {name}\n")
    STAMP.write_text("".join(lines), encoding="utf-8", newline="\n")
    print(f"已重建 {STAMP.relative_to(ROOT)}（{len(lines)} 个文件）")
    return 0


def main() -> int:
    if "--write" in sys.argv:
        return write_stamp()
    stamp = read_stamp()
    print(f"对照基准：{STAMP.relative_to(ROOT)}（拷贝时记录的板端校验值）\n")

    changed, missing, forked_changed = [], [], []
    for name, old in sorted(stamp.items()):
        src = BOARD / name
        if not src.exists():
            missing.append(name)
            continue
        if sha256(src) == old:
            continue
        (forked_changed if name in FORKED else changed).append(name)

    # 板端专有、手机端**故意不编**的文件：
    #   main.c / gbk*     串口 REPL 与 GBK 转换（手机是 POSIX 终端，用不到）
    #   feng_cuda.*       PC 的 GPU 后端，nvcc 专属
    #   feng_pie_*.S      Xtensa PIE 汇编，只有 ESP32-S3 有
    #   feng_gemv_bench.c 板端周期级剖析（-DFENG_BENCH_GEMV=1），
    #                     专门定位 S3 上 flash 流式读造成的流水线停顿——手机权重走
    #                     mmap 的 page cache，没有这个问题；手机端的基准是 phone_chat --bench
    skip = {"main.c", "gbk.c", "gbk_table.c", "gbk.h", "gbk_table.h",
            "feng_cuda.cu", "feng_cuda.h", "feng_pie_bench.S", "feng_pie_gemv.S",
            "feng_gemv_bench.c", "CMakeLists.txt"}
    added = sorted(p.name for p in BOARD.glob("*.[ch]")
                   if p.name not in stamp and p.name not in skip)

    ok = True
    if changed:
        ok = False
        print("‼ [已改] 板端改了、手机端没跟（需要人工搬过来）：")
        for n in changed:
            print(f"    esp32s3-feng-llm/main/{n}")
        print("    搬法：diff 两边，把改动贴进 phone/engine/ 对应文件，再跑 phone/cli/verify.sh\n")
    if forked_changed:
        print("⚠ [已改·分叉文件] 板端改了这些文件，但手机端**有意**改过它们，不会自动同步：")
        for n in forked_changed:
            print(f"    esp32s3-feng-llm/main/{n}")
        print("    照 phone/README.md 的「分叉清单」逐条判断：只搬不属于那四处的改动\n")
    if missing:
        ok = False
        print("‼ [缺失] 板端已经没有这些文件了：")
        for n in missing:
            print(f"    {n}")
        print()
    if added:
        print("⚠ [新增] 板端多了这些文件，手机端可能要一起编：")
        for n in added:
            print(f"    esp32s3-feng-llm/main/{n}")
        print()

    if ok and not forked_changed and not added:
        print("✓ 没有漂移：板端引擎与 phone/engine 拷贝时完全一致")
        return 0
    if ok and not added:
        print("✓ 无实质漂移（只动过手机端有意分叉的那四个文件）")
        return 0
    print("→ 有需要人工确认的漂移，见上面")
    return 1


if __name__ == "__main__":
    sys.exit(main())
