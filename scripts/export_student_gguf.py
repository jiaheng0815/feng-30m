"""Export the distilled student to GGUF (f16) and quantize it (Q8_0 / Q4_K_M / IQ3_XXS)."""
import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import ROOT  # noqa: E402
from paths import LLAMA_BIN as BIN  # noqa: E402
from paths import CONVERT_HF_TO_GGUF as CONVERT  # noqa: E402
from paths import PY  # noqa: E402
from paths import require  # noqa: E402


def run(cmd, log):
    with open(log, "w", encoding="utf-8", errors="replace") as f:
        p = subprocess.run([str(c) for c in cmd], stdout=f, stderr=subprocess.STDOUT)
    return p.returncode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(ROOT / "student" / "stageB" / "final"))
    ap.add_argument("--out-dir", default=str(ROOT / "student" / "gguf"))
    ap.add_argument("--quants", nargs="*", default=["Q8_0", "Q4_K_M"])
    args = ap.parse_args()
    require(CONVERT, "llama.cpp 的 convert_hf_to_gguf.py", "FENG_LLAMA_DIR")
    require(PY, "Python 解释器", "FENG_PY")
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    logs = ROOT / "logs"
    f16 = out / "feng-30m-f16.gguf"
    rc = run([PY, CONVERT, args.model, "--outtype", "f16", "--outfile", f16], logs / "convert_student.log")
    if rc != 0 or not f16.exists():
        print("convert failed:", (logs / "convert_student.log").read_text(encoding="utf-8", errors="replace")[-800:])
        return 2
    print(f"f16 gguf: {f16.stat().st_size/1024**2:.1f} MB")
    made = []
    for q in args.quants:
        dst = out / f"feng-30m-{q}.gguf"
        rc = run([BIN / "llama-quantize.exe", f16, dst, q, "8"], logs / f"quant_student_{q}.log")
        if rc == 0 and dst.exists():
            made.append((dst.name, round(dst.stat().st_size / 1024**2, 1)))
            print(f"  {q}: {made[-1][1]} MB")
        else:
            print(f"  {q}: FAILED -> {(logs / f'quant_student_{q}.log').read_text(encoding='utf-8', errors='replace')[-300:]}")
    (out / "export_summary.json").write_text(json.dumps(
        {"source": args.model, "f16_mb": round(f16.stat().st_size / 1024**2, 1), "quants": made},
        ensure_ascii=False, indent=2), encoding="utf-8")
    print("saved ->", out)


if __name__ == "__main__":
    main()
