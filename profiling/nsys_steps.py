"""Phase 3.4 C: true GPU time per engine step, from an Nsight Systems capture.

vLLM's "cuda" profiler wraps every engine step in an NVTX range named
"execute_context_<n>(<prefill tokens>)_generation_<m>(<decode tokens>)". This script reads the
SQLite export of the capture, assigns every GPU kernel to the step whose range was open (or most
recently opened) when the kernel was launched, and reports per step: GPU kernel time and its
split into GEMM / attention / other. It also reports how busy the GPU was over the window and
fits  gpu_ms = a + b * prefill_tokens / 1000 + c * decode_requests  (cf. iteration_log.py).

Usage:
  python profiling/nsys_steps.py profiling/nsys/lru_r12.sqlite [--plot out.png] [--csv steps.csv]
(the .sqlite is written next to the .nsys-rep by `nsys stats`, or by `nsys export --type sqlite`)
"""
import argparse
import bisect
import re
import sqlite3

import numpy as np

NAME = re.compile(r"execute_context_(\d+)\((\d+)\)_generation_(\d+)\((\d+)\)")


def category(kernel):
    k = kernel.lower()
    if "flash" in k or "attn" in k or "attention" in k or "paged" in k:
        return "attention"
    if "gemm" in k or "cutlass" in k or "nvjet" in k or "cublas" in k or "matmul" in k:
        return "gemm"
    return "other"


def load(db):
    con = sqlite3.connect(db)
    tables = {r[0] for r in con.execute("select name from sqlite_master where type='table'")}
    has_strings = "StringIds" in tables
    nvtx_text = ("coalesce(n.text, s.value)" if has_strings else "n.text")
    join = " left join StringIds s on n.textId = s.id" if has_strings else ""
    ranges = []
    for start, end, text in con.execute(f"select n.start, n.end, {nvtx_text} from NVTX_EVENTS n{join}"):
        m = NAME.search(text or "")
        if m and end:
            ranges.append((start, end, *map(int, m.groups())))
    ranges.sort()
    kname = "coalesce(s.value, '')" if has_strings else "''"
    kjoin = " left join StringIds s on k.demangledName = s.id" if has_strings else ""
    kernels = con.execute(
        f"select k.start, k.end, r.start, {kname} from CUPTI_ACTIVITY_KIND_KERNEL k"
        f" join CUPTI_ACTIVITY_KIND_RUNTIME r on k.correlationId = r.correlationId{kjoin}").fetchall()
    return ranges, kernels


def analyze(ranges, kernels):
    starts = [r[0] for r in ranges]
    n = len(ranges)
    gpu = np.zeros((n, 3))  # gemm, attention, other (ms)
    first = np.full(n, np.inf)
    last = np.zeros(n)
    unassigned = 0.0
    cats = {"gemm": 0, "attention": 1, "other": 2}
    for k_start, k_end, launch, name in kernels:
        i = bisect.bisect_right(starts, launch) - 1
        dur = (k_end - k_start) / 1e6
        if i < 0 or (i == n - 1 and launch > ranges[-1][1] + 50e6):
            unassigned += dur
            continue
        gpu[i, cats[category(name)]] += dur
        first[i] = min(first[i], k_start)
        last[i] = max(last[i], k_end)
    steps = []
    for i, (s, e, ctx_req, ctx_tok, gen_req, gen_tok) in enumerate(ranges):
        if not np.isfinite(first[i]):
            continue
        steps.append({"ctx_req": ctx_req, "ctx_tok": ctx_tok, "gen_req": gen_req, "gen_tok": gen_tok,
                      "cpu_ms": (e - s) / 1e6, "gpu_ms": gpu[i].sum(), "gemm_ms": gpu[i, 0],
                      "attn_ms": gpu[i, 1], "other_ms": gpu[i, 2],
                      "gpu_span_ms": (last[i] - first[i]) / 1e6})
    # GPU busy fraction over the window: union of kernel intervals / window length
    iv = sorted((k[0], k[1]) for k in kernels)
    busy, cur_s, cur_e = 0, iv[0][0], iv[0][1]
    for s, e in iv[1:]:
        if s > cur_e:
            busy += cur_e - cur_s
            cur_s, cur_e = s, e
        else:
            cur_e = max(cur_e, e)
    busy += cur_e - cur_s
    window = iv[-1][1] - iv[0][0]
    return steps, busy / window, window / 1e9, unassigned


def main():
    p = argparse.ArgumentParser()
    p.add_argument("sqlite")
    p.add_argument("--plot", default=None)
    p.add_argument("--csv", default=None)
    a = p.parse_args()
    ranges, kernels = load(a.sqlite)
    assert ranges, "no execute_context_* NVTX ranges: was vLLM started with --profiler-config '{\"profiler\": \"cuda\"}'?"
    steps, busy_frac, window_s, unassigned = analyze(ranges, kernels)
    S = {k: np.array([s[k] for s in steps]) for k in steps[0]}
    pre = S["ctx_tok"] > 0
    total = S["gpu_ms"].sum()
    print(f"window {window_s:.2f} s, {len(steps)} steps ({len(steps) / window_s:.1f} steps/s), "
          f"GPU busy {100 * busy_frac:.1f}% of the window, unassigned kernel time {unassigned:.1f} ms")
    print(f"kernel time: GEMM {100 * S['gemm_ms'].sum() / total:.1f}%, attention "
          f"{100 * S['attn_ms'].sum() / total:.1f}%, other {100 * S['other_ms'].sum() / total:.1f}%")
    print(f"{'':<22}{'steps':>7}{'gpu ms p50':>11}{'gpu ms mean':>12}{'cpu ms p50':>11}{'share of GPU time':>19}")
    for label, mask in (("decode-only steps", ~pre), ("steps with prefill", pre)):
        if mask.any():
            print(f"{label:<22}{mask.sum():>7}{np.median(S['gpu_ms'][mask]):>11.2f}"
                  f"{S['gpu_ms'][mask].mean():>12.2f}{np.median(S['cpu_ms'][mask]):>11.2f}"
                  f"{100 * S['gpu_ms'][mask].sum() / total:>18.1f}%")
    X = np.column_stack([np.ones(len(steps)), S["ctx_tok"] / 1000.0, S["gen_req"]])
    coef, *_ = np.linalg.lstsq(X, S["gpu_ms"], rcond=None)
    pred = X @ coef
    r2 = 1 - np.sum((S["gpu_ms"] - pred) ** 2) / np.sum((S["gpu_ms"] - S["gpu_ms"].mean()) ** 2)
    print(f"fit gpu_ms = {coef[0]:.2f} + {coef[1]:.1f} * prefill_tokens/1000 + {coef[2]:.3f} * decode_requests"
          f"   (R2 {r2:.2f})")
    print(f"mean GPU ms per step {S['gpu_ms'].mean():.2f} vs wall ms per step {1000 * window_s / len(steps):.2f}")
    if a.csv:
        import csv
        with open(a.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(steps[0]))
            w.writeheader()
            w.writerows(steps)
        print("wrote", a.csv)
    if a.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 5))
        sc = ax.scatter(S["ctx_tok"], S["gpu_ms"], c=S["gen_req"], s=8, cmap="viridis", alpha=0.7)
        fig.colorbar(sc, label="decode requests in the step")
        ax.set_xlabel("prefill tokens in the step")
        ax.set_ylabel("GPU kernel time of the step (ms, Nsight Systems)")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(a.plot, dpi=130)
        print("wrote", a.plot)


if __name__ == "__main__":
    main()
