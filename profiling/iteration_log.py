"""Phase 3.4: cost of one engine step, from vLLM's per-iteration log.

Needs runs whose server was started with --enable-logging-iteration-details: every engine
step then logs how many prefill ("context") tokens and decode ("generation") requests it ran
and how long the engine waited for the GPU result ("elapsed"). This script fits

    elapsed_ms = a + b * prefill_tokens / 1000 + c * decode_requests

and splits the run's total step time into a fixed part (a), prefill (b) and decode (c).
Caveat: with async scheduling (vLLM's default) the scheduler prepares the next step while the
GPU runs, so "elapsed" is the GPU time not hidden behind that CPU work: a slight underestimate.

Usage:
  python profiling/iteration_log.py results/raw/phase3_profile_b1600            # one row per run
  python profiling/iteration_log.py results/raw/phase3_profile_b1600/<run> --plot steps.png
"""
import argparse
import glob
import os
import re

import numpy as np

PAT = re.compile(r"Iteration\((\d+)\): (\d+) context requests, (\d+) context tokens, "
                 r"(\d+) generation requests, (\d+) generation tokens, "
                 r"iteration elapsed time: ([\d.]+) ms( \(dummy\))?")


def read_steps(run_dir):
    rows = []
    with open(os.path.join(run_dir, "server.log"), errors="replace") as f:
        for line in f:
            m = PAT.search(line)
            if m and not m.group(7):
                _, ctx_req, ctx_tok, gen_req, gen_tok, ms = (float(x) for x in m.groups()[:6])
                if ctx_tok + gen_tok > 0:
                    rows.append((ctx_req, ctx_tok, gen_req, gen_tok, ms))
    return np.array(rows) if rows else np.zeros((0, 5))


def fit(steps):
    ctx_tok, gen_req, ms = steps[:, 1], steps[:, 2], steps[:, 4]
    X = np.column_stack([np.ones(len(ms)), ctx_tok / 1000.0, gen_req])
    coef, *_ = np.linalg.lstsq(X, ms, rcond=None)
    pred = X @ coef
    r2 = 1 - np.sum((ms - pred) ** 2) / max(np.sum((ms - ms.mean()) ** 2), 1e-12)
    a, b, c = coef
    total = ms.sum()
    return {"a_ms": a, "b_ms_per_1k_prefill_tokens": b, "c_ms_per_decode_request": c, "r2": r2,
            "share_fixed": a * len(ms) / total, "share_prefill": b * ctx_tok.sum() / 1000 / total,
            "share_decode": c * gen_req.sum() / total}


def summarize(run_dir):
    s = read_steps(run_dir)
    if len(s) < 10:
        return None
    pre = s[:, 1] > 0
    out = {"run": os.path.basename(os.path.normpath(run_dir)), "steps": len(s),
           "prefill_step_pct": 100 * pre.mean(),
           "decode_only_ms_p50": float(np.median(s[~pre, 4])) if (~pre).any() else float("nan"),
           "with_prefill_ms_p50": float(np.median(s[pre, 4])) if pre.any() else float("nan"),
           "decode_reqs_mean": float(s[:, 2].mean()),
           "prefill_tokens_per_prefill_step": float(s[pre, 1].mean()) if pre.any() else 0.0}
    out.update(fit(s))
    return out


def plot(run_dir, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    s = read_steps(run_dir)
    fig, ax = plt.subplots(figsize=(8, 5))
    sc = ax.scatter(s[:, 1], s[:, 4], c=s[:, 2], s=6, cmap="viridis", alpha=0.6)
    fig.colorbar(sc, label="decode requests in the step")
    ax.set_xlabel("prefill tokens in the step")
    ax.set_ylabel("step elapsed (ms)")
    ax.set_title(os.path.basename(os.path.normpath(run_dir)), fontsize=10)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    print("wrote", out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("paths", nargs="+", help="run directories or experiment folders")
    p.add_argument("--plot", default=None, help="scatter of step time vs prefill tokens (first run only)")
    a = p.parse_args()
    runs = []
    for path in a.paths:
        if os.path.exists(os.path.join(path, "server.log")):
            runs.append(path)
        else:
            runs += sorted(os.path.dirname(x) for x in glob.glob(os.path.join(path, "*", "server.log")))
    rows = [r for r in (summarize(d) for d in runs) if r]
    assert rows, "no iteration lines found: start the server with --enable-logging-iteration-details"
    print(f"{'run':<34}{'steps':>7}{'pf%':>6}{'dec p50':>8}{'pf p50':>8}{'a':>7}{'b/1k':>7}{'c':>7}"
          f"{'R2':>6}{'fixed%':>7}{'pf%':>6}{'dec%':>6}")
    for r in rows:
        print(f"{r['run'][:33]:<34}{r['steps']:>7}{r['prefill_step_pct']:>6.1f}{r['decode_only_ms_p50']:>8.1f}"
              f"{r['with_prefill_ms_p50']:>8.1f}{r['a_ms']:>7.2f}{r['b_ms_per_1k_prefill_tokens']:>7.1f}"
              f"{r['c_ms_per_decode_request']:>7.3f}{r['r2']:>6.2f}{100 * r['share_fixed']:>7.1f}"
              f"{100 * r['share_prefill']:>6.1f}{100 * r['share_decode']:>6.1f}")
    print("ms per step; a = fixed cost per step, b = per 1000 prefill tokens, c = per decode request; "
          "last three columns split the run's total step time")
    if a.plot:
        plot(runs[0], a.plot)


if __name__ == "__main__":
    main()
