"""Phase 2.5: metrics for one or more replay runs.

Reads requests.jsonl (+ run.json) from each run directory, computes latency percentiles,
goodput under the given SLOs, and prefix-cache hit statistics, writes summary.json into
the run directory, and prints a comparison table when several runs are given.

Definitions (same as vLLM's own benchmark):
  TTFT  time from sending a request to its first output token
  TPOT  (end-to-end time - TTFT) / (output tokens - 1): average time per later token
  ITL   gap between two consecutive output tokens (pooled over all requests)
  goodput  requests per second that met BOTH SLOs (failed requests count as not met)

Usage:
  python client/metrics.py results/raw/<run_dir> [more run dirs ...] --slo-ttft-ms 500 --slo-tpot-ms 50
"""
import argparse
import json
import os

import numpy as np


def pct(xs, q):
    xs = [x for x in xs if x is not None]
    return round(float(np.percentile(xs, q)), 2) if xs else None


def lat_stats(xs):
    xs = [x for x in xs if x is not None]
    return {"p50": pct(xs, 50), "p95": pct(xs, 95), "p99": pct(xs, 99), "n": len(xs)}


def tpot_ms(r):
    n = r.get("completion_tokens") or r["n_chunks"]
    if not r["ok"] or not n or n < 2:
        return None
    return (r["e2e_ms"] - r["ttft_ms"]) / (n - 1)


def summarize(run_dir, slo_ttft, slo_tpot):
    with open(os.path.join(run_dir, "requests.jsonl")) as f:
        reqs = [json.loads(line) for line in f]
    run = {}
    if os.path.exists(os.path.join(run_dir, "run.json")):
        with open(os.path.join(run_dir, "run.json")) as f:
            run = json.load(f)
    ok = [r for r in reqs if r["ok"]]
    for r in ok:
        r["tpot_ms"] = tpot_ms(r)
    duration = max(r["end_s"] for r in reqs) - min(r["start_s"] for r in reqs)
    good = [r for r in ok
            if r["ttft_ms"] <= slo_ttft and (r["tpot_ms"] is None or r["tpot_ms"] <= slo_tpot)]

    s = {
        "run": os.path.basename(os.path.normpath(run_dir)),
        "trace": run.get("trace"),
        "n_requests": len(reqs),
        "n_errors": len(reqs) - len(ok),
        "duration_s": round(duration, 2),
        "throughput_req_s": round(len(ok) / duration, 3),
        "output_tokens_s": round(sum(r.get("completion_tokens") or 0 for r in ok) / duration, 1),
        "slo": {"ttft_ms": slo_ttft, "tpot_ms": slo_tpot},
        "goodput_req_s": round(len(good) / duration, 3),
        "slo_attainment": round(len(good) / len(reqs), 4),
        "ttft_ms": lat_stats([r["ttft_ms"] for r in ok]),
        "tpot_ms": lat_stats([r["tpot_ms"] for r in ok]),
        "itl_ms": lat_stats([x for r in ok for x in r["itl_ms"]]),
        "token_hit_rate_server": run.get("prefix_cache_hit_rate"),  # from /metrics deltas
    }

    if ok and all(r.get("cached_tokens") is not None for r in ok):
        hits = [r for r in ok if r["cached_tokens"] > 0]
        misses = [r for r in ok if r["cached_tokens"] == 0]
        s["request_hit_rate"] = round(len(hits) / len(ok), 4)
        s["cached_token_share"] = round(
            sum(r["cached_tokens"] for r in ok) / sum(r["prompt_tokens"] for r in ok), 4)
        s["ttft_ms_hit"] = lat_stats([r["ttft_ms"] for r in hits])
        s["ttft_ms_miss"] = lat_stats([r["ttft_ms"] for r in misses])

    phases = sorted({r["phase"] for r in reqs})
    if len(phases) > 1:
        s["by_phase"] = {}
        for ph in phases:
            sub = [r for r in ok if r["phase"] == ph]
            entry = {"n": len(sub), "ttft_ms_p50": pct([r["ttft_ms"] for r in sub], 50)}
            if "request_hit_rate" in s and sub:
                entry["request_hit_rate"] = round(sum(r["cached_tokens"] > 0 for r in sub) / len(sub), 4)
            s["by_phase"][ph] = entry
    return s


def main():
    p = argparse.ArgumentParser()
    p.add_argument("run_dirs", nargs="+")
    p.add_argument("--slo-ttft-ms", type=float, default=500.0, help="provisional until frozen in Phase 3")
    p.add_argument("--slo-tpot-ms", type=float, default=50.0, help="provisional until frozen in Phase 3")
    a = p.parse_args()

    rows = []
    for d in a.run_dirs:
        s = summarize(d, a.slo_ttft_ms, a.slo_tpot_ms)
        with open(os.path.join(d, "summary.json"), "w") as f:
            json.dump(s, f, indent=2)
        rows.append(s)
    if len(rows) == 1:
        print(json.dumps(rows[0], indent=2))
        return
    head = f"{'run':<44}{'n':>5}{'goodput':>9}{'SLO%':>7}{'TTFT50':>8}{'TTFT99':>8}{'TPOT50':>8}{'TPOT99':>8}{'hit%':>7}"
    print(head)
    for s in rows:
        hit = s.get("request_hit_rate")
        print(f"{s['run'][-44:]:<44}{s['n_requests']:>5}{s['goodput_req_s']:>9.3f}"
              f"{100 * s['slo_attainment']:>7.1f}{s['ttft_ms']['p50']:>8.1f}{s['ttft_ms']['p99']:>8.1f}"
              f"{s['tpot_ms']['p50']:>8.1f}{s['tpot_ms']['p99']:>8.1f}"
              f"{(100 * hit if hit is not None else float('nan')):>7.1f}")


if __name__ == "__main__":
    main()
