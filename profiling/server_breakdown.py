"""Phase 3.4: where a request's time goes, as seen by the vLLM server.

Reads metrics_before.prom / metrics_after.prom (full /metrics snapshots that client/replay.py
saves at the start and end of every run) and takes the difference, so only this run counts.
Per request: queue time (arrival -> first scheduled), prefill time (first scheduled -> first
token), decode time (first -> last token). Per engine step: tokens per step and steps per second.
Percentiles are interpolated within histogram buckets, so they are approximate.

Usage:
  python profiling/server_breakdown.py results/raw/phase3_profile_b1600            # table, one row per run
  python profiling/server_breakdown.py results/raw/phase3_profile_b1600/<run> --json
"""
import argparse
import glob
import json
import math
import os
import re
from collections import defaultdict

LINE = re.compile(r'^([a-zA-Z_:][a-zA-Z0-9_:]*)(\{[^}]*\})?\s+([-+0-9.eEinfNa]+)')
LE = re.compile(r'le="([^"]+)"')


def parse(path):
    """Metric name -> value summed over label sets; histogram buckets kept per upper bound."""
    values, buckets = defaultdict(float), defaultdict(lambda: defaultdict(float))
    with open(path) as f:
        for line in f:
            m = LINE.match(line)
            if not m:
                continue
            name, labels, v = m.group(1), m.group(2) or "", float(m.group(3))
            if name.endswith("_bucket"):
                le = LE.search(labels)
                if le:
                    buckets[name[:-7]][float(le.group(1))] += v
            else:
                values[name] += v
    return values, buckets


def quantile(q, bucket_diff):
    """Histogram quantile (linear inside the bucket), like Prometheus histogram_quantile."""
    bounds = sorted(bucket_diff)
    if not bounds or bucket_diff[bounds[-1]] <= 0:
        return float("nan")
    rank, prev_b, prev_c = q * bucket_diff[bounds[-1]], 0.0, 0.0
    for b in bounds:
        c = bucket_diff[b]
        if c >= rank:
            if math.isinf(b):
                return prev_b
            return prev_b + (b - prev_b) * (rank - prev_c) / max(c - prev_c, 1e-12)
        prev_b, prev_c = b, c
    return prev_b


def breakdown(run_dir):
    vb, bb = parse(os.path.join(run_dir, "metrics_before.prom"))
    va, ba = parse(os.path.join(run_dir, "metrics_after.prom"))

    def diff(name):
        return va.get(name, 0.0) - vb.get(name, 0.0)

    def hist(base, scale=1000.0):
        n = diff(base + "_count")
        bd = {b: ba[base][b] - bb[base].get(b, 0.0) for b in ba.get(base, {})}
        return {"n": int(n),
                "mean": round(scale * diff(base + "_sum") / n, 2) if n else None,
                "p50": round(scale * quantile(0.5, bd), 2) if n else None,
                "p99": round(scale * quantile(0.99, bd), 2) if n else None}

    wall = None
    if os.path.exists(os.path.join(run_dir, "run.json")):
        with open(os.path.join(run_dir, "run.json")) as f:
            wall = json.load(f).get("wall_s")
    prompt = diff("vllm:prompt_tokens_total")
    cached = diff("vllm:prompt_tokens_cached_total")
    gen = diff("vllm:generation_tokens_total")
    steps = hist("vllm:iteration_tokens_total", scale=1.0)
    out = {
        "run": os.path.basename(os.path.normpath(run_dir)),
        "queue_ms": hist("vllm:request_queue_time_seconds"),
        "prefill_ms": hist("vllm:request_prefill_time_seconds"),
        "decode_ms": hist("vllm:request_decode_time_seconds"),
        "ttft_ms_server": hist("vllm:time_to_first_token_seconds"),
        "e2e_ms_server": hist("vllm:e2e_request_latency_seconds"),
        "steps": steps["n"],
        "steps_per_s": round(steps["n"] / wall, 1) if wall else None,
        "tokens_per_step_mean": steps["mean"],
        "prompt_tokens": int(prompt), "prompt_tokens_cached": int(cached),
        "prompt_tokens_computed": int(prompt - cached), "generation_tokens": int(gen),
        "preemptions": int(diff("vllm:num_preemptions_total")),
    }
    work = out["prompt_tokens_computed"] + gen
    out["computed_prompt_token_share"] = round(out["prompt_tokens_computed"] / work, 3) if work else None
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("paths", nargs="+", help="run directories or experiment folders")
    p.add_argument("--json", action="store_true", help="print the full breakdown as JSON")
    a = p.parse_args()
    runs = []
    for path in a.paths:
        if os.path.exists(os.path.join(path, "metrics_after.prom")):
            runs.append(path)
        else:
            runs += sorted(os.path.dirname(x) for x in glob.glob(os.path.join(path, "*", "metrics_after.prom")))
    assert runs, "no run with metrics_before.prom / metrics_after.prom found (runs from before Phase 3.4 have none)"
    rows = [breakdown(r) for r in runs]
    if a.json:
        print(json.dumps(rows, indent=2))
        return
    print(f"{'run':<34}{'n':>6}{'queue':>8}{'q p99':>8}{'prefill':>9}{'pf p99':>8}{'decode':>8}"
          f"{'steps/s':>8}{'tok/step':>9}{'new-prompt%':>12}")
    for r in rows:
        q, pf, dc = r["queue_ms"], r["prefill_ms"], r["decode_ms"]
        print(f"{r['run'][:33]:<34}{q['n']:>6}{q['mean'] or 0:>8.1f}{q['p99'] or 0:>8.0f}{pf['mean'] or 0:>9.1f}"
              f"{pf['p99'] or 0:>8.0f}{dc['mean'] or 0:>8.0f}{r['steps_per_s'] or 0:>8.1f}"
              f"{r['tokens_per_step_mean'] or 0:>9.1f}{100 * (r['computed_prompt_token_share'] or 0):>12.1f}")
    print("times in ms, means unless marked p99; new-prompt% = share of computed tokens that are "
          "uncached prompt tokens (the rest are generated tokens)")


if __name__ == "__main__":
    main()
