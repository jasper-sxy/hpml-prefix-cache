"""Phase 2.6: timeline plot for one replay run.

Panels share the time axis (seconds since the trace started):
  1. request hit rate per window (fraction of requests that reused a cached prefix)
  2. TTFT of every request, hit vs. miss, with the TTFT SLO line
  3. KV-cache usage (from the server's /metrics samples)
  4. running and waiting requests
  5. GPU utilization (if nvidia-smi samples exist)
Drift and burst times from the trace's .meta.json are marked on every panel.

Usage:
  python client/plot_run.py results/raw/<run_dir> [--window 15] [--slo-ttft-ms 300]
Writes <run_dir>/timeline.png
"""
import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

BLUE, ORANGE = "#2a78d6", "#eb6834"  # categorical slots 1 and 2
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e6e5e0", "#fcfcfb"


def load_jsonl(path):
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [json.loads(line) for line in f]


def events_from_meta(run):
    meta_path = (run.get("trace") or "").replace(".jsonl", ".meta.json")
    if not os.path.exists(meta_path):
        return []
    with open(meta_path) as f:
        args = json.load(f)["args"]
    ev = []
    if args.get("drift_at") is not None:
        ev.append((args["drift_at"] * args["duration"], "popularity drift"))
    if args.get("burst_at") is not None:
        ev.append((args["burst_at"] * args["duration"], "one-off burst"))
    return ev


def style(ax, ylabel):
    ax.set_facecolor(SURFACE)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.set_ylabel(ylabel, color=INK2, fontsize=9)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("run_dir")
    p.add_argument("--window", type=float, default=15.0, help="seconds per hit-rate window")
    p.add_argument("--slo-ttft-ms", type=float, default=300.0)
    a = p.parse_args()

    reqs = [r for r in load_jsonl(os.path.join(a.run_dir, "requests.jsonl")) if r["ok"]]
    series = load_jsonl(os.path.join(a.run_dir, "timeseries.jsonl"))
    with open(os.path.join(a.run_dir, "run.json")) as f:
        run = json.load(f)
    events = events_from_meta(run)
    has_cached = all(r.get("cached_tokens") is not None for r in reqs)
    has_gpu = any("gpu_util_pct" in s for s in series)

    n_panels = 4 + has_gpu
    fig, axes = plt.subplots(n_panels, 1, figsize=(10, 2.1 * n_panels), sharex=True)
    fig.patch.set_facecolor(SURFACE)
    title = os.path.basename(os.path.normpath(a.run_dir))
    fig.suptitle(title, color=INK, fontsize=11, x=0.01, ha="left")

    # 1. hit rate per window
    ax = axes[0]
    style(ax, "request hit rate")
    if has_cached and reqs:
        end = max(r["start_s"] for r in reqs)
        xs, ys, t = [], [], 0.0
        while t <= end:
            win = [r for r in reqs if t <= r["start_s"] < t + a.window]
            if win:
                xs.append(t + a.window / 2)
                ys.append(sum(r["cached_tokens"] > 0 for r in win) / len(win))
            t += a.window
        ax.plot(xs, ys, color=BLUE, linewidth=2)
        ax.set_ylim(0, 1)
    else:
        ax.text(0.5, 0.5, "no per-request cached_tokens in this run", transform=ax.transAxes,
                ha="center", color=INK2, fontsize=9)

    # 2. TTFT per request
    ax = axes[1]
    style(ax, "TTFT (ms)")
    if has_cached:
        hit = [r for r in reqs if r["cached_tokens"] > 0]
        miss = [r for r in reqs if r["cached_tokens"] == 0]
        ax.scatter([r["start_s"] for r in miss], [r["ttft_ms"] for r in miss], s=9, color=ORANGE,
                   label=f"miss ({len(miss)})", linewidths=0)
        ax.scatter([r["start_s"] for r in hit], [r["ttft_ms"] for r in hit], s=9, color=BLUE,
                   label=f"hit ({len(hit)})", linewidths=0)
        ax.legend(loc="upper right", fontsize=8, frameon=False, labelcolor=INK2)
    else:
        ax.scatter([r["start_s"] for r in reqs], [r["ttft_ms"] for r in reqs], s=9, color=BLUE,
                   linewidths=0)
    ax.axhline(a.slo_ttft_ms, color=INK2, linewidth=1, linestyle="--")
    ax.text(0.0, a.slo_ttft_ms, f" SLO {a.slo_ttft_ms:.0f} ms", color=INK2, fontsize=8,
            va="bottom", transform=ax.get_yaxis_transform())
    ax.set_ylim(bottom=0)

    ts = [s["t_s"] for s in series]

    # 3. KV-cache usage
    ax = axes[2]
    style(ax, "KV-cache usage (%)")
    key = "kv_cache_usage_perc" if any("kv_cache_usage_perc" in s for s in series) else "gpu_cache_usage_perc"
    vals = [s.get(key) for s in series]
    if any(v is not None for v in vals):
        scale = 100.0 if max(v for v in vals if v is not None) <= 1.0 else 1.0
        ax.plot(ts, [v * scale if v is not None else None for v in vals], color=BLUE, linewidth=2)
        ax.set_ylim(0, 105)

    # 4. running / waiting
    ax = axes[3]
    style(ax, "requests")
    run_v = [s.get("num_requests_running") for s in series]
    wait_v = [s.get("num_requests_waiting") for s in series]
    if any(v is not None for v in run_v):
        ax.plot(ts, run_v, color=BLUE, linewidth=2, label="running")
        ax.plot(ts, wait_v, color=ORANGE, linewidth=2, label="waiting")
        ax.legend(loc="upper right", fontsize=8, frameon=False, labelcolor=INK2)
    ax.set_ylim(bottom=0)

    # 5. GPU utilization
    if has_gpu:
        ax = axes[4]
        style(ax, "GPU util (%)")
        ax.plot(ts, [s.get("gpu_util_pct") for s in series], color=BLUE, linewidth=2)
        ax.set_ylim(0, 105)

    for ax in axes:
        for t, label in events:
            ax.axvline(t, color=INK2, linewidth=1, linestyle=":")
    for t, label in events:
        axes[0].text(t, 1.0, f" {label}", color=INK2, fontsize=8, va="top")
    axes[-1].set_xlabel("time since trace start (s)", color=INK2, fontsize=9)

    fig.tight_layout()
    out = os.path.join(a.run_dir, "timeline.png")
    fig.savefig(out, dpi=130, facecolor=SURFACE)
    print("wrote", out)


if __name__ == "__main__":
    main()
