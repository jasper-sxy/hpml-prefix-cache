"""Phase 3.3: load-sweep plot.

Reads every run under the given folders (summary.json + experiment.json + trace_meta.json)
and draws, against the arrival rate, one line per configuration:
  goodput (with the "every request on time" diagonal), SLO attainment, TTFT p99, preemptions.
A configuration is the run label without its "_r<rate>" suffix, plus the KV budget when
several budgets are plotted together.

Usage:
  python client/plot_sweep.py results/raw/phase3_sweep_b800 [more experiment folders] [-o out.png]
"""
import argparse
import glob
import json
import os
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]  # fixed order
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e6e5e0", "#fcfcfb"
NICE = {"stable_off": "Stable, caching off", "stable_lru": "Stable, LRU",
        "oneoff30_lru": "30% one-off, LRU", "oneoff15_lru": "15% one-off, LRU"}


def load(path):
    with open(path) as f:
        return json.load(f)


def collect(folders):
    rows = []
    for folder in folders:
        for s_path in glob.glob(os.path.join(folder, "**", "summary.json"), recursive=True):
            d = os.path.dirname(s_path)
            exp_path, meta_path = os.path.join(d, "experiment.json"), os.path.join(d, "trace_meta.json")
            if not (os.path.exists(exp_path) and os.path.exists(meta_path)):
                continue
            s, exp, meta = load(s_path), load(exp_path), load(meta_path)
            cfg = exp.get("config", {})
            rows.append({
                "base": re.sub(r"_r[\d.]+$", "", cfg.get("label", "run")),
                "blocks": cfg.get("num_gpu_blocks"),
                "rate": meta["args"]["rate"],
                "goodput": s["goodput_req_s"],
                "slo": 100 * s["slo_attainment"],
                "ttft_p99_s": s["ttft_ms"]["p99"] / 1000 if s["ttft_ms"]["p99"] else None,
                "preempt": s.get("preemptions") or 0,
            })
    return rows


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
    ax.set_xlabel("arrival rate (requests/s)", color=INK2, fontsize=9)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("folders", nargs="+")
    p.add_argument("-o", "--out", default=None, help="default: <first folder>/sweep.png")
    p.add_argument("--slo-ttft-ms", type=float, default=300.0)
    a = p.parse_args()
    rows = collect(a.folders)
    assert rows, "no finished runs found"
    several_budgets = len({r["blocks"] for r in rows}) > 1
    keys = []
    for r in sorted(rows, key=lambda r: (list(NICE).index(r["base"]) if r["base"] in NICE else 99,
                                         r["base"], r["blocks"] or 0)):
        k = (r["base"], r["blocks"])
        if k not in keys:
            keys.append(k)

    def name(k):
        n = NICE.get(k[0], k[0])
        return f"{n}, {k[1]} blocks" if several_budgets else n

    fig, axes = plt.subplots(2, 2, figsize=(11, 7.2))
    fig.patch.set_facecolor(SURFACE)
    panels = [("goodput", "goodput (requests/s meeting SLO)"), ("slo", "requests meeting SLO (%)"),
              ("ttft_p99_s", "TTFT p99 (s, log scale)"), ("preempt", "preemptions per run")]
    max_rate = max(r["rate"] for r in rows)
    for ax, (field, label) in zip(axes.flat, panels):
        style(ax, label)
        for i, k in enumerate(keys):
            pts = sorted((r["rate"], r[field]) for r in rows if (r["base"], r["blocks"]) == k and r[field] is not None)
            if pts:
                ax.plot([x for x, _ in pts], [y for _, y in pts], color=PALETTE[i % len(PALETTE)],
                        linewidth=2, marker="o", markersize=5, label=name(k))
        if field == "goodput":
            ax.plot([0, max_rate], [0, max_rate], color=INK2, linewidth=1, linestyle=":")
            ax.text(max_rate * 0.98, max_rate * 0.98, "every request on time ", color=INK2,
                    fontsize=8, ha="right", va="bottom")
            ax.set_ylim(bottom=0)
        if field == "slo":
            ax.set_ylim(0, 105)
        if field == "ttft_p99_s":
            ax.set_yscale("log")
            ax.axhline(a.slo_ttft_ms / 1000, color=INK2, linewidth=1, linestyle="--")
            ax.text(0.2, a.slo_ttft_ms / 1000, f" SLO {a.slo_ttft_ms:.0f} ms", color=INK2,
                    fontsize=8, va="bottom")
        if field == "preempt":
            ax.set_ylim(bottom=0)
        ax.set_xlim(0, max_rate * 1.05)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(labels), frameon=False, fontsize=9,
               labelcolor=INK2, bbox_to_anchor=(0.5, 0.995))
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = a.out or os.path.join(a.folders[0], "sweep.png")
    fig.savefig(out, dpi=130, facecolor=SURFACE)
    print("wrote", out)
    # text table, one row per configuration
    for k in keys:
        pts = sorted((r["rate"], r["goodput"]) for r in rows if (r["base"], r["blocks"]) == k)
        best = max(pts, key=lambda x: x[1])
        print(f"{name(k):<34} peak goodput {best[1]:.2f} req/s at {best[0]:g} req/s")


if __name__ == "__main__":
    main()
