"""Phase 3.3: prefix-cache hit rate vs peak goodput, one point per configuration.

A configuration is the run label without its "_r<rate>" suffix, at one KV budget. For each
one: peak goodput over the rates it was run at, and its median request hit rate over those
runs. A hollow marker means the curve had no knee in the tested range (the peak is at the
highest rate and that run still met the SLO for >= 95% of requests), so the true peak is higher.
Colors = prefix length, marker = KV budget.

Usage:
  python client/plot_hit_goodput.py results/raw/phase3_sweep_b1600 [more experiment folders] -o out.png
      [--only REGEX]   keep configurations whose name matches
      [--skip REGEX]   drop configurations whose name matches, e.g. 'drift|burst'
"""
import argparse
import glob
import json
import os
import re
import statistics

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

PALETTE = {512: "#2a78d6", 1024: "#eb6834", 2048: "#1baf7a"}  # same hues as plot_sweep.py
MARKER = {800: "s", 1600: "o", 3200: "D"}
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e6e5e0", "#fcfcfb"


def load(path):
    with open(path) as f:
        return json.load(f)


def short(base):
    """Point label: stable_a0.6_lru -> "stable alpha 0.6", oneoff30_L512_lru -> "one-off"."""
    if base == "stable_off":
        return "caching off"
    s = re.sub(r"_lru$", "", base)
    s = s.replace("driftoneoff30", "drift + one-off").replace("oneoff30", "one-off")
    s = re.sub(r"_L\d+", "", s)  # prefix length is the color
    return s.replace("_a", " alpha ").replace("_", " ")


def collect(folders):
    groups = {}
    for folder in folders:
        for s_path in glob.glob(os.path.join(folder, "**", "summary.json"), recursive=True):
            d = os.path.dirname(s_path)
            exp_path, meta_path = os.path.join(d, "experiment.json"), os.path.join(d, "trace_meta.json")
            if not (os.path.exists(exp_path) and os.path.exists(meta_path)):
                continue
            s, cfg, args = load(s_path), load(exp_path).get("config", {}), load(meta_path)["args"]
            if not cfg.get("prefix_caching", True) and s.get("request_hit_rate") is None:
                s["request_hit_rate"] = 0.0
            key = (re.sub(r"_r[\d.]+$", "", cfg.get("label", "run")), cfg.get("num_gpu_blocks"),
                   args.get("prefix_len", 1024))
            groups.setdefault(key, []).append(
                (args["rate"], s["goodput_req_s"], s["slo_attainment"], s.get("request_hit_rate")))
    return groups


def main():
    p = argparse.ArgumentParser()
    p.add_argument("folders", nargs="+")
    p.add_argument("-o", "--out", required=True)
    p.add_argument("--only", default=None, help="regex on the configuration name")
    p.add_argument("--skip", default=None, help="regex on the configuration name to leave out")
    p.add_argument("--title", default="Hit rate vs peak goodput (native LRU)")
    a = p.parse_args()
    groups = collect(a.folders)
    if a.only:
        groups = {k: v for k, v in groups.items() if re.search(a.only, k[0])}
    if a.skip:
        groups = {k: v for k, v in groups.items() if not re.search(a.skip, k[0])}
    assert groups, "no finished runs found"

    points = []
    for (base, blocks, plen), runs in sorted(groups.items()):
        runs.sort()
        rate, good, slo, _ = max(runs, key=lambda r: r[1])
        open_ended = rate == runs[-1][0] and slo >= 0.95
        hits = [r[3] for r in runs if r[3] is not None]
        hit = 100 * statistics.median(hits) if hits else 0.0
        points.append((base, blocks, plen, hit, good, rate, open_ended))

    fig, ax = plt.subplots(figsize=(8.5, 5.6))
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9)
    seen = set()
    for base, blocks, plen, hit, good, rate, open_ended in sorted(points, key=lambda p: (p[2], p[1])):
        color = PALETTE.get(plen, INK2)
        legend = f"prefix {plen}, {blocks} blocks"
        ax.scatter([hit], [good], s=64, marker=MARKER.get(blocks, "o"), linewidths=2, zorder=3,
                   facecolors="none" if open_ended else color, edgecolors=color,
                   label=None if legend in seen else legend)
        seen.add(legend)
        ax.annotate(short(base) + (" (no knee)" if open_ended else ""), (hit, good),
                    xytext=(6, 4), textcoords="offset points", fontsize=7.5, color=INK2)
    ax.set_xlim(-2, max(p[3] for p in points) + 18)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("request hit rate (%, median over the runs of a configuration)", color=INK2, fontsize=9)
    ax.set_ylabel("peak goodput (requests/s meeting SLO)", color=INK2, fontsize=9)
    ax.set_title(a.title, color=INK, fontsize=11)
    ax.legend(frameon=False, fontsize=8.5, labelcolor=INK2, loc="upper left")
    fig.tight_layout()
    fig.savefig(a.out, dpi=130, facecolor=SURFACE)
    print("wrote", a.out)
    print(f"{'configuration':<26}{'blocks':>7}{'prefix':>7}{'hit%':>7}{'peak':>7}{'@rate':>7}")
    for base, blocks, plen, hit, good, rate, open_ended in sorted(points, key=lambda p: (p[1], p[2], p[3])):
        print(f"{base:<26}{blocks:>7}{plen:>7}{hit:>7.1f}{good:>7.2f}{rate:>7g}" + ("  no knee" if open_ended else ""))


if __name__ == "__main__":
    main()
