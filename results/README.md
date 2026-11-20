# results/
Small summary tables and figures. Raw runs (`results/raw/<experiment>/<run>/`) are not in git;
they are uploaded to W&B with `client/upload_wandb.py`.

| Path | Content |
|---|---|
| `tables/phase3/<experiment>.txt` | `client/metrics.py` table for each Phase 3 experiment (one row per run) |
| `figures/phase3/budget.png` | caching off / stable LRU / 30% one-off LRU at 800 and 1600 blocks |
| `figures/phase3/scenarios_b1600.png` | stable, one-off, drift, burst, drift + one-off at 1600 blocks |
| `figures/phase3/alpha_b1600.png` | popularity skew alpha 0.6 / 0.9 / 1.2 |
| `figures/phase3/prefix_b1600.png` | prefix length 512 / 1024 / 2048 |
| `figures/phase3/hit_vs_goodput.png` | request hit rate vs peak goodput, one point per configuration |
| `figures/phase3/timeline_drift_r12.png`, `timeline_burst_r12.png` | single runs at 12 req/s over time |
| `figures/phase3/steps_lru_r12.png` | step time vs prefill tokens from vLLM's iteration log (Phase 3.4) |
| `figures/phase3/nsys_steps_lru_r12.png` | true GPU time per step from Nsight Systems (Phase 3.4) |

Regenerate the sweep figures (on the server, from the repository root):

    R=results/raw; B="$R/phase3_sweep_b1600 $R/phase3_sweep_b1600_ext"
    D="$R/phase3_sensitivity_b1600 $R/phase3_sensitivity_b1600_fill"; F=results/figures/phase3
    python client/plot_sweep.py $R/phase3_sweep_b800 $B --title "KV budget: 800 vs 1600 blocks" -o $F/budget.png
    python client/plot_sweep.py $B $R/phase3_scenarios_b1600 --title "Scenarios at 1600 blocks" -o $F/scenarios_b1600.png
    python client/plot_sweep.py $B $D --only 'a0\.6|a1\.2|^stable_lru$|^oneoff30_lru$' -o $F/alpha_b1600.png
    python client/plot_sweep.py $B $D --only 'L512|L2048|^stable_lru$|^oneoff30_lru$' -o $F/prefix_b1600.png
    python client/plot_hit_goodput.py $R/phase3_sweep_b800 $B $D -o $F/hit_vs_goodput.png
    for e in $R/phase3_*/; do python client/metrics.py $e > results/tables/phase3/$(basename $e).txt; done
