# client/  (Phase 2)

| File | Purpose |
|---|---|
| `replay.py` | sends each trace request at its timestamp (streaming), times every token, samples server metrics every second |
| `metrics.py` | TTFT / TPOT / ITL percentiles, goodput under SLOs, hit vs. miss TTFT, per-phase stats, preemptions |
| `plot_run.py` | timeline plot: hit rate per window, TTFT per request, KV usage, running/waiting requests, GPU utilization |
| `run_experiment.py` | one-click runner: for each run in a config, restart vLLM, replay, score, plot, record the setup |
| `upload_wandb.py` | uploads finished run directories to W&B from any machine; skips runs already uploaded |

`metrics.py` usage:
```bash
python client/metrics.py results/raw/<exp>/<run_dir>    # one run: full summary (JSON)
python client/metrics.py results/raw/<exp>              # all runs in an experiment: comparison table
python client/metrics.py <run_dir_1> <run_dir_2> ...    # any runs side by side
```
Every run's full summary is also written to `<run_dir>/summary.json`.
Paths are relative to the repository root (no leading `/`).

A run directory (`results/raw/<experiment>/<label>_rep<k>_<time>/`) contains
`experiment.json`, `trace_meta.json`, `server.log`, `runner.log`, `requests.jsonl`,
`timeseries.jsonl`, `run.json`, `summary.json`, and `timeline.png`.
