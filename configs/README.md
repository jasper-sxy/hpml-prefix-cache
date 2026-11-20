# configs/

One YAML file per experiment, run with `python client/run_experiment.py configs/<file>.yaml`
(add `--dry-run` to print the plan only). `defaults` apply to every run; each entry in `runs`
overrides any of them.

Keys: `model`, `num_gpu_blocks`, `prefix_caching`, `policy` (exported as `KV_EVICT_POLICY`,
used from Phase 6), `max_model_len`, `max_num_seqs`, `block_size`, `seed`, `port`, `repeats`,
`sample_interval`, `slo_ttft_ms`, `slo_tpot_ms`, `extra_server_args`, `env`, `trace`, `label`.

| File | Experiment |
|---|---|
| `phase2_three_scenarios.yaml` | native LRU on the stable, drift, and burst traces at 2 requests/s |
| `fork_check.yaml` | stable trace after switching to the editable install of our fork (must match Phase 2) |
| `phase3_calibration.yaml` | prefix caching off at 1 request/s, 3 repeats: SLO calibration |
| `phase3_oneoff_check.yaml` | 15% and 30% one-off traffic at 2 requests/s, 800 blocks |
| `phase3_sweep_b800.yaml` | caching off, stable LRU, 30% one-off LRU at 2-12 requests/s, 800 blocks |
| `phase3_sweep_b1600.yaml` | the same three at 6-12 requests/s, 1600 blocks |
| `phase3_sweep_b1600_ext.yaml` | stable LRU at 14 and 16 requests/s, 1600 blocks |
| `phase3_scenarios_b1600.yaml` | drift, burst, drift + 30% one-off at the base sweep rates, 1600 blocks |
| `phase3_sensitivity_b1600.yaml` | alpha 0.6 / 1.2 and prefix 512 / 2048, stable and 30% one-off, 1600 blocks |
| `phase3_sensitivity_b1600_fill.yaml` | extra rates so that every sensitivity curve has a knee |
| `phase3_profile_b1600.yaml` | 7 profiling runs with vLLM's per-iteration log (Phase 3.4) |
