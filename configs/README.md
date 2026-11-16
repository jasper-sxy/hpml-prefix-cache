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
