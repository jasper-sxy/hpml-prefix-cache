# Workload-Aware Prefix KV-Cache Management (COMS 6998 HPML, Fall 2026)

We study prefix-cache eviction in vLLM under a fixed KV-cache budget, and test whether a
workload-aware policy (reuse history + recency, optionally weighted by prefill cost)
improves goodput over vLLM's native LRU eviction.

## Status
- [x] Phase 1: vLLM running with a capped KV cache; prefix-cache smoke test passed
- [x] Phase 2: document pool, trace generator, replay client, metrics, server sampling,
      one-click runner, W&B upload
- [ ] Phase 3: load sweep, SLO calibration, profiling, correctness baseline
- [ ] Phase 4: instrument vLLM's eviction path (block event log)
- [ ] Phase 5: offline simulator and policy selection
- [ ] Phase 6: implement the policy in vLLM, tests, overhead measurement
- [ ] Phase 7: held-out experiments and cross-model validation
- [ ] Phase 8: analysis, demo, report

## Environment (tested)
| Item | Value |
|---|---|
| GPU | NVIDIA RTX 4080 SUPER 32 GB (AutoDL), driver 580 |
| vLLM | 0.33.0 (commit 42d9d569) |
| PyTorch | 2.14.0+cu130 |
| Python | 3.12 (conda env `vllm`), plus matplotlib, pyyaml |
| Model | Qwen/Qwen2.5-3B-Instruct (Llama-3.2-3B-Instruct pending access) |

## Known issues
- If the system `nvcc` is older than CUDA 12.8, FlashInfer's sampler JIT build fails
  (`nvcc fatal: Unknown option '--compress-mode=size'`). Fix: `export VLLM_USE_FLASHINFER_SAMPLER=0`
  (the launch script and the experiment runner set it).
- On AutoDL: use `source /etc/network_turbo` for GitHub/Hugging Face, and
  `unset http_proxy https_proxy` before pip or local HTTP requests.
- W&B is not reachable from the AutoDL server (HTTP 403). Experiments save everything locally;
  `client/upload_wandb.py` uploads them from any machine that can reach W&B.
- Run only one vLLM server per GPU. The experiment runner stops any running server first.

## Pipeline
workload/build_doc_pool.py -> workload/data/<model>/docs_{dev,test}.jsonl WikiText-103 prefixes as token IDs
workload/make_trace.py -> workload/traces/<model>/<trace>.jsonl Zipf popularity, Poisson arrivals,
popularity drift, one-off bursts
client/run_experiment.py configs/<exp>.yaml on the GPU server; per run:
restart vLLM (empty cache) -> client/replay.py -> client/metrics.py -> client/plot_run.py
-> results/raw/<exp>/<label>rep<k><time>/ self-contained run directory
client/upload_wandb.py results/raw/<exp> on any machine that reaches W&B
Generated data, traces, and raw results are not in git; they are reproducible from the scripts
and seeds. The document pool for Qwen2.5-3B (seed 0, 100 dev + 100 test, 2048 tokens) has
`token_digest` `0274c44d8d4065e1`.

## Quick start
```bash
# on the GPU server (conda env vllm)
python workload/build_doc_pool.py --model Qwen/Qwen2.5-3B-Instruct
python workload/make_trace.py                                    # stable popularity
python workload/make_trace.py --drift-at 0.5                     # re-rank documents halfway
python workload/make_trace.py --burst-at 0.5 --burst-size 20     # one-off burst halfway
python client/run_experiment.py configs/phase2_three_scenarios.yaml   # inside screen, ~20 min
python client/metrics.py results/raw/phase2_three_scenarios           # comparison table

# on a machine that reaches W&B, after copying results/raw/<exp> there
python client/upload_wandb.py results/raw/phase2_three_scenarios
```
Manual alternative: `bash scripts/start_server.sh` in one screen, then
`python client/replay.py --trace <trace>` and `python client/metrics.py <run_dir>`.

## Results

### Phase 1: prefix-cache smoke test
Qwen2.5-3B, 800 KV blocks (12,800 tokens), one ~2,300-token shared prefix.

| Request | Prefix cache ON | Prefix cache OFF |
|---|---|---|
| A: doc1 + q1 (cold) | 201.7 ms | 200.1 ms |
| B: doc1 + q2 (reuse) | 59.4 ms | 189.2 ms |
| C: doc2 + q1 (cold) | 186.2 ms | 186.4 ms |
| Prefix-cache hit tokens | 2,288 (143 full blocks) | 0 |

### Phase 2: native LRU on three scenarios
Qwen2.5-3B, 800 KV blocks (12,800 tokens = 12 prefixes of 1,024 tokens), 80 documents,
Zipf alpha 0.9, 2 requests/s for 300 s, 64 output tokens, provisional SLO TTFT 500 ms /
TPOT 50 ms, one run each. Config: `configs/phase2_three_scenarios.yaml`.

| Scenario | Requests | Request hit rate | Ideal (infinite cache) | TTFT p50 | TTFT p99 | SLO met | Goodput | Preemptions |
|---|---|---|---|---|---|---|---|---|
| stable | 595 | 43.5% | 87.2% | 92.9 ms | 166.3 ms | 100.0% | 1.99 req/s | 0 |
| drift (re-rank at 150 s) | 595 | 42.9% | 87.2% | 93.5 ms | 162.5 ms | 100.0% | 1.99 req/s | 0 |
| burst (20 one-off docs at 150 s) | 615 | 41.1% | 84.4% | 93.9 ms | 242.5 ms | 99.2% | 2.04 req/s | 1 |

Findings:
- LRU reuses about half of the requests that could reuse a prefix, in all three scenarios.
- A hit cuts TTFT p50 from 97 ms to 42 ms (stable run). At 2 requests/s almost every request
  meets the provisional SLO, so goodput equals throughput; Phase 3 raises the load.
- Drift and one-off bursts barely change LRU's hit rate: with room for 12 prefixes, LRU only
  remembers the last 10-15 s of traffic, so it loses hot prefixes all the time, not just at events.
- The burst caused one preemption and about 5 SLO misses: 800 blocks hold only about 11
  concurrent requests. Phase 3 must size the KV budget together with the load so that
  preemption does not hide the effect of eviction.
- Runs are deterministic: the automated runner reproduced the manual runs' hit rates exactly
  (0.4353 stable, 0.4286 drift).

## Repository layout
| Path | Content | Phase |
|---|---|---|
| `scripts/` | server launch script, prefix-cache smoke test | 1 |
| `workload/` | document pool, question templates, trace generator | 2 |
| `client/` | replay client, metrics, timeline plot, experiment runner, W&B uploader | 2 |
| `configs/` | experiment configs (YAML) | 2+ |
| `profiling/` | profiler scripts and summaries | 3 |
| `eval/` | correctness and quality checks | 3, 7 |
| `sim/` | offline eviction simulator | 5 |
| `results/` | summary tables and figures (raw runs are not in git) | 3+ |
| `vllm/` | our vLLM fork as a git submodule (to be added) | 4, 6 |

## W&B
Project `hpml-prefix-cache` (personal workspace for now; pass `--entity <team>` to
`upload_wandb.py` once a team exists). Each run is grouped by experiment name and carries the
full config, summary metrics, server time series, per-window hit rate, the request table, and
the timeline plot.

## Contributions
To be filled in by each member.
