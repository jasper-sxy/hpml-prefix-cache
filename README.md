# Workload-Aware Prefix KV-Cache Management (COMS 6998 HPML, Fall 2026)

We study prefix-cache eviction in vLLM under a fixed KV-cache budget, and test whether a
workload-aware policy (reuse history + recency, optionally weighted by prefill cost)
improves goodput over vLLM's native LRU eviction.

## Status
- [x] Phase 1: vLLM running with a capped KV cache; prefix-cache smoke test passed
- [ ] Phase 2: workload generator, replay client, metrics, one-click runner
- [ ] Phase 3: baselines, SLO calibration, profiling, correctness baseline
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
| Python | 3.12 (conda env `vllm`) |
| Model | Qwen/Qwen2.5-3B-Instruct (Llama-3.2-3B-Instruct pending access) |

## Known issues
- If the system `nvcc` is older than CUDA 12.8, FlashInfer's sampler JIT build fails
  (`nvcc fatal: Unknown option '--compress-mode=size'`). Fix: `export VLLM_USE_FLASHINFER_SAMPLER=0`.
- On AutoDL: use `source /etc/network_turbo` for GitHub/Hugging Face, and
  `unset http_proxy https_proxy` before pip or local HTTP requests.
- Run only one vLLM server per GPU; stop the old one before starting a new one.

## Quick start
```bash
bash scripts/start_server.sh Qwen/Qwen2.5-3B-Instruct 800   # inside screen
python scripts/smoke_prefix.py                               # in another terminal
```

## Phase 1 results (Qwen2.5-3B, 800 KV blocks = 12,800 tokens, ~2,300-token shared prefix)
| Request | Prefix cache ON | Prefix cache OFF |
|---|---|---|
| A: doc1 + q1 (cold) | 201.7 ms | 200.1 ms |
| B: doc1 + q2 (reuse) | 59.4 ms | 189.2 ms |
| C: doc2 + q1 (cold) | 186.2 ms | 186.4 ms |
| Prefix-cache hit tokens | 2,288 (143 full blocks) | 0 |

Single runs; final experiments use at least 3 repetitions.
Uncapped, this GPU holds 40,473 KV blocks for Qwen2.5-3B; we cap it with `--num-gpu-blocks-override`.

## Repository layout
| Path | Content | Phase |
|---|---|---|
| `scripts/` | server launch script, smoke test | 1 |
| `workload/` | document pool, trace generator | 2 |
| `client/` | replay client, metrics, one-click runner | 2 |
| `configs/` | per-run YAML configs | 2+ |
| `profiling/` | profiler scripts and summaries | 3 |
| `eval/` | correctness and quality checks | 3, 7 |
| `sim/` | offline eviction simulator | 5 |
| `results/` | summary tables and figures | 3+ |
| `vllm/` | our vLLM fork as a git submodule (to be added) | 4, 6 |

## Contributions
To be filled in by each member.
