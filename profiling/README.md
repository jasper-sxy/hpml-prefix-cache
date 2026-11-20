# profiling/ (Phase 3.4)
Where the time goes when vLLM serves our workload with native LRU prefix caching, and how close
each part runs to the hardware limits.

Setup: RTX 4080 SUPER (32 GB version, 80 SMs, sm_89; 736 GB/s DRAM bandwidth, about 104 TFLOPS
BF16 dense at the official boost clock), Qwen2.5-3B-Instruct in BF16, vLLM 0.33.0 (our fork),
1600 KV blocks, 1024-token prefixes, Zipf alpha 0.9, async scheduling on (vLLM default).

## Scripts
| Script | Input | What it gives |
|---|---|---|
| `server_breakdown.py` | `metrics_before.prom` / `metrics_after.prom`, saved by `client/replay.py` in every run since Phase 3.4 | per request: queue, prefill and decode time; steps per second, tokens per step |
| `iteration_log.py` | `server.log` of runs started with `--enable-logging-iteration-details` | fit of step time vs prefill tokens and decode requests (relative numbers only, see below) |
| `nsys_capture.sh` | a trace | runs vLLM under Nsight Systems and records a few hundred engine steps during a replay |
| `nsys_steps.py` | the `.sqlite` export of a capture | true GPU time per step, kernel mix, GPU busy fraction, step-cost fit |

Reproduce (on the GPU server, from the repo root):

    python client/run_experiment.py configs/phase3_profile_b1600.yaml      # 7 runs, about 50 min
    python profiling/server_breakdown.py results/raw/phase3_profile_b1600
    python profiling/iteration_log.py results/raw/phase3_profile_b1600
    bash profiling/nsys_capture.sh workload/traces/Qwen__Qwen2.5-3B-Instruct/L1024_a0.9_r12.0_T300_s0.jsonl lru_r12
    python profiling/nsys_steps.py profiling/nsys/lru_r12.sqlite --plot results/figures/phase3/nsys_steps_lru_r12.png

Nsight reports are large and stay out of git (`profiling/nsys/` is in `.gitignore`).
The nsys in `/root/autodl-tmp/cuda-12.4` (2023.4.4) works on AutoDL; ncu does not (no access to
GPU performance counters, `ERR_NVGPUCTRPERM`).

## Results

### Request time, server side (`configs/phase3_profile_b1600.yaml`)
Same traces as the Phase 3.3 baselines plus vLLM's iteration log; goodput matches Phase 3.3 within
0.2%, so the logging does not change the results.

| Run | Mean queue | Queue p99 | Mean prefill |
|---|---|---|---|
| caching off, 2 req/s | 0.1 ms | | 88.9 ms |
| LRU, 2 req/s | 0.0 ms | | 53.9 ms |
| stable LRU, 12 req/s (busy) | 4.1 ms | | 77.6 ms |
| stable LRU, 14 req/s (near collapse) | 24.1 ms | 576 ms | 87.6 ms |
| 30% one-off LRU, 10 req/s (near collapse) | 29.2 ms | 665 ms | 103.5 ms |

A cache hit cuts the mean prefill from 89 to 54 ms at low load. Near the knee, requests that miss
the SLO miss it mostly by queueing. vLLM's histograms have a first bucket of 0-300 ms, so
percentiles below 300 ms are not meaningful (left blank); use the means.

### Cost of one engine step (Nsight Systems, stable trace at 12 req/s, 500 steps)

    gpu_ms = 9.6 + 61.5 * prefill_tokens / 1000 + 0.171 * decode_requests      (R^2 = 1.00)

| | Steps | GPU time p50 | CPU time p50 (execute_model) |
|---|---|---|---|
| decode-only steps | 374 | 13.0 ms | 3.7 ms |
| steps with prefill | 126 | 76.6 ms | 30.7 ms (includes waiting for the GPU) |

Kernel time: GEMM about 82%, attention about 13%, other about 5%. The GPU was busy 98.2% of the
12.2 s window (seconds 60-72 of the run, which include one arrival burst).

vLLM's own per-iteration log gives `3.0 + 42.2 * prefill_tokens/1000 + 0.089 * decode_requests`:
about a third too low, because with async scheduling it only counts the GPU time that is not
overlapped with preparing the next step. Use `iteration_log.py` for relative comparisons only.

### Roofline estimates
| Part | Work | Time | Achieved | Peak | Bound |
|---|---|---|---|---|---|
| decode step, fixed part | read about 6.2 GB of weights | 9.6 ms | about 640 GB/s | 736 GB/s | memory, about 87% |
| prefill, per 1000 tokens | about 5.5 TFLOP (2 x 2.77 B non-embedding params x 1000) | 61.5 ms | about 90 TFLOPS | about 104 TFLOPS | compute, 80-90% |

Peaks are the official figures at boost clock; the card can run faster than that, so the
utilization numbers are upper estimates.

### What this means for the project
- Both prefill and decode already run close to the hardware limits, so there is little to gain
  per token. The lever is to avoid recomputing prefixes, i.e. a better eviction policy.
- At 12 req/s the GPU is close to fully busy: with the fitted costs, the run's ~17k steps,
  prefill tokens and decode requests add up to roughly the 300 s of the run. The knee at about
  14 req/s (Phase 3.3) is the GPU running out of throughput.
- One miss of a 1024-token prefix costs about 64 ms of GPU time, about five decode steps, and
  every request decoding in that step waits for it (TPOT) while new arrivals queue (TTFT).
- With async scheduling the CPU has roughly 9 ms of slack per decode step. That is the budget for
  a more expensive eviction policy; scheduler and block-allocation time are measured in Phase 6
  for LRU and the new policy with the same instrumentation.

## Caveats
- One run per configuration and one 12 s nsys window; numbers are for this GPU and this model.
- The step-cost fit mixes prefill and decode in the same steps (chunked prefill); it describes
  the average cost, not individual kernels.
