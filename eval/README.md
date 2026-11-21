# eval/  (Phases 3.5, 6, 7)
Correctness of prefix caching: generated tokens (temperature 0, 64 tokens) compared against a
reference made with caching off and one request at a time.

| File | Purpose |
|---|---|
| `make_unique_trace.py` | every distinct prompt of the regular documents, sent once and far apart (batch size 1), document by document |
| `compare_outputs.py` | matches requests to the reference by (document, question, prefix length); identical share, split by hit / miss; position of the first differing token; prompts with more than one output within a run |

`client/replay.py` records the generated token IDs of every request (`output_ids` in
`requests.jsonl`, via vLLM's `return_token_ids`), so any run can be checked afterwards.

Reproduce (`configs/phase3_correctness.yaml`, 1600 blocks, about 50 min):

    T=workload/traces/Qwen__Qwen2.5-3B-Instruct
    python eval/make_unique_trace.py $T/L1024_a0.9_r6.0_T300_s0.jsonl $T/L1024_a0.9_r12.0_T300_s0.jsonl $T/L1024_a0.9_r10.0_T300_oneoff0.3_s0.jsonl
    python client/run_experiment.py configs/phase3_correctness.yaml
    python eval/compare_outputs.py --ref 'results/raw/phase3_correctness/ref_off_seq_a_*' results/raw/phase3_correctness --show 3

## Phase 3.5 baseline (native LRU)
| Run | Compared | Identical | Hit / miss identical | First difference (median token) |
|---|---|---|---|---|
| caching off, one at a time, again | 400 | 100% | - / 100% | - |
| LRU, one at a time | 400 | 71.2% | 64.1% / 100% | 29 |
| caching off, 6 req/s | 1,753 | 61.0% | - / 61.0% | 34 |
| LRU, stable, 12 req/s | 3,554 | 61.3% | 62.2% / 60.0% | 32 |
| LRU, 30% one-off, 10 req/s | 2,115 | 60.3% | 60.3% / 60.3% | 29 |

The reference is exactly reproducible. Reusing cached KV, and batching with other requests,
each change about 4 in 10 outputs, almost always as a different but plausible continuation
after a few dozen identical tokens (a near-tie between the top two tokens flips under slightly
different floating-point order). Under load, batching dominates: hit and miss requests differ
from the reference equally often, and the same prompt can give different outputs within one run.

Use in Phase 6: the new policy, on the same trace, should match the reference about as often as
LRU (61%, about +-1 point at 3,554 requests) and should not raise the share of early divergences
(first difference within 8 tokens: about 3% of requests with LRU). Unit tests of the block pool
remain the main correctness check.
