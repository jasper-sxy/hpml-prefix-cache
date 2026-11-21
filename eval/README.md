# eval/  (Phases 3.5, 6, 7)
Correctness of prefix caching: generated tokens (temperature 0, 64 tokens) compared against a
reference made with caching off and one request at a time.

| File | Purpose |
|---|---|
| `make_unique_trace.py` | every distinct prompt of the regular documents, sent once and far apart (batch size 1), document by document |
| `build_qa_set.py` | 50 SQuAD v1.1 questions in our prompt format (10 articles cut to 1,024 tokens), plus one-at-a-time and mixed-into-load traces |
| `score_qa.py` | QA accuracy per run (gold answer in the first output line, whole words, SQuAD normalization), split by hit / miss |
| `qa_squad_n50.json` | the question set with gold answers (SQuAD, CC BY-SA 4.0) |
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

## Phase 3.5 quality check (SQuAD)
50 questions from SQuAD v1.1 validation (`qa_squad_n50.json`): the first 10 articles, each cut to
1,024 tokens, 5 questions per article whose paragraph lies inside the cut and whose answer has
at most 4 words. Prompt: document + "Question: ... Answer with a few words from the article.
Answer:". Correct = a gold answer appears, as whole words after SQuAD normalization, in the
first non-empty line of the output. `configs/phase3_qa.yaml`, 1600 blocks, about 20 min:

    export HF_ENDPOINT=https://hf-mirror.com
    T=workload/traces/Qwen__Qwen2.5-3B-Instruct
    python eval/build_qa_set.py --mix-base $T/L1024_a0.9_r12.0_T300_s0.jsonl $T/L1024_a0.9_r6.0_T300_s0.jsonl
    python client/run_experiment.py configs/phase3_qa.yaml
    python eval/score_qa.py --qa eval/qa_squad_n50.json results/raw/phase3_qa --show 5

| Run | QA requests | Accuracy |
|---|---|---|
| caching off, one question at a time | 50 | 78.0% |
| LRU, one at a time (questions 2-5 of each article hit) | 50 | 78.0% |
| caching off, each question 4 times inside the 6 req/s stable trace | 200 | 78.0% |
| LRU, each question 4 times inside the 12 req/s stable trace | 200 | 77.5% |

Accuracy is the same in all four conditions (one answer out of 200 apart), although under load
only about 30% of QA outputs are identical to the reference: the answer comes in the first
line, and outputs usually diverge later (median first difference at token 21-25). Within a run,
hit and miss accuracies cover different questions (in the one-at-a-time run the first question
of each article is the miss), so they are not a like-for-like comparison. Mixed into the load,
the QA documents (each asked 20 times in 300 s) are rarely hot enough to stay cached under LRU:
31 of 200 QA requests hit. The scoring is strict ("Yes, No" for "yes or no" counts as wrong) but
the same in every condition.
