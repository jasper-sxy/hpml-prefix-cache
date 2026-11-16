# workload/  (Phase 2)

| File | Purpose |
|---|---|
| `build_doc_pool.py` | WikiText-103 articles -> first 2,048 token IDs per article; 100 dev + 100 test documents per model |
| `questions.py` | five question templates appended after a document prefix |
| `peek_prompt.py` | sanity check: sends one document with every question, checks token counts and prefix reuse |
| `make_trace.py` | request trace: Zipf popularity, Poisson arrivals, popularity drift, one-off burst |

Generated, not in git: `data/<model>/` (document pools) and `traces/<model>/` (traces plus
`.meta.json` with parameters and statistics such as the infinite-cache hit rate).

Design choices: prefixes are multiples of 16 tokens (the KV block size); prompts are sent as
token IDs; there is no shared header before the document (the instruction lives in the
question suffix), so different documents never share a first block.
