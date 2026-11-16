# workload/  (Phase 2)
Document pool: WikiText articles cut into fixed-length prefixes (multiple of 16 tokens), stored as token IDs, with dev/test split.
Trace generator: Zipf popularity (alpha), Poisson arrivals (lambda), popularity drift, one-off bursts. Output: JSONL.
Status: TODO
