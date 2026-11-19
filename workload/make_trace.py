"""Phase 2.3: trace generator.

Writes a JSONL trace: one request per line, sorted by arrival time, with the full prompt
token IDs (document prefix + question suffix) so replay needs no tokenizer.

Knobs: Zipf popularity (alpha), Poisson arrivals (rate), popularity drift (re-rank all
documents at a given time), and a one-off burst of documents that are used only once.

Usage:
  python workload/make_trace.py                                  # stable popularity
  python workload/make_trace.py --drift-at 0.5                   # re-rank halfway
  python workload/make_trace.py --burst-at 0.5 --burst-size 20   # one-off burst halfway
  python workload/make_trace.py --oneoff-frac 0.3                # 30% of requests are one-off
"""
import argparse
import json
import math
import os
import random
from collections import Counter

from transformers import AutoTokenizer

from questions import TEMPLATES, suffix_ids


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    p.add_argument("--split", default="dev")
    p.add_argument("--prefix-len", type=int, default=1024)
    p.add_argument("--n-docs", type=int, default=80, help="documents in the regular traffic")
    p.add_argument("--alpha", type=float, default=0.9, help="Zipf exponent; larger = more skewed")
    p.add_argument("--rate", type=float, default=2.0, help="mean requests per second (Poisson)")
    p.add_argument("--duration", type=float, default=300.0, help="trace length in seconds")
    p.add_argument("--max-tokens", type=int, default=64, help="fixed output length")
    p.add_argument("--drift-at", type=float, default=None, help="fraction of duration; re-rank docs")
    p.add_argument("--burst-at", type=float, default=None, help="fraction of duration; one-off burst")
    p.add_argument("--burst-size", type=int, default=20, help="number of one-off documents")
    p.add_argument("--burst-window", type=float, default=5.0, help="seconds the burst is spread over")
    p.add_argument("--oneoff-frac", type=float, default=0.0,
                   help="fraction of arrivals that request a document never requested again")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", default="workload/traces")
    a = p.parse_args()
    assert a.prefix_len % 16 == 0, "prefix length must be a multiple of 16"

    slug = a.model.replace("/", "__")
    with open(f"workload/data/{slug}/docs_{a.split}.jsonl") as f:
        docs = [json.loads(line) for line in f]
    n_burst = a.burst_size if a.burst_at is not None else 0
    assert a.n_docs + n_burst <= len(docs), "not enough documents in this split"
    assert all(len(d["token_ids"]) >= a.prefix_len for d in docs), "prefix-len longer than pool"
    regular, one_off = docs[: a.n_docs], docs[a.n_docs : a.n_docs + n_burst]
    oneoff_pool = []
    if a.oneoff_frac > 0:
        path = f"workload/data/{slug}/docs_oneoff_{a.split}.jsonl"
        assert os.path.exists(path), f"{path} missing: rebuild the pool with --n-oneoff"
        with open(path) as f:
            oneoff_pool = [json.loads(line) for line in f]
    oneoff_iter = iter(oneoff_pool)

    rng = random.Random(a.seed)
    weights = [1.0 / (r ** a.alpha) for r in range(1, a.n_docs + 1)]  # Zipf over ranks 1..N
    rank_before = list(range(a.n_docs))
    rng.shuffle(rank_before)  # rank_before[r] = which regular doc has popularity rank r
    rank_after = list(range(a.n_docs))
    rng.shuffle(rank_after)
    drift_t = a.drift_at * a.duration if a.drift_at is not None else math.inf

    events = []  # (arrival time, doc, phase)
    t = 0.0
    while True:
        t += rng.expovariate(a.rate)  # Poisson arrivals: exponential gaps
        if t >= a.duration:
            break
        if a.oneoff_frac > 0 and rng.random() < a.oneoff_frac:
            d = next(oneoff_iter, None)
            assert d is not None, "not enough one-off documents: rebuild the pool with a larger --n-oneoff"
            events.append((t, d, "oneoff"))
            continue
        ranks, phase = (rank_before, "before") if t < drift_t else (rank_after, "after")
        r = rng.choices(range(a.n_docs), weights=weights)[0]
        events.append((t, regular[ranks[r]], phase))
    if n_burst:
        t0 = a.burst_at * a.duration
        for d in one_off:
            events.append((t0 + rng.uniform(0, a.burst_window), d, "burst"))
    events.sort(key=lambda e: e[0])

    tok = AutoTokenizer.from_pretrained(a.model)
    sfx_cache = {}
    name = f"L{a.prefix_len}_a{a.alpha}_r{a.rate}_T{int(a.duration)}"
    if a.drift_at is not None:
        name += f"_drift{a.drift_at}"
    if n_burst:
        name += f"_burst{a.burst_at}x{n_burst}"
    if a.oneoff_frac > 0:
        name += f"_oneoff{a.oneoff_frac}"
    name += f"_s{a.seed}"
    out_dir = os.path.join(a.out_dir, slug)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, name + ".jsonl")

    seen, ideal_hit, total_prompt = set(), 0, 0
    doc_counts, phase_counts = Counter(), Counter()
    with open(path, "w") as f:
        for i, (t, d, phase) in enumerate(events):
            qid = rng.randrange(len(TEMPLATES))
            key = (d["doc_id"], qid)
            if key not in sfx_cache:
                sfx_cache[key] = suffix_ids(tok, d["title"], qid)
            ids = d["token_ids"][: a.prefix_len] + sfx_cache[key]
            if d["doc_id"] in seen:  # with an infinite cache the whole prefix would be reused
                ideal_hit += a.prefix_len
            seen.add(d["doc_id"])
            total_prompt += len(ids)
            doc_counts[d["doc_id"]] += 1
            phase_counts[phase] += 1
            rec = {"req_id": i, "t": round(t, 4), "doc_id": d["doc_id"], "qid": qid,
                   "phase": phase, "prefix_len": a.prefix_len, "max_tokens": a.max_tokens,
                   "prompt_ids": ids}
            f.write(json.dumps(rec) + "\n")

    n = len(events)
    top10 = sum(c for _, c in doc_counts.most_common(10))
    meta = {
        "trace": path,
        "args": vars(a),
        "n_requests": n,
        "requests_per_phase": dict(phase_counts),
        "distinct_docs": len(seen),
        "top10_doc_share": round(top10 / n, 3),  # how concentrated popularity is
        "mean_prompt_tokens": round(total_prompt / n, 1),
        "ideal_prefix_hit_rate": round(ideal_hit / total_prompt, 3),  # infinite-cache upper bound
        "prefixes_fitting_in_800_blocks": 800 * 16 // a.prefix_len,
    }
    with open(os.path.join(out_dir, name + ".meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(json.dumps({k: v for k, v in meta.items() if k != "args"}, indent=2))


if __name__ == "__main__":
    main()
