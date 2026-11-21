"""Phase 3.5: a "one request at a time" trace for reference outputs.

Collects every distinct prompt (document, question, prefix length) of the regular documents in
the given traces and sends each once, `--spacing` seconds apart, so requests never overlap
(batch size 1). Prompts are ordered document by document: with prefix caching on, the first
question about a document misses and the next ones hit, without any other request in the batch.

Usage:
  python eval/make_unique_trace.py workload/traces/Qwen__Qwen2.5-3B-Instruct/L1024_a0.9_r12.0_T300_s0.jsonl \
      [more traces] [--spacing 1.5]
"""
import argparse
import json
import os


def main():
    p = argparse.ArgumentParser()
    p.add_argument("traces", nargs="+")
    p.add_argument("--spacing", type=float, default=1.5,
                   help="seconds between requests; must exceed one request's latency at batch size 1")
    p.add_argument("--phases", default="before,after", help="trace phases to take prompts from")
    a = p.parse_args()
    phases = set(a.phases.split(","))
    prompts = {}
    for path in a.traces:
        with open(path) as f:
            for line in f:
                r = json.loads(line)
                if r["phase"] in phases:
                    prompts.setdefault((r["doc_id"], r["qid"], r["prefix_len"]), r)
    keys = sorted(prompts)
    plen = {k[2] for k in keys}
    out_dir = os.path.dirname(a.traces[0])
    name = f"unique_L{'-'.join(str(x) for x in sorted(plen))}_n{len(keys)}_sp{a.spacing}"
    path = os.path.join(out_dir, name + ".jsonl")
    with open(path, "w") as f:
        for i, k in enumerate(keys):
            r = dict(prompts[k], req_id=i, t=round(i * a.spacing, 4), phase="unique")
            f.write(json.dumps(r) + "\n")
    with open(a.traces[0].replace(".jsonl", ".meta.json")) as f:
        args = json.load(f)["args"]
    args.update({"rate": round(1 / a.spacing, 4), "duration": round(len(keys) * a.spacing, 1),
                 "drift_at": None, "burst_at": None, "oneoff_frac": 0.0})
    meta = {"trace": path, "args": args, "n_requests": len(keys), "unique_of": a.traces,
            "distinct_docs": len({k[0] for k in keys}), "spacing_s": a.spacing}
    with open(path.replace(".jsonl", ".meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"wrote {path}: {len(keys)} prompts from {meta['distinct_docs']} documents, "
          f"{len(keys) * a.spacing / 60:.1f} min at {a.spacing} s spacing")


if __name__ == "__main__":
    main()
