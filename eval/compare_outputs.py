"""Phase 3.5: compare generated tokens (temperature 0) against a reference run.

Requests are matched by prompt (document, question, prefix length), so a run under load can be
checked against a one-request-at-a-time reference made with eval/make_unique_trace.py. For each
run: share of requests whose 64 output tokens are identical to the reference, split by prefix
cache hit / miss, and where the first differing token is. Also checks that the same prompt gave
the same output every time within the run.

Usage:
  python eval/compare_outputs.py --ref results/raw/phase3_correctness/ref_off_seq_a_* \
      results/raw/phase3_correctness            # every run in the folder
  ... --show 3     # print the first 3 mismatches as text (needs the tokenizer)
"""
import argparse
import glob
import json
import os

import numpy as np


def load(run_dir):
    out = []
    with open(os.path.join(run_dir, "requests.jsonl")) as f:
        for line in f:
            r = json.loads(line)
            if r.get("ok") and r.get("output_ids"):
                out.append(r)
    return out


def key(r):
    return (r["doc_id"], r["qid"], r["prefix_len"])


def first_diff(a, b):
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return min(len(a), len(b)) if len(a) != len(b) else None


def compare(ref, run):
    rows, mism = [], []
    for r in run:
        k = key(r)
        if k not in ref:
            continue
        d = first_diff(r["output_ids"], ref[k])
        hit = (r.get("cached_tokens") or 0) > 0
        rows.append((d is None, hit, d))
        if d is not None:
            mism.append((r, ref[k], d))
    return rows, mism


def self_consistency(run):
    outs = {}
    for r in run:
        outs.setdefault(key(r), set()).add(tuple(r["output_ids"]))
    repeated = [v for v in outs.values()]
    multi = sum(len(v) > 1 for v in repeated)
    return multi, len(repeated)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ref", required=True, help="reference run directory (glob allowed)")
    p.add_argument("paths", nargs="+", help="run directories or experiment folders")
    p.add_argument("--show", type=int, default=0, help="print the first N mismatches as text")
    a = p.parse_args()
    ref_dir = sorted(glob.glob(a.ref))[0]
    ref_reqs = load(ref_dir)
    assert ref_reqs, f"{ref_dir}: no output_ids (was it run with the Phase 3.5 replay.py?)"
    ref = {key(r): r["output_ids"] for r in ref_reqs}
    runs = []
    for path in a.paths:
        if os.path.exists(os.path.join(path, "requests.jsonl")):
            runs.append(path)
        else:
            runs += sorted(os.path.dirname(x) for x in glob.glob(os.path.join(path, "*", "requests.jsonl")))
    print(f"reference: {os.path.basename(ref_dir)} ({len(ref)} prompts)")
    print(f"{'run':<34}{'compared':>9}{'same%':>8}{'hit same%':>10}{'miss same%':>11}"
          f"{'1st diff p50':>13}{'diff<8 tok':>11}{'prompts w/ >1 output':>22}")
    all_mism = []
    for d in runs:
        if os.path.samefile(d, ref_dir):
            continue
        reqs = load(d)
        if not reqs:
            print(f"{os.path.basename(d)[:33]:<34}  no output_ids recorded")
            continue
        rows, mism = compare(ref, reqs)
        if not rows:
            continue
        same = np.array([x[0] for x in rows])
        hit = np.array([x[1] for x in rows])
        divs = np.array([x[2] for x in rows if x[2] is not None])
        multi, n_keys = self_consistency(reqs)

        def pct(mask):
            return f"{100 * same[mask].mean():.1f}" if mask.any() else "-"
        print(f"{os.path.basename(d)[:33]:<34}{len(rows):>9}{100 * same.mean():>8.1f}{pct(hit):>10}{pct(~hit):>11}"
              f"{(np.median(divs) if len(divs) else float('nan')):>13.0f}"
              f"{(100 * (divs < 8).mean() if len(divs) else 0):>10.1f}%{f'{multi}/{n_keys}':>22}")
        all_mism += [(os.path.basename(d),) + m for m in mism]
    print("same% = identical 64-token output; 1st diff = position of the first differing token "
          "among mismatches; last column = prompts that produced different outputs within the run")
    if a.show and all_mism:
        from transformers import AutoTokenizer
        meta = json.load(open(os.path.join(ref_dir, "trace_meta.json")))
        tok = AutoTokenizer.from_pretrained(meta["args"]["model"])
        for name, r, ref_ids, d in all_mism[: a.show]:
            print(f"\n--- {name} req {r['req_id']} doc {r['doc_id']} q {r['qid']} "
                  f"cached {r.get('cached_tokens')} first diff at token {d}")
            print("same so far:", repr(tok.decode(ref_ids[:d])))
            print("reference  :", repr(tok.decode(ref_ids[d:d + 20])))
            print("this run   :", repr(tok.decode(r["output_ids"][d:d + 20])))


if __name__ == "__main__":
    main()
