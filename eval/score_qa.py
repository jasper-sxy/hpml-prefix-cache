"""Phase 3.5 (quality): accuracy on the SQuAD questions made by eval/build_qa_set.py.

A request is correct when a gold answer (normalized as in SQuAD) appears in the first non-empty
line of the generated text. Reports accuracy per run, split by prefix-cache hit / miss.

Usage:
  python eval/score_qa.py --qa eval/qa_squad_n50.json results/raw/phase3_qa  [--show 5]
"""
import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_qa_set import normalize  # noqa: E402


def first_line(text):
    for line in text.split("\n"):
        if line.strip():
            return line
    return ""


def score(run_dir, gold, tok):
    rows = []
    with open(os.path.join(run_dir, "requests.jsonl")) as f:
        for line in f:
            r = json.loads(line)
            k = (r["doc_id"], r["qid"])
            if r.get("phase") != "qa" or k not in gold or not r.get("ok"):
                continue
            ans = first_line(tok.decode(r["output_ids"]))
            ok = any(f" {normalize(g)} " in f" {normalize(ans)} " for g in gold[k]["answers"])  # whole words only
            rows.append({"key": k, "correct": ok, "hit": (r.get("cached_tokens") or 0) > 0, "answer": ans})
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--qa", required=True)
    p.add_argument("paths", nargs="+", help="run directories or experiment folders")
    p.add_argument("--show", type=int, default=0, help="print N wrong answers of the first run")
    a = p.parse_args()
    qa = json.load(open(a.qa))
    gold = {(it["doc_id"], it["qid"]): it for it in qa["items"]}
    runs = []
    for path in a.paths:
        if os.path.exists(os.path.join(path, "requests.jsonl")):
            runs.append(path)
        else:
            runs += sorted(os.path.dirname(x) for x in glob.glob(os.path.join(path, "*", "requests.jsonl")))
    meta = json.load(open(os.path.join(runs[0], "trace_meta.json")))
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(meta["args"]["model"])
    print(f"{'run':<36}{'QA requests':>12}{'accuracy':>10}{'hit acc (n)':>16}{'miss acc (n)':>16}")
    shown = 0
    for d in runs:
        rows = score(d, gold, tok)
        if not rows:
            continue
        acc = lambda rs: f"{100 * sum(r['correct'] for r in rs) / len(rs):.1f} ({len(rs)})" if rs else "-"
        hits = [r for r in rows if r["hit"]]
        miss = [r for r in rows if not r["hit"]]
        print(f"{os.path.basename(d)[:35]:<36}{len(rows):>12}{100 * sum(r['correct'] for r in rows) / len(rows):>9.1f}%"
              f"{acc(hits):>16}{acc(miss):>16}")
        for r in rows:
            if shown < a.show and not r["correct"]:
                g = gold[r["key"]]
                print(f"    Q: {g['question']}\n    gold: {g['answers']}\n    got:  {r['answer'][:120]!r}")
                shown += 1
    print("correct = a gold answer appears in the first non-empty line of the output (SQuAD normalization)")


if __name__ == "__main__":
    main()
