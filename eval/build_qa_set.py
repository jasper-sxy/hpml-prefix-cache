"""Phase 3.5 (quality): a small SQuAD question set in our prompt format, plus traces.

SQuAD v1.1 validation (Wikipedia paragraphs, human-annotated answers). For each of the first
`--n-articles` articles: consecutive paragraphs are joined into one document and cut to exactly
`--prefix-len` tokens; `--per-article` questions are drawn (seed 0) among those whose paragraph
lies entirely inside the cut and whose answer has at most 4 words. Prompt = document tokens +
"\\n\\nQuestion: <q> Answer with a few words from the article.\\nAnswer:".

Writes
  eval/qa_squad_n<N>.json                    questions, gold answers, document ids (doc_id 10000+)
  <trace dir>/qa_squad_n<N>_seq_sp1.5.jsonl  each question once, 1.5 s apart (batch size 1)
  <trace dir>/qa_squad_n<N>_mix_<base>.jsonl a base trace plus every question --repeat times at
                                             random moments (QA competes for the cache under load)

Usage:
  python eval/build_qa_set.py --mix-base workload/traces/Qwen__Qwen2.5-3B-Instruct/L1024_a0.9_r12.0_T300_s0.jsonl \
      workload/traces/Qwen__Qwen2.5-3B-Instruct/L1024_a0.9_r6.0_T300_s0.jsonl
"""
import argparse
import json
import os
import random
import re
import string

SUFFIX = "\n\nQuestion: {q} Answer with a few words from the article.\nAnswer:"
DOC_ID_BASE = 10000


def normalize(s):
    """SQuAD answer normalization: lower case, no punctuation, no articles, single spaces."""
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def load_squad():
    from huggingface_hub import hf_hub_download
    import pyarrow.parquet as pq
    path = hf_hub_download("rajpurkar/squad", "plain_text/validation-00000-of-00001.parquet",
                           repo_type="dataset")
    return pq.read_table(path).to_pylist()


def build(rows, tok, n_articles, per_article, prefix_len, seed=0):
    articles = {}
    for r in rows:  # keep dataset order: articles, then paragraphs, then questions
        a = articles.setdefault(r["title"], {"contexts": [], "questions": {}})
        if r["context"] not in a["questions"]:
            a["contexts"].append(r["context"])
            a["questions"][r["context"]] = []
        a["questions"][r["context"]].append(r)
    rng = random.Random(seed)
    docs, items = [], []
    for title, a in articles.items():
        if len(docs) == n_articles:
            break
        text, inside = "", []
        for c in a["contexts"]:
            cand = (text + "\n\n" + c) if text else c
            if len(tok(cand, add_special_tokens=False)["input_ids"]) > prefix_len:
                break
            text = cand
            inside.append(c)
        ids = tok(text + "\n\n" + "\n\n".join(a["contexts"][len(inside):]),
                  add_special_tokens=False)["input_ids"][:prefix_len]
        if len(ids) < prefix_len:
            continue  # article too short
        pool = [q for c in inside for q in a["questions"][c]
                if len(q["answers"]["text"][0].split()) <= 4
                and normalize(q["answers"]["text"][0]) in normalize(text)]
        if len(pool) < per_article:
            continue
        doc_id = DOC_ID_BASE + len(docs)
        docs.append({"doc_id": doc_id, "title": title, "token_ids": ids})
        for qid, q in enumerate(rng.sample(pool, per_article)):
            items.append({"doc_id": doc_id, "qid": qid, "title": title, "squad_id": q["id"],
                          "question": q["question"], "answers": sorted(set(q["answers"]["text"]))})
    return docs, items


def records(docs, items, tok, prefix_len, max_tokens):
    by_doc = {d["doc_id"]: d for d in docs}
    out = []
    for it in items:
        sfx = tok(SUFFIX.format(q=it["question"]), add_special_tokens=False)["input_ids"]
        out.append({"doc_id": it["doc_id"], "qid": it["qid"], "phase": "qa", "prefix_len": prefix_len,
                    "max_tokens": max_tokens, "prompt_ids": by_doc[it["doc_id"]]["token_ids"] + sfx})
    return out


def write_trace(path, recs, meta):
    with open(path, "w") as f:
        for i, r in enumerate(recs):
            f.write(json.dumps(dict(r, req_id=i)) + "\n")
    with open(path.replace(".jsonl", ".meta.json"), "w") as f:
        json.dump(dict(meta, trace=path, n_requests=len(recs)), f, indent=2)
    print("wrote", path, f"({len(recs)} requests)")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    p.add_argument("--n-articles", type=int, default=10)
    p.add_argument("--per-article", type=int, default=5)
    p.add_argument("--prefix-len", type=int, default=1024)
    p.add_argument("--max-tokens", type=int, default=64)
    p.add_argument("--spacing", type=float, default=1.5)
    p.add_argument("--mix-base", nargs="*", default=[], help="load traces to mix the questions into")
    p.add_argument("--repeat", type=int, default=4, help="times each question is asked in a mixed trace")
    p.add_argument("--trace-dir", default="workload/traces/Qwen__Qwen2.5-3B-Instruct")
    a = p.parse_args()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model)
    docs, items = build(load_squad(), tok, a.n_articles, a.per_article, a.prefix_len)
    n = len(items)
    with open(f"eval/qa_squad_n{n}.json", "w") as f:
        json.dump({"source": "SQuAD v1.1 validation (rajpurkar/squad)", "prefix_len": a.prefix_len,
                   "suffix": SUFFIX, "items": items}, f, indent=1, ensure_ascii=False)
    print(f"wrote eval/qa_squad_n{n}.json: {n} questions from {len(docs)} articles")
    recs = records(docs, items, tok, a.prefix_len, a.max_tokens)
    args = {"model": a.model, "prefix_len": a.prefix_len, "max_tokens": a.max_tokens, "alpha": None,
            "rate": round(1 / a.spacing, 4), "duration": round(n * a.spacing, 1), "seed": 0,
            "drift_at": None, "burst_at": None, "oneoff_frac": 0.0}
    seq = [dict(r, t=round(i * a.spacing, 4)) for i, r in enumerate(recs)]
    write_trace(os.path.join(a.trace_dir, f"qa_squad_n{n}_seq_sp{a.spacing}.jsonl"), seq,
                {"args": args, "qa": f"eval/qa_squad_n{n}.json"})
    for base in a.mix_base:
        with open(base) as f:
            base_recs = [json.loads(line) for line in f]
        with open(base.replace(".jsonl", ".meta.json")) as f:
            base_meta = json.load(f)
        dur = base_meta["args"]["duration"]
        rng = random.Random(1)
        extra = [dict(r, t=round(rng.uniform(0, dur), 4)) for r in recs for _ in range(a.repeat)]
        mixed = sorted(base_recs + extra, key=lambda r: r["t"])
        name = f"qa_squad_n{n}_mix_" + os.path.basename(base)
        write_trace(os.path.join(a.trace_dir, name), mixed,
                    dict(base_meta, qa=f"eval/qa_squad_n{n}.json", qa_repeat=a.repeat, mixed_into=base))


if __name__ == "__main__":
    main()
