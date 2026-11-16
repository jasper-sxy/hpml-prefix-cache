"""Sanity check: send one document with every question template as token IDs.

Checks that (1) vLLM sees exactly the token IDs we send, (2) the shared prefix is reused,
and (3) the answers are about the document. Requires the vLLM server to be running.

Usage:
  python workload/peek_prompt.py --doc 0 --prefix-len 1024
"""
import argparse
import json

import requests
from transformers import AutoTokenizer

from questions import TEMPLATES, suffix_ids, suffix_text

URL = "http://localhost:8000"


def hits(s):
    for line in s.get(URL + "/metrics").text.splitlines():
        if line.startswith("vllm:prefix_cache_hits_total"):
            return float(line.split()[-1])
    return float("nan")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    p.add_argument("--split", default="dev")
    p.add_argument("--doc", type=int, default=0)
    p.add_argument("--prefix-len", type=int, default=1024)
    p.add_argument("--max-tokens", type=int, default=48)
    a = p.parse_args()
    assert a.prefix_len % 16 == 0, "prefix length must be a multiple of 16"

    tok = AutoTokenizer.from_pretrained(a.model)
    data_dir = f"workload/data/{a.model.replace('/', '__')}"
    with open(f"{data_dir}/docs_{a.split}.jsonl") as f:
        docs = [json.loads(line) for line in f]
    d = docs[a.doc]
    prefix = d["token_ids"][: a.prefix_len]
    print(f"doc {a.doc}: {d['title']}  (prefix {a.prefix_len} tokens)\n")

    s = requests.Session()
    s.trust_env = False  # talk to localhost directly, not through a proxy
    h0 = hits(s)
    for qid in range(len(TEMPLATES)):
        ids = prefix + suffix_ids(tok, d["title"], qid)
        body = {"model": a.model, "prompt": ids, "max_tokens": a.max_tokens, "temperature": 0}
        out = s.post(URL + "/v1/completions", json=body).json()
        n = out["usage"]["prompt_tokens"]
        flag = "OK" if n == len(ids) else "MISMATCH"
        print(f"Q{qid}: {suffix_text(d['title'], qid).strip()}")
        print(f"    prompt_tokens={n} (sent {len(ids)}) {flag}")
        print(f"    answer: {out['choices'][0]['text'].strip()[:200]}\n")
    gained = hits(s) - h0
    expected = (len(TEMPLATES) - 1) * a.prefix_len
    print(f"prefix-cache hit tokens gained: {gained:.0f} (expected about {expected})")


if __name__ == "__main__":
    main()
