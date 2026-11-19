"""Phase 2.1: build the document pool.

WikiText-103 articles -> fixed-length prefixes stored as token IDs (one pool per model).
Each record keeps the first MAX_LEN token IDs of an article. Shorter prefixes are taken
later as token_ids[:L], so every prefix length uses exactly the same tokens.

Usage:
  python workload/build_doc_pool.py --model Qwen/Qwen2.5-3B-Instruct
"""
import argparse
import hashlib
import json
import os
import random
import re

from datasets import load_dataset
from transformers import AutoTokenizer

# Top-level article title line in WikiText, e.g. " = Valkyria Chronicles III = "
HEADING = re.compile(r"^ = [^=].* = $")


def clean(text):
    # WikiText escapes some characters: "2 @,@ 011" -> "2,011", "game @-@ like" -> "game-like"
    for a, b in ((" @-@ ", "-"), (" @,@ ", ","), (" @.@ ", ".")):
        text = text.replace(a, b)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def split_articles(lines):
    articles, title, buf = [], None, []
    for line in lines:
        if HEADING.match(line.rstrip("\n")):
            if title is not None:
                articles.append((title, clean("".join(buf))))
            title, buf = line.strip().strip("=").strip(), []
        elif title is not None:
            buf.append(line)
    if title is not None:
        articles.append((title, clean("".join(buf))))
    return articles


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    p.add_argument("--max-len", type=int, default=2048, help="tokens kept per article")
    p.add_argument("--n-dev", type=int, default=100)
    p.add_argument("--n-test", type=int, default=100)
    p.add_argument("--n-oneoff", type=int, default=0,
                   help="extra documents for one-off traffic (half dev, half test)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="workload/data")
    a = p.parse_args()
    assert a.max_len % 16 == 0, "max-len must be a multiple of the KV block size (16)"

    tok = AutoTokenizer.from_pretrained(a.model)
    ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="train")
    articles = split_articles(ds["text"])
    print(f"articles in WikiText-103 train: {len(articles)}")

    rng = random.Random(a.seed)
    rng.shuffle(articles)
    need = a.n_dev + a.n_test + a.n_oneoff
    pool, seen_first_blocks = [], set()
    for title, text in articles:
        if len(text) < a.max_len * 3:  # cheap filter: clearly too short
            continue
        # add_special_tokens=True: Llama gets its BOS token, Qwen gets nothing
        ids = tok(text, add_special_tokens=True)["input_ids"]
        if len(ids) < a.max_len:
            continue
        first_block = tuple(ids[:16])  # one KV block; documents sharing it would share cache
        if first_block in seen_first_blocks:
            continue
        seen_first_blocks.add(first_block)
        pool.append({"doc_id": len(pool), "title": title, "token_ids": ids[: a.max_len]})
        if len(pool) == need:
            break
    assert len(pool) == need, f"only found {len(pool)} long-enough articles"

    out_dir = os.path.join(a.out, a.model.replace("/", "__"))
    os.makedirs(out_dir, exist_ok=True)
    n_main = a.n_dev + a.n_test
    half = a.n_oneoff // 2
    splits = [("dev", pool[: a.n_dev]), ("test", pool[a.n_dev:n_main])]
    if a.n_oneoff:
        splits += [("oneoff_dev", pool[n_main:n_main + half]), ("oneoff_test", pool[n_main + half:])]
    for name, docs in splits:
        with open(os.path.join(out_dir, f"docs_{name}.jsonl"), "w") as f:
            for d in docs:
                f.write(json.dumps(d, ensure_ascii=False) + "\n")

    first_blocks = {tuple(d["token_ids"][:16]) for d in pool}
    # digest over dev + test only, so adding one-off documents keeps it unchanged
    digest = hashlib.sha256(json.dumps([d["token_ids"] for d in pool[:n_main]]).encode()).hexdigest()[:16]
    meta = {
        "model": a.model,
        "dataset": "Salesforce/wikitext wikitext-103-raw-v1 (train split)",
        "max_len": a.max_len,
        "seed": a.seed,
        "n_dev": a.n_dev,
        "n_test": a.n_test,
        "n_oneoff": a.n_oneoff,
        "distinct_first_blocks": len(first_blocks),  # should equal n_dev + n_test + n_oneoff
        "token_digest": digest,  # same settings -> same digest on every machine
    }
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(json.dumps(meta, indent=2))
    print("example:", pool[0]["title"], "|", repr(tok.decode(pool[0]["token_ids"][:60])))


if __name__ == "__main__":
    main()
