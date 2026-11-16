"""Phase 2.2: question suffixes appended after a document prefix.

A request = document prefix (shared, cacheable) + question suffix (differs per request).
The instruction lives in the suffix, so different documents never share a first block.
"""

TEMPLATES = [
    "What is this article about? Answer in one sentence.",
    "According to the article, when did the main events described take place?",
    "Name two people, places, or organizations mentioned in the article.",
    "What is one notable fact the article states about {title}?",
    "Summarize the first part of the article in two sentences.",
]
SUFFIX_FORMAT = "\n\nQuestion: {q}\nAnswer:"


def suffix_text(title, qid):
    return SUFFIX_FORMAT.format(q=TEMPLATES[qid].format(title=title))


def suffix_ids(tok, title, qid):
    # No special tokens: the suffix is appended directly after the prefix token IDs.
    return tok(suffix_text(title, qid), add_special_tokens=False)["input_ids"]
