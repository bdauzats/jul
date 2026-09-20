"""Helpers shared by the dev experiment scripts."""


def short_names(labels):
    """Drop the words shared by all labels at the start and end: 'This example ... is about sports' -> 'sports'."""
    words = [l.split() for l in labels]
    pre = 0
    while all(len(w) > pre + 1 for w in words) and len({w[pre] for w in words}) == 1:
        pre += 1
    suf = 0
    while all(len(w) > pre + suf + 1 for w in words) and len({w[-1 - suf] for w in words}) == 1:
        suf += 1
    return [" ".join(w[pre:len(w) - suf]).strip(" .:") for w in words]


def match(answer, names):
    """Index of the label named in a generated answer (earliest mention, after any thinking), or None."""
    answer = answer.split("</think>")[-1].lower()
    hits = [(answer.find(n.lower()), i) for i, n in enumerate(names) if n.lower() in answer]
    return min(hits)[1] if hits else None
