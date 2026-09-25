"""Encoders (BERT, XLM-R, e5...) as backbones: the micro models of jul.

A decoder is read at the last token of a prompt that ends on the answer. An encoder has no such
token: every position sees the whole text, and a sentence embedding model is trained so that the
mean of its last layer over the text is the text's meaning. So an encoder is read as it was trained:
the features of a layer are [mean over every token of the sequence ; mean over the input's tokens],
and the vector method keeps the first half, as it keeps the last token of a decoder. The option
vectors are the embeddings of the options, the state's is the embedding of the state: zero-shot is
plain embedding similarity, and a tuned head (`autotune`, lexical or hybrid) does the rest.

What changes against a decoder:
- the tokens are wrapped in the tokenizer's special tokens ([CLS] ... [SEP], <s> ... </s>), and a
  batch is padded with an attention mask (attention is bidirectional);
- nothing can be cached: a prefix changes every position after it and is changed by them. The
  prefix of a prompt is therefore run again with each query. Encoder prompts are a few tokens.
- a sequence is capped at the model's positions (512 for BERT-like models);
- the prompts (see `templates`): no chat template, no "in one word" cue; the model's own input
  convention (e5: "query: ") is prepended.

On e5-small (21M parameters outside the embedding), a head over these vectors and TF-IDF scored
0.784 on dair-ai/emotion and 0.890 on Banking77, against 0.797 and 0.900 for Harrier 0.6B (440M),
in 4 ms against 55 ms per value on one CPU thread (jul-lambda, 2026-09-25).
"""

from __future__ import annotations

import warnings

import numpy as np

#: Hugging Face `model_type`s read as encoders.
MODEL_TYPES = {"bert", "xlm-roberta", "roberta", "distilbert", "camembert", "deberta-v2", "electra",
               "mpnet", "modernbert", "nomic_bert", "new"}

#: Input conventions of known embedding models, by repo prefix (the source repo of an export).
TEXT_PREFIXES = {"intfloat/multilingual-e5": "query: ", "intfloat/e5": "query: "}

#: Removed from a "question_options" template to get the "question" one (presets.formulations_for).
OPTIONS_CLAUSES = ("\nPossible answers: {options}.", " ({options})")


def text_prefix(repo: str) -> str:
    return next((p for r, p in TEXT_PREFIXES.items() if repo.startswith(r)), "")


def templates(prefix: str = "") -> dict[str, str]:
    """The encoder counterparts of presets.ONE_WORD / QUESTION_OPTIONS / QUESTION."""
    return {"one_word": prefix + "{state}",
            "question_options": prefix + "{instructions} ({options}): {state}",
            "question": prefix + "{instructions}: {state}"}


def special_tokens(encode) -> tuple[list[int], list[int]]:
    """(ids before, ids after) the text, from `encode(text, add_special_tokens=True)`."""
    ids = list(encode("", add_special_tokens=True))
    return ids[:1], ids[1:]


def batch(seqs: list[list[int]], pools: list, head: list[int], tail: list[int], pad: int
          ) -> tuple[np.ndarray, np.ndarray, list[tuple[int, int]]]:
    """(input_ids, attention_mask, pools shifted past `head`), right-padded."""
    rows = [head + list(s) + tail for s in seqs]
    ids = np.full((len(rows), max(map(len, rows))), pad, dtype=np.int64)
    mask = np.zeros_like(ids)
    for i, r in enumerate(rows):
        ids[i, : len(r)] = r
        mask[i, : len(r)] = 1
    shifted = [((p or (0, len(s)))[0] + len(head), (p or (0, len(s)))[1] + len(head)) for s, p in zip(seqs, pools)]
    return ids, mask, shifted


def features(hidden: dict[int, np.ndarray], mask: np.ndarray, pools: list[tuple[int, int]]
             ) -> list[dict[int, np.ndarray]]:
    """Per row, {layer: (2d,) [mean over the sequence ; mean over its pool]} from (B, T, d) states."""
    out = []
    for i, (start, end) in enumerate(pools):
        n = int(mask[i].sum())
        row = {}
        for layer, h in hidden.items():
            h = np.asarray(h[i], dtype=np.float32)
            row[layer] = np.concatenate([h[:n].mean(0), h[start:end].mean(0)])
            if not np.isfinite(row[layer]).all():
                raise FloatingPointError(f"non-finite hidden state at layer {layer}")
        out.append(row)
    return out


def fit(prefix: list[int], query: list[int], pool, limit: int) -> tuple[list[int], tuple[int, int]]:
    """prefix + query cut to `limit` tokens, and the pool in the result's positions. The end of the
    input is cut first (the suffix is kept); if the prompt alone is still too long (a question listing
    many options), the end of the prefix is cut too."""
    start, end = pool or (0, len(query))
    over = len(prefix) + len(query) - limit
    if over > 0:
        cut = min(over, end - start - 1)
        query = query[: end - cut] + query[end:]
        end -= cut
        cut_prefix = max(0, over - cut)
        prefix = prefix[: len(prefix) - cut_prefix]
        warnings.warn(f"a prompt of {limit + over} tokens is over the encoder's {limit}: its input is cut by "
                      f"{cut} tokens" + (f", its prefix by {cut_prefix}" if cut_prefix else ""), stacklevel=4)
    return prefix + query, (len(prefix) + start, len(prefix) + end)
