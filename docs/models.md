# Models

## Every model measured

Jev scores 0.753 on the same benchmark. `wemm-4b-4bit` and `minicpm5-2b` are built in; any other is
one command away,
`jul models add <name> --repo <repository>`, which fits it on the dev sets.

| Name | Repository | Memory | Jev bench, zero-shot | + autotune |
| --- | --- | ---: | ---: | ---: |
| `wemm-4b-4bit` | [`usejul/WeMM-Embedding-4B-mlx-4bit`](https://huggingface.co/usejul/WeMM-Embedding-4B-mlx-4bit) | 2.6 GB | 0.857 | **0.897** |
| `wemm-4b` | [`tencent/WeMM-Embedding-4B`](https://huggingface.co/tencent/WeMM-Embedding-4B) | 4.5 GB | **0.877** | 0.862 |
| `wemm-9b` | [`tencent/WeMM-Embedding-9B`](https://huggingface.co/tencent/WeMM-Embedding-9B) | 9.0 GB | 0.863 | 0.857 |
| `wemm-2b` | [`hfadam/WeMM-Embedding-2B-MLX-4bit`](https://huggingface.co/hfadam/WeMM-Embedding-2B-MLX-4bit) | 1.5 GB | 0.777 | 0.805 |
| `f2llm-8b` | converted locally, not published | 4.5 GB | 0.820 | 0.850 |
| `f2llm-4b` | [`fcmeyer/F2LLM-v2-4B-mlx-6bit`](https://huggingface.co/fcmeyer/F2LLM-v2-4B-mlx-6bit) | 3.0 GB | 0.840 | 0.853 |
| `f2llm-1.7b` | converted locally, not published | 1.0 GB | 0.817 | 0.833 |
| `f2llm-0.6b` | [`fcmeyer/F2LLM-v2-0.6B-bf16-mlx`](https://huggingface.co/fcmeyer/F2LLM-v2-0.6B-bf16-mlx) | 1.1 GB | 0.623 | 0.817 |
| `qwen3-embedding-8b` | [`mlx-community/Qwen3-Embedding-8B-4bit-DWQ`](https://huggingface.co/mlx-community/Qwen3-Embedding-8B-4bit-DWQ) | 4.0 GB | 0.773 | 0.806 |
| `qwen3-embedding-4b` | [`mlx-community/Qwen3-Embedding-4B-4bit-DWQ`](https://huggingface.co/mlx-community/Qwen3-Embedding-4B-4bit-DWQ) | 2.1 GB | 0.733 | 0.775 |
| `qwen3-embedding-0.6b` | [`mlx-community/Qwen3-Embedding-0.6B-4bit-DWQ`](https://huggingface.co/mlx-community/Qwen3-Embedding-0.6B-4bit-DWQ) | 0.32 GB | 0.637 | 0.760 |
| `harrier-0.6b` | [`majentik/harrier-oss-v1-0.6b-MLX-4bit`](https://huggingface.co/majentik/harrier-oss-v1-0.6b-MLX-4bit) | **0.31 GB** | 0.667 | 0.814 |
| `minicpm5-2b` | [`openbmb/MiniCPM5-2B-MLX`](https://huggingface.co/openbmb/MiniCPM5-2B-MLX) | 2.7 GB | 0.617 | 0.757 |
| `minicpm5-2b-decision` | [`usejul/minicpm5-2b-decision-mlx-4bit`](https://huggingface.co/usejul/minicpm5-2b-decision-mlx-4bit) | 1.3 GB | 0.796 |  |
| `ternary-bonsai-1.7b` | [`prism-ml/Ternary-Bonsai-1.7B-mlx-2bit`](https://huggingface.co/prism-ml/Ternary-Bonsai-1.7B-mlx-2bit) | 0.46 GB | 0.640 | 0.760 |
| `ternary-bonsai-8b` | [`prism-ml/Ternary-Bonsai-8B-mlx-2bit`](https://huggingface.co/prism-ml/Ternary-Bonsai-8B-mlx-2bit) | 1.75 GB | 0.563 | 0.753 |
| `bitnet-2b` | [`mlx-community/bitnet-b1.58-2B-4T`](https://huggingface.co/mlx-community/bitnet-b1.58-2B-4T) | 1.1 GB | 0.617 | 0.723 |
| `e5-small` | [`intfloat/multilingual-e5-small`](https://huggingface.co/intfloat/multilingual-e5-small), onnx 8-bit | **0.09 GB** | 0.587 | 0.670 (**0.770** hybrid) |

`+ autotune` is a head on the vectors (`features="vector"`). With a hybrid head (vectors + TF-IDF, see
[`features=`](tuning.md#what-the-head-reads-features)), `e5-small` — an encoder, see [Micro models](#micro-models-encoders) —
reaches **0.770**, above Jev's 0.753 (AG News 0.92, Banking77 0.84, Emotion 0.55), at 21 ms per text on
an M4 Pro. The other models were not measured with a hybrid head.

## Adding a model

```bash
jul models add my-model --repo org/Some-Instruct-3B            # on the default backend
jul models add minicpm5-2b --backend torch                     # a known preset, on another backend
```

One command runs the protocol that produced the built-in presets, on the dev datasets only (never on
the Jev benchmark):

1. **checks** — the model loads, the prefix cache leaves the vectors unchanged, a call does not
   disturb the next one, the letters reading (Noul, Score) has single-token markers;
2. **extraction** — 4 dev sets x 50 examples and 200 generic texts, every layer of the upper half
   read in the same pass;
3. **choice** — layers and center by dev accuracy averaged over neighbouring layers (a plateau, not a
   lucky peak), then tau by pooled NLL;
4. **output** — `~/.jul/presets/<name>@<backend>.json` and its generic center, used from then on by
   `--model <name>` on that backend. `jul models` lists it with what was measured.

The calibration data is downloaded once from BTZSC into `~/.jul/calibration-data` (needs
`pip install "jul[calibrate]"`), or taken from `--data <dir>`. The dev accuracy it reports comes
with its standard error (±3.5 points at n=200): it orients, it does not rank close models. Measure
on the Jev bench separately, once.

## Micro models: encoders

An encoder (BERT, XLM-R, multilingual-e5…) is a backbone like any other: `jul models add`,
`autotune` and `jul pack` run on it unchanged, on the torch and onnx backends. JuL recognizes one by
its `model_type` and reads it as it was trained, not as a decoder (`lib/jul/encoder.py`):

- the vector is the mean of the layer over the whole sequence (the sentence embedding e5 was trained to
  produce), not a last token; zero-shot is plain embedding similarity between state and options;
- the prompts are the model's own input convention (`query: {state}` for e5), no chat template and no
  "in one word" cue (`Backbone.templates`);
- attention is bidirectional, so no prefix can be cached: the prefix runs again with each query, and a
  sequence is cut to the model's positions (512), the input first, then the prefix;
- no logits: the letters reading and decision models need a decoder.

```bash
python -m jul.backends.onnx_export intfloat/multilingual-e5-small models/e5-small-onnx
python -m jul.backends.onnx_export models/e5-small-onnx models/e5-small-onnx-w8 --int8   # 86 MB
jul models add e5-small --repo models/e5-small-onnx-w8 --backend onnx
```

Why bother: on a support-triage task (jul-lambda, 2026-09-25), multilingual-e5-small (21 M
parameters outside its embedding) with a hybrid head matched Harrier 0.6B (440 M) — emotion 0.787
against 0.791, Banking77 0.890 against 0.890 — in **4 ms per message against 37 ms** on an M4 Pro,
and 17 ms on a 1,769 MB AWS Lambda ($0.59 per million calls, cold start 2.4 s). Zero-shot, on the
calibration dev sets, it scored 0.590 against 0.475 for Harrier 0.6B. The heads carry it: alone,
zero-shot, a small encoder is no match for a 4B embedding model.

## Decision models

A *decision model* is a model trained to answer questions about a state, rather than to write text. It
reads the state and the options through delimiter tokens it learned, and a small head scores each
option against the question. `jul` runs one with no code of its own: everything that model needs sits
next to its weights, in a `decision.json` (delimiters, layout, readout, head file, temperature, and the
longest state and question it was trained on).

```bash
jul models add minicpm5-2b-decision --repo usejul/minicpm5-2b-decision-mlx-4bit   # MLX, 1.3 GB
jul models add minicpm5-2b-decision --repo usejul/minicpm5-2b-decision --backend torch
jul ask choice "Which team should handle this ticket?" -o billing -o shipping -o access \
    --state "I was charged twice for order 4411" --model minicpm5-2b-decision
```

A repo (or a local directory) holding a `decision.json` is registered as it is: there is nothing to
fit, no layer to choose and no tau, so the command returns at once. The API is the same as for any other model, and all three
question types go through the same format. The state is encoded once per call and every question
continues from it, so questions never see each other.

The state is paid once per call: a ticket with four questions (two `Choice`, a `Noul` and a `Score`)
answers in **180 ms** on an M4 Pro, against 65 ms for the first question alone. What costs is the
options — they are re-read on every request — so a three-option question runs in 64 ms where a
fifty-nine-option one takes 596 ms. Weights: [`usejul/minicpm5-2b-decision-mlx-4bit`](https://huggingface.co/usejul/minicpm5-2b-decision-mlx-4bit)
(MLX, 1.3 GB) and [`usejul/minicpm5-2b-decision`](https://huggingface.co/usejul/minicpm5-2b-decision)
(PyTorch, bf16).

Two differences with the presets above: `autotune(...)` does not apply (its heads are trained on the
vectors of the other method, and such a model needs a full fine-tune instead), and a state longer than
the limit in its `decision.json` is truncated rather than stretched.

## The built-in presets

| Preset                          | Model                               | Layers  |    tau | p50, M4 Pro | p50, M5 Max | Jev bench, zero-shot |
| ------------------------------- | ----------------------------------- | ------- | -----: | ----------: | ----------: | -------------------- |
| `wemm-4b-4bit` (alias `accurate`, default) | `usejul/WeMM-Embedding-4B-mlx-4bit` | 31 / 31 | 0.0553 |      146 ms |       55 ms | **0.857**            |
| `minicpm5-2b` (alias `fast`)    | `openbmb/MiniCPM5-2B-MLX`           | 39 / 40 | 0.0413 |   **62 ms** |             | 0.617                |

`wemm-4b-4bit` is the default because it is the most accurate: 10 points above Jev with no training.
On the same M4 Pro it is 2.4 times slower than `minicpm5-2b`, which stays the fast option.

A third option does not read a general model at all: `minicpm5-2b-decision` is MiniCPM5-2B *trained*
to answer typed questions (a merged LoRA and a pointer head). It has no layer and no tau — it brings
its own format — and it is 6 points ahead on the development sets, at a latency that depends on how
many options a question has. It is not built in: `jul models add` registers it in a second.

## How it answers

For each formulation of the preset, the state and every option go through the same prompt; the answer
is the option whose vector is closest (cosine, after subtracting a center). The scores of the two
formulations are averaged, then `softmax(cosine / tau)`.

```
1.  This text: "{state}" means in one word: "
2.  {instructions}\nPossible answers: {options}.\nText: "{state}"\nIn one word, the answer is: "
```

The model never writes anything. Everything state-independent — the prompt prefix, the option vectors,
the center — is computed once, so a call only pays for its own tokens.

## Every reading, and every setting

Four ways to read a model. A preset picks one; a call may override it.

| Reading | What it compares | Chosen by | Available on |
| --- | --- | --- | --- |
| **vector** (default) | cosine between the state's hidden state and each option's, `softmax(cos / tau)` | preset `method: "vector"` | any model |
| **letters** | the logits of the option letters (A, B, C…) at the next position | `method="letters"`, per call or per client | any model; a tuned head overrides it |
| **pointer** | a trained pointer head, at the delimiter tokens of the format in `decision.json` | preset `method: "pointer"` | decision models only |
| **tuned head** | a logistic head fitted by `autotune` on vector features | `autotune()` plus a `Context` | pins the reading to vectors |

A decision model may also **route by option count**: below the threshold the pointer head answers, above
it the vector reading does. The pointer reads every option on every call, so its cost grows with the option
list (65 ms at 3 options, 868 ms at 59) while the vector reading is flat; past ~30 options it stops earning
that latency. Routing is per question, so one call can mix both. Measured on `massive`, 59 options, same
weights: **868 ms → 77 ms at equal accuracy**.

| Setting | Where it lives | Default | What it does |
| --- | --- | --- | --- |
| `formulations` | preset | 2 built in | the prompts, and the layer each is read at |
| `tau` | preset | per model | temperature of `softmax(cos / tau)` |
| `center` | preset | `"options"` | what is subtracted before the cosine: `options`, `generic`, `none` |
| `one_word` | preset | — | layer and tau of the single-formulation variant (`one_word_only=True`) |
| `head.temperature` | `decision.json` | 1.954 for ours | divides the pointer logits; never changes an answer |
| `limits.max_state_tokens` / `max_branch_tokens` | `decision.json` | 384 / 1024 | where a too-long state or question is cut |
| `routing.above_options` | preset, measured by `jul models add` | measured | option count above which the vector reading answers |
| `routing` formulations, `tau`, `center` | preset, fitted by `jul models add` | — | the fallback reading, fitted on these very weights |
| `route_above=N` | per call | the model's value | overrides that threshold; `0` disables routing |
| `method=` | per call or client | `"vector"` | `vector` or `letters`; ignored on a pointer preset |
| `one_word_only=` | client | `False` | one formulation instead of two: faster, a little less accurate |
| `backend=` | client, or `JUL_BACKEND` | auto | `mlx` or `torch`; `onnx` only when asked |
| `JUL_BATCH_TOKENS` / `JUL_BATCH_SIZE` | environment | per backend | how many rows the backbone batches at once |
| `features=` | `autotune` | `"vector"` | what a tuned head reads: `vector`, `lexical` (TF-IDF) or `hybrid` |
| `formulations=` | `autotune` | the preset's | the prompts a question is read with, by name, per question if a dict |
| `JUL_HOME` | environment | `~/.jul` | where presets, contexts and calibration data live |
| `JUL_ONNX_MODEL` | environment | the export's graph | another graph for the onnx backend: a path or `s3://bucket/key` |
| `JUL_ONNX_THREADS` | environment | ONNX Runtime's | intra-op threads (on Lambda: one per whole vCPU) |
| `JUL_ONNX_MAX_TOKENS` / `JUL_ONNX_BATCH_TOKENS` | environment | 2048 / 2048 | longest row, and rows × longest per run, on onnx |

**Routing trades accuracy for speed, and the trade is not free.** Measured on massive by subsampling one
set's own options — so the option count is not confounded with the task — the pointer head is better at
*every* count, by about 5 points, while costing 105 ms at 3 options and 832 ms at 59. There is no count
above which it stops earning its answer; there is only a count above which its speed stops being worth
those points.

So `jul models add` **measures** the threshold rather than guessing one: it takes the smallest option
count at which the vector reading is three times faster, and records in the preset what that costs in
accuracy. A model where that never happens gets no routing at all. `--route-above N` sets it by hand and
`--no-routing` skips the whole fitting; the model's own `decision.json` may also carry a threshold.

**For maximum accuracy, pass `route_above=0`** on the call: every question then goes to the pointer head,
whatever its option count, and you pay the latency in the table above. The default is a compromise, and
`jul models add` prints exactly what it costs on the dev set it measured.
