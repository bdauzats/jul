# JuL — Juste un LLM

Jev is a hosted _System One model_. This is just an LLM, on your Mac.

Typed decisions with the interface of the TypeSafe (Jev) Python SDK — same imports, same calls, same
response shapes. Everything runs through MLX: no API key, no network, and not one token generated.

That last part is not a limitation. JuL is stopped one step before its first syllable and the answer
is taken straight out of its head: no monologue, no reasoning trace, no opinion on the matter —
nobody asked for one. It has nothing to say, and it says it in 64 milliseconds.

And when it is off-key, there is always `client.autotune(...)` — or `jul autotune` from the shell.

## Install

Apple Silicon only: everything runs through MLX.

```bash
pip install -e .              # add [yaml] for YAML question files, [dev] for the tests
```

### The models

No weights are committed here. `mlx-lm` downloads them from the Hugging Face Hub the first time a
preset is used, into `~/.cache/huggingface`, and they are cached for good after that.

| Preset        | Repository                                                                              | On disk |
| ------------- | --------------------------------------------------------------------------------------- | ------: |
| `minicpm5-2b` | [`openbmb/MiniCPM5-2B-MLX`](https://huggingface.co/openbmb/MiniCPM5-2B-MLX)             |  2.7 GB |
| `qwen3.5-9b`  | [`mlx-community/Qwen3.5-9B-4bit`](https://huggingface.co/mlx-community/Qwen3.5-9B-4bit) |   11 GB |

You only need the preset you actually use, and only one is ever held in memory. To fetch them ahead
of time instead of on the first call:

```bash
huggingface-cli download openbmb/MiniCPM5-2B-MLX
huggingface-cli download mlx-community/Qwen3.5-9B-4bit
```

`jul models` shows which ones are already downloaded.

## Use it as a drop-in for Jev

Change the import; nothing else.

```python
# from typesafe_sdk import TypeSafeClient, Choice, Noul, Score
from jul import TypeSafeClient, Choice, Noul, Score

client = TypeSafeClient(model="minicpm5-2b")          # or "qwen3.5-9b"

response = client.system_one(
    state={"ticket": "I was charged twice for my subscription this month."},
    questions={
        "team": Choice(instructions="Which team should handle this ticket?",
                       criteria={"billing": "payments, invoices, refunds",
                                 "technical": "bugs, errors, crashes",
                                 "sales": "pricing, plans, demos"}),
        "is_bug": Noul(instructions="Does the message report a software bug?"),
        "frustration": Score(instructions="How frustrated is the customer?",
                             criteria=["Calm", "Frustrated but civil", "Very angry"]),
    },
)

response.choices["team"].choice           # "billing"
response.choices["team"].probabilities    # {"billing": 0.88, "technical": 0.11, "sales": 0.01}
response.choices["team"].confidence       # 0.88
response.nouls["is_bug"].noul             # 0.12
response.scores["frustration"].score      # 1.15
response.model, response.usage.input_tokens
```

`AsyncTypeSafeClient` has the same API, awaitable. The arguments that only mean something remotely
(`api_key`, `response_model`, `retry`, `extra_body`, …) are accepted and ignored.

**The option description is what the model compares against**, so options must describe themselves.
The key (`"billing"`) is only the identifier you get back.

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

## Two presets

| Preset                          | Model                           | Layers  |    tau | Latency (p50) | Jev bench, zero-shot |
| ------------------------------- | ------------------------------- | ------- | -----: | ------------: | -------------------- |
| `minicpm5-2b` (alias `fast`)    | `openbmb/MiniCPM5-2B-MLX`       | 39 / 40 | 0.0413 |     **64 ms** | 0.617                |
| `qwen3.5-9b` (alias `accurate`) | `mlx-community/Qwen3.5-9B-4bit` | 31 / 31 | 0.0483 |        273 ms | 0.660                |

## Results

The full published benchmark, 300 examples, run through this library (`scripts/bench_jul.py`).
**Only the zero-shot block compares to Jev** — the other rows receive task data and Jev receives none.

| Zero-shot             |  AG News | Banking77 | Emotion |      Mean |     ECE ↓ |       p50 |
| --------------------- | -------: | --------: | ------: | --------: | --------: | --------: |
| Jev (published)       | **0.91** |  **0.87** |    0.48 | **0.753** |     0.156 |    246 ms |
| jul `qwen3.5-9b`      |     0.79 |      0.74 |    0.45 |     0.660 |     0.175 |    273 ms |
| jul `minicpm5-2b`     |     0.80 |      0.59 |    0.46 |     0.617 | **0.113** | **64 ms** |
| GLiNER2.5 (published) |     0.70 |      0.61 |    0.44 |     0.583 |     0.101 |    128 ms |

| With task data (not comparable)        |  AG News | Banking77 |  Emotion |      Mean |     ECE ↓ |       p50 |
| -------------------------------------- | -------: | --------: | -------: | --------: | --------: | --------: |
| `qwen3.5-9b` + head (1000 labeled)     | **0.94** |      0.79 | **0.58** | **0.770** | **0.077** |    221 ms |
| `minicpm5-2b` + head (1000 labeled)    |     0.92 |      0.77 |     0.58 |     0.757 |     0.119 | **65 ms** |
| `qwen3.5-9b` + context (50 unlabeled)  |     0.91 |      0.78 |     0.55 |     0.747 |     0.096 |    242 ms |
| `minicpm5-2b` + context (50 unlabeled) |     0.84 |      0.63 |     0.47 |     0.647 |     0.140 |     65 ms |

### The baseline worth remembering

Before any of this earns its cost, here is what a bag of words does on the same 300 rows, trained on
the same 1000 labeled examples, with no LLM at all (`scripts/bench_tfidf.py`):

| No LLM | AG News | Banking77 | Emotion | Mean | p50 | Training |
|---|---:|---:|---:|---:|---:|---:|
| TF-IDF + linear SVM | 0.88 | 0.76 | 0.43 | **0.690** | **0.17 ms** | 0.1 s on CPU |

It is 6.3 points behind Jev at roughly **1400× lower latency**, and it beats the zero-shot LLM
outright. It loses on one dataset only — Emotion, where recognising a feeling needs meaning rather
than vocabulary. That is exactly, and only, where the model earns its keep.

If you have labels and your task looks like topic or intent sorting, try this first. It takes a
minute and it may be the end of the story.

Read honestly:

- **Zero-shot, jul beats GLiNER and stays 9 points behind Jev.** Almost all of that gap is Banking77
  and its 72 fine-grained intents (0.74 vs 0.87); on AG News and Emotion the gap is 2 to 12 points.
- **jul is better calibrated than Jev** nearly everywhere. On Emotion Jev's ECE is 0.351: more often
  right, but badly overconfident.
- **With 1000 labeled examples, the small model is enough**: MiniCPM reaches 0.757 at 65 ms, Jev's
  level for a quarter of its latency, and within 1.3 points of tuned Qwen which costs 3.4× more.
- 100 rows per dataset, so ±5 points per cell and ±3 on the mean.

Every layer and temperature was fitted on dev datasets the Jev benchmark never uses. Models load on
first use, one at a time (Qwen3.5-9B is about 5.5 GB).

## Context — what the data looks like

Jev has no equivalent. A context acts at three levels.

```python
from jul import TypeSafeClient, Context

tickets = Context(
    description="Customer support tickets of an online bank, written in English by customers.",
    examples=open("sample_tickets.txt").read().splitlines(),   # ~50–200 real texts, no labels
)

client = TypeSafeClient(model="qwen3.5-9b", context=tickets)   # for every call
client.system_one(state, questions, context=tickets)           # or per call
```

| Element       | What it does                                     | Cost                            | Status                                                                                                        |
| ------------- | ------------------------------------------------ | ------------------------------- | ------------------------------------------------------------------------------------------------------------- |
| `description` | prepended as `Context: …` to both formulations   | nil (sits in the cached prefix) | **off by default, measured harmful on average** — `use_description=True` to try it                            |
| `examples`    | their mean vector becomes the center of the task | computed once                   | measured: helps on topics (AG News 0.66 → 0.75), slightly hurts on fine-grained tasks (Banking77 0.55 → 0.53) |

A task center is the best center measured for both presets (JOURNAL §9 octies), which is the main
reason to bother with a context at all. **Ten examples already capture most of the gain, fifty is the
sweet spot, two hundred adds nothing** (JOURNAL §9 decies). Below ten the center is noise and can be
worse than no context at all — with 5 examples, a ticket rated `billing` at 0.98 flipped to a wrong
`technical`. `Context` warns under ten.

The gain is uneven: Yahoo Answers topics jumped 0.480 → 0.600 with Qwen and ten examples, while
financial sentences lost 3 points. A task center helps most when your texts have a style of their own.

**The `description` is off by default.** Measured on both presets over four datasets, it failed the
plan's bar: MiniCPM gained 1.3 points on average with one dataset losing 4, and Qwen lost on all four
(−5.2 on average). Worse, the sign flips between models on the same data. Switch it on with
`use_description=True` only if you measure a gain on your own.
| `labeled` | fits a temperature and a per-option bias | computed once | proven, but makes any comparison with Jev unfair |

Adding information to a prompt does not always help: listing the options helped on topics and
emotions but hurt on sentiment (JOURNAL §9 quater). Measure before trusting the description.

Contexts are cached on disk under `~/.jul/contexts/<name>/` and can be reused by name:

```python
client.system_one(state, questions, context="tickets")
```

## `autotune(...)` — when it is off-key

Give it labeled examples and it stops missing the note. A small head over the vectors the method
already computes: **the LLM itself is not modified**, training takes seconds, and inference stays
exactly as fast — the correction is applied after the fact, and nobody hears it.

```python
report = client.autotune("tickets", questions, labeled)   # labeled: [(state, {"team": "billing"}), …]
print(report["team"])
client.system_one(state, questions, context="tickets")   # uses the head automatically
```

```
question 'topic': 200 labeled examples, 4 options
  held out for judging : 40
  zero-shot accuracy   : 0.800
  tuned head accuracy  : 0.925
  result               : ACTIVE (beats zero-shot on the held-out examples (+0.125))
```

**Safety net**: the head is judged by stratified cross-validation, so every labeled example is
predicted by a head that never saw it. If it does not beat the zero-shot method there, it is not
activated and the report says so. On six measured curves it decides right 19 times out of 22, and its
three errors are each worth under 3 points.

| Labeled examples per question | What happens                                                               |
| ----------------------------- | -------------------------------------------------------------------------- |
| below `max(20, 3 × options)`  | calibration only (temperature + per-option bias)                           |
| above it                      | a head is trained, and kept only if it beats zero-shot in cross-validation |

The floor scales with the number of options because that is what the measurement showed: 200 examples
is plenty for 4 options and not enough for Banking77's 72. Measured gains at 1000 examples (JOURNAL
§9 nonies, on a validation split, ±3.5 points):

|             | AG News       | Emotion       | Banking77     |
| ----------- | ------------- | ------------- | ------------- |
| MiniCPM5-2B | 0.770 → 0.870 | 0.480 → 0.630 | 0.555 → 0.695 |
| Qwen3.5-9B  | 0.745 → 0.875 | 0.505 → 0.570 | 0.715 → 0.830 |

**Qwen benefits more, and sooner** — +9 points from 24 examples on AG News, its plateau by 100, where
MiniCPM needs 500 to 1000. The bigger model's vectors are more separable, so a linear probe learns
from fewer examples.

A head only knows the options it saw, and only the preset whose vectors it saw: change either and you
call `autotune(...)` again.

## Command line

```bash
jul ask choice "Which team should handle this ticket?" \
    -o billing:"payments, invoices" -o technical:"bugs, errors" \
    --state "I was charged twice" --model minicpm5-2b

jul run questions.yaml --input tickets.jsonl --output answers.jsonl --context tickets
jul context create tickets --description "Support tickets of an online bank" --examples sample.txt
jul context list | show tickets | delete tickets
jul autotune tickets --questions questions.yaml --labeled labeled.jsonl
jul models
jul lab ...        # the research commands of the prototype
```

The JSON printed has the same shape as a Jev API response. A question file is YAML or JSON:

```yaml
team:
  type: choice
  instructions: Which team should handle this ticket?
  criteria:
    billing: payments, invoices, refunds
    technical: bugs, errors, crashes
is_bug:
  type: noul
  instructions: Does the message report a software bug?
frustration:
  type: score
  instructions: How frustrated is the customer?
  criteria: [Calm, Frustrated but civil, Very angry]
```

## What is measured, and what is not

Measured (see `docs/JOURNAL.md`):

- Both presets' layer and temperature, on dev datasets only.
- Vectors beating letters for `Noul` and `Score`, on data disjoint from the benchmark.
- Which center to subtract, per model (both ship a generic one for `one word`).
- That the context `description` hurts on average, and how many `examples` a task center needs.
- `Choice` by vectors, on the full Jev benchmark.
- The effect of `examples` as a task center.
- A per-task head, on both presets and three datasets, with the example-count curve.

Not yet measured — do not rely on these without checking:

- **`Noul` and `Score` thresholds**: vectors are now the default for all three types, measured on 480
  class-balanced yes/no examples (accuracy 0.771 vs 0.692 for letters, AUC 0.938 vs 0.904, ECE 0.148
  vs 0.238) and on the 9-case ordinal set (7/9 vs 6/9). But **both readings still have a threshold
  bias** — they rank well (AUC 0.87–0.99) and decide badly. Fix it with `client.autotune(...)` on a
  few dozen labeled examples. `method="letters"` keeps the old reading.
- **Centering**: measured (JOURNAL §9 octies). Not centering costs 4.5–8.5 points, so always centre.
  Which centre matters less, and differently per model: on Qwen the three are tied, on MiniCPM the
  generic centre gains 4 points over the option mean. **The best centre is the task's own**, i.e.
  `Context(examples=…)`, for both models — 200 examples per strategy though, so ±3.5 points.
- Everything was tuned in English.

## Layout

```
jul/
  lib/jul/          the library — the CLI never imports anything else
    types.py        questions and answers, same fields as the Jev SDK
    presets.py      the two presets: repo, layers, tau, centers
    engine.py       the vector method: formulations, cached prefixes, combination
    client.py       TypeSafeClient / AsyncTypeSafeClient
    context.py      Context: description, examples, labeled; disk cache
    tuning.py       `autotune(...)`: per-task head, cross-validated, with a safety net
    backbone.py     MLX pass, stop at a layer, cached prefix
    calibration.py  temperature and per-option bias
    lab/            research code, kept out of the public API
  cli/jul_cli/      the command line
  scripts/          data preparation, tuning experiments, the benchmark
  tests/            fast tests, plus a slow suite that loads the models
```

## Tests

```bash
pytest tests                       # 61 tests, under a second, no model and no data
JUL_SLOW=1 pytest tests            # all 80, downloads and loads both presets (~90 s)
JUL_SLOW=1 pytest tests -m slow    # only the 19 that need a model
```

**`JUL_SLOW` is a test-only switch**, read by `tests/conftest.py` and by nothing in the library. The
19 tests marked `@pytest.mark.slow` load a real MLX model, so a plain `pytest` skips them rather than
pulling 13 GB of weights on someone who just cloned the repo. They are reported as skipped, with the
reason, never silently dropped. Set `JUL_SLOW=1` to run them.

The slow suite includes a per-preset non-regression check against `tests/fixtures/baseline.json`,
which ships with the repo — that one needs the models but no dataset download.

## Reproducing the measurements

Every number in this README comes from a script in `scripts/`. None of the data is committed; these
steps fetch it. Expect around 2 hours end to end on an M-series Mac, most of it Qwen.

```bash
pip install -e ".[repro]"
git clone https://github.com/AbdelStark/jev-benchmarks external/jev-benchmarks
pip install -e "external/jev-benchmarks[data]"
```

Then build the datasets. They all come from [`btzsc/btzsc`](https://huggingface.co/datasets/btzsc/btzsc)
on the Hub, split by the benchmark's own config so that tuning data and benchmark rows never overlap:

```bash
python scripts/prepare_dev.py        # 4 tuning datasets, never used by the benchmark
python scripts/prepare_btzsc.py      # 1000 train / 200 val / 100 test per benchmark dataset
python scripts/prepare_generic.py    # varied texts, for the generic centers
```

Then the experiments, in order. Each writes to `runs/` and prints its table:

```bash
python scripts/dev_fit_tau.py minicpm5-2b 39 40 50     # temperatures          (~4 min)
python scripts/dev_generic_center.py minicpm5-2b       # centering             (~4 min)
python scripts/dev_generic_center.py qwen3.5-9b        #                       (~10 min)
python scripts/dev_context_effect.py minicpm5-2b 100   # does a Context help?  (~7 min)
python scripts/dev_context_effect.py qwen3.5-9b 100    #                       (~25 min)
python scripts/dev_tuning_curve.py minicpm5-2b         # how many labels?      (~5 min)
python scripts/dev_tuning_curve.py qwen3.5-9b          #                       (~17 min)
```

`dev_generic_center.py` rewrites the shipped centers in `lib/jul/assets/`, and `dev_tuning_curve.py`
caches encoded vectors in `features/` so it can be re-run for free.

Finally the benchmark itself. It checks the published manifest's SHA-256 and verifies that no
training row appears in it, then runs the three variants **through the public API**:

```bash
python scripts/bench_jul.py minicpm5-2b                # (~7 min)
python scripts/bench_jul.py qwen3.5-9b                 # (~25 min)
python scripts/bench_tfidf.py                          # the no-LLM baseline (~5 s)
```

Reports land in `runs/jev-bench-jul/`. Tune on the dev datasets, and run the benchmark once, at the
end — that is the whole point of keeping the two apart.

## License

MIT — see [LICENSE](LICENSE).
