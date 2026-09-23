# JuL — Juste un LLM

A headless decision runtime: same typed-decision interface as Jev's SDK, but the model underneath is
yours to pick, swap, or fine-tune. No hosted API, no fixed backbone — point it at any local LLM (MLX
or PyTorch) and it becomes a calibrated decision head for that model.

Jev is a hosted _System One model_. JuL is the same idea, un-hosted and un-fixed: same imports, same
calls, same response shapes as the TypeSafe (Jev) Python SDK — but everything runs on your machine, on
whichever backbone you choose, and not one token gets generated.

That last part is not a limitation. JuL is stopped one step before its first syllable and the answer
is taken straight out of its head: no monologue, no reasoning trace, no opinion on the matter —
nobody asked for one. It has nothing to say, and it says it in 64 milliseconds.

And when it is off-key, there is always `client.autotune(...)` — or `jul autotune` from the shell.

## Install

Pick a backend: MLX on Apple Silicon, PyTorch (transformers) anywhere else — CUDA, CPU, or MPS.

```bash
pip install -e ".[mlx]"       # Apple Silicon
pip install -e ".[torch]"     # Linux / Windows / any GPU
# add [yaml] for YAML question files, [dev] for the tests
```

Or let `jul setup` do the rest: it picks the backend (MLX on Apple Silicon, else PyTorch), installs
it if missing, downloads MiniCPM5-2B once, and checks one real decision. Running it again redoes only
the check.

```bash
pip install -e .
jul setup                          # --backend torch, --model qwen3.5-9b, --no-install, --no-check
```

Or have a big LLM install the small one. Paste this into Claude Code, Codex, or any agent that has
a shell:

```text
Install jul (https://github.com/bdauzats/jul), a local library that answers typed questions
with a 2B model, and check that it works on this machine.

1. Clone it (or use the checkout I am in) and create a virtualenv with Python >= 3.10 inside it.
2. In that venv: pip install -e .   then   jul setup
   jul setup picks the backend (MLX on Apple Silicon, else PyTorch), installs it, downloads
   MiniCPM5-2B (a few GB) and runs one test decision.
3. Then run:
   jul ask choice "Which team should handle this ticket?" -o billing:"payments, invoices" -o technical:"bugs, errors" --state "I was charged twice"
   and show me the JSON.

Do not use sudo and do not install anything outside the venv. If a step fails, show me the error and
what you suggest before trying something else. At the end, tell me in two lines: the backend, the
latency per decision, and the answer to the ticket.
```

Yes: a model that writes essays, installing one that answers in a single word. Neither of them minds.

The backend defaults to MLX when it is installed on Apple Silicon, else PyTorch. Force it with
`TypeSafeClient(backend="torch")`, `jul ask ... --backend torch`, or `JUL_BACKEND=torch`. The torch
device defaults to cuda > mps > cpu (`JUL_DEVICE` overrides it), in bfloat16 — float32 on CPU, float16
on GPUs older than Ampere such as the T4, which have no bfloat16 tensor cores (`JUL_DTYPE` overrides it).

**The two backends do not run the same weights.** The MLX presets are 4-bit; PyTorch loads the
original bf16 weights. On the same weights the two backends read the same vectors (cosine > 0.9999,
`tests/test_backends.py`), but 4-bit moves them to a cosine of ~0.95 with bf16. The presets' tau and
generic centers were fitted on the 4-bit MLX weights, so on PyTorch they are a starting point, not
measured values. A backend-specific center is picked up from `assets/<preset>.<backend>.<formulation>.center.npy`
when it exists. Centers, heads and calibrations saved in a context are keyed per backend, so a head
trained on MLX is never applied to PyTorch vectors.

The research code under `jul.lab` still trains its heads with MLX.

### The models

No weights are committed here. `jul setup` downloads them from the Hugging Face Hub into
`~/.cache/huggingface`, once. The CLI asks for it when they are missing; the Python API downloads them
on the first call.

| Preset        | MLX repository                                                                          | On disk | PyTorch repository                                                  |
| ------------- | --------------------------------------------------------------------------------------- | ------: | ------------------------------------------------------------------- |
| `minicpm5-2b` | [`openbmb/MiniCPM5-2B-MLX`](https://huggingface.co/openbmb/MiniCPM5-2B-MLX)             |  2.7 GB | [`openbmb/MiniCPM5-2B`](https://huggingface.co/openbmb/MiniCPM5-2B) |
| `qwen3.5-9b`  | [`mlx-community/Qwen3.5-9B-4bit`](https://huggingface.co/mlx-community/Qwen3.5-9B-4bit) |   11 GB | [`Qwen/Qwen3.5-9B`](https://huggingface.co/Qwen/Qwen3.5-9B) ¹       |
| `minicpm5-2b-decision` ² | [`bdauzats/minicpm5-2b-decision-mlx-4bit`](https://huggingface.co/bdauzats/minicpm5-2b-decision-mlx-4bit) | 1.3 GB | [`bdauzats/minicpm5-2b-decision`](https://huggingface.co/bdauzats/minicpm5-2b-decision) |

¹ Not tested yet on PyTorch.
² A decision model, read differently from the two presets: see [Decision models](#decision-models). It
is not built in; add it once with `jul models add` (below).

You only need the preset you actually use, and only one is ever held in memory:

```bash
jul setup                          # minicpm5-2b
jul setup --model qwen3.5-9b
```

`jul models` shows which ones are already downloaded.

### Adding a model

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
`pip install -e ".[calibrate]"`), or taken from `--data <dir>`. The dev accuracy it reports comes
with its standard error (±3.5 points at n=200): it orients, it does not rank close models. Measure
on the Jev bench separately, once.

### Decision models

A *decision model* is a model trained to answer questions about a state, rather than to write text. It
reads the state and the options through delimiter tokens it learned, and a small head scores each
option against the question. `jul` runs one with no code of its own: everything that model needs sits
next to its weights, in a `decision.json` (delimiters, layout, readout, head file, temperature, and the
longest state and question it was trained on).

```bash
jul models add minicpm5-2b-decision --repo bdauzats/minicpm5-2b-decision-mlx-4bit   # MLX, 1.3 GB
jul models add minicpm5-2b-decision --repo bdauzats/minicpm5-2b-decision --backend torch
jul ask choice "Which team should handle this ticket?" -o billing -o shipping -o access \
    --state "I was charged twice for order 4411" --model minicpm5-2b-decision
```

A repo (or a local directory) holding a `decision.json` is registered as it is: there is nothing to
fit, no layer to choose and no tau, so the command returns at once. The API is the same as for any other model, and all three
question types go through the same format. The state is encoded once per call and every question
continues from it, so questions never see each other.

The state is paid once per call: a ticket with four questions (two `Choice`, a `Noul` and a `Score`)
answers in **180 ms** on an M4 Pro, against 65 ms for the first question alone. What costs is the
options — they are re-read on every request — so a three-option question runs in 65 ms where a
fifty-nine-option one takes 613 ms. Weights: [`bdauzats/minicpm5-2b-decision-mlx-4bit`](https://huggingface.co/bdauzats/minicpm5-2b-decision-mlx-4bit)
(MLX, 1.3 GB) and [`bdauzats/minicpm5-2b-decision`](https://huggingface.co/bdauzats/minicpm5-2b-decision)
(PyTorch, bf16).

Two differences with the presets above: `autotune(...)` does not apply (its heads are trained on the
vectors of the other method, and such a model needs a full fine-tune instead), and a state longer than
the limit in its `decision.json` is truncated rather than stretched.

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

## Two presets, and a decision model

| Preset                          | Model                           | Layers  |    tau | Latency (p50) | Jev bench, zero-shot |
| ------------------------------- | ------------------------------- | ------- | -----: | ------------: | -------------------- |
| `minicpm5-2b` (alias `fast`)    | `openbmb/MiniCPM5-2B-MLX`       | 39 / 40 | 0.0413 |     **64 ms** | 0.617                |
| `qwen3.5-9b` (alias `accurate`) | `mlx-community/Qwen3.5-9B-4bit` | 31 / 31 | 0.0483 |        273 ms | 0.660                |

A third option does not read a general model at all: `minicpm5-2b-decision` is MiniCPM5-2B *trained*
to answer typed questions (a merged LoRA and a pointer head). It has no layer and no tau — it brings
its own format — and it is 6 points ahead on the development sets, at a latency that depends on how
many options a question has. It is not built in: `jul models add` registers it in a second.

## Results

The full published benchmark, 300 examples, run through this library (`scripts/bench_jul.py`, and
`bench_jul_decision.py` for the decision model). **Only the zero-shot block compares to Jev** — the
rows below it receive task data at call time and Jev receives none.

| Zero-shot                      |  AG News | Banking77 |  Emotion |      Mean |     ECE ↓ |       p50 |
| ------------------------------ | -------: | --------: | -------: | --------: | --------: | --------: |
| jul `minicpm5-2b-decision` ¹   | **0.91** |      0.79 | **0.69** | **0.796** |     0.133 |    217 ms |
| Jev (published)                | **0.91** |  **0.87** |     0.48 |     0.753 |     0.156 |    246 ms |
| jul `qwen3.5-9b`               |     0.79 |      0.74 |     0.45 |     0.660 |     0.175 |    273 ms |
| jul `minicpm5-2b`              |     0.80 |      0.59 |     0.46 |     0.617 | **0.113** | **64 ms** |
| GLiNER2.5 (published)          |     0.70 |      0.61 |     0.44 |     0.583 |     0.101 |    128 ms |

Zero-shot means no example of the task at call time: every row here gets the text, the question and the
option list, nothing else. The first two rows are models *trained* to decide, the next two are general
LLMs read without training, and GLiNER is a trained zero-shot tagger.

¹ `minicpm5-2b-decision` learned these three tasks during training, on their training splits (the
benchmark rows come from the test splits). Jev's training data is not published, so whether it saw them
too is unknown. On six sources neither it nor Kev ever trained on, it scores 0.721 against Jev's 0.857:
see [the development sets](#the-decision-model-on-the-development-sets).

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

- **The decision model passes Jev on the mean (0.796 against 0.753)**, and it is the only row here
  trained for this job, like Jev. See footnote ¹ before reading it as a like-for-like win.
- **Read without training, jul beats GLiNER and stays 9 points behind Jev.** Almost all of that gap is
  Banking77 and its 72 fine-grained intents (0.74 vs 0.87); on AG News and Emotion the gap is 2 to 12
  points.
- **jul is better calibrated than Jev** nearly everywhere. On Emotion Jev's ECE is 0.351: more often
  right, but badly overconfident.
- **With 1000 labeled examples, the small model is enough**: MiniCPM reaches 0.757 at 65 ms, Jev's
  level for a quarter of its latency, and within 1.3 points of tuned Qwen which costs 3.4× more.
- 100 rows per dataset, so ±5 points per cell and ±3 on the mean.

Every layer and temperature was fitted on dev datasets the Jev benchmark never uses. Models load on
first use, one at a time (Qwen3.5-9B is about 5.5 GB).

### The decision model, on the development sets

Here are the four development sets
(BTZSC, 200 examples each), both models read through this library (`scripts/dev_decision_jul.py` in the
research repo), on an M4 Pro (24 GB) on mains power:

Cells are accuracy / ECE (lower is better) / p50 latency.

| Development set      |   `minicpm5-2b` (vectors) |      `minicpm5-2b-decision` |
| -------------------- | ------------------------: | --------------------------: |
| FinancialPhraseBank  |   0.705 / 0.129 / 105 ms  | **0.755** / 0.127 / **65 ms** |
| Yahoo Topics         |   0.450 / **0.045** / 203 ms | **0.610** / 0.107 / 140 ms |
| Empathetic           |   0.345 / 0.147 / 208 ms  | **0.395** / **0.134** / 208 ms |
| Massive (59 options) | **0.670** / **0.114** / **63 ms** |   0.665 / 0.241 / 613 ms |
| **Mean**             |             0.542 / 0.109 |           **0.606** / 0.152 |

- **+6 points**, and the ranking holds whichever way the options are written (short names, as above, or
  the full label sentences: 0.542 against 0.613).
- **Latency depends on the options.** The vector method encodes them once and caches them; the decision
  model re-reads all of them on every request. Three short options: 65 ms against 105. Fifty-nine long
  ones: 613 ms against 63.
- **Calibration is the one place it loses.** Its probabilities come from a temperature (1.954) fitted
  once, on data of the sources it was trained on; the vector method's tau was fitted on these very dev
  sets. Mean ECE 0.152 against 0.109, and Massive is the bad one at 0.241 — quantizing to 4 bits moves
  probabilities by up to 0.3, enough to flip a borderline decision. Fit a `Context` calibration on your
  own data if you need the probabilities themselves, not only the answer.
- On the data it was trained on it stands between the two published Kev models (`transfer-v4`, sources
  never trained on: 0.721, against 0.652 for Kev-0.8B and 0.797 for Kev-4B; Jev 0.857).
### The decision model, on the Jev benchmark

Its row sits in the table above, measured on the same 300 rows with the same metrics
(`scripts/bench_jul_decision.py` in the research repo). Read honestly:

- **Emotion is where it wins** (0.69 against 0.48), and Emotion is one of the tasks it trained on.
- **Banking77 stays 8 points behind Jev (0.79 against 0.87) although it trained on that one too.**
  Seventy-two fine-grained intents remain the hard part — the vector method scores 0.59 there.
- AG News is a tie, calibration is better than Jev's (0.133 against 0.156) and latency comparable.
- **AG News 111 ms, Emotion 111 ms, Banking77 431 ms** — the 217 ms above is their mean. `minicpm5-2b`
  answers all three in 64 ms, and the reason is structural: the vector method encodes each option once
  and caches it, so a call only pays for its own text, and it stops the forward at layer 39/40. The
  decision model re-reads the instructions and the whole option list on every call, and runs all 42
  layers, since it reads the last one. Cost therefore scales with the options: four labels cost 111 ms,
  seventy-two cost 431 ms. Caching them would need the format to put the options before the text, which
  means retraining.

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

### Not enough labeled examples? `jul synth`

Writes a synthetic labeled dataset from a few real examples, in the format `autotune` reads. It only
writes the data: tuning stays a separate step.

```bash
jul synth questions.yaml --seeds sample.jsonl --per-option 30 --output synth.jsonl   # writer: qwen3.5-9b
jul autotune tickets --questions questions.yaml --labeled synth.jsonl
```

The writer first describes the source from the seeds (who writes, form, tone), then writes texts option
by option, shown the whole option list and the seeds carrying that option. Seeds are `{state, answers}`
lines; `answers` may be left out, the text then only informs the style. Synthetic texts are cleaner
than real ones: keep real labeled examples aside to check what a head trained on them is worth.

## Command line

```bash
jul ask choice "Which team should handle this ticket?" \
    -o billing:"payments, invoices" -o technical:"bugs, errors" \
    --state "I was charged twice" --model minicpm5-2b

jul run questions.yaml --input tickets.jsonl --output answers.jsonl --context tickets
jul context create tickets --description "Support tickets of an online bank" --examples sample.txt
jul context list | show tickets | delete tickets
jul synth questions.yaml --seeds sample.jsonl --per-option 30 --output synth.jsonl
jul autotune tickets --questions questions.yaml --labeled labeled.jsonl
jul models
jul models add minicpm5-2b-decision --repo bdauzats/minicpm5-2b-decision-mlx-4bit   # a decision model
jul models add my-model --repo org/Some-Instruct-3B                                 # fits a preset
jul setup --model minicpm5-2b-decision    # backend, weights and one timed decision
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

## Next steps

### Images and video, with Qwen

`qwen3.5-9b` is a multimodal checkpoint. Its config declares a vision tower of 27 blocks, an
`image_token_id` and a `video_token_id` — and the weights are already on your disk: **333
`vision_tower.*` tensors**, part of the 11 GB the preset downloads. They are never loaded.
`mlx_lm.load()` instantiates the language model alone (`children()` returns `['language_model']`),
which is exactly why `backbone.py` reaches through the multimodal wrapper to find the text model.

The method should transpose. The state vector and the option vectors meet in the same residual
stream, at the same position, so the cosine stays defined whether the state arrived as text or as
pixels — the trick CLIP plays across two aligned encoders, except here the model does the fusion
itself and the options stay plain text.

What it would take:

- load through `mlx-vlm` rather than `mlx-lm`, to instantiate the vision tower and the processor;
- **measure everything again**: an image-conditioned hidden state has a different distribution, so the
  center, `tau` and most likely the layer all have to be refitted;
- rewrite the formulations — `This text: "…" means in one word:` is absurd in front of a photograph;
- accept the latency. One image is hundreds of visual tokens on top of 27 tower blocks. Still a
  single pass with nothing generated, but the "four times faster than the hosted option" argument
  would not survive it.

The prefix cache does survive: the prefix stays text and the image takes the state's place.

`minicpm5-2b` is out of this — `model_type: llama`, no vision config, no image preprocessor.

And the honest question is the same one as for text: a small vision model trained on your own images
would probably do as well for a fraction of the cost. ResNet plus a logistic regression has been the
image equivalent of TF-IDF for a decade, and it deserves the same benchmark row before anything else
is built.

### Smaller leads, already measured

- **A generic center for the `question + options` formulation.** Worth about 2.5 points on MiniCPM
  (0.560 against 0.535), but it depends on the question, so it costs 195 extra passes every time a
  new question appears. Left off: a `Context` with fifty examples is cheaper and scores better.
- **Banking77, zero-shot.** This is where the whole gap with the hosted option sits: 0.74 against
  0.87, on 72 fine-grained intents. `autotune(...)` closes most of it; nothing else has.
- **One batch instead of two passes.** The two formulations run one after the other. Batching them
  should cut latency without touching a single accuracy figure.
- **English only.** Every layer, temperature and center here was fitted on English text.

## Layout

```
jul/
  lib/jul/          the library — the CLI never imports anything else
    types.py        questions and answers, same fields as the Jev SDK
    presets.py      the two presets: repo, layers, tau, centers
    engine.py       the vector method: formulations, cached prefixes, combination
    decision.py     the pointer method: a decision model read with its own decision.json
    client.py       TypeSafeClient / AsyncTypeSafeClient
    context.py      Context: description, examples, labeled; disk cache
    synth.py        `jul synth`: synthetic labeled data for autotune
    tuning.py       `autotune(...)`: per-task head, cross-validated, with a safety net
    calibrate.py    `jul models add`: checks, extraction, choice of layers / center / tau
    backbone.py     the backend interface: tap layers, stop early, cached prefix; picks the backend
    backends/       mlx.py (mlx-lm) and torch.py (transformers)
    calibration.py  temperature and per-option bias
    lab/            research code, kept out of the public API
  cli/jul_cli/      the command line
  scripts/          data preparation, tuning experiments, the benchmark
  tests/            fast tests, plus a slow suite that loads the models
```

## Tests

```bash
pytest tests                       # 86 tests, under a second, no model and no data
JUL_SLOW=1 pytest tests            # all 112, downloads and loads both presets (~2 min)
JUL_SLOW=1 pytest tests -m slow    # only the 26 that need a model
JUL_SLOW=1 pytest tests -m torch   # MLX against PyTorch on the same weights
```

**`JUL_SLOW` is a test-only switch**, read by `tests/conftest.py` and by nothing in the library. The
22 tests marked `@pytest.mark.slow` load a real model, so a plain `pytest` skips them rather than
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

Apache License 2.0 — see [LICENSE](LICENSE) and [NOTICE](NOTICE).

The decision-model format and its pointer readout come from [Kev](https://github.com/jaredpalmer/kev)
(Jared Palmer, Apache 2.0), and `minicpm5-2b-decision` is [MiniCPM5-2B](https://huggingface.co/openbmb/MiniCPM5-2B)
(OpenBMB, Apache 2.0) trained with Kev's code on its data plus ours.
