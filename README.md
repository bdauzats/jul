# JuL — Juste un LLM

**[usejul.github.io/jul](https://usejul.github.io/jul/)** · [PyPI](https://pypi.org/project/jul/)

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
pip install "jul[mlx]"        # Apple Silicon
pip install "jul[torch]"      # Linux / Windows / any GPU
# add [yaml] for YAML question files
```

From a checkout, to work on jul itself: `pip install -e ".[dev]"`.

Or let `jul setup` do the rest: it picks the backend (MLX on Apple Silicon, else PyTorch), installs
it if missing, downloads WeMM-Embedding-4B (2.6 GB, 4-bit) once, and checks one real decision. Running it again redoes only
the check.

```bash
pip install jul
jul setup                          # --backend torch, --model minicpm5-2b, --no-install, --no-check
```

Or have a big LLM install the small one. Paste this into Claude Code, Codex, or any agent that has
a shell:

```text
Install jul (https://pypi.org/project/jul/), a local library that answers typed questions
with a 4B model, and check that it works on this machine.

1. Create a virtualenv with Python >= 3.10.
2. In that venv: pip install jul   then   jul setup
   jul setup picks the backend (MLX on Apple Silicon, else PyTorch), installs it, downloads
   WeMM-Embedding-4B (2.6 GB, 4-bit) and runs one test decision.
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

Both backends read many texts through one template in batches: compiling a question's options, a
context's center, `autotune` and `jul models add`. A group holds at most `JUL_BATCH_TOKENS` tokens
(rows × longest prompt, 16384) and `JUL_BATCH_SIZE` rows (64). A single call (`ask`) is read alone.

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
| `wemm-4b-4bit` (default) | [`usejul/WeMM-Embedding-4B-mlx-4bit`](https://huggingface.co/usejul/WeMM-Embedding-4B-mlx-4bit) | 2.6 GB | [`tencent/WeMM-Embedding-4B`](https://huggingface.co/tencent/WeMM-Embedding-4B) |
| `minicpm5-2b` | [`openbmb/MiniCPM5-2B-MLX`](https://huggingface.co/openbmb/MiniCPM5-2B-MLX)             |  2.7 GB | [`openbmb/MiniCPM5-2B`](https://huggingface.co/openbmb/MiniCPM5-2B) |
| `minicpm5-2b-decision` ¹ | [`usejul/minicpm5-2b-decision-mlx-4bit`](https://huggingface.co/usejul/minicpm5-2b-decision-mlx-4bit) | 1.3 GB | [`usejul/minicpm5-2b-decision`](https://huggingface.co/usejul/minicpm5-2b-decision) |

¹ A decision model, read differently from the presets: see [Decision models](#decision-models). It
is not built in; add it once with `jul models add` (below).

You only need the preset you actually use, and only one is ever held in memory:

```bash
jul setup                          # wemm-4b-4bit
jul setup --model minicpm5-2b
```

`jul models` shows which ones are already downloaded.

### Every model measured

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
`pip install "jul[calibrate]"`), or taken from `--data <dir>`. The dev accuracy it reports comes
with its standard error (±3.5 points at n=200): it orients, it does not rank close models. Measure
on the Jev bench separately, once.

### Decision models

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

## Use it as a drop-in for Jev

Change the import; nothing else.

```python
# from typesafe_sdk import TypeSafeClient, Choice, Noul, Score
from jul import TypeSafeClient, Choice, Noul, Score

client = TypeSafeClient()                             # wemm-4b-4bit, or model="minicpm5-2b"

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

### Showcases

[**jul-showcases**](https://github.com/guyon-it-consulting/jul-showcases), by
[Jérôme Guyon](https://github.com/JeromeGuyon): seven small demos, each one an idea from
[jevable.com](https://jevable.com/) running locally through JuL. They include a form that branches
itself, re-ranking by intent, notification triage, prompt-difficulty routing, 50,000 real support
tickets triaged in 668 s at $0, `autotune` taking a fast model from 82.0% to 96.5%, and an on-device
browser agent that books a train on SNCF Connect.

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
| `backend=` | client, or `JUL_BACKEND` | auto | `mlx` or `torch` |
| `JUL_BATCH_TOKENS` / `JUL_BATCH_SIZE` | environment | per backend | how many rows the backbone batches at once |

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

## Two presets, and a decision model

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

## Results

Jev's published benchmark, 300 examples, every row read through this library
(`scripts/bench_jul.py`). **Only the zero-shot table compares to Jev**: the second one receives task
data at call time, and Jev receives none.

| Zero-shot                      |  AG News | Banking77 |  Emotion |      Mean |     ECE ↓ |       p50 |  Memory |
| ------------------------------ | -------: | --------: | -------: | --------: | --------: | --------: | ------: |
| jul `wemm-4b`                  |     0.95 |  **0.88** |     0.80 | **0.877** |     0.112 |     78 ms |  4.5 GB |
| jul `wemm-9b`                  | **0.97** |      0.84 |     0.78 |     0.863 |     0.114 |    138 ms |  9.0 GB |
| jul `wemm-4b-4bit`             |     0.90 |      0.87 |     0.80 |     0.857 |     0.084 |     55 ms |  2.6 GB |
| jul `f2llm-4b`                 |     0.89 |      0.82 |     0.81 |     0.840 |     0.090 |     46 ms |  3.0 GB |
| jul `f2llm-1.7b`               |     0.91 |      0.67 | **0.87** |     0.817 | **0.082** | **24 ms** |  1.0 GB |
| jul `minicpm5-2b-decision` ¹   |     0.91 |      0.79 |     0.69 |     0.796 |     0.133 |    217 ms |         |
| **Jev (published)**            |     0.91 |      0.87 |     0.48 |     0.753 |     0.156 |    246 ms |  hosted |
| jul `minicpm5-2b`              |     0.80 |      0.59 |     0.46 |     0.617 |     0.113 |     64 ms |  2.7 GB |
| GLiNER2.5 (published)          |     0.70 |      0.61 |     0.44 |     0.583 |     0.101 |    128 ms |         |

- **Nine `jul` models beat Jev zero-shot, the best by 12.4 points** (`wemm-4b`, 0.877 against
  0.753). `wemm-4b-4bit` is still 10 points ahead in 2.6 GB at 55 ms, and `f2llm-1.7b` 6 points ahead
  in 1 GB at 24 ms, ten times faster than Jev.
- **Better calibrated too**: the five leaders sit between 0.082 and 0.114 of ECE, Jev at 0.156.
- **AG News is the clean comparison.** Banking77 and Emotion are in MTEB, which embedding models train
  on, so `wemm-*` and `f2llm-*` have probably seen them. AG News is not, and there `wemm-9b` scores
  0.97 against Jev's 0.91.

| With task data (not comparable to Jev)        |  AG News | Banking77 |  Emotion |      Mean |     ECE ↓ |       p50 |
| --------------------------------------------- | -------: | --------: | -------: | --------: | --------: | --------: |
| `wemm-4b-4bit` + autotune (1000 labeled)      |     0.94 |  **0.94** |     0.81 | **0.897** | **0.067** |     59 ms |
| `f2llm-1.7b` + autotune (1000 labeled)        |     0.89 |      0.72 | **0.89** |     0.833 |           |     24 ms |
| `harrier-0.6b` + autotune (1000 labeled)      |     0.91 |      0.71 |     0.83 |     0.814 |           | **13 ms** |
| `minicpm5-2b` + autotune (1000 labeled)       |     0.92 |      0.76 |     0.59 |     0.757 |     0.102 |     38 ms |
| `minicpm5-2b` + context (50 unlabeled)        |     0.84 |      0.62 |     0.47 |     0.643 |     0.140 |     44 ms |

With a thousand labels, `wemm-4b-4bit` reaches **0.897** and 0.94 on Banking77, and `harrier-0.6b`
passes Jev in 0.31 GB at 13 ms.

¹ Trained on these three tasks' training splits (the benchmark rows come from the test splits); Jev's
training data is not published. On six sources neither ever trained on, Jev leads, 0.857 against
0.721: see [the development sets](#the-decision-model-on-the-development-sets). Measured on v1.0.

On anything that reads two things together (a paraphrase, a policy against a case), an embedding
model falls behind the decision model: see [Embedding models](#embedding-models).

Seventeen models were read on these same 300 rows; this table keeps the leaders and the presets. 100 rows per dataset means ±5 points per cell and ±3 on the mean. Every layer
and temperature was fitted on development sets the benchmark never uses. Latencies are p50: `wemm-*`, `f2llm-*` and the tuned rows on an M5 Max, `minicpm5-2b`
and the decision model on an M4 Pro, where `wemm-4b-4bit` takes 146 ms. Jev's includes
the network.

### The baseline worth remembering

Before any of this earns its cost, here is what a bag of words does on the same 300 rows, trained on
the same 1000 labeled examples, with no LLM at all (`scripts/bench_tfidf.py`):

| No LLM | AG News | Banking77 | Emotion | Mean | p50 | Training |
|---|---:|---:|---:|---:|---:|---:|
| TF-IDF + linear SVM | 0.88 | 0.76 | 0.43 | **0.690** | **0.17 ms** | 0.1 s on CPU |

It is 6.3 points behind Jev at roughly **1400× lower latency**. It loses clearly on Emotion only,
where recognising a feeling needs meaning rather than vocabulary. If you have labels and your task
looks like topic or intent sorting, try it first: it takes a minute.

### The decision model, on the development sets

Here are the four development sets
(BTZSC, 200 examples each), both models read through this library (`scripts/dev_decision_jul.py` in the
research repo), on an M4 Pro (24 GB) on mains power:

Cells are accuracy / ECE (lower is better) / p50 latency.

| Development set      |   `minicpm5-2b` (vectors) |      `minicpm5-2b-decision` |
| -------------------- | ------------------------: | --------------------------: |
| FinancialPhraseBank  |   0.705 / 0.129 / 105 ms  | **0.740** / 0.188 / **64 ms** |
| Yahoo Topics         |   0.450 / **0.045** / 203 ms | **0.565** / 0.087 / 136 ms |
| Empathetic           |   0.345 / 0.147 / 208 ms  | **0.460** / 0.214 / 234 ms |
| Massive (59 options) |   0.670 / 0.114 / **63 ms** | **0.715** / **0.069** / 596 ms |
| **Mean**             |             0.542 / 0.109 |           **0.620** / 0.140 |

- **+8 points on the mean**, and it wins on all four sets.
- **Latency depends on the options.** The vector method encodes them once and caches them; the decision
  model re-reads all of them on every request. Three short options: 64 ms against 105. Fifty-nine long
  ones: 596 ms against 63. Above a measured threshold `jul` routes a question to the vector reading
  instead; `jul models add` finds that threshold and says what it costs.
- **Calibration is no longer where it loses.** v1.1 ships a temperature of 1.289 fitted on one epoch's
  weights, and mean ECE is 0.140 against the vector method's 0.109 — Massive, the bad one at 0.241 in
  v1.0, is now the good one at 0.069. Quantizing to 4 bits still moves probabilities by up to 0.23,
  enough to flip a borderline decision, so fit a `Context` calibration on your own data if you need the
  probabilities themselves and not only the answer.
- **It reads French.** On MASSIVE's parallel French and English splits, 0.710 against 0.815 — v1.0 scored
  0.485 in French, below `jul`'s untrained vector method. That reversal is what v1.1 was trained for.
- On sources it was never trained on (`transfer-v4`) it scores 0.739, against 0.652 for Kev-0.8B and
  0.797 for Kev-4B; Jev 0.857.

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

### Embedding models

WeMM-Embedding (Tencent), F2LLM-v2 (CodeFuse-AI), Qwen3-Embedding (Alibaba) and Harrier-OSS
(Microsoft) are ordinary decoder-only LLMs, fine-tuned by contrastive learning to put texts with the
same meaning close together. `jul` reads them like any other model, with no code of its own, and
`wemm-4b-4bit` is the default preset. For the others,
`jul models add` finds their layers, center and tau on the dev sets. They are the top of the table
above.

```bash
jul models add wemm-4b      --repo tencent/WeMM-Embedding-4B               # 4.5 GB, the zero-shot best
jul models add f2llm-4b     --repo fcmeyer/F2LLM-v2-4B-mlx-6bit            # 3.0 GB, 46 ms
jul models add harrier-0.6b --repo majentik/harrier-oss-v1-0.6b-MLX-4bit   # 0.31 GB, 13 ms
```

The measurements below come from an earlier run of WeMM-Embedding-4B as a plain embedding (pooled at
its `<embedding>` token, by its own script), on the development sets and on Kev's decision questions.

**Who it is for: anyone sorting one text into labels described in words, with no labeled data.**
Ticket routing, topics, intents, sentiment, emotions. On the development sets it beats everything
else here, the trained decision model included, and French costs it nothing:

| Development set (label sentences) | `minicpm5-2b` (vectors) | `minicpm5-2b-decision` | WeMM-Embedding-4B |
| --- | ---: | ---: | ---: |
| FinancialPhraseBank | 0.705 | 0.740 | **0.800** |
| Yahoo Topics | 0.450 | 0.565 | **0.655** |
| Empathetic | 0.345 | 0.460 | **0.510** |
| Massive (59 options) | 0.670 | 0.715 | **0.780** |
| **Mean** (ECE) | 0.542 (0.109) | 0.620 (0.140) | **0.686** (0.099) |
| MASSIVE English / French, option names | 0.640 / 0.535 | **0.815 / 0.710** ³ | 0.595 / 0.590 |

³ Trained on MASSIVE, English and French.

**What it is not for: any question that needs the text and something else read together.** Is this
sentence a paraphrase of that one, does this case satisfy the policy, is the report late. An embedding
never sees the option while it reads the text, so it cannot compare them. On Kev's `transfer-v4`
(656 decision questions from sources never trained on) it scores **0.643 against 0.739** for
`minicpm5-2b-decision`, and falls *below the majority class* on paraphrase (0.45), two of three policy
compositions and deadlines. Use the decision model for those.

One temperature, 0.0219 fitted on the development sets, calibrates it across tasks (fitted on three
sets, scored on the fourth, it stays within 0.021–0.023). Latency follows the text length, not the
option count: options are embedded once, so 52 ms for a short utterance, about 240 ms for a paragraph.

## Context — what the data looks like

Jev has no equivalent. A context acts at three levels.

```python
from jul import TypeSafeClient, Context

tickets = Context(
    description="Customer support tickets of an online bank, written in English by customers.",
    examples=open("sample_tickets.txt").read().splitlines(),   # ~50–200 real texts, no labels
)

client = TypeSafeClient(context=tickets)                       # for every call
client.system_one(state, questions, context=tickets)           # or per call
```

| Element       | What it does                                     | Cost                            | Status                                                                                                        |
| ------------- | ------------------------------------------------ | ------------------------------- | ------------------------------------------------------------------------------------------------------------- |
| `description` | prepended as `Context: …` to both formulations   | nil (sits in the cached prefix) | **off by default, measured harmful on average** — `use_description=True` to try it                            |
| `examples`    | their mean vector becomes the center of the task | computed once                   | measured: helps on topics (AG News 0.66 → 0.75), slightly hurts on fine-grained tasks (Banking77 0.55 → 0.53) |

A task center is the best center measured (JOURNAL §9 octies), which is the main
reason to bother with a context at all. **Ten examples already capture most of the gain, fifty is the
sweet spot, two hundred adds nothing** (JOURNAL §9 decies). Below ten the center is noise and can be
worse than no context at all — with 5 examples, a ticket rated `billing` at 0.98 flipped to a wrong
`technical`. `Context` warns under ten.

The gain is uneven: topics gain the most, while financial sentences lost 3 points. A task center helps most when your texts have a style of their own.

**The `description` is off by default.** Measured over four datasets, it failed the plan's bar:
MiniCPM gained 1.3 points on average with one dataset losing 4, and a 9B model lost on all four
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

MiniCPM needs 500 to 1000 examples to plateau. A model with more separable vectors learns from
fewer: on the Jev bench, `wemm-4b-4bit` goes from 0.857 to 0.897 with 1000.

A head only knows the options it saw, and only the preset whose vectors it saw: change either and you
call `autotune(...)` again.

### Not enough labeled examples? `jul synth`

Writes a synthetic labeled dataset from a few real examples, in the format `autotune` reads. It only
writes the data: tuning stays a separate step.

```bash
jul synth questions.yaml --seeds sample.jsonl --per-option 30 --output synth.jsonl --writer <mlx-lm repo>
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
jul models add minicpm5-2b-decision --repo usejul/minicpm5-2b-decision-mlx-4bit   # a decision model
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
  Which centre matters less, and differently per model: on MiniCPM the generic centre gains 4 points
  over the option mean. **The best centre is the task's own**, i.e.
  `Context(examples=…)`, for both models — 200 examples per strategy though, so ±3.5 points.
- Everything was tuned in English.

## Next steps

### Routing by question type

Embedding models lead on sorting one text into labels; the decision model leads on questions that read
two things together. The natural shape is to route by question type: a `Choice` over a single text to
`wemm-4b-4bit`, the rest to `minicpm5-2b-decision`. `Noul` and `Score` still have to be measured on the
embedding models — only `Choice` is so far.

### Smaller leads, already measured

- **A generic center for the `question + options` formulation.** Worth about 2.5 points on MiniCPM
  (0.560 against 0.535), but it depends on the question, so it costs 195 extra passes every time a
  new question appears. Left off: a `Context` with fifty examples is cheaper and scores better.
- **Banking77 on MiniCPM.** Read zero-shot, `minicpm5-2b` stays at 0.59 against Jev's 0.87 on 72
  fine-grained intents. `wemm-4b` closes it (0.88), and `autotune(...)` closes most of it.
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
pytest tests                       # 94 tests, a few seconds, no model and no data
JUL_SLOW=1 pytest tests            # all 127, downloads and loads the presets
JUL_SLOW=1 pytest tests -m slow    # only the 33 that need a model
JUL_SLOW=1 pytest tests -m torch   # MLX against PyTorch on the same weights
```

**`JUL_SLOW` is a test-only switch**, read by `tests/conftest.py` and by nothing in the library. The
33 tests marked `@pytest.mark.slow` load a real model, so a plain `pytest` skips them rather than
pulling over 15 GB of weights on someone who just cloned the repo. They are reported as skipped, with the
reason, never silently dropped. Set `JUL_SLOW=1` to run them.

The slow suite includes a per-preset non-regression check against `tests/fixtures/baseline.json`,
which ships with the repo — that one needs the models but no dataset download.

## Reproducing the measurements

Every number in this README comes from a script in `scripts/`. None of the data is committed; these
steps fetch it. Durations are given per command where measured.

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
python scripts/dev_context_effect.py minicpm5-2b 100   # does a Context help?  (~7 min)
python scripts/dev_tuning_curve.py minicpm5-2b         # how many labels?      (~5 min)
```

`dev_generic_center.py` rewrites the shipped centers in `lib/jul/assets/`, and `dev_tuning_curve.py`
caches encoded vectors in `features/` so it can be re-run for free.

Finally the benchmark itself. It checks the published manifest's SHA-256 and verifies that no
training row appears in it, then runs the three variants **through the public API**:

```bash
python scripts/bench_jul.py minicpm5-2b                # (~7 min)
python scripts/bench_jul.py wemm-4b-4bit
python scripts/bench_tfidf.py                          # the no-LLM baseline (~5 s)
```

Reports land in `runs/jev-bench-jul/`. Tune on the dev datasets, and run the benchmark once, at the
end — that is the whole point of keeping the two apart.

## License

Apache License 2.0 — see [LICENSE](LICENSE) and [NOTICE](NOTICE).

The decision-model format and its pointer readout come from [Kev](https://github.com/jaredpalmer/kev)
(Jared Palmer, Apache 2.0), and `minicpm5-2b-decision` is [MiniCPM5-2B](https://huggingface.co/openbmb/MiniCPM5-2B)
(OpenBMB, Apache 2.0) trained with Kev's code on its data plus ours.
