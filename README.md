<div align="center">

# JuL — Juste un LLM

**Typed decisions on your machine, with the model of your choice.<br>No training, no API, no task learned by heart.**

[![PyPI](https://img.shields.io/pypi/v/jul?color=blue&label=PyPI)](https://pypi.org/project/jul/)
[![Python](https://img.shields.io/badge/python-%E2%89%A53.10-3776AB?logo=python&logoColor=white)](https://pypi.org/project/jul/)
[![CI](https://img.shields.io/github/actions/workflow/status/usejul/jul/ci.yml?branch=main&label=CI)](https://github.com/usejul/jul/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-Apache--2.0-green)](https://github.com/usejul/jul/blob/main/LICENSE)
<br>
[![Backends](https://img.shields.io/badge/backends-MLX%20%C2%B7%20PyTorch%20%C2%B7%20ONNX-orange)](https://github.com/usejul/jul/blob/main/docs/installation.md)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20models-usejul-yellow)](https://huggingface.co/usejul)
[![Website](https://img.shields.io/badge/site-usejul.github.io%2Fjul-black)](https://usejul.github.io/jul/)

[Website](https://usejul.github.io/jul/) · [Quickstart](#quickstart) · [Results](#results) · [Docs](#documentation) · [Showcases](#showcases)

</div>

---

JuL answers typed questions about a piece of text: pick one option, say yes or no, give a score on a
scale. Every answer comes with a probability, and everything runs on your machine.

It doesn't learn your task. It takes a general-purpose model, reads the vector the model builds for
your text, and compares it with the vectors of your option descriptions. Your task never ends up in
the weights, so you can change the options and the next call answers the new question. You can also
swap the model underneath without touching your code.

JuL is stopped one step before its first syllable and the answer is read straight out of its hidden
states: no monologue, no reasoning trace, no opinion on the matter — nobody asked for one. It has
nothing to say, and it says it in 55 milliseconds.

And when it is off-key on your data, there is `client.autotune(...)`, or `jul autotune` from the shell.

## Why JuL

- It works zero-shot. The default model scores 0.857 on Jev's public benchmark (300 examples)
  without seeing a single example of the three tasks. Two of those tasks are in MTEB, which embedding
  models train on, so read the clean one first: AG News, where `wemm-4b` gets 0.95. [Results](#results)
- It holds up with many options. Banking77 has 72 intents and the default model gets 0.87 on it
  (same MTEB caveat). Option vectors are computed once and cached, so adding options barely changes
  the cost of a call.
- Its probabilities mean something. ECE is 0.084 on the same benchmark: on average, the confidence
  it reports is within about 8 points of how often it's actually right. That's what lets you automate
  above a threshold and send the rest to a human.
- It runs locally. Apache 2.0, 2.6 GB, 55 ms per decision on a Mac. `f2llm-1.7b` fits in 1 GB and
  answers in 24 ms. You pay nothing per call and the text stays on your machine.
- The model is replaceable. We've measured 17 so far, from a 90 MB encoder to 9B embedding models.
  When a better embedding model ships, `jul models add` wires it in and your application code stays
  as it is.
- You can tune it when you have labels. `autotune(...)` fits a small head on top of the vectors,
  keeps it only if it beats zero-shot in cross-validation, and leaves the model alone.
- It's a drop-in for the Jev SDK: same imports, same calls, same response shapes. Change the import
  and you're done.
  For callers that aren't Python, `jul serve` speaks the Jev HTTP protocol on your machine:
  [serving over HTTP](https://github.com/usejul/jul/blob/main/docs/serve.md).

## Install

```bash
pip install jul
jul setup          # picks MLX or PyTorch, installs it, downloads the default model, runs one decision
```

Or choose the backend yourself:

```bash
pip install "jul[mlx]"      # Apple Silicon
pip install "jul[torch]"    # Linux / Windows / CUDA / CPU
pip install "jul[onnx]"     # CPU only, no torch — for deploying a packed bundle
```

Python ≥ 3.10. Weights are downloaded once from the Hugging Face Hub (2.6 GB for the default).
Backends, devices, extras and an agent-ready install prompt: [docs/installation.md](https://github.com/usejul/jul/blob/main/docs/installation.md).

## Quickstart

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
response.nouls["is_bug"].noul             # 0.12
response.scores["frustration"].score      # 1.15
```

`AsyncTypeSafeClient` has the same API, awaitable. Arguments that only make sense for a remote API
(`api_key`, `retry`, …) are accepted and ignored.

One thing to know before writing your own questions: the model compares the text with the option
*description*, not with its key. `"billing"` is just the name you get back; `"payments, invoices,
refunds"` is what does the work. Write descriptions a colleague would understand.

From the shell:

```bash
jul ask choice "Which team should handle this ticket?" \
    -o billing:"payments, invoices" -o technical:"bugs, errors" \
    --state "I was charged twice"
```

## How it works

Each option description goes through the same prompt template as the text, and the answer is the
option whose hidden-state vector is closest to the text's (cosine similarity, after subtracting a
center). A softmax over `cosine / tau` turns those similarities into probabilities. The prompt prefix,
the option vectors and the center don't depend on the input, so they're computed once and cached; a
call only pays for its own tokens.

Layers, temperatures and centers were fitted for each model on development sets that the benchmark
never touches. Other ways of reading an answer (option-letter logits, trained decision models, tuned
heads) are in [docs/models.md](https://github.com/usejul/jul/blob/main/docs/models.md).

## Results

This is Jev's published benchmark: 300 examples over three tasks. Every row was produced through the
public API with `scripts/bench_jul.py`, zero-shot, without any example of these tasks.

| Model | AG News | Banking77 (72 options) | Emotion | Mean | ECE ↓ | p50 | Memory |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `wemm-4b` | 0.95 | **0.88** | 0.80 | **0.877** | 0.112 | 78 ms | 4.5 GB |
| `wemm-9b` | **0.97** | 0.84 | 0.78 | 0.863 | 0.114 | 138 ms | 9.0 GB |
| `wemm-4b-4bit` (default) | 0.90 | 0.87 | 0.80 | 0.857 | 0.084 | 55 ms | 2.6 GB |
| `f2llm-4b` | 0.89 | 0.82 | 0.81 | 0.840 | 0.090 | 46 ms | 3.0 GB |
| `f2llm-1.7b` | 0.91 | 0.67 | **0.87** | 0.817 | **0.082** | **24 ms** | 1.0 GB |
| `minicpm5-2b` (`fast`) | 0.80 | 0.59 | 0.46 | 0.617 | 0.113 | 64 ms ¹ | 2.7 GB |
| *Jev, published, for reference* | *0.91* | *0.87* | *0.48* | *0.753* | *0.156* | *246 ms* ² | *hosted* |

A few things to keep in mind. With 100 rows per task, each cell is ±5 points and the mean ±3.
Banking77 and Emotion are in MTEB, and the embedding models here (`wemm-*`, `f2llm-*`) were trained
on MTEB, so they've probably seen those texts before. AG News is the clean comparison: 0.95 for
`wemm-4b`, 0.91 for Jev. Latencies are p50 on an M5 Max (¹ on an M4 Pro); Jev's (²) includes the
network.

We read seventeen models on these rows. The full table, the tuned results, the decision model and
the commands to reproduce all of it are in [docs/benchmarks.md](https://github.com/usejul/jul/blob/main/docs/benchmarks.md).

We haven't tested domains far from this kind of text yet (sensor logs, chemistry and so on), and
every setting was fitted on English.

### The baseline worth remembering

Before paying for any of this, look at what a bag of words does. Same 300 rows, trained on the same
1000 labeled examples, no LLM at all (`scripts/bench_tfidf.py`):

| No LLM | AG News | Banking77 | Emotion | Mean | p50 | Training |
|---|---:|---:|---:|---:|---:|---:|
| TF-IDF + linear SVM | 0.88 | 0.76 | 0.43 | **0.690** | **0.17 ms** | 0.1 s on CPU |

It lands 6.3 points behind Jev and is about 1400× faster. The only task where it clearly loses is
Emotion, because telling feelings apart takes meaning, not vocabulary. If you have labels and your
problem looks like sorting by topic or intent, try this first. It takes a minute.

CI re-measures these accuracies on every PR ([numbers.yml](https://github.com/usejul/jul/blob/main/.github/workflows/numbers.yml)). The
expected values are also written in that workflow (`CLAIMED`), so update both together.

## Models

| Preset | Size | Jev bench, zero-shot | + autotune, 1000 labels | Notes |
| --- | ---: | ---: | ---: | --- |
| `wemm-4b-4bit` (default, alias `accurate`) | 2.6 GB | 0.857 | 0.897 | built in |
| `minicpm5-2b` (alias `fast`) | 2.7 GB | 0.617 | 0.757 | built in, 62 ms on an M4 Pro |
| `minicpm5-2b-decision` | 1.3 GB | see [benchmarks](https://github.com/usejul/jul/blob/main/docs/benchmarks.md) | — | trained decision model, `jul models add` |
| `e5-small` (ONNX, 8-bit) | 0.09 GB | 0.543 | 0.713 (0.790 hybrid head) | encoder, 6 ms per text on an M4 Pro |

To use another model, run `jul models add <name> --repo <hf-repo>`. It fits the layer, center and
temperature on the dev sets. The 17 models we measured, encoders, decision models and every setting
are listed in [docs/models.md](https://github.com/usejul/jul/blob/main/docs/models.md).

## Documentation

| Guide | What's inside |
| --- | --- |
| [Installation](https://github.com/usejul/jul/blob/main/docs/installation.md) | backends (MLX, PyTorch, ONNX), devices, batching, `jul setup`, model downloads |
| [Models](https://github.com/usejul/jul/blob/main/docs/models.md) | every model measured, adding a model, encoders, decision models, readings and settings |
| [Adapting to your data](https://github.com/usejul/jul/blob/main/docs/tuning.md) | `Context`, `autotune(...)`, hybrid heads, formulations, `jul synth` |
| [Deployment](https://github.com/usejul/jul/blob/main/docs/deployment.md) | `jul pack`, bundles, the ONNX backend, AWS Lambda numbers |
| [Serving over HTTP](https://github.com/usejul/jul/blob/main/docs/serve.md) | `jul serve`: a local server speaking the Jev HTTP protocol, for non-Python callers |
| [Command line](https://github.com/usejul/jul/blob/main/docs/cli.md) | every command and file format (questions, labeled data, I/O) |
| [Benchmarks](https://github.com/usejul/jul/blob/main/docs/benchmarks.md) | full results, what is and isn't measured, reproducing every number |
| [Development](https://github.com/usejul/jul/blob/main/docs/development.md) | repository layout, test suites |
| [Publishing](https://github.com/usejul/jul/blob/main/docs/publishing.md) | versioning and the release pipeline |

<details>
<summary>Looking for a section of the old README? Everything moved to <code>docs/</code>.</summary>

- Install, backends, `jul setup` → [installation](https://github.com/usejul/jul/blob/main/docs/installation.md)
- The models, every model measured, adding a model, micro models, decision models, how it answers,
  every reading and setting, the two presets → [models](https://github.com/usejul/jul/blob/main/docs/models.md)
- Use it as a drop-in for Jev → [Quickstart](#quickstart)
- Context, `autotune(...)`, `features=`, `formulations=`, `jul synth` → [tuning](https://github.com/usejul/jul/blob/main/docs/tuning.md)
- `jul pack` and the onnx backend → [deployment](https://github.com/usejul/jul/blob/main/docs/deployment.md)
- Command line, file formats → [cli](https://github.com/usejul/jul/blob/main/docs/cli.md)
- Results, the decision model, embedding models, what is measured, next steps, reproducing the
  measurements → [benchmarks](https://github.com/usejul/jul/blob/main/docs/benchmarks.md)
- Layout, tests → [development](https://github.com/usejul/jul/blob/main/docs/development.md)

</details>

## Showcases

[jul-showcases](https://github.com/guyon-it-consulting/jul-showcases) is a set of seven small demos
by [Jérôme Guyon](https://github.com/JeromeGuyon). Each one takes an idea from
[jevable.com](https://jevable.com/) and runs it locally with JuL. There's a form that branches on its
own answers, re-ranking by intent, notification triage and prompt-difficulty routing. One demo triages
50,000 real support tickets in 668 s for $0, another uses `autotune` to take a fast model from 82.0%
to 96.5%, and the last is an on-device browser agent that books a train on SNCF Connect.

## Contributing

Bug reports, measurements on your own data and pull requests are all welcome on
[GitHub](https://github.com/usejul/jul/issues).

```bash
git clone https://github.com/usejul/jul && cd jul
pip install -e ".[dev]"
pytest tests                  # fast suite, no model, ~30 s
JUL_SLOW=1 pytest tests       # full suite, downloads the presets
```

One rule for experiments: tune on the dev datasets, and run the benchmark once at the end. The steps
are in [reproducing the measurements](https://github.com/usejul/jul/blob/main/docs/benchmarks.md#reproducing-the-measurements).

## License

Apache License 2.0. See [LICENSE](https://github.com/usejul/jul/blob/main/LICENSE) and [NOTICE](https://github.com/usejul/jul/blob/main/NOTICE).

The decision-model format and its pointer readout come from [Kev](https://github.com/jaredpalmer/kev)
(Jared Palmer, Apache 2.0). `minicpm5-2b-decision` is [MiniCPM5-2B](https://huggingface.co/openbmb/MiniCPM5-2B)
(OpenBMB, Apache 2.0), trained with Kev's code on Kev's data plus ours. JuL follows TypeSafe's public
System One API. It uses no TypeSafe or Jev code, weights or outputs.
