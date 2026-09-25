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

JuL answers **typed questions** about a piece of text — a choice among options, a yes/no, a score on
a scale — with calibrated probabilities, on your machine.

It learns no task. It reads the hidden-state vector of a general-purpose model and compares it with
the *descriptions* of your options. Nothing about your task is frozen into weights: change the
options, and the next call answers the new question. Change the model, and your code stays the same.

JuL is stopped one step before its first syllable and the answer is read straight out of its hidden
states: no monologue, no reasoning trace, no opinion on the matter — nobody asked for one. It has
nothing to say, and it says it in 55 milliseconds.

And when it is off-key, there is always `client.autotune(...)` — or `jul autotune` from the shell.

## Why JuL

- **Zero-shot, no task training.** 0.857 on Jev's public benchmark (300 examples) with the default
  model, and JuL is given no example of the tasks. Two of the three sets are in MTEB, which embedding
  models train on; the clean one, AG News, gives 0.95 with `wemm-4b`. [Results](#results)
- **Many options, no collapse.** On Banking77, 72 fine-grained intents, the default model scores
  0.87 zero-shot (MTEB caveat above). Options are encoded once, so their number costs almost nothing
  at call time.
- **Probabilities you can put a threshold on.** ECE 0.084 on the same benchmark: a confidence is close
  to how often it turns out right, so you can automate above a threshold and escalate below it.
- **Local and open.** Apache 2.0, 2.6 GB, 55 ms per decision on a Mac; `f2llm-1.7b` fits in 1 GB at
  24 ms. Nothing to pay per call, and the text never leaves the machine.
- **The model is a part, not the product.** 17 backbones measured, from 90 MB encoders to 9B
  embedding models. When a better embedding model comes out, `jul models add` plugs it in the same
  day; your application code does not change.
- **Off-key? Autotune.** `autotune(...)` fits a small head on a few labels, keeps it only if it beats
  zero-shot in cross-validation, and never touches the model.
- **Drop-in for the Jev SDK.** Same imports, same calls, same response shapes: change one import.

## Install

```bash
pip install jul
jul setup          # picks MLX or PyTorch, installs it, downloads the default model, runs one decision
```

Yes: a model that writes essays, installing one that answers in a single word. Neither of them minds.

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

`AsyncTypeSafeClient` has the same API, awaitable. Arguments that only mean something remotely
(`api_key`, `retry`, …) are accepted and ignored. **The option description is what the model compares
against**, so options must describe themselves; the key is only the identifier you get back.

From the shell:

```bash
jul ask choice "Which team should handle this ticket?" \
    -o billing:"payments, invoices" -o technical:"bugs, errors" \
    --state "I was charged twice"
```

## How it works

For each prompt formulation, the state and every option go through the same template; the answer is
the option whose hidden-state vector is closest (cosine, after subtracting a center), turned into
probabilities by `softmax(cosine / tau)`. Everything that does not depend on the input — prompt
prefix, option vectors, center — is computed once and cached, so a call only pays for its own tokens.
Layers, temperatures and centers are fitted per model on development sets the benchmark never sees.

Other readings (option-letter logits, trained decision models with a pointer head, tuned heads) are
described in [docs/models.md](https://github.com/usejul/jul/blob/main/docs/models.md).

## Results

Jev's published benchmark: 300 examples over three tasks, every row read through the public API
(`scripts/bench_jul.py`), zero-shot — no example of these tasks was used.

| Model | AG News | Banking77 (72 options) | Emotion | Mean | ECE ↓ | p50 | Memory |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `wemm-4b` | 0.95 | **0.88** | 0.80 | **0.877** | 0.112 | 78 ms | 4.5 GB |
| `wemm-9b` | **0.97** | 0.84 | 0.78 | 0.863 | 0.114 | 138 ms | 9.0 GB |
| `wemm-4b-4bit` (default) | 0.90 | 0.87 | 0.80 | 0.857 | 0.084 | 55 ms | 2.6 GB |
| `f2llm-4b` | 0.89 | 0.82 | 0.81 | 0.840 | 0.090 | 46 ms | 3.0 GB |
| `f2llm-1.7b` | 0.91 | 0.67 | **0.87** | 0.817 | **0.082** | **24 ms** | 1.0 GB |
| `minicpm5-2b` (`fast`) | 0.80 | 0.59 | 0.46 | 0.617 | 0.113 | 64 ms ¹ | 2.7 GB |
| *Jev, published, for reference* | *0.91* | *0.87* | *0.48* | *0.753* | *0.156* | *246 ms* ² | *hosted* |

**How to read it.** 100 rows per task: ±5 points per cell, ±3 on the mean. Banking77 and Emotion are
in MTEB, which embedding models (`wemm-*`, `f2llm-*`) train on, so they have probably seen these
texts; **AG News is the clean comparison** (0.95 for `wemm-4b`, 0.91 for Jev). p50 on an M5 Max,
¹ on an M4 Pro; ² includes the network. Seventeen models were read on these rows; the full table,
the tuned rows, the decision model and how to reproduce every number are in
[docs/benchmarks.md](https://github.com/usejul/jul/blob/main/docs/benchmarks.md).

**Not measured yet.** Domains far from this benchmark's text (sensor logs, chemistry…) have not been
tested. Every layer, temperature and center was fitted on English.

### The baseline worth remembering

Before any of this earns its cost, here is what a bag of words does on the same 300 rows, trained on
the same 1000 labeled examples, with no LLM at all (`scripts/bench_tfidf.py`):

| No LLM | AG News | Banking77 | Emotion | Mean | p50 | Training |
|---|---:|---:|---:|---:|---:|---:|
| TF-IDF + linear SVM | 0.88 | 0.76 | 0.43 | **0.690** | **0.17 ms** | 0.1 s on CPU |

It is 6.3 points behind Jev at roughly **1400× lower latency**. It loses clearly on Emotion only,
where recognising a feeling needs meaning rather than vocabulary. If you have labels and your task
looks like topic or intent sorting, try it first: it takes a minute. CI re-measures these accuracies
on every PR ([numbers.yml](https://github.com/usejul/jul/blob/main/.github/workflows/numbers.yml));
change `CLAIMED` there if you change them here.

## Models

| Preset | Size | Jev bench, zero-shot | + autotune, 1000 labels | Notes |
| --- | ---: | ---: | ---: | --- |
| `wemm-4b-4bit` (default, alias `accurate`) | 2.6 GB | 0.857 | 0.897 | built in |
| `minicpm5-2b` (alias `fast`) | 2.7 GB | 0.617 | 0.757 | built in, 62 ms on an M4 Pro |
| `minicpm5-2b-decision` | 1.3 GB | see [benchmarks](https://github.com/usejul/jul/blob/main/docs/benchmarks.md) | — | trained decision model, `jul models add` |
| `e5-small` (ONNX, 8-bit) | 0.09 GB | 0.543 | 0.713 (0.790 hybrid head) | encoder, 6 ms per text on an M4 Pro |

Any other model is one command away — `jul models add <name> --repo <hf-repo>` fits its layers,
center and temperature on the dev sets. All 17 measured models, encoders, decision models and every
setting: [docs/models.md](https://github.com/usejul/jul/blob/main/docs/models.md).

## Documentation

| Guide | What's inside |
| --- | --- |
| [Installation](https://github.com/usejul/jul/blob/main/docs/installation.md) | backends (MLX, PyTorch, ONNX), devices, batching, `jul setup`, model downloads |
| [Models](https://github.com/usejul/jul/blob/main/docs/models.md) | every model measured, adding a model, encoders, decision models, readings and settings |
| [Adapting to your data](https://github.com/usejul/jul/blob/main/docs/tuning.md) | `Context`, `autotune(...)`, hybrid heads, formulations, `jul synth` |
| [Deployment](https://github.com/usejul/jul/blob/main/docs/deployment.md) | `jul pack`, bundles, the ONNX backend, AWS Lambda numbers |
| [Command line](https://github.com/usejul/jul/blob/main/docs/cli.md) | every command and file format (questions, labeled data, I/O) |
| [Benchmarks](https://github.com/usejul/jul/blob/main/docs/benchmarks.md) | full results, what is and isn't measured, reproducing every number |
| [Development](https://github.com/usejul/jul/blob/main/docs/development.md) | repository layout, test suites |
| [Publishing](https://github.com/usejul/jul/blob/main/docs/publishing.md) | versioning and the release pipeline |

<details>
<summary>Looking for a section of the old README? It moved to <code>docs/</code>.</summary>

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

[**jul-showcases**](https://github.com/guyon-it-consulting/jul-showcases), by
[Jérôme Guyon](https://github.com/JeromeGuyon): seven small demos, each one an idea from
[jevable.com](https://jevable.com/) running locally through JuL — a form that branches itself,
re-ranking by intent, notification triage, prompt-difficulty routing, 50,000 real support tickets
triaged in 668 s at $0, `autotune` taking a fast model from 82.0% to 96.5%, and an on-device browser
agent that books a train on SNCF Connect.

## Contributing

Issues and pull requests are welcome on [GitHub](https://github.com/usejul/jul/issues).

```bash
git clone https://github.com/usejul/jul && cd jul
pip install -e ".[dev]"
pytest tests                  # fast suite, no model, ~30 s
JUL_SLOW=1 pytest tests       # full suite, downloads the presets
```

Tune on the dev datasets, measure on the benchmark once: see
[reproducing the measurements](https://github.com/usejul/jul/blob/main/docs/benchmarks.md#reproducing-the-measurements).

## License

Apache License 2.0 — see [LICENSE](https://github.com/usejul/jul/blob/main/LICENSE) and [NOTICE](https://github.com/usejul/jul/blob/main/NOTICE).

The decision-model format and its pointer readout come from [Kev](https://github.com/jaredpalmer/kev)
(Jared Palmer, Apache 2.0), and `minicpm5-2b-decision` is
[MiniCPM5-2B](https://huggingface.co/openbmb/MiniCPM5-2B) (OpenBMB, Apache 2.0) trained with Kev's code
on its data plus ours. JuL follows TypeSafe's public System One API; no TypeSafe or Jev code, weights
or outputs are used.
