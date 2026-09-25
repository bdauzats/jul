<div align="center">

# JuL — Juste un LLM

**Typed decisions from a local LLM. Same interface as Jev's SDK, any model, zero tokens generated.**

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

JuL is a headless decision runtime. It answers **typed questions** — a choice among options, a yes/no,
a score on a scale — about a piece of text, with calibrated probabilities, on your machine.

It is a drop-in for the [TypeSafe (Jev)](https://jevable.com/) Python SDK: same imports, same calls,
same response shapes. But there is no hosted API and no fixed backbone: point it at any local LLM
(MLX, PyTorch or ONNX) and it becomes a decision head for that model.

JuL is stopped one step before its first syllable and the answer is read straight out of its hidden
states: no monologue, no reasoning trace, no opinion on the matter — nobody asked for one. It has
nothing to say, and it says it in 64 milliseconds.

## Why JuL

- **Competitive with Jev, locally.** On Jev's own benchmark, `wemm-4b` scores 0.877 against 0.753
  zero-shot. Two of its three sets are in MTEB, which embedding models train on; on the clean one,
  AG News, it is 0.95 against 0.91. [Results](#results)
- **Drop-in.** Change one import; your `system_one(...)` calls keep working.
- **Any backbone.** 17 models measured, from 90 MB encoders to 9B embedding models. Add yours with
  `jul models add`.
- **Tunable in seconds.** `autotune(...)` fits a small head on your labels, keeps it only if it beats
  zero-shot in cross-validation, and never touches the LLM.
- **Deployable anywhere.** `jul pack` + the ONNX backend: 17 ms per message and $0.59 per million
  calls on AWS Lambda, no torch.
- **Honest numbers.** The benchmark figures come from scripts in `scripts/`, the rest is labelled
  where it was measured, and the no-LLM baseline is re-measured by CI on every PR.

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

Jev's published benchmark, 300 examples, every row read through the public API
(`scripts/bench_jul.py`). Zero-shot only — the only setting comparable to Jev.

| Model | AG News | Banking77 | Emotion | Mean | ECE ↓ | p50 | Memory |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| JuL `wemm-4b` | 0.95 | **0.88** | 0.80 | **0.877** | 0.112 | 78 ms | 4.5 GB |
| JuL `wemm-9b` | **0.97** | 0.84 | 0.78 | 0.863 | 0.114 | 138 ms | 9.0 GB |
| JuL `wemm-4b-4bit` (default) | 0.90 | 0.87 | 0.80 | 0.857 | 0.084 | 55 ms | 2.6 GB |
| JuL `f2llm-4b` | 0.89 | 0.82 | 0.81 | 0.840 | 0.090 | 46 ms | 3.0 GB |
| JuL `f2llm-1.7b` | 0.91 | 0.67 | **0.87** | 0.817 | **0.082** | **24 ms** | 1.0 GB |
| JuL `minicpm5-2b-decision` ¹ | 0.91 | 0.79 | 0.69 | 0.796 | 0.133 | 217 ms ² | |
| **Jev (published)** | 0.91 | 0.87 | 0.48 | 0.753 | 0.156 | 246 ms ³ | hosted |
| JuL `minicpm5-2b` (`fast`) | 0.80 | 0.59 | 0.46 | 0.617 | 0.113 | 64 ms ² | 2.7 GB |
| GLiNER2.5 (published) | 0.70 | 0.61 | 0.44 | 0.583 | 0.101 | 128 ms | |

Latencies are p50 on an M5 Max, except ² on an M4 Pro (where `wemm-4b-4bit` takes 146 ms); ³ includes
the network. ¹ Trained on these three tasks' training splits (the rows come from the test splits); on
six sources neither ever trained on, Jev leads, 0.857 against 0.721.

100 rows per dataset: ±5 points per cell, ±3 on the mean. Seventeen models were read on these rows;
this table keeps the leaders and the presets. Banking77 and Emotion are in MTEB, which embedding models
train on; **AG News is the clean comparison**. With 1000 labels (not comparable to Jev), `autotune` takes
`wemm-4b-4bit` to 0.897. Full tables, the decision model, calibration, latencies and how to reproduce
everything: [docs/benchmarks.md](https://github.com/usejul/jul/blob/main/docs/benchmarks.md).

### The baseline worth remembering

Before any of this earns its cost, here is what a bag of words does on the same 300 rows, trained on
the same 1000 labeled examples, with no LLM at all (`scripts/bench_tfidf.py`):

| No LLM | AG News | Banking77 | Emotion | Mean | p50 | Training |
|---|---:|---:|---:|---:|---:|---:|
| TF-IDF + linear SVM | 0.88 | 0.76 | 0.43 | **0.690** | **0.17 ms** | 0.1 s on CPU |

It is 6.3 points behind Jev at roughly **1400× lower latency**. It loses clearly on Emotion only,
where recognising a feeling needs meaning rather than vocabulary. If you have labels and your task
looks like topic or intent sorting, try it first: it takes a minute. CI re-measures these accuracies
on every PR ([numbers.yml](https://github.com/usejul/jul/blob/main/.github/workflows/numbers.yml)).

## Models

| Preset | Size | Jev bench, zero-shot | + autotune, 1000 labels | Notes |
| --- | ---: | ---: | ---: | --- |
| `wemm-4b-4bit` (default, alias `accurate`) | 2.6 GB | 0.857 | 0.897 | built in |
| `minicpm5-2b` (alias `fast`) | 2.7 GB | 0.617 | 0.757 | built in, 62 ms on an M4 Pro |
| `minicpm5-2b-decision` | 1.3 GB | 0.796 | — | trained decision model, `jul models add` |
| `e5-small` (ONNX, 8-bit) | 0.09 GB | 0.587 | 0.670 (0.770 hybrid head) | encoder, 21 ms per text on an M4 Pro |

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
