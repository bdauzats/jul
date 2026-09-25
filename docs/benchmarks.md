# Benchmarks

Jev's published benchmark, 300 examples, every row read through this library
(`scripts/bench_jul.py`). **Only the zero-shot table compares to Jev**: the second one receives task
data at call time, and Jev receives none.

| Zero-shot                      |  AG News | Banking77 |  Emotion |      Mean |     ECE ↓ |       p50 |  Memory |
| ------------------------------ | -------: | --------: | -------: | --------: | --------: | --------: | ------: |
| JuL `wemm-4b`                  |     0.95 |  **0.88** |     0.80 | **0.877** |     0.112 |     78 ms |  4.5 GB |
| JuL `wemm-9b`                  | **0.97** |      0.84 |     0.78 |     0.863 |     0.114 |    138 ms |  9.0 GB |
| JuL `wemm-4b-4bit`             |     0.90 |      0.87 |     0.80 |     0.857 |     0.084 |     55 ms |  2.6 GB |
| JuL `f2llm-4b`                 |     0.89 |      0.82 |     0.81 |     0.840 |     0.090 |     46 ms |  3.0 GB |
| JuL `f2llm-1.7b`               |     0.91 |      0.67 | **0.87** |     0.817 | **0.082** | **24 ms** |  1.0 GB |
| JuL `minicpm5-2b-decision` ¹   |     0.91 |      0.79 |     0.69 |     0.796 |     0.133 |    217 ms |         |
| **Jev (published)**            |     0.91 |      0.87 |     0.48 |     0.753 |     0.156 |    246 ms |  hosted |
| JuL `minicpm5-2b`              |     0.80 |      0.59 |     0.46 |     0.617 |     0.113 |     64 ms |  2.7 GB |
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

## The baseline worth remembering

Before any of this earns its cost, here is what a bag of words does on the same 300 rows, trained on
the same 1000 labeled examples, with no LLM at all (`scripts/bench_tfidf.py`):

| No LLM | AG News | Banking77 | Emotion | Mean | p50 | Training |
|---|---:|---:|---:|---:|---:|---:|
| TF-IDF + linear SVM | 0.88 | 0.76 | 0.43 | **0.690** | **0.17 ms** | 0.1 s on CPU |

It is 6.3 points behind Jev at roughly **1400× lower latency**. It loses clearly on Emotion only,
where recognising a feeling needs meaning rather than vocabulary. If you have labels and your task
looks like topic or intent sorting, try it first: it takes a minute.

## The decision model, on the development sets

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

## The decision model, on the Jev benchmark

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

## Embedding models

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

## What is measured, and what is not

Measured:

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
- **Centering**: measured. Not centering costs 4.5–8.5 points, so always centre.
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

## Reproducing the measurements

Every number in these docs comes from a script in `scripts/`. None of the data is committed; these
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
