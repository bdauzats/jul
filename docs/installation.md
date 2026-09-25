# Installation

Pick a backend: MLX on Apple Silicon, PyTorch (transformers) anywhere else — CUDA, CPU, or MPS.

```bash
pip install "jul[mlx]"        # Apple Silicon
pip install "jul[torch]"      # Linux / Windows / any GPU
pip install "jul[onnx]"       # CPU only, no torch: to deploy an exported model (AWS Lambda, containers)
# add [yaml] for YAML question files, [tune] for lexical and hybrid autotune heads,
# [onnx-export] to export a model for the onnx backend
```

From a checkout, to work on JuL itself: `pip install -e ".[dev]"`.

## `jul setup`

`jul setup` does the rest: it picks the backend (MLX on Apple Silicon, else PyTorch), installs
it if missing, downloads WeMM-Embedding-4B (2.6 GB, 4-bit) once, and checks one real decision. Running it again redoes only
the check.

```bash
pip install jul
jul setup                          # --backend torch, --model minicpm5-2b, --no-install, --no-check
```

## Let an agent install it

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

## Backend, device and batching

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

A third backend, **onnx** (ONNX Runtime on CPU), is never picked by default: it reads a model
exported for it, and exists to deploy JuL where torch does not fit (see [Deploying a fixed
need](deployment.md)). `JUL_HOME` moves everything JuL
writes (presets, contexts, calibration data) from `~/.jul` elsewhere, e.g. a read-only Lambda package.

## The models

No weights are committed here. `jul setup` downloads them from the Hugging Face Hub into
`~/.cache/huggingface`, once. The CLI asks for it when they are missing; the Python API downloads them
on the first call.

| Preset        | MLX repository                                                                          | On disk | PyTorch repository                                                  |
| ------------- | --------------------------------------------------------------------------------------- | ------: | ------------------------------------------------------------------- |
| `wemm-4b-4bit` (default) | [`usejul/WeMM-Embedding-4B-mlx-4bit`](https://huggingface.co/usejul/WeMM-Embedding-4B-mlx-4bit) | 2.6 GB | [`tencent/WeMM-Embedding-4B`](https://huggingface.co/tencent/WeMM-Embedding-4B) |
| `minicpm5-2b` | [`openbmb/MiniCPM5-2B-MLX`](https://huggingface.co/openbmb/MiniCPM5-2B-MLX)             |  2.7 GB | [`openbmb/MiniCPM5-2B`](https://huggingface.co/openbmb/MiniCPM5-2B) |
| `minicpm5-2b-decision` ¹ | [`usejul/minicpm5-2b-decision-mlx-4bit`](https://huggingface.co/usejul/minicpm5-2b-decision-mlx-4bit) | 1.3 GB | [`usejul/minicpm5-2b-decision`](https://huggingface.co/usejul/minicpm5-2b-decision) |

¹ A decision model, read differently from the presets: see [Decision models](models.md#decision-models). It
is not built in; add it once with `jul models add` (below).

You only need the preset you actually use, and only one is ever held in memory:

```bash
jul setup                          # wemm-4b-4bit
jul setup --model minicpm5-2b
```

`jul models` shows which ones are already downloaded.
