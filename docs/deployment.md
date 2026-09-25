# Deployment: `jul pack` and the onnx backend

When the questions are known in advance, everything that does not depend on the message can be computed
once: the prompts, their prefix caches, the option vectors, the centers, the heads and calibrations of a
context. `jul pack` packs all of it into a directory, a bundle; `Bundle` answers with the message as its
only input, in the format of `system_one`.

Packing trains nothing and changes no model: a bundle holds no weights, it names the model it was
packed on and is loaded on that model, wherever it runs.

```bash
jul pack bundle/ --questions questions.yaml --context tickets --model e5-small --backend onnx
```

```python
from jul import Bundle, pack
pack(client, questions, "bundle/", context="tickets")   # the same, from Python
bundle = Bundle.load("bundle/")
bundle.system_one("I was charged twice").answers["team"].choice
bundle.system_one_batch(messages)          # every prompt read over all messages in batches
```

A prompt shared by several questions (`one_word`) is read once per message. Packing is not tied to a
backend — a bundle names the backend its vectors came from, and the code is the same on all three
(tested on torch and onnx) — but the vectors are: pack on the backend you deploy on (loading on
another one warns).

**The onnx backend** runs a model exported by JuL on ONNX Runtime, without torch or transformers (the
tokenizer is read with `tokenizers` alone). With e5-small in 8 bits, the whole Lambda package, model
included, is 225 MB.

```bash
pip install "jul[onnx-export]"
python -m jul.backends.onnx_export intfloat/multilingual-e5-small models/e5-small-onnx
python -m jul.backends.onnx_export models/e5-small-onnx models/e5-small-onnx-w8 --int8
jul models add e5-small --repo models/e5-small-onnx-w8 --backend onnx
jul autotune tickets --questions questions.yaml --labeled labeled.jsonl --features hybrid \
    --model e5-small --backend onnx
jul pack bundle/ --questions questions.yaml --context tickets --model e5-small --backend onnx
```

- The graph returns the raw hidden states of the upper half of the layers (`--layers` narrows it), and
  for a decoder takes and returns a **KV cache**: each prompt prefix runs once, a message pays for its
  own tokens only (266 → 134 ms per message on Harrier 0.6B, same vectors).
- `--int8`: 8-bit weights (MatMulNBits, blocks of 32, int8 compute), within a cosine of 0.9995 of
  float32 on an M4 and on AWS Graviton alike. Dynamic int8 quantization was tried first and returned
  wrong vectors on Graviton2 (cosine ~0.85). An encoder's embedding table is also quantized, at 4 bits
  (e5-small: 470 → 86 MB).
- The exported attention is not memory-efficient: rows are capped at `JUL_ONNX_MAX_TOKENS` (the end of
  the input is cut, the prompt kept) and batches at `JUL_ONNX_BATCH_TOKENS`. Without those caps, one
  17k-token text asked for ~20 GB.
- No logits and no final norm: the letters reading and decision models need mlx or torch.
- `JUL_ONNX_MODEL=s3://bucket/key` reads the graph from S3 into memory, for a graph too big for a
  package (Harrier 0.6B in 8 bits: 1.1 GB).

Measured on AWS Lambda arm64, eu-west-1 prices, two questions per message (intent among 72, emotion
among 6), by the jul-lambda showcase (CDK, not published yet):

| Model | Reading (formulations) | Head (features) | Passes | Accuracy (emotion / intent) | Lambda | Per message | Per million |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **multilingual-e5-small** | direct: `one_word` for both | hybrid | 1 | 0.787 / 0.890 | 1,769 MB | **17 ms** | **$0.59** |
| Harrier 0.6B, 8-bit | direct: `one_word` for both | hybrid | 1 | 0.791 / 0.890 | 4 GB | 164 ms | $8.95 |
| Harrier 0.6B, 8-bit | mixed: emotion `question_options`, intent `question` | hybrid | 2 | 0.883 / 0.910 | 4 GB, batch 32 | — | $13.76 |

The reading is which prompts a message is read with (`formulations=`); the head is what the tuned head
reads (`features=`). All three bundles use hybrid heads.

e5-small is the one to deploy: same accuracy as Harrier read the same way, ten times faster, fifteen times
cheaper, a 2.4 s cold start with the model in the package. Harrier's extra passes (`mixed`) buy emotion
points at 23 times the price.
