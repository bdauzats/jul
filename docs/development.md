# Development

## Layout

```
jul/
  lib/jul/          the library — the CLI never imports anything else
    types.py        questions and answers, same fields as the Jev SDK
    presets.py      the presets: repos, formulations, layers, tau, centers
    engine.py       the vector method: formulations, cached prefixes, combination
    decision.py     the pointer method: a decision model read with its own decision.json
    client.py       TypeSafeClient / AsyncTypeSafeClient
    context.py      Context: description, examples, labeled; disk cache
    synth.py        `jul synth`: synthetic labeled data for autotune
    tuning.py       `autotune(...)`: per-task head (vector, lexical, hybrid), cross-validated, with a safety net
    lexical.py      TF-IDF in numpy, for the lexical and hybrid heads
    bundle.py       `jul pack`: a fixed need packed into a bundle, and Bundle
    encoder.py      encoders (BERT, XLM-R, e5) as backbones: the micro models
    home.py         JUL_HOME
    calibrate.py    `jul models add`: checks, extraction, choice of layers / center / tau
    backbone.py     the backend interface: tap layers, stop early, cached prefix; picks the backend
    backends/       mlx.py (mlx-lm), torch.py (transformers), onnx.py (ONNX Runtime) and
                    onnx_export.py (the export, KV cache, 8-bit weights)
    calibration.py  temperature and per-option bias
  cli/jul_cli/      the command line
  scripts/          data preparation, tuning experiments, the benchmark
  tests/            fast tests, plus a slow suite that loads the models
```

## Tests

```bash
pytest tests                       # 126 tests, ~30 seconds, no model and no data
JUL_SLOW=1 pytest tests            # all 159, downloads and loads the presets
JUL_SLOW=1 pytest tests -m slow    # only the 33 that need a model
JUL_SLOW=1 pytest tests -m torch   # MLX against PyTorch on the same weights
```

The onnx, encoder and bundle tests build tiny random models on the fly (a 4-layer Qwen3 and a
4-layer XLM-R), export them with JuL and compare onnx with torch on the same weights; only the
tokenizers are downloaded (`JUL_TEST_TOKENIZER`, `JUL_TEST_ENCODER_TOKENIZER`). They need
`jul[onnx-export]` and are skipped without it.

**`JUL_SLOW` is a test-only switch**, read by `tests/conftest.py` and by nothing in the library. The
33 tests marked `@pytest.mark.slow` load a real model, so a plain `pytest` skips them rather than
pulling over 15 GB of weights on someone who just cloned the repo. They are reported as skipped, with the
reason, never silently dropped. Set `JUL_SLOW=1` to run them.

The slow suite includes a per-preset non-regression check against `tests/fixtures/baseline.json`,
which ships with the repo — that one needs the models but no dataset download.
