"""Shared fixtures. Tests that load an MLX model are marked `slow` and skipped by default.

    pytest tests                 # fast tests only (the default)
    JUL_SLOW=1 pytest tests      # everything, loads MiniCPM5-2B and WeMM-Embedding-4B
    JUL_SLOW=1 pytest tests -m slow   # only the ones that need a model
"""

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "lib"))
sys.path.insert(0, str(ROOT / "cli"))

SLOW = os.environ.get("JUL_SLOW") == "1"


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: loads a model; run with JUL_SLOW=1")


def pytest_collection_modifyitems(config, items):
    if SLOW:
        return
    skip = pytest.mark.skip(reason="needs a model; set JUL_SLOW=1")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip)


@pytest.fixture(scope="session")
def client():
    from jul import TypeSafeClient
    c = TypeSafeClient(model="minicpm5-2b")
    yield c
    c.close()


@pytest.fixture
def home(tmp_path):
    """An isolated context store, so tests never touch ~/.jul."""
    return tmp_path / "contexts"


# --- a tiny random Qwen3, exported to onnx by jul: for the onnx and compiled tests ---------------

@pytest.fixture(scope="session")
def tiny_models(tmp_path_factory):
    """Directories of the same tiny random weights: ("hf" for torch, "onnx" exported by jul).
    Only the tokenizer is downloaded (`JUL_TEST_TOKENIZER`, default Qwen/Qwen3-0.6B)."""
    import torch
    from transformers import AutoTokenizer, Qwen3Config, Qwen3ForCausalLM

    from jul.backends.onnx_export import export
    root = tmp_path_factory.mktemp("tiny")
    tokenizer = AutoTokenizer.from_pretrained(os.environ.get("JUL_TEST_TOKENIZER", "Qwen/Qwen3-0.6B"))
    config = Qwen3Config(vocab_size=len(tokenizer), hidden_size=64, intermediate_size=128,
                         num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2,
                         head_dim=16, max_position_embeddings=1024)
    torch.manual_seed(0)
    Qwen3ForCausalLM(config).save_pretrained(root / "hf")
    tokenizer.save_pretrained(root / "hf")
    export(str(root / "hf"), root / "onnx")
    os.environ.setdefault("JUL_DEVICE", "cpu")
    return root / "hf", root / "onnx"


@pytest.fixture(scope="module")
def pair(tiny_models):
    """(torch, onnx) backbones on the tiny weights."""
    from jul.backbone import Backbone
    hf, onnx = tiny_models
    return Backbone(str(hf), "torch"), Backbone(str(onnx), "onnx")


# --- a tiny random XLM-R encoder (the architecture of multilingual-e5), exported by jul ------------

@pytest.fixture(scope="session")
def tiny_encoder(tmp_path_factory):
    """Directories of the same tiny random encoder: ("hf" for torch, "onnx" exported by jul). Only the
    tokenizer is downloaded (`JUL_TEST_ENCODER_TOKENIZER`, default intfloat/multilingual-e5-small)."""
    import torch
    from transformers import AutoTokenizer, XLMRobertaConfig, XLMRobertaModel

    from jul.backends.onnx_export import export
    root = tmp_path_factory.mktemp("tiny-encoder")
    tokenizer = AutoTokenizer.from_pretrained(os.environ.get("JUL_TEST_ENCODER_TOKENIZER",
                                                             "intfloat/multilingual-e5-small"))
    config = XLMRobertaConfig(vocab_size=len(tokenizer), hidden_size=32, intermediate_size=64,
                              num_hidden_layers=4, num_attention_heads=2, max_position_embeddings=66,
                              pad_token_id=tokenizer.pad_token_id)
    torch.manual_seed(0)
    XLMRobertaModel(config).save_pretrained(root / "hf")
    tokenizer.model_max_length = 64
    tokenizer.save_pretrained(root / "hf")
    export(str(root / "hf"), root / "onnx")
    os.environ.setdefault("JUL_DEVICE", "cpu")
    return root / "hf", root / "onnx"
