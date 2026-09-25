"""Lexical features (jul/lexical.py) and the "lexical" / "hybrid" autotune heads (jul/tuning.py)."""

import importlib.util

import numpy as np
import pytest

from jul import tuning
from jul.context import Context
from jul.lexical import Tfidf

HAS_SKLEARN = importlib.util.find_spec("sklearn") is not None
needs_sklearn = pytest.mark.skipif(not HAS_SKLEARN, reason="needs scikit-learn (jul[tune])")
TEXTS = ["I was charged TWICE for my subscription!", "the app crashes on export",
         "Je voudrais annuler ma commande, merci", "ok", "a b c", "  spaces   and\ttabs  "]
WORDS = {0: ["refund", "charged", "invoice"], 1: ["crash", "error", "bug"], 2: ["price", "plan", "upgrade"]}
FILLER = ["please", "today", "my", "account", "again", "hello", "thanks", "quickly"]


def labeled(n: int, seed: int = 0) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Texts whose label is in their words; vectors that know nothing about it."""
    rng = np.random.default_rng(seed)
    y = np.arange(n) % 3
    texts = [" ".join(rng.choice(FILLER, 4).tolist() + [rng.choice(WORDS[int(k)])]) for k in y]
    return texts, y, rng.normal(size=(n, 16))


@needs_sklearn
def test_the_tfidf_is_scikit_learns():
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.pipeline import make_union
    reference = make_union(TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True),
                           TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True)).fit(TEXTS)
    ours = Tfidf.fit(TEXTS)
    assert np.allclose(ours.transform(TEXTS).scipy().toarray(), reference.transform(TEXTS).toarray())


def test_the_tfidf_round_trips_through_arrays_and_ignores_unknown_terms():
    model = Tfidf.fit(TEXTS)
    again = Tfidf.from_arrays(model.to_arrays())
    rows, rows_again = model.transform(TEXTS + ["zzz qqq"]), again.transform(TEXTS + ["zzz qqq"])
    assert np.array_equal(rows.indices, rows_again.indices) and np.allclose(rows.data, rows_again.data)
    assert rows.indptr[-1] == rows.indptr[-2]          # nothing known in "zzz qqq"
    weights = np.random.default_rng(0).normal(size=(model.n_features, 3))
    assert np.allclose(rows.dot(weights)[:-1], rows.scipy()[:-1] @ weights)


@needs_sklearn
@pytest.mark.parametrize("mode", ["lexical", "hybrid"])
def test_a_lexical_head_reads_the_words_the_vectors_miss(mode):
    texts, y, vectors = labeled(90)
    zero_shot = np.random.default_rng(1).normal(size=(90, 3))
    head, report = tuning.train(vectors, y, zero_shot, ["a", "b", "c"], "q", "preset", texts=texts, mode=mode)
    assert report.activated and report.head_accuracy > 0.9 and report.features == mode
    new_texts, new_y, new_vectors = labeled(30, seed=7)
    probabilities = tuning.apply(head, new_vectors, new_texts)
    assert probabilities.shape == (30, 3) and np.allclose(probabilities.sum(1), 1)
    assert (probabilities.argmax(1) == new_y).mean() > 0.9
    single = tuning.apply(head, new_vectors[0], new_texts[0])
    assert np.allclose(single, probabilities[0])


def test_the_vector_head_is_unchanged_and_needs_no_text():
    texts, y, vectors = labeled(90)
    vectors[np.arange(90), y] += 4.0                    # now the vectors do know
    zero_shot = np.random.default_rng(1).normal(size=(90, 3))
    head, report = tuning.train(vectors, y, zero_shot, ["a", "b", "c"], "q", "preset")
    assert report.features == "vector" and report.activated
    assert tuning.apply(head, vectors).argmax(1).tolist() == tuning.apply(head, vectors, texts).argmax(1).tolist()


@needs_sklearn
def test_a_hybrid_head_saves_and_loads_with_its_context(tmp_path):
    texts, y, vectors = labeled(90)
    zero_shot = np.random.default_rng(1).normal(size=(90, 3))
    head, _ = tuning.train(vectors, y, zero_shot, ["a", "b", "c"], "q", "preset", texts=texts, mode="hybrid")
    before = tuning.apply(head, vectors[:5], texts[:5])     # caches the parsed vocabulary in the payload
    ctx = Context(name="t")
    ctx.heads["digest"] = head
    ctx.save(home=tmp_path)
    loaded = Context.load("t", home=tmp_path).heads["digest"]
    assert loaded["meta"]["features"] == "hybrid"
    assert np.allclose(tuning.apply(loaded, vectors[:5], texts[:5]), before)


def test_an_unknown_mode_is_refused():
    texts, y, vectors = labeled(30)
    with pytest.raises(ValueError, match="features must be one of"):
        tuning.train(vectors, y, np.zeros((30, 3)), ["a", "b", "c"], "q", "p", texts=texts, mode="tfidf")
