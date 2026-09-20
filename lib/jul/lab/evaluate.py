"""Train and compare the decision heads on cached features.

Variants:
  A        letter logits of the options prompt (zero-shot, no training)
  B        linear probe on the best single layer (plain prompt / options prompt)
  B+C      hybrid head: query x label embeddings in a shared space (plain prompt / options prompt)

Every variant gets its temperature fitted on the validation split; metrics are reported on test.
"""

from __future__ import annotations

import json
from pathlib import Path

import mlx.core as mx
import numpy as np

from . import features
from ..calibration import fit_temperature, metrics, softmax
from .heads import Normalizer, TrainConfig, train_hybrid, train_probe


def _save(run_dir: Path, head, norm: Normalizer, config: dict):
    run_dir.mkdir(parents=True, exist_ok=True)
    head.save_weights(str(run_dir / "head.safetensors"))
    np.savez(run_dir / "norm.npz", mean=np.array(norm.mean), std=np.array(norm.std))
    (run_dir / "config.json").write_text(json.dumps(config, indent=2))


def run(feat_dir: Path, runs_dir: Path, holdout: str | None = None, cfg: TrainConfig = TrainConfig(),
        tied: bool = True, verbose: bool = True) -> dict:
    meta, splits, label_feats = features.load(feat_dir)
    names, layers = meta["labels"], meta["layers"]
    K = len(names)
    tr, va, te = splits["train"], splits["val"], splits["test"]

    # Holdout: drop one label from train/val entirely; test still covers all labels.
    keep = list(range(K))
    if holdout:
        h = names.index(holdout)
        keep = [i for i in keep if i != h]
    remap = {old: new for new, old in enumerate(keep)}

    def train_view(split):
        m = np.isin(split["y"], keep)
        return m, np.array([remap[v] for v in split["y"][m]], np.int64)

    m_tr, y_tr = train_view(tr)
    m_va, y_va = train_view(va)
    y_te = te["y"]

    def expand(logits_sub):
        """Map logits over training labels back to all K labels (unseen ones get -inf)."""
        full = np.full((len(logits_sub), K), -1e4, np.float32)
        full[:, keep] = logits_sub
        return full

    results, layer_table, test_probs = {}, {}, {}

    def record(name, val_logits, y_val, test_logits, extra=None):
        T = fit_temperature(val_logits, y_val)
        r = {"T": T, "raw": metrics(test_logits, y_te), "cal": metrics(test_logits / T, y_te)}
        if holdout:
            pred = (test_logits / T).argmax(-1)
            r["holdout_recall"] = float((pred[y_te == names.index(holdout)] == names.index(holdout)).mean())
        r.update(extra or {})
        results[name] = r
        test_probs[name] = softmax(test_logits / T)
        return T

    # --- A: letter logits (restricted to training labels on val, all labels on test) ---
    if meta["has_options"]:
        va_logits = va["letter_logits"][m_va][:, keep]
        record("A  logits", va_logits, y_va, te["letter_logits"],
               {"letter_mass": float(te["letter_mass"].mean())})

    prompts = [("plain", "q_plain")] + ([("options", "q_opts")] if meta["has_options"] else [])
    for prompt, key in prompts:
        norm = Normalizer.fit(tr[key][m_tr])
        x_tr, x_va, x_te = norm(tr[key][m_tr]), norm(va[key][m_va]), norm(te[key])

        # --- B: one linear probe per tapped layer, keep the best on validation ---
        best = None
        for j, layer in enumerate(layers):
            probe = train_probe(x_tr[:, j], y_tr, x_va[:, j], y_va, len(keep), cfg)
            acc = float((np.array(probe(x_va[:, j])).argmax(-1) == y_va).mean())
            layer_table.setdefault(prompt, {})[layer] = acc
            if best is None or acc > best[0]:
                best = (acc, j, layer, probe)
        _, j, layer, probe = best
        T = record(f"B  probe/{prompt} L{layer}", np.array(probe(x_va[:, j])), y_va,
                   expand(np.array(probe(x_te[:, j]))), {"layer": layer})
        _save(runs_dir / f"probe-{prompt}", probe, Normalizer(np.array(norm.mean[j]), np.array(norm.std[j])),
              {"kind": "probe", "prompt": prompt, "layers": [layer], "T": T, "labels": [names[i] for i in keep],
               "model": meta["model"]})

        # --- B+C: hybrid head over all tapped layers ---
        lf = norm(label_feats)
        head = train_hybrid(x_tr, y_tr, x_va, y_va, lf[mx.array(keep)], cfg, tied=tied)
        mix = np.array(mx.softmax(head.q_mix))
        T = record(f"B+C hybrid/{prompt}", np.array(head(x_va, lf[mx.array(keep)])), y_va,
                   np.array(head(x_te, lf)), {"layer_mix": dict(zip(map(str, layers), mix.round(3).tolist()))})
        _save(runs_dir / f"hybrid-{prompt}", head, norm,
              {"kind": "hybrid", "prompt": prompt, "layers": layers, "T": T, "tied": tied,
               "labels": names, "trained_labels": [names[i] for i in keep], "model": meta["model"],
               "n_layers": len(layers), "d": int(x_tr.shape[-1])})

    report = {"model": meta["model"], "holdout": holdout, "n": {s: meta["splits"][s]["n"] for s in meta["splits"]},
              "layer_probe_val_acc": layer_table, "results": results}
    runs_dir.mkdir(parents=True, exist_ok=True)
    (runs_dir / "report.json").write_text(json.dumps(report, indent=2))
    np.savez(runs_dir / "test_probs.npz", y=y_te, **test_probs)  # calibrated test probabilities per variant
    if verbose:
        print_report(report)
    return report


def print_report(report: dict):
    print(f"\n=== {report['model']}" + (f"  (holdout: {report['holdout']})" if report["holdout"] else "") + " ===")
    for prompt, table in report["layer_probe_val_acc"].items():
        print(f"  probe val acc per layer [{prompt}]: " + "  ".join(f"L{l}={a:.3f}" for l, a in table.items()))
    hdr = f"  {'variant':<26}{'acc':>7}{'conf':>7}{'nll':>8}{'ECE raw→cal':>15}{'T':>7}{'cover@.9':>10}{'acc@.9':>8}"
    if report["holdout"]:
        hdr += f"{'unseen rec':>12}"
    print(hdr)
    for name, r in report["results"].items():
        c, raw = r["cal"], r["raw"]
        line = (f"  {name:<26}{c['acc']:>7.3f}{c['conf']:>7.3f}{c['nll']:>8.3f}{raw['ece']:>8.3f}→{c['ece']:.3f}{r['T']:>7.2f}"
                f"{c['cover@0.9']:>10.2f}{c['acc@0.9']:>8.3f}")
        if report["holdout"]:
            line += f"{r['holdout_recall']:>12.3f}"
        print(line)
