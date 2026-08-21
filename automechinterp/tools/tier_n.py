"""Tier N -- Network-Level tools (Network Analyst, Sec. 3.4).

Stage-0 whole-model profiling: everything here is architecture-agnostic
(built only on tools/adapter.py primitives), so these four functions run
unchanged whether `handle` wraps GPT-2 or a 7B Qwen2 model.
"""
from __future__ import annotations

import numpy as np

from . import adapter
from .digest import scan_digest
from .cache import CACHE


def profile_network(handle: adapter.ModelHandle) -> str:
    p = adapter.profile_network(handle)
    return (f"model={p['model_id']} layers={p['n_layers']} hidden={p['hidden_size']} "
            f"heads={p['n_heads']} params={p['n_params']:,} "
            f"layer_stack={p['layer_stack_path']}")


def layerwise_cka_scan(handle: adapter.ModelHandle, probe_texts: list[str]):
    """CKA(l, l+1) across depth -- sharp drops mark representation shifts.

    Returns (digest_str, boundaries, drop_scores): the digest is what an
    agent sees in its evidence log; the two lists are structured data the
    calling agent uses to *decide* which layers to investigate next (Sec.
    3.6: "Full arrays persist ... retrievable on demand" -- here, retained
    in-process rather than serialized, same principle)."""
    def compute():
        acts = adapter.capture_activations(handle, list(range(handle.n_layers)), probe_texts)
        return [adapter.layer_similarity(acts[i], acts[i + 1]) for i in range(handle.n_layers - 1)]

    scores = CACHE.get_or_compute("layerwise_cka_scan",
                                   {"model": handle.model_id, "texts": probe_texts}, compute)
    boundaries = list(range(handle.n_layers - 1))
    drop_scores = [1.0 - s for s in scores]  # "interesting" = low CKA (representation shifted a lot)
    digest = scan_digest("layerwise_cka_scan (1-CKA per boundary)", boundaries, drop_scores, 0.15)
    return digest, boundaries, drop_scores


def redundancy_scan(handle: adapter.ModelHandle, layer_indices: list[int], run_fn_factory):
    """Behavioral impact of dropping each layer individually (mode='skip').
    Returns (digest_str, layer_indices, abs_deltas)."""
    xs, ys = [], []
    for idx in layer_indices:
        run_fn = run_fn_factory()
        result = adapter.ablate_layer(handle, idx, "skip", run_fn)
        xs.append(idx)
        ys.append(abs(result["delta"]))
    digest = scan_digest("redundancy_scan (|delta| from skip-ablating each layer)", xs, ys, 0.05)
    return digest, xs, ys


def depth_probe_sweep(handle: adapter.ModelHandle, concept_texts: dict[str, list[str]],
                       layer_indices: list[int]):
    """For a binary concept (two labeled text lists), the layer at which it
    first becomes linearly decodable. Returns (digest_str, layer_indices, accs)."""
    labels_pos, labels_neg = list(concept_texts.values())
    texts = labels_pos + labels_neg
    y = np.array([1] * len(labels_pos) + [0] * len(labels_neg))
    accs = []
    for idx in layer_indices:
        acts = adapter.capture_activations(handle, [idx], texts)[idx]
        r = adapter.freeze_and_probe(acts, y)
        accs.append(r["probe_accuracy"])
    digest = scan_digest("depth_probe_sweep (linear-probe accuracy per layer)", layer_indices, accs, 0.85)
    return digest, layer_indices, accs
