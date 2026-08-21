"""Tensor-to-text summarization (Sec. 3.6): tools never hand raw tensors back
to an agent's context. Every wrapper in tier_*.py returns a compact digest
through these helpers; full arrays stay in the run's artifact store.
"""
from __future__ import annotations

import numpy as np


def topk_digest(names: list[str], scores: np.ndarray, k: int = 5, label: str = "component") -> str:
    order = np.argsort(-np.abs(scores))[:k]
    lines = [f"{names[i]}: {scores[i]:+.3f}" for i in order]
    return f"top-{min(k, len(order))} {label} by |effect|: " + "; ".join(lines)


def attention_digest(head_name: str, mass_to_target: float, target_desc: str) -> str:
    return f"{head_name}: {mass_to_target * 100:.0f}% attention mass -> {target_desc}"


def scan_digest(name: str, xs: list, ys: list[float], flag_threshold: float) -> str:
    flagged = [x for x, y in zip(xs, ys) if y >= flag_threshold]
    peak_i = int(np.argmax(ys)) if ys else -1
    peak = f"{xs[peak_i]}={ys[peak_i]:.3f}" if peak_i >= 0 else "n/a"
    return f"{name}: peak {peak}; {len(flagged)}/{len(xs)} points >= {flag_threshold} -> {flagged}"


def metric_delta_digest(label: str, baseline: float, after: float) -> str:
    return f"{label}: baseline={baseline:.4f} after={after:.4f} delta={after - baseline:+.4f}"
