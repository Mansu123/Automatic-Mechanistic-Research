"""MI for AI safety -- refusal direction, deception/sleeper-agent scans (stubs
w/ requirements), simple activation-anomaly monitor.
(learnmechinterp / MI for AI Safety)
"""
from __future__ import annotations

import torch

from ..tools import adapter
from . import _common as C


def refusal_direction(handle, harmful_prompts: list[str], harmless_prompts: list[str],
                       test_prompt: str, layer_idx: int | None = None) -> dict:
    """Arditi et al. 2024 -- a single diff-of-means direction (harmful vs
    harmless, mid-layer) mediates refusal. Build it, then (a) project it out
    and (b) add it, reading the next-token shift. Only meaningful on an
    instruction-tuned model; runs on any model but the readout may be flat."""
    layer_idx = layer_idx if layer_idx is not None else handle.n_layers // 2
    d = C.diff_of_means_direction(handle, layer_idx, harmful_prompts, harmless_prompts)
    b = handle.tokenizer([test_prompt], return_tensors="pt").to(handle.device)

    def top(hook):
        with adapter.MODEL_LOCK:
            hs = [handle.layers[layer_idx].register_forward_hook(hook)] if hook else []
            try:
                with torch.no_grad():
                    lg = handle.model(**b).logits[0, -1]
            finally:
                for h in hs:
                    h.remove()
        t = torch.topk(lg, 5)
        return [(handle.tokenizer.decode([i]).strip(), round(float(v), 2))
                for i, v in zip(t.indices.tolist(), t.values.tolist())]
    return {"layer": layer_idx, "direction_top_tokens": C.top_tokens(handle, d, 6),
            "next_token_baseline": top(None),
            "next_token_direction_removed": top(C.project_out_hook(d)),
            "next_token_direction_added": top(C.add_direction_hook(d, 6.0)),
            "note": "chat model needed for a clean refusal readout"}


def activation_anomaly_monitor(handle, reference_texts: list[str], test_texts: list[str],
                                layer_idx: int | None = None) -> dict:
    """Runtime-style monitor: fit mean/precision of reference activations at a
    layer, score test inputs by Mahalanobis distance -- a cheap OOD / anomaly
    flag of the kind used for deployed safety monitoring."""
    layer_idx = layer_idx if layer_idx is not None else 2 * handle.n_layers // 3
    R = C.capture_resid(handle, layer_idx, reference_texts).double()
    mu = R.mean(0)
    # Woodbury identity: invert the small sample Gram matrix, not a d_model^2 matrix.
    centered = R - mu
    lam = 1e-2
    denom = max(1, R.shape[0] - 1)
    gram = centered @ centered.T + lam * denom * torch.eye(
        R.shape[0], device=R.device, dtype=R.dtype)
    def distances(X):
        x = X - mu
        projected = x @ centered.T
        correction = (projected * torch.linalg.solve(gram, projected.T).T).sum(-1)
        return ((x.square().sum(-1) - correction) / lam).clamp_min(0).sqrt()
    T = C.capture_resid(handle, layer_idx, test_texts).double()
    dist = distances(T)
    ref_d = distances(R)
    thr = float(ref_d.mean() + 3 * ref_d.std())
    return {"layer": layer_idx, "ref_mahalanobis_mean": round(float(ref_d.mean()), 2),
            "anomaly_threshold(3sigma)": round(thr, 2),
            "test_scores": [(t[:40], round(float(dd), 2), bool(dd > thr))
                            for t, dd in zip(test_texts, dist)]}


def evaluation_awareness_probe(handle, eval_like_texts: list[str], deploy_like_texts: list[str],
                                layer_idx: int | None = None) -> dict:
    """Needham et al. / Anthropic evals -- does the model linearly represent
    'this looks like an evaluation' vs 'this looks like real deployment'? Fit
    a diff-of-means direction between the two context styles, report linear
    separability and the direction's decoded tokens."""
    layer_idx = layer_idx if layer_idx is not None else 2 * handle.n_layers // 3
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    E = C.capture_resid(handle, layer_idx, eval_like_texts).cpu().numpy()
    D = C.capture_resid(handle, layer_idx, deploy_like_texts).cpu().numpy()
    import numpy as np
    X = np.vstack([E, D]); y = np.array([1] * len(E) + [0] * len(D))
    acc = float(cross_val_score(LogisticRegression(max_iter=1000), X, y,
                                cv=min(5, len(y) // 2 or 2)).mean())
    d = C.diff_of_means_direction(handle, layer_idx, eval_like_texts, deploy_like_texts)
    return {"layer": layer_idx, "eval_vs_deploy_separability": round(acc, 3),
            "eval_direction_top_tokens": C.top_tokens(handle, d, 6),
            "note": "acc >> 0.5 => the model has a linear 'being evaluated' feature"}


def deception_detection(*a, **k) -> dict:
    """MacDiarmid et al. / Goldowsky-Dill et al. -- NOT IMPLEMENTED. Needs a
    labelled dataset of matched honest vs deceptive/roleplay completions from
    the target model to build the probe. activation_anomaly_monitor +
    linear_probe are the primitives to build it once such data exists."""
    raise NotImplementedError(deception_detection.__doc__)


def sleeper_agent_scan(*a, **k) -> dict:
    """Hubinger et al. / Price et al. -- NOT IMPLEMENTED. Detecting a latent
    backdoor requires either the trigger distribution or a residual-stream
    probe trained on known-triggered vs clean activations; neither is
    available for an arbitrary off-the-shelf model."""
    raise NotImplementedError(sleeper_agent_scan.__doc__)


ALL = {
    "refusal_direction": refusal_direction,
    "activation_anomaly_monitor": activation_anomaly_monitor,
    "evaluation_awareness_probe": evaluation_awareness_probe,
    "deception_detection": deception_detection,
    "sleeper_agent_scan": sleeper_agent_scan,
}
