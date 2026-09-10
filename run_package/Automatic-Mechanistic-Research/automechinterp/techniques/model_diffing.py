"""Model diffing -- what changed between two models (usually base vs a
fine-tune / a later checkpoint): logit-diff amplification, per-layer
activation drift, fine-tuning traces, feature-level diffing (stub).
(learnmechinterp / Model Diffing)

Every function takes `other_model_id: str | None`. When it's None the code
tries a sensible default (for GPT-2: the biomedical fine-tuned stand-in from
`automechinterp.finetune_stand_in`, same relationship BioMistral has to
Mistral); for any other target it raises NotImplementedError naming what a
real diff needs -- a second checkpoint of the *same architecture*.
"""
from __future__ import annotations

import numpy as np
import torch

from ..tools import adapter
from . import _common as C


def _resolve_other(handle, other_model_id: str | None) -> str:
    if other_model_id:
        return other_model_id
    if handle.model_id == "gpt2":
        from ..finetune_stand_in import build_finetuned_stand_in
        return build_finetuned_stand_in("gpt2")   # builds + caches on first call
    raise NotImplementedError(
        f"model diffing needs a second model of the SAME architecture as "
        f"'{handle.model_id}' (a fine-tune or a later checkpoint). Pass "
        f"other_model_id=..., or see automechinterp/stage_c.py and "
        f"automechinterp/finetune_stand_in.py for building a stand-in pair.")


def _both_resids(handle, other_model_id, texts, layers):
    other = adapter.register_model(_resolve_other(handle, other_model_id), device=handle.device)
    if other.n_layers != handle.n_layers:
        raise ValueError(f"architecture mismatch: {handle.model_id} has {handle.n_layers} layers, "
                         f"{other.model_id} has {other.n_layers} -- model diffing needs the same stack")
    a = {l: C.capture_resid(handle, l, texts) for l in layers}
    b = {l: C.capture_resid(other, l, texts) for l in layers}
    return other, a, b


def logit_diff_amplification(handle, texts: list[str], other_model_id: str | None = None,
                              alpha: float = 4.0, top_k: int = 6) -> dict:
    """Anthropic 2024 -- amplify the base->other logit change,
    logits* = logits_other + alpha * (logits_other - logits_base), and read
    which next-token predictions the fine-tune moved most. Surfaces
    behavioural divergence that a single greedy decode would hide."""
    other = adapter.register_model(_resolve_other(handle, other_model_id), device=handle.device)
    rows = []
    for txt in texts[:8]:
        b = handle.tokenizer([txt], return_tensors="pt").to(handle.device)
        with adapter.MODEL_LOCK, torch.no_grad():
            lg_base = handle.model(**b).logits[0, -1].float()
        b2 = other.tokenizer([txt], return_tensors="pt").to(other.device)
        with adapter.MODEL_LOCK, torch.no_grad():
            lg_other = other.model(**b2).logits[0, -1].float()
        amp = lg_other + alpha * (lg_other - lg_base)
        gained = torch.topk(amp - lg_base, top_k).indices.tolist()
        rows.append({"prompt": txt[:50],
                     "base_top": [handle.tokenizer.decode([i]).strip()
                                  for i in lg_base.topk(3).indices.tolist()],
                     "amplified_gainers": [other.tokenizer.decode([i]).strip() for i in gained]})
    return {"other_model": other.model_id, "alpha": alpha, "per_prompt": rows}


def activation_drift(handle, texts: list[str], other_model_id: str | None = None) -> dict:
    """Per-layer mean L2 distance between the two models' last-token residual
    streams on the same inputs, normalised by the base activation norm --
    where in the stack the fine-tune moved representations."""
    layers = list(range(handle.n_layers))
    other, a, b = _both_resids(handle, other_model_id, texts, layers)
    rows = []
    for l in layers:
        drift = float((a[l] - b[l]).norm(dim=1).mean())
        scale = max(float(a[l].norm(dim=1).mean()), 1e-6)
        rows.append((l, round(drift, 3), round(drift / scale, 3)))
    peak = max(rows, key=lambda r: r[2])
    return {"other_model": other.model_id,
            "per_layer_(abs_drift, rel_drift)": [(l, d, r) for l, d, r in rows],
            "peak_relative_drift_layer": peak[0]}


def finetuning_traces(handle, texts: list[str], other_model_id: str | None = None,
                       layer_idx: int | None = None) -> dict:
    """Byun et al. / Prakash et al. -- localise the fine-tune. Which token
    positions and which layer's directions carry the change: top principal
    directions of the (other - base) activation-difference matrix at one
    layer, decoded to vocabulary."""
    layer_idx = layer_idx if layer_idx is not None else 2 * handle.n_layers // 3
    other, a, b = _both_resids(handle, other_model_id, texts, [layer_idx])
    D = (b[layer_idx] - a[layer_idx]).cpu()               # [n_texts, d]
    Dc = D - D.mean(0)
    U, S, Vt = torch.linalg.svd(Dc, full_matrices=False)
    comps = [{"pc": i, "sv": round(float(S[i]), 3),
              "tokens": C.top_tokens(handle, Vt[i].to(handle.device), 6)}
             for i in range(min(3, Vt.shape[0]))]
    return {"other_model": other.model_id, "layer": layer_idx,
            "mean_shift_tokens": C.top_tokens(handle, D.mean(0).to(handle.device), 6),
            "difference_principal_components": comps}


def feature_level_model_diffing(*a, **k) -> dict:
    """Bricken / Lindsey et al. -- match SAE (or crosscoder) features between
    the two models and report features that appeared, vanished, or changed
    meaning. NOT IMPLEMENTED: needs a trained dictionary for each model (or a
    joint crosscoder) -- see superposition.crosscoder. activation_drift +
    finetuning_traces are the dictionary-free stand-ins here."""
    raise NotImplementedError(feature_level_model_diffing.__doc__)


ALL = {
    "logit_diff_amplification": logit_diff_amplification,
    "activation_drift": activation_drift,
    "finetuning_traces": finetuning_traces,
    "feature_level_model_diffing": feature_level_model_diffing,
}
