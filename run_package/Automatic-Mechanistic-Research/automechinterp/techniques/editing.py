"""Model editing -- LEACE (closed-form linear concept erasure), inference-time
concept ablation, ROME (stub).  (learnmechinterp / Model Editing)
"""
from __future__ import annotations

import numpy as np
import torch

from ..tools import adapter
from . import _common as C


def _erasure_factors(X, y, ridge=1e-3):
    """Thin-SVD factors for the ridge-whitened row-vector erasure operator."""
    X = X.double().cpu(); y = y.cpu(); mu = X.mean(0); Xc = X-mu
    _, singular, Vt = torch.linalg.svd(Xc, full_matrices=False)
    eigen = singular.square()/len(X)+ridge
    delta = Xc[y == 1].mean(0)-Xc[y == 0].mean(0)
    coefficients = delta @ Vt.T / eigen.sqrt()
    coefficients = coefficients / coefficients.norm().clamp_min(1e-12)
    left = Vt.T @ (coefficients/eigen.sqrt())
    right = Vt.T @ (coefficients*eigen.sqrt())
    return mu.float(), left.float(), right.float()


def leace(handle, layer_idx, pos_texts, neg_texts, test_prompt, pos_token, neg_token) -> dict:
    """Ridge-whitened erasure diagnostic, computed without a d_model-cubed eigensolve.

    The training-fit probe is descriptive and explicitly not independent validation.
    """
    if min(len(set(pos_texts)),len(set(neg_texts))) < 2:
        return {"layer":layer_idx,"status":"insufficient_data",
                "skipped":"Need two distinct training examples in each class for erasure"}
    X=torch.cat([C.capture_resid(handle,layer_idx,pos_texts),C.capture_resid(handle,layer_idx,neg_texts)],0).cpu()
    y=torch.tensor([1]*len(pos_texts)+[0]*len(neg_texts))
    mu,left,right=_erasure_factors(X,y)
    mu,left,right=[v.to(handle.device) for v in (mu,left,right)]
    def hook(m,i,o):
        is_t=isinstance(o,tuple);hs=o[0] if is_t else o
        centered=hs.float()-mu
        changed=(hs.float()-(centered@left).unsqueeze(-1)*right).to(hs.dtype)
        return (changed,)+o[1:] if is_t else changed
    from contextlib import contextmanager
    from ..eval.causal_measurements import score_contrast
    @contextmanager
    def intervene():
        with adapter.MODEL_LOCK:
            token=handle.layers[layer_idx].register_forward_hook(hook)
            try: yield
            finally: token.remove()
    before=score_contrast(handle,test_prompt,pos_token,neg_token)
    after=score_contrast(handle,test_prompt,pos_token,neg_token,intervene)
    Xe=X.float()-((X.float()-mu.cpu())@left.cpu()).unsqueeze(-1)*right.cpu()
    from sklearn.linear_model import LogisticRegression
    classifier=LogisticRegression(max_iter=1000).fit(Xe.numpy(),y.numpy())
    return {"layer":layer_idx,"metric":"full_continuation_log_probability_margin",
            "margin_before":before["margin"],"margin_after":after["margin"],
            "training_fit_probe_accuracy_after":float(classifier.score(Xe.numpy(),y.numpy())),
            "note":"Low-rank ridge-whitened erasure; training-fit accuracy is not held-out erasure validation."}


def concept_ablation_edit(handle, layer_idx, pos_texts, neg_texts, test_prompt) -> dict:
    """Cheapest 'edit': zero the concept direction at inference (a 1-D LEACE)."""
    d = C.diff_of_means_direction(handle, layer_idx, pos_texts, neg_texts)
    b = handle.tokenizer([test_prompt], return_tensors="pt").to(handle.device)
    with adapter.MODEL_LOCK:
        h = handle.layers[layer_idx].register_forward_hook(C.project_out_hook(d))
        try:
            with torch.no_grad():
                lg = handle.model(**b).logits[0, -1]
        finally:
            h.remove()
    t = torch.topk(lg, 5)
    return {"layer": layer_idx,
            "top_tokens_after_ablating_direction":
                [(handle.tokenizer.decode([i]).strip(), round(float(v), 2))
                 for i, v in zip(t.indices.tolist(), t.values.tolist())]}


def localized_fact_editing(handle, subject_prompt, old_object: str, new_object: str) -> dict:
    """Meng et al. 2022 (the causal-tracing half of ROME) done with a hook,
    not a weight write. (1) causal-trace which layer's residual state at the
    last subject token most restores the old fact after corruption, (2) at
    inference add (U[new] - U[old]) direction into that layer at the last
    position, (3) report the probability mass moved from old_object to
    new_object. A reversible, model-agnostic stand-in for the rank-one edit."""
    tok = handle.tokenizer
    b = tok([subject_prompt], return_tensors="pt").to(handle.device)
    old_id = tok.encode(" " + old_object.strip())[-1]
    new_id = tok.encode(" " + new_object.strip())[-1]
    # (1) crude localisation: per-layer logit-lens boost for old_object at the last position
    store = {}
    with adapter.MODEL_LOCK:
        hs = [handle.layers[i].register_forward_hook(
            (lambda i: (lambda m, inp, o: store.__setitem__(
                i, (o[0] if isinstance(o, tuple) else o).detach()[:, -1, :])))(i))
            for i in range(handle.n_layers)]
        try:
            with torch.no_grad():
                handle.model(**b)
        finally:
            for h in hs:
                h.remove()
    ln, W = C.final_norm(handle), C.unembed(handle)
    lens = []
    for i in range(handle.n_layers):
        h = store[i]
        h = ln(h) if ln is not None else h
        lens.append(float((h @ W.T)[0][old_id]))
    # the decisive layer is the one that most *increases* the old fact's logit
    # (logit-lens delta), i.e. where the association is written -- not just where
    # it is already high (Meng et al.'s causal-tracing peak, approximated)
    deltas = [(i, lens[i] - (lens[i - 1] if i else lens[0])) for i in range(handle.n_layers)]
    edit_layer = max(deltas, key=lambda t: t[1])[0]
    # (2) inference-time direction edit at that layer
    direction = (W[new_id] - W[old_id]).detach()
    direction = direction / direction.norm().clamp_min(1e-8)
    rms = float(store[edit_layer].norm() / (direction.numel() ** 0.5))

    def hook(m, i, o):
        is_t = isinstance(o, tuple)
        x = (o[0] if is_t else o).clone()
        x[:, -1, :] = x[:, -1, :] + 6.0 * rms * direction.to(x.dtype)
        return (x,) + o[1:] if is_t else x

    def probs():
        with torch.no_grad():
            lg = handle.model(**b).logits[0, -1].softmax(-1)
        return float(lg[old_id]), float(lg[new_id])

    p_old_before, p_new_before = probs()
    with adapter.MODEL_LOCK:
        hh = handle.layers[edit_layer].register_forward_hook(hook)
        try:
            p_old_after, p_new_after = probs()
        finally:
            hh.remove()
    return {"edit_layer": edit_layer,
            "p(old_object)": [round(p_old_before, 4), round(p_old_after, 4)],
            "p(new_object)": [round(p_new_before, 4), round(p_new_after, 4)],
            "note": "hook-based (reversible) stand-in for the ROME rank-one weight edit"}


def machine_unlearning(*a, **k) -> dict:
    """Eldan & Russinovich / Li et al. -- remove a target fact or corpus from
    the weights while preserving everything else. NOT IMPLEMENTED: needs a
    forget set + a retain set and a gradient procedure (ascent on the forget
    set, constrained on the retain set) that writes weights. leace and
    concept_ablation_edit are the reversible inference-time analogues here."""
    raise NotImplementedError(machine_unlearning.__doc__)


def rome(*a, **k) -> dict:
    """Meng et al. 2022 -- Rank-One Model Editing. NOT IMPLEMENTED: needs
    (1) causal tracing to locate the decisive MLP layer for a given fact,
    (2) a rank-one weight update W += (v* - Wk) k^T / (k^T C^-1 k) using the
    layer's key covariance C estimated from a corpus, (3) a paired
    (subject, old fact, new fact) edit request. This is a weight edit, not a
    hook -- out of scope for the model-agnostic hook toolkit. Use the ROME
    reference implementation (github.com/kmeng01/rome) for GPT-2/GPT-J."""
    raise NotImplementedError(rome.__doc__)


ALL = {
    "leace": leace,
    "concept_ablation_edit": concept_ablation_edit,
    "localized_fact_editing": localized_fact_editing,
    "machine_unlearning": machine_unlearning,
    "rome": rome,
}
