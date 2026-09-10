"""Probing -- linear probes, sparse (k-) probing, MDL probing, probe-direction
causal test.  (learnmechinterp / Probing)
"""
from __future__ import annotations

import numpy as np
import torch

from ..tools import adapter
from . import _common as C


def _xy(handle, layer_idx, pos_texts, neg_texts):
    P = C.capture_resid(handle, layer_idx, pos_texts).cpu().numpy()
    N = C.capture_resid(handle, layer_idx, neg_texts).cpu().numpy()
    X = np.vstack([P, N])
    y = np.array([1] * len(P) + [0] * len(N))
    return X, y


def linear_probe(handle, layer_idx, pos_texts, neg_texts, concept: str = "concept") -> dict:
    """Alain & Bengio / Hewitt. Train a linear classifier on frozen
    activations; accuracy = is the concept linearly accessible here."""
    if min(len(set(pos_texts)),len(set(neg_texts))) < 2:
        return {"layer":layer_idx,"cv_accuracy":None,"status":"insufficient_data",
                "skipped":"At least two distinct training examples per class are required; no estimate computed."}
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    X, y = _xy(handle, layer_idx, pos_texts, neg_texts)
    clf = LogisticRegression(max_iter=1000)
    acc = float(cross_val_score(clf, X, y, cv=min(5, len(y) // 2 or 2)).mean())
    clf.fit(X, y)
    w = torch.tensor(clf.coef_[0], dtype=torch.float32)
    return {"concept": concept, "layer": layer_idx, "cv_accuracy": round(acc, 3),
            "probe_top_tokens": C.top_tokens(handle, w / w.norm(), 6)}


def sparse_probe(handle, layer_idx, pos_texts, neg_texts, ks=(1, 3, 5, 10)) -> dict:
    """Gurnee et al. -- constrain the probe to k neurons (hard L0). How
    distributed is the feature across the neuron basis."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    X, y = _xy(handle, layer_idx, pos_texts, neg_texts)
    def ranking(features, labels):
        p, n = features[labels == 1], features[labels == 0]
        t = np.abs(p.mean(0)-n.mean(0))/(np.sqrt(p.var(0)/len(p)+n.var(0)/len(n))+1e-8)
        return np.argsort(-t)
    folds = list(StratifiedKFold(n_splits=min(5, int(np.bincount(y).min()))).split(X,y))
    rows=[]
    for k in ks:
        scores=[]
        for train, test in folds:
            idx=ranking(X[train],y[train])[:k]  # Select features inside each training fold.
            clf=LogisticRegression(max_iter=1000).fit(X[train][:,idx],y[train])
            scores.append(clf.score(X[test][:,idx],y[test]))
        rows.append((k,round(float(np.mean(scores)),3)))
    return {"layer":layer_idx,"accuracy_by_k":rows,"top_neurons":ranking(X,y)[:10].tolist(),
            "selection":"training-fold-only; descriptive small-sample probe"}


def mdl_probe(handle, layer_idx, pos_texts, neg_texts) -> dict:
    """Voita & Titov 2020 -- online-code MDL: how many bits to transmit the
    labels given the representation. Lower = more accessible. Reported vs the
    uniform codelength."""
    from sklearn.linear_model import LogisticRegression
    X, y = _xy(handle, layer_idx, pos_texts, neg_texts)
    n = len(y)
    if n < 8:
        return {"layer": layer_idx, "skipped": f"only {n} samples; MDL needs >= 8 "
                "(pass larger pos_texts / neg_texts)"}
    rng = np.random.default_rng(0)
    perm = rng.permutation(n); X, y = X[perm], y[perm]
    cuts = sorted(set(int(f * n) for f in np.linspace(0.25, 1.0, 6)))
    bits = float(cuts[0])                # only the first block uses uniform coding
    prev = cuts[0]
    for cut in cuts[1:]:
        if cut <= prev:
            continue
        if len(set(y[:prev])) < 2:
            bits += cut - prev          # uniform fallback still transmits every label
            prev = cut
            continue
        clf = LogisticRegression(max_iter=1000).fit(X[:prev], y[:prev])
        pr = np.clip(clf.predict_proba(X[prev:cut]), 1e-6, 1 - 1e-6)
        bits += float(-np.log2(pr[np.arange(cut - prev), y[prev:cut]]).sum())
        prev = cut
    uniform = float(n)
    return {"layer": layer_idx, "mdl_bits": round(float(bits), 1),
            "uniform_codelength_bits": uniform,
            "compression_ratio": round(uniform / float(bits), 3)}


def probe_direction_causal_test(handle, layer_idx, pos_texts, neg_texts,
                                 test_prompt, pos_token, neg_token, strength=6.0) -> dict:
    """Does the probe direction actually *do* anything? Add +/- it to the
    residual stream and measure the logit-diff shift (correlation -> causation
    check)."""
    from sklearn.linear_model import LogisticRegression
    X, y = _xy(handle, layer_idx, pos_texts, neg_texts)
    clf = LogisticRegression(max_iter=1000).fit(X, y)
    d = torch.tensor(clf.coef_[0], dtype=torch.float32, device=handle.device)
    d = d / d.norm().clamp_min(1e-8)
    b = handle.tokenizer([test_prompt], return_tensors="pt").to(handle.device)
    io = handle.tokenizer.encode(" " + pos_token.strip())[-1]
    s = handle.tokenizer.encode(" " + neg_token.strip())[-1]

    def run(strength):
        with adapter.MODEL_LOCK:
            h = handle.layers[layer_idx].register_forward_hook(C.add_direction_hook(d, strength))
            try:
                with torch.no_grad():
                    lg = handle.model(**b).logits[0, -1]
            finally:
                h.remove()
        return float(lg[io] - lg[s])
    base = C.last_token_logit_diff(handle, test_prompt, pos_token, neg_token)
    return {"layer": layer_idx, "baseline_logit_diff": round(base, 3),
            "with_+dir": round(run(strength), 3), "with_-dir": round(run(-strength), 3)}


def geometry_of_truth(handle, layer_idx, true_texts, false_texts) -> dict:
    """Marks & Tegmark 2023 -- is there a linear 'truth' direction? Fit a
    diff-of-means direction (true vs false statements) at one layer, report
    its linear separability and how the two clouds project onto it."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    X, y = _xy(handle, layer_idx, true_texts, false_texts)
    acc = float(cross_val_score(LogisticRegression(max_iter=1000), X, y,
                                cv=min(5, len(y) // 2 or 2)).mean())
    d = C.diff_of_means_direction(handle, layer_idx, true_texts, false_texts)
    T = C.capture_resid(handle, layer_idx, true_texts) @ d
    F = C.capture_resid(handle, layer_idx, false_texts) @ d
    return {"layer": layer_idx, "linear_separability_acc": round(acc, 3),
            "true_proj_mean": round(float(T.mean()), 3), "false_proj_mean": round(float(F.mean()), 3),
            "truth_direction_top_tokens": C.top_tokens(handle, d, 6),
            "note": "Small-sample separability on these examples; this does not establish a general truth direction"}


def attention_probe(handle, layer_idx, pos_texts, neg_texts, concept: str = "concept") -> dict:
    """Kantamneni et al. -- probe the per-head attention outputs (the z
    vector, before the output projection) instead of the residual stream:
    tells you which *heads* at this layer carry the concept."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    op = adapter._find_attn_out_proj(handle.layers[layer_idx])
    nh = getattr(handle.model.config, "num_attention_heads", None) or handle.model.config.n_head
    hd = adapter.get_head_dim(handle)

    def z_feats(texts):
        rows = []
        for t in texts:
            b = handle.tokenizer([t], return_tensors="pt").to(handle.device)
            store = {}
            with adapter.MODEL_LOCK:
                h = op.register_forward_pre_hook(
                    lambda m, a, k: store.__setitem__("z", (a[0] if a else k["input"]).detach()),
                    with_kwargs=True)
                try:
                    with torch.no_grad():
                        handle.model(**b)
                finally:
                    h.remove()
            rows.append(store["z"][0, -1].float().cpu().numpy())
        return np.array(rows)
    P, N = z_feats(pos_texts), z_feats(neg_texts)
    X = np.vstack([P, N]); yv = np.array([1] * len(P) + [0] * len(N))
    per_head = []
    for h in range(nh):
        sl = slice(h * hd, (h + 1) * hd)
        acc = float(cross_val_score(LogisticRegression(max_iter=1000), X[:, sl], yv,
                                    cv=min(5, len(yv) // 2 or 2)).mean())
        per_head.append((h, round(acc, 3)))
    per_head.sort(key=lambda t: -t[1])
    return {"concept": concept, "layer": layer_idx, "per_head_probe_acc": per_head,
            "best_head": per_head[0][0]}


def lat_reading_vectors(handle, pos_texts, neg_texts, layers: list[int] | None = None) -> dict:
    """Zou et al. 2023, Representation Engineering / Linear Artificial
    Tomography -- for each layer, take the contrastive pair differences,
    PCA them, and use the top component as the 'reading vector'. Report its
    class separation per layer to find where the concept is most linear."""
    layers = layers or list(range(1, handle.n_layers, max(1, handle.n_layers // 8)))
    n = min(len(pos_texts), len(neg_texts))
    rows = []
    for l in layers:
        P = C.capture_resid(handle, l, pos_texts[:n])
        N = C.capture_resid(handle, l, neg_texts[:n])
        diffs = (P - N).cpu()
        diffs = diffs - diffs.mean(0)
        _, _, Vt = torch.linalg.svd(diffs, full_matrices=False)
        r = Vt[0].to(handle.device)
        sep = float((P @ r).mean() - (N @ r).mean())
        rows.append((l, round(abs(sep), 3)))
    best = max(rows, key=lambda t: t[1])
    return {"per_layer_reading_vector_separation": rows, "peak_layer": best[0]}


ALL = {
    "linear_probe": linear_probe,
    "sparse_probe": sparse_probe,
    "mdl_probe": mdl_probe,
    "probe_direction_causal_test": probe_direction_causal_test,
    "geometry_of_truth": geometry_of_truth,
    "attention_probe": attention_probe,
    "lat_reading_vectors": lat_reading_vectors,
}
