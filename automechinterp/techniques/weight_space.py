"""Weight-space interpretability -- parameter SVD / effective rank, weight-norm
profile, OV/QK weight-only structure.
(learnmechinterp / Weight-Space Interpretability)
"""
from __future__ import annotations

import numpy as np
import torch

from ..tools import adapter
from . import _common as C


def parameter_svd(handle, which: str = "attn_out", max_layers: int = 6) -> dict:
    """Decompose a weight matrix per layer: singular-value spectrum, effective
    rank (entropy of normalised singular values), and the top left-singular
    direction decoded to vocabulary (for output-side matrices)."""
    rows = []
    U = C.unembed(handle)
    layers = list(range(0, handle.n_layers, max(1, handle.n_layers // max_layers)))
    for l in layers:
        layer = handle.layers[l]
        if which == "attn_out":
            W = adapter._find_attn_out_proj(layer).weight.detach().float()
        elif which == "mlp_out":
            mlp = getattr(layer, "mlp", None)
            W = (getattr(mlp, "c_proj", None) or getattr(mlp, "down_proj", None)).weight.detach().float()
        else:
            raise ValueError(which)
        if W.shape[0] < W.shape[1]:
            W = W.T
        s = torch.linalg.svdvals(W)
        p = (s / s.sum()).clamp_min(1e-12)
        eff_rank = float(torch.exp(-(p * p.log()).sum()))
        rows.append((l, round(float(s[0]), 2), round(eff_rank, 1), int(min(W.shape))))
    return {"matrix": which, "per_layer_(top_sv, effective_rank, full_rank)": rows}


def weight_norm_profile(handle) -> dict:
    """Per-layer parameter-norm distribution -- where the model concentrates
    capacity."""
    rows = []
    for l in range(handle.n_layers):
        layer = handle.layers[l]
        by = {}
        for n, p in layer.named_parameters():
            key = n.split(".")[0]
            by[key] = by.get(key, 0.0) + float(p.detach().float().norm() ** 2)
        rows.append((l, {k: round(v ** 0.5, 1) for k, v in by.items()}))
    return {"per_layer_submodule_frobenius_norm": rows}


def embedding_unembedding_alignment(handle) -> dict:
    """Are E and U tied / aligned? Mean cosine of matched rows, and the
    'always-promoted' tokens (rows of U with largest norm)."""
    E = handle.model.get_input_embeddings().weight.detach().float()
    Uw = C.unembed(handle).detach().float()
    n = min(E.shape[0], Uw.shape[0])
    cos = torch.nn.functional.cosine_similarity(E[:n], Uw[:n], dim=1)
    norms = Uw.norm(dim=1)
    top = torch.topk(norms, 8).indices.tolist()
    return {"mean_row_cosine(E,U)": round(float(cos.mean()), 3),
            "tied": bool(cos.mean() > 0.99),
            "largest_unembed_norm_tokens": [handle.tokenizer.decode([i]).strip() for i in top]}


def parameter_space_circuits(handle, max_layers: int = 8, sim_threshold: float = 0.4) -> dict:
    """Bushnaq et al. / weight-space circuits -- group attention heads by what
    their OV output writes to the vocabulary, from weights alone (no forward
    pass). Heads whose output-direction / vocab projection is highly similar
    are doing a related job; the clusters are candidate parameter-space
    circuits."""
    U = C.unembed(handle).detach().float()
    hd = adapter.get_head_dim(handle)
    nh = getattr(handle.model.config, "num_attention_heads", None) or handle.model.config.n_head
    layers = list(range(0, handle.n_layers, max(1, handle.n_layers // max_layers)))
    labels, vecs = [], []
    for l in layers:
        Wo = adapter._find_attn_out_proj(handle.layers[l]).weight.detach().float()
        if Wo.shape[0] != U.shape[1]:
            Wo = Wo.T
        for h in range(nh):
            col = Wo[:, h * hd:(h + 1) * hd]                 # d_model x head_dim
            v = (U @ col).norm(dim=1)                        # vocab-space energy per token
            v = v / v.norm().clamp_min(1e-8)
            labels.append(f"L{l}H{h}"); vecs.append(v)
    M = torch.stack(vecs)
    S = (M @ M.T)
    groups, seen = [], set()
    order = list(range(len(labels)))
    for i in order:
        if i in seen:
            continue
        grp = [labels[i]]
        seen.add(i)
        for j in order:
            if j not in seen and float(S[i, j]) > sim_threshold:
                grp.append(labels[j]); seen.add(j)
        if len(grp) > 1:
            groups.append(grp)
    return {"n_heads_scanned": len(labels), "similarity_threshold": sim_threshold,
            "parameter_space_head_groups": groups[:10],
            "note": "grouped by cosine of the head's OV->vocab energy profile, weights only"}


def interpretable_training_note(*a, **k) -> dict:
    """Sharkey et al. / weight-sparsity training -- constrain weights during
    training (L1 / group sparsity / low-rank) so computation localises. NOT
    IMPLEMENTED: this is a training-time intervention, out of scope for a
    read-only toolkit. parameter_svd's effective-rank column is the metric
    you would watch to check whether such training worked."""
    raise NotImplementedError(interpretable_training_note.__doc__)


ALL = {
    "parameter_svd": parameter_svd,
    "weight_norm_profile": weight_norm_profile,
    "embedding_unembedding_alignment": embedding_unembedding_alignment,
    "parameter_space_circuits": parameter_space_circuits,
    "interpretable_training_note": interpretable_training_note,
}
