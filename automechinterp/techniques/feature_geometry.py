"""Feature geometry -- dimensionality of the residual stream, feature-direction
angles / cliques, activation manifolds.
(learnmechinterp / Feature Geometry)
"""
from __future__ import annotations

import numpy as np
import torch

from ..tools import adapter
from . import _common as C
from .superposition import train_toy_sae, _TinySAE


def activation_dimensionality(handle, texts, layers: list[int] | None = None) -> dict:
    """Participation ratio (effective dimension) of last-token residuals per
    layer -- how many directions the layer actually uses."""
    layers = layers or list(range(0, handle.n_layers, max(1, handle.n_layers // 6)))
    rows = []
    for l in layers:
        X = C.capture_resid(handle, l, texts).cpu().numpy()
        Xc = X - X.mean(0)
        ev = np.linalg.svd(Xc, compute_uv=False) ** 2
        pr = (ev.sum() ** 2) / (np.square(ev).sum() + 1e-12)
        rows.append((l, round(float(pr), 2), X.shape[1]))
    return {"participation_ratio_by_layer": rows,
            "note": "PR << d_model => activations live on a low-dim manifold"}


def feature_direction_geometry(handle, layer_idx, texts, expansion: int = 6) -> dict:
    """Elhage et al. 'toy models' / Bricken -- train a toy SAE, then look at
    how its feature directions pack: near-orthogonal? antipodal pairs?
    small cliques of high mutual cosine?"""
    torch.manual_seed(0)
    b = handle.tokenizer(texts, return_tensors="pt", padding=True).to(handle.device)
    store = {}
    with adapter.MODEL_LOCK:
        h = handle.layers[layer_idx].register_forward_hook(
            lambda m, i, o: store.__setitem__("h", (o[0] if isinstance(o, tuple) else o).detach()))
        try:
            with torch.no_grad():
                handle.model(**b)
        finally:
            h.remove()
    acts = store["h"][b["attention_mask"].bool()].float()
    d = acts.shape[1]
    sae = _TinySAE(d, d * expansion)
    opt = torch.optim.Adam(sae.parameters(), lr=1e-3)
    for _ in range(300):
        idx = torch.randint(0, acts.shape[0], (min(256, acts.shape[0]),))
        x = acts[idx]
        recon, f = sae(x)
        (((recon - x) ** 2).mean() + 1e-3 * f.abs().mean()).backward()
        opt.step(); opt.zero_grad()
    D = torch.nn.functional.normalize(sae.dec.weight.detach().T, dim=1)   # [m, d]
    cos = (D @ D.T)
    cos.fill_diagonal_(0)
    off = cos.flatten()
    antipodal = int((off < -0.7).sum() // 2)
    cliques = int((off > 0.5).sum() // 2)
    return {"layer": layer_idx, "n_features": D.shape[0],
            "mean_abs_offdiag_cosine": round(float(off.abs().mean()), 3),
            "antipodal_pairs(<-0.7)": antipodal, "high_cosine_pairs(>0.5)": cliques,
            "interpretation": "near-0 mean cosine => features ~orthogonal (little superposition "
                              "at this width); many high-cosine pairs => packed in cliques"}


def neural_manifold(handle, texts, layer_idx: int | None = None) -> dict:
    """Coarse manifold probe: top principal directions of the layer's
    activations, decoded to vocabulary."""
    layer_idx = layer_idx if layer_idx is not None else 2 * handle.n_layers // 3
    X = C.capture_resid(handle, layer_idx, texts)
    Xc = (X - X.mean(0)).cpu()
    U, S, Vt = torch.linalg.svd(Xc, full_matrices=False)
    comps = []
    for i in range(min(4, Vt.shape[0])):
        comps.append({"pc": i, "sv": round(float(S[i]), 2),
                      "tokens": C.top_tokens(handle, Vt[i].to(handle.device), 5)})
    return {"layer": layer_idx, "principal_components": comps}


def manifold_steering(handle, texts, test_prompt, layer_idx: int | None = None,
                       strength: float = 4.0) -> dict:
    """Steer along the activation manifold, not across it. Take the top
    principal direction of the layer's activations (an on-manifold axis) and
    a random off-manifold direction of equal norm; add each to the residual
    stream and compare how far each pushes the next-token distribution and
    how in-distribution the resulting state stays (residual norm ratio)."""
    layer_idx = layer_idx if layer_idx is not None else 2 * handle.n_layers // 3
    X = C.capture_resid(handle, layer_idx, texts)
    Xc = (X - X.mean(0)).cpu()
    _, _, Vt = torch.linalg.svd(Xc, full_matrices=False)
    on = Vt[0].to(handle.device)
    on = on / on.norm()
    rng = torch.Generator(device="cpu").manual_seed(0)
    off = torch.randn(on.shape[0], generator=rng).to(handle.device)
    off = off - (off @ on) * on
    off = off / off.norm()
    rms = float(X.norm(dim=1).mean() / (on.numel() ** 0.5))

    def probe(direction):
        b = handle.tokenizer([test_prompt], return_tensors="pt").to(handle.device)
        with adapter.MODEL_LOCK:
            h = handle.layers[layer_idx].register_forward_hook(
                C.add_direction_hook(direction, strength * rms))
            try:
                with torch.no_grad():
                    lg = handle.model(**b).logits[0, -1]
            finally:
                h.remove()
        t = torch.topk(lg, 5)
        return [(handle.tokenizer.decode([i]).strip(), round(float(v), 2))
                for i, v in zip(t.indices.tolist(), t.values.tolist())]

    return {"layer": layer_idx,
            "next_token_baseline": probe(torch.zeros_like(on)),
            "next_token_on_manifold_(top_PC)": probe(on),
            "next_token_off_manifold_(random)": probe(off),
            "on_manifold_direction_tokens": C.top_tokens(handle, on, 5)}


ALL = {
    "activation_dimensionality": activation_dimensionality,
    "feature_direction_geometry": feature_direction_geometry,
    "neural_manifold": neural_manifold,
    "manifold_steering": manifold_steering,
}
