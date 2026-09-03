"""Steering -- activation addition (CAA), ablation steering, affine steering,
multi-layer steering, function vectors.  (learnmechinterp / Steering)
"""
from __future__ import annotations

import torch

from ..tools import adapter
from . import _common as C


def _steer_dir(handle, layer_idx, pos_texts, neg_texts):
    """Unit concept direction, plus the residual RMS at this layer so a
    `strength` of ~1-8 is a meaningful multiple of the activation scale."""
    u = C.diff_of_means_direction(handle, layer_idx, pos_texts, neg_texts)
    rms = float(C.capture_resid(handle, layer_idx, pos_texts).norm(dim=1).mean()
                / (u.numel() ** 0.5))
    return u, max(rms, 1e-3)


def _next_top(handle, prompt, hook_specs):
    b = handle.tokenizer([prompt], return_tensors="pt").to(handle.device)
    with adapter.MODEL_LOCK:
        hs = [handle.layers[l].register_forward_hook(fn) for (l, fn) in hook_specs]
        try:
            with torch.no_grad():
                lg = handle.model(**b).logits[0, -1]
        finally:
            for h in hs:
                h.remove()
    t = torch.topk(lg, 5)
    return [(handle.tokenizer.decode([i]).strip(), round(float(v), 2))
            for i, v in zip(t.indices.tolist(), t.values.tolist())]


def activation_addition(handle, layer_idx, pos_texts, neg_texts, test_prompt, strength=5.0) -> dict:
    """Turner et al. / Rimsky et al. CAA -- add a diff-of-means concept vector
    to the residual stream and read the shifted next-token distribution."""
    d, rms = _steer_dir(handle, layer_idx, pos_texts, neg_texts)
    before = _next_top(handle, test_prompt, [])
    after = _next_top(handle, test_prompt, [(layer_idx, C.add_direction_hook(d, strength * rms))])
    return {"layer": layer_idx, "strength_x_rms": round(strength * rms, 2),
            "concept_direction_top_tokens": C.top_tokens(handle, d, 6),
            "next_token_before": before, "next_token_after": after}


def ablation_steering(handle, layer_idx, pos_texts, neg_texts, test_prompt) -> dict:
    """Project the concept direction OUT of the residual stream (zero its
    component) and see what the model does without it."""
    d = C.diff_of_means_direction(handle, layer_idx, pos_texts, neg_texts)
    before = _next_top(handle, test_prompt, [])
    after = _next_top(handle, test_prompt, [(layer_idx, C.project_out_hook(d))])
    return {"layer": layer_idx, "next_token_before": before, "next_token_after_projecting_out": after}


def affine_steering(handle, layer_idx, pos_texts, neg_texts, test_prompt,
                     scale=1.0, shift=4.0) -> dict:
    """Scale the component along the concept direction and add a shift --
    a * <h,u> u + b u -- the general affine intervention."""
    u, rms = _steer_dir(handle, layer_idx, pos_texts, neg_texts)

    def hook(m, i, o):
        is_t = isinstance(o, tuple)
        hs = o[0] if is_t else o
        coef = (hs.to(u.dtype) * u).sum(-1, keepdim=True)
        hs = hs + ((scale - 1.0) * coef + shift * rms) * u.to(hs.dtype)
        return (hs,) + o[1:] if is_t else hs
    before = _next_top(handle, test_prompt, [])
    after = _next_top(handle, test_prompt, [(layer_idx, hook)])
    return {"layer": layer_idx, "scale": scale, "shift": shift,
            "next_token_before": before, "next_token_after": after}


def multi_layer_steering(handle, pos_texts, neg_texts, test_prompt, strength=3.0,
                          layers: list[int] | None = None) -> dict:
    """Apply the concept vector at several consecutive layers at once."""
    layers = layers or list(range(handle.n_layers // 3, 2 * handle.n_layers // 3))
    specs = []
    for l in layers:
        d, rms = _steer_dir(handle, l, pos_texts, neg_texts)
        specs.append((l, C.add_direction_hook(d, strength * rms)))
    return {"layers": layers, "strength": strength,
            "next_token_before": _next_top(handle, test_prompt, []),
            "next_token_after": _next_top(handle, test_prompt, specs)}


def function_vector(handle, icl_prompt, zeroshot_prompt, layer_idx: int | None = None) -> dict:
    """Todd et al. 2023 -- capture the mean last-token residual state on an
    in-context-learning prompt, add it to a zero-shot prompt, and see if the
    task transfers without the demonstrations."""
    layer_idx = layer_idx if layer_idx is not None else handle.n_layers // 2
    fv = C.capture_resid(handle, layer_idx, [icl_prompt])[0]      # keep natural scale
    before = _next_top(handle, zeroshot_prompt, [])
    after = _next_top(handle, zeroshot_prompt, [(layer_idx, C.add_direction_hook(fv, 1.0))])
    return {"layer": layer_idx, "icl_prompt": icl_prompt, "zeroshot_prompt": zeroshot_prompt,
            "next_token_before": before, "next_token_after_fv": after}


def unsupervised_steering_vector(handle, layer_idx, prompt, target_layer: int | None = None,
                                  steps: int = 60, radius: float = 6.0) -> dict:
    """Mack & Turner 2024 (MELBO) -- find a steering direction with NO
    labelled data: the norm-constrained direction added at `layer_idx` that
    maximally changes the residual stream at `target_layer` downstream.
    Power-iteration-style: gradient-ascend a unit vector on that objective."""
    target_layer = target_layer if target_layer is not None else min(handle.n_layers - 1, layer_idx + 3)
    b = handle.tokenizer([prompt], return_tensors="pt").to(handle.device)
    d = handle.model.config.hidden_size if hasattr(handle.model.config, "hidden_size") else handle.model.config.n_embd
    theta = torch.randn(d, device=handle.device)
    theta = theta / theta.norm()
    theta.requires_grad_(True)
    opt = torch.optim.Adam([theta], lr=0.05)

    with adapter.MODEL_LOCK:
        base_store = {}
        h0 = handle.layers[target_layer].register_forward_hook(
            lambda m, i, o: base_store.__setitem__("h", (o[0] if isinstance(o, tuple) else o).detach()))
        try:
            with torch.no_grad():
                handle.model(**b)
        finally:
            h0.remove()
        base_down = base_store["h"][:, -1, :]

        for _ in range(steps):
            u = radius * theta / theta.norm().clamp_min(1e-8)
            tgt = {}
            ha = handle.layers[layer_idx].register_forward_hook(C.add_direction_hook(u, 1.0))
            hb = handle.layers[target_layer].register_forward_hook(
                lambda m, i, o: tgt.__setitem__("h", (o[0] if isinstance(o, tuple) else o)))
            try:
                handle.model(**b)
            finally:
                ha.remove(); hb.remove()
            loss = -(tgt["h"][:, -1, :] - base_down).pow(2).mean()
            opt.zero_grad(); loss.backward(); opt.step()

    vec = (radius * theta / theta.norm()).detach()
    before = _next_top(handle, prompt, [])
    after = _next_top(handle, prompt, [(layer_idx, C.add_direction_hook(vec, 1.0))])
    return {"inject_layer": layer_idx, "objective_layer": target_layer, "radius": radius,
            "found_direction_top_tokens": C.top_tokens(handle, vec, 6),
            "next_token_before": before, "next_token_after": after}


ALL = {
    "activation_addition": activation_addition,
    "ablation_steering": ablation_steering,
    "affine_steering": affine_steering,
    "multi_layer_steering": multi_layer_steering,
    "function_vector": function_vector,
    "unsupervised_steering_vector": unsupervised_steering_vector,
}
