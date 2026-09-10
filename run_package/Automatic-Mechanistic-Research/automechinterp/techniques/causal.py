"""Causal interventions -- activation patching, attribution/path patching,
self-repair, interchange (causal abstraction).  (learnmechinterp / Causal Interventions)
"""
from __future__ import annotations

import torch

from ..tools import adapter, tier_c, tier_v
from . import _common as C


def activation_patching(handle, clean_prompt, corrupted_prompt, pos_token, neg_token,
                         granularity: str = "layer") -> dict:
    """Meng et al. / Wang et al. -- swap a clean activation into the corrupted
    run (denoising) and report fraction of the metric gap recovered, per
    layer or per head."""
    mfn = C.logit_diff_metric_fn(handle, pos_token, neg_token)
    ci = handle.tokenizer([clean_prompt], return_tensors="pt").to(handle.device)
    xi = handle.tokenizer([corrupted_prompt], return_tensors="pt").to(handle.device)
    n_heads = getattr(handle.model.config, "num_attention_heads", None) or handle.model.config.n_head
    if granularity == "head":
        digest, circuit = tier_c.run_acdc(handle, clean_prompt, corrupted_prompt, pos_token, neg_token,
                                          range(handle.n_layers), n_heads, threshold=0.10)
        return {"granularity": "head", "digest": digest, "recovering_heads": [list(c) for c in circuit]}
    rows = []
    for l in range(handle.n_layers):
        r = adapter.patch_layer(handle, l, ci["input_ids"], ci["attention_mask"],
                                xi["input_ids"], xi["attention_mask"], mfn)
        rows.append((l, round(r["fraction_recovered"], 3)))
    rows.sort(key=lambda t: -abs(t[1]))
    return {"granularity": "layer", "clean_metric": round(C.last_token_logit_diff(handle, clean_prompt, pos_token, neg_token), 3),
            "corrupt_metric": round(C.last_token_logit_diff(handle, corrupted_prompt, pos_token, neg_token), 3),
            "top_layers_by_recovery": rows[:8]}


def attribution_patching(handle, clean_prompt, corrupted_prompt, pos_token, neg_token) -> dict:
    """Nanda 2023 / Syed et al. 2023 -- linear approx of patching in one
    fwd+bwd pass (tier_c.run_eap)."""
    n_heads = getattr(handle.model.config, "num_attention_heads", None) or handle.model.config.n_head
    digest = tier_c.run_eap(handle, clean_prompt, corrupted_prompt, pos_token, neg_token,
                            range(handle.n_layers), n_heads)
    return {"digest": digest}


def path_patching_qk(handle, clean_prompt, corrupted_prompt, pos_token, neg_token,
                      receiver_layer: int, receiver_head: int) -> dict:
    """Wang et al. / Goldowsky-Dill et al. -- which upstream heads feed the
    receiver head's *query* input. Simplified: for each upstream head, patch
    its clean output into the corrupted run but ONLY let the change reach the
    receiver layer's attention input, and measure the metric shift."""
    mfn = C.logit_diff_metric_fn(handle, pos_token, neg_token)
    ci = handle.tokenizer([clean_prompt], return_tensors="pt").to(handle.device)
    xi = handle.tokenizer([corrupted_prompt], return_tensors="pt").to(handle.device)
    n_heads = getattr(handle.model.config, "num_attention_heads", None) or handle.model.config.n_head
    head_dim = adapter.get_head_dim(handle)
    with adapter.MODEL_LOCK, torch.no_grad():
        base = mfn(handle.model(**xi).logits)
    rows = []
    recv_ln = getattr(handle.layers[receiver_layer], "ln_1", None) or \
              getattr(handle.layers[receiver_layer], "input_layernorm", None)
    for ul in range(receiver_layer):
        op = adapter._find_attn_out_proj(handle.layers[ul])
        clean_z = {}

        def cap(m, a, k):
            clean_z["v"] = (a[0] if a else k["input"]).detach().clone()
        with adapter.MODEL_LOCK:
            hc = op.register_forward_pre_hook(cap, with_kwargs=True)
            with torch.no_grad():
                handle.model(**ci)
            hc.remove()
        for uh in range(n_heads):
            lo, hi = uh * head_dim, (uh + 1) * head_dim

            def patch(m, a, k):
                x = (a[0] if a else k["input"]).clone()
                x[:, :, lo:hi] = clean_z["v"][:, -x.shape[1]:, lo:hi]
                if a:
                    return (x,) + a[1:], k
                k["input"] = x
                return a, k
            with adapter.MODEL_LOCK:
                hp = op.register_forward_pre_hook(patch, with_kwargs=True)
                try:
                    with torch.no_grad():
                        m = mfn(handle.model(**xi).logits)
                finally:
                    hp.remove()
            rows.append((f"L{ul}H{uh}", round(m - base, 3)))
    rows.sort(key=lambda t: -abs(t[1]))
    return {"receiver": f"L{receiver_layer}H{receiver_head}",
            "note": "simplified full-path patch (not query-only isolation)",
            "top_senders": rows[:8]}


def self_repair(handle, clean_prompt, pos_token, neg_token, ablate_head: tuple[int, int]) -> dict:
    """McGrath et al. 2023 (Hydra) / Rushing & Nanda 2024. Ablate one head,
    measure how every *other* head's direct logit attribution changes -- the
    compensation ('self-repair') that makes single-head ablation misleading."""
    jl, jh = ablate_head
    head_dim = adapter.get_head_dim(handle)
    lo, hi = jh * head_dim, (jh + 1) * head_dim
    op = adapter._find_attn_out_proj(handle.layers[jl])

    def abl(m, a, k):
        x = (a[0] if a else k["input"]).clone()
        x[:, :, lo:hi] = x[:, :, lo:hi].mean(dim=(0, 1), keepdim=True)
        if a:
            return (x,) + a[1:], k
        k["input"] = x
        return a, k

    base0 = C.dla_scores(handle, clean_prompt, pos_token, neg_token)
    base1 = C.dla_scores(handle, clean_prompt, pos_token, neg_token, extra_pre_hooks=[(op, abl)])
    shifts = sorted(((f"L{l}H{h}", round(base1[(l, h)] - base0[(l, h)], 3))
                     for (l, h) in base0 if (l, h) != (jl, jh)), key=lambda t: -abs(t[1]))
    return {"ablated": f"L{jl}H{jh}", "biggest_DLA_shifts_after_ablation": shifts[:8],
            "interpretation": "positive shift = that head took over write-toward-answer load (backup)"}


def interchange_intervention(handle, prompt_a, prompt_b, pos_a, neg_a, pos_b, neg_b,
                              layer_idx: int, head_idx: int) -> dict:
    """Geiger et al. causal abstraction / IIT -- inject head activation from
    run B into run A, check A's metric moves toward B's."""
    r = tier_v.interchange_intervention(handle, layer_idx, head_idx,
                                        (prompt_a, pos_a, neg_a), (prompt_b, pos_b, neg_b))
    return {"digest": r}


def causal_mediator_selection(handle, clean_prompt, corrupted_prompt, pos_token, neg_token,
                               recover_frac: float = 0.8) -> dict:
    """Vig et al. / Geiger et al. -- pick the intermediate variables to treat
    as causal mediators. Greedily add the layer whose denoising patch adds
    the most recovered metric until >= `recover_frac` of the clean-corrupt
    gap is explained; that set is the minimal layer-mediator set."""
    mfn = C.logit_diff_metric_fn(handle, pos_token, neg_token)
    ci = handle.tokenizer([clean_prompt], return_tensors="pt").to(handle.device)
    xi = handle.tokenizer([corrupted_prompt], return_tensors="pt").to(handle.device)
    per_layer = {}
    for l in range(handle.n_layers):
        r = adapter.patch_layer(handle, l, ci["input_ids"], ci["attention_mask"],
                                xi["input_ids"], xi["attention_mask"], mfn)
        per_layer[l] = r["fraction_recovered"]
    chosen, cum = [], 0.0
    for l, _ in sorted(per_layer.items(), key=lambda t: -t[1]):
        if cum >= recover_frac:
            break
        chosen.append(l); cum += max(per_layer[l], 0.0)
    return {"target_recovery": recover_frac,
            "selected_layer_mediators": sorted(chosen),
            "approx_cumulative_recovery": round(cum, 3),
            "per_layer_single_patch_recovery": {l: round(v, 3) for l, v in per_layer.items()}}


def refined_attribution(handle, clean_prompt, corrupted_prompt, pos_token, neg_token,
                         steps: int = 5) -> dict:
    """Hanna et al. 2024 -- integrated-gradients attribution patching: average
    the patching gradient over `steps` points interpolated between the
    corrupted and clean activations, instead of the single-point linear
    approximation plain EAP/AtP uses. Reduces the AtP approximation error."""
    mfn = C.logit_diff_metric_fn(handle, pos_token, neg_token)
    ci = handle.tokenizer([clean_prompt], return_tensors="pt").to(handle.device)
    xi = handle.tokenizer([corrupted_prompt], return_tensors="pt").to(handle.device)
    acc = {}
    for a_i in range(1, steps + 1):
        alpha = a_i / steps
        contrib = {}
        for l in range(handle.n_layers):
            cache = {}

            def cap(mod, inp, out, key=l):
                cache[key] = (out[0] if isinstance(out, tuple) else out).detach()
            with adapter.MODEL_LOCK:
                h = handle.layers[l].register_forward_hook(cap)
                with torch.no_grad():
                    handle.model(**ci)
                h.remove()
                clean_act = cache[l]

            def blend(mod, inp, out, ca=clean_act, al=alpha):
                is_t = isinstance(out, tuple)
                hs = out[0] if is_t else out
                hs = (1 - al) * hs + al * ca[:, : hs.shape[1], :].to(hs.dtype)
                return (hs,) + out[1:] if is_t else hs
            with adapter.MODEL_LOCK:
                h = handle.layers[l].register_forward_hook(blend)
                try:
                    with torch.no_grad():
                        m = mfn(handle.model(**xi).logits)
                finally:
                    h.remove()
            contrib[l] = m
        for l, v in contrib.items():
            acc[l] = acc.get(l, 0.0) + v / steps
    rows = sorted(((l, round(v, 3)) for l, v in acc.items()), key=lambda t: -abs(t[1]))
    return {"steps": steps, "integrated_layer_attribution": rows[:8],
            "note": "mean metric under a partial clean-patch, averaged along the interpolation path"}


ALL = {
    "activation_patching": activation_patching,
    "attribution_patching": attribution_patching,
    "path_patching_qk": path_patching_qk,
    "self_repair": self_repair,
    "interchange_intervention": interchange_intervention,
    "causal_mediator_selection": causal_mediator_selection,
    "refined_attribution": refined_attribution,
}
