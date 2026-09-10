"""Prompt-position causal measurements using complete answer continuations.

These are head-level approximations/searches, not published edge-graph ACDC.
The cache is cleared between behaviors by the Colab worker; it holds immutable
unintervened reference states only. Intervened forwards are never cached.
"""
from contextlib import nullcontext
import numpy as np
import torch

from . import adapter
from ..eval.causal_measurements import ActivationBank, score_contrast, _candidate_batch, ablate_heads


def reference(handle, clean, corrupted, positive, negative):
    cache = getattr(handle, "_causal_reference_cache", {})
    key = (clean, corrupted, positive, negative)
    if key not in cache:
        cache[key] = (ActivationBank.capture(handle, clean),
                      score_contrast(handle, clean, positive, negative)["margin"],
                      score_contrast(handle, corrupted, positive, negative)["margin"])
        handle._causal_reference_cache = cache
    return cache[key]


def layer_effect(handle, layer, clean, corrupted, positive, negative):
    bank, cm, xm = reference(handle, clean, corrupted, positive, negative)
    position = len(handle.tokenizer.encode(corrupted.rstrip(), add_special_tokens=True)) - 1
    pm = score_contrast(handle, corrupted, positive, negative,
                       lambda: bank.patch("block_update", layer, position))["margin"]
    gap = cm - xm
    fraction = (pm-xm)/gap if gap > 1e-6 else float("nan")
    return {"clean_metric": cm, "corrupted_metric": xm, "patched_metric": pm,
            "fraction_recovered": fraction, "intervention": "last_prompt_position_block_update",
            "metric": "full_continuation_log_probability_margin", "gap_status": "positive" if gap > 1e-6 else "unresolved"}


def exact_head_scan(handle, clean, corrupted, positive, negative, layers, n_heads, threshold=.10, min_gap=.50):
    bank, cm, xm = reference(handle, clean, corrupted, positive, negative)
    gap = cm-xm
    if gap < min_gap:
        return f"unresolved positive clean-corrupted margin ({gap:.4f}); no head claim", []
    pos = len(handle.tokenizer.encode(corrupted.rstrip(), add_special_tokens=True))-1
    rows = []
    for layer in layers:
        for head in range(n_heads):
            value = score_contrast(handle, corrupted, positive, negative,
                         lambda: bank.patch("head", layer, pos, head=head))["margin"]
            rows.append((layer, head, (value-xm)/gap))
    circuit = [(l, h) for l, h, v in rows if v >= threshold]
    ordered = sorted(rows, key=lambda r: -abs(r[2]))
    # Preserve every head score, not only the text digest.
    handle._last_head_scan = {"rows": rows, "threshold": threshold, "margin_gap": gap,
                              "scope": "single-head prompt-position sufficiency"}
    return f"exact head scan (full continuation; prompt position): top={ordered[:8]} | candidates={circuit}", circuit


def gradients(handle, prompt, positive, negative, layers, ablated=(), ablate_positions=None):
    """Full-answer margin derivative at each connected attention projection input.

    Multiple activation tensors remain connected. Detaching each selected layer
    would sever the gradient paths to earlier layers and silently zero their scores.
    """
    layers = list(layers)
    scores = {l: None for l in layers}
    activations = {}
    for sign, answer in ((1., positive), (-1., negative)):
        batch, start = _candidate_batch(handle, prompt, answer)
        captured, hooks = {}, []
        def make_hook(layer):
            def hook(module, args, kwargs):
                x = (args[0] if args else kwargs["input"]).clone().requires_grad_(True)
                captured[layer] = x
                return ((x,)+args[1:], kwargs) if args else (args, {**kwargs, "input": x})
            return hook
        with adapter.MODEL_LOCK, (ablate_heads(handle, ablated, positions=ablate_positions) if ablated else nullcontext()), torch.enable_grad():
            try:
                for layer in layers:
                    hooks.append(adapter._find_attn_out_proj(handle.layers[layer]).register_forward_pre_hook(
                        make_hook(layer), with_kwargs=True))
                output = handle.model(**batch, use_cache=False).logits
                logits = output[0, start-1:-1].float()
                labels = batch["input_ids"][0, start:]
                metric = logits.log_softmax(-1).gather(-1, labels[:, None]).sum()
                gs = torch.autograd.grad(metric, [captured[l] for l in layers])
            finally:
                for hook in hooks:
                    hook.remove()
        for l, g in zip(layers, gs):
            value = sign * g[:, start-1, :].detach().float()
            scores[l] = value if scores[l] is None else scores[l] + value
            activations[l] = captured[l][:, start-1, :].detach().float()
            if not torch.isfinite(scores[l]).all():
                raise FloatingPointError("Nonfinite full-continuation activation gradient")
    return scores, activations


def eap(handle, clean, corrupted, positive, negative, layers, n_heads):
    layers = list(layers)
    bank, _, _ = reference(handle, clean, corrupted, positive, negative)
    grad, corr = gradients(handle, corrupted, positive, negative, layers)
    dim = adapter.get_head_dim(handle)
    rows = []
    for l in layers:
        delta = bank.head_input[l].float() - corr[l]
        for h in range(n_heads):
            lo, hi = h*dim, (h+1)*dim
            rows.append((float((grad[l][:, lo:hi]*delta[:, lo:hi]).sum()), (l, h)))
    rows.sort(key=lambda r: -abs(r[0]))
    handle._last_eap = rows
    return f"head attribution approximation (full continuation; prompt position): {rows[:8]}"


def synergy(handle, clean, corrupted, positive, negative, layers, n_heads, candidates=None, k=10):
    layers = list(layers)
    base, acts = gradients(handle, clean, positive, negative, layers)
    dim = adapter.get_head_dim(handle)
    first = {(l, h): float((base[l][:, h*dim:(h+1)*dim]*acts[l][:, h*dim:(h+1)*dim]).sum())
             for l in layers for h in range(n_heads)}
    candidates = candidates or sorted(first, key=lambda h: -abs(first[h]))[:8]
    position = len(handle.tokenizer.encode(clean.rstrip(), add_special_tokens=True))-1
    rows = []
    for j in candidates:
        if j not in first:
            continue
        changed, changed_acts = gradients(handle, clean, positive, negative, layers, [j],
                                          ablate_positions=[position])
        for i in first:
            if i == j:
                continue
            l, h = i; lo, hi = h*dim, (h+1)*dim
            score = float((base[l][:,lo:hi]*acts[l][:,lo:hi]
                           - changed[l][:,lo:hi]*changed_acts[l][:,lo:hi]).sum())
            rows.append((score, i, j))
    rows.sort(key=lambda r: -abs(r[0]))
    return (f"S-EAP candidate interactions (full continuation; last-prompt-position zero-ablation reference; "
            f"prompt-position derivative; not exact interaction proof): {rows[:k]}"), rows
