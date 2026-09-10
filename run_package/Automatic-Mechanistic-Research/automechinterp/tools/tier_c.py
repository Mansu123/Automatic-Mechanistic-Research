"""Tier C -- Component-Level tools (Component Agents, Sec. 3.4).

Transformer-specific: circuit discovery needs attention-head structure, so
unlike Tier N/L these assume `handle.layers[i]` exposes an attention
sub-module with a locatable out-projection (see adapter._find_attn_out_proj).
`run_acdc` and `run_eap` are simplified, honest re-implementations of the
methods they name -- exhaustive per-head causal search rather than the full
iterative-graph-pruning algorithms, which is the right scope for a codebase
meant to be read and run rather than a reproduction of the acdc library.
"""
from __future__ import annotations

import torch

from . import adapter, sae as _sae
from .digest import topk_digest


def run_sae_decompose(handle: adapter.ModelHandle, layer_idx: int, text: str, token_idx: int = -1) -> str:
    """Tool: run_sae_decompose(). Sparse interpretable feature directions for
    a flagged activation -- gives semantic content to a causally-important
    head/circuit ("L9H9 matters" -> "L9H9's output correlates with SAE
    features {...}"), not just its causal effect size."""
    return _sae.run_sae_decompose(layer_idx, text, token_idx=token_idx,
                                   model_id=handle.model_id, device=handle.device)


def _ioi_metric_fn(handle, io_token: str, s_token: str):
    # [-1] not [0] -- see behaviors.py's _make_task for why (cross-tokenizer safety).
    io_id = handle.tokenizer.encode(" " + io_token.strip())[-1]
    s_id = handle.tokenizer.encode(" " + s_token.strip())[-1]

    def metric_fn(logits: torch.Tensor) -> float:
        last = logits[0, -1]
        return float((last[io_id] - last[s_id]).item())
    return metric_fn


def run_acdc(handle, clean_prompt, corrupted_prompt, io_token, s_token, layer_range, n_heads,
             threshold=.10, min_gap=.50):
    """Exhaustive head sufficiency scan, not edge-graph ACDC or proof of minimality."""
    from .colab_causal import exact_head_scan
    return exact_head_scan(handle, clean_prompt, corrupted_prompt, io_token, s_token,
                           layer_range, n_heads, threshold, min_gap)


def run_eap(handle, clean_prompt, corrupted_prompt, io_token, s_token, layer_range, n_heads):
    from .colab_causal import eap
    return eap(handle, clean_prompt, corrupted_prompt, io_token, s_token, layer_range, n_heads)


def direct_logit_attribution(handle: adapter.ModelHandle, clean_prompt: str,
                              io_token: str, s_token: str, layer_range: range, n_heads: int,
                              k: int = 8) -> tuple[str, list[tuple[int, int]]]:
    """Tool: direct_logit_attribution(). Projects each attention head's write
    onto the logit-difference direction W_U[:,IO] - W_U[:,S] in one forward
    pass (Elhage et al. 2021). A positive score = the head writes toward the
    correct name, negative = it suppresses it. This is a *write-direction*
    signal, orthogonal to ablation effect -- a name-mover-type head shows up
    here even when its marginal ablation effect is ~0 (the regime run_acdc /
    run_eap are blind to). Returns a digest and the top-|score| heads."""
    import numpy as np

    io_id = handle.tokenizer.encode(" " + io_token.strip())[-1]
    s_id = handle.tokenizer.encode(" " + s_token.strip())[-1]
    W_U = handle.model.get_output_embeddings().weight  # [vocab, d_model]
    dir_vec = (W_U[io_id] - W_U[s_id]).detach()        # [d_model]
    head_dim = adapter.get_head_dim(handle)
    batch = handle.tokenizer([clean_prompt], return_tensors="pt").to(handle.device)
    layers = list(layer_range)

    names, scores = [], []
    with adapter.MODEL_LOCK, torch.no_grad():
        # final-residual RMS at the last position, to approximate the frozen
        # final-LayerNorm gain that a full DLA implementation folds in
        fin = {}
        ln = (getattr(getattr(handle.model, "transformer", None), "ln_f", None)
              or getattr(getattr(handle.model, "model", None), "norm", None))

        def cap_final(module, inp, out):
            fin["h"] = (inp[0] if isinstance(inp, tuple) else inp).detach()
        h_ln = ln.register_forward_hook(cap_final) if ln is not None else None
        handle.model(**batch)
        if h_ln is not None:
            h_ln.remove()
        rms = fin["h"][0, -1].pow(2).mean().sqrt().clamp_min(1e-6) if "h" in fin else torch.tensor(1.0)

        for l in layers:
            out_proj = adapter._find_attn_out_proj(handle.layers[l])
            z = {}

            def cap_z(module, args, kwargs):
                z["v"] = (args[0] if args else kwargs["input"]).detach().clone()
            hz = out_proj.register_forward_pre_hook(cap_z, with_kwargs=True)
            handle.model(**batch)
            hz.remove()
            zc = z["v"]                                 # [1, seq, n_heads*head_dim]
            zero_out = out_proj(torch.zeros_like(zc[:, -1:, :]))[0, -1]  # bias only
            for hd in range(n_heads):
                lo, hi = hd * head_dim, (hd + 1) * head_dim
                masked = torch.zeros_like(zc[:, -1:, :])
                masked[:, :, lo:hi] = zc[:, -1:, lo:hi]
                r_h = out_proj(masked)[0, -1] - zero_out  # this head's write to resid, last pos
                score = float((r_h / rms) @ dir_vec)
                names.append(f"L{l}H{hd}")
                scores.append(score)

    order = np.argsort(-np.abs(scores))
    top = [tuple(int(x) for x in names[i][1:].split("H")) for i in order[:k]]
    digest = topk_digest(names, np.array(scores), k=k, label="head (direct logit-diff attribution)")
    digest += f" | top-|DLA| heads={top}"
    return digest, top


def _headwise_grad_and_acts(handle: adapter.ModelHandle, prompt: str, layers: list[int],
                             io_id: int, s_id: int, ablate_heads: list[tuple[int, int]],
                             ablate_mode: str = "mean"):
    """One forward+backward on `prompt` with `ablate_heads` (mean/zero) ablated.
    Returns, per layer: d(logit_diff)/d(out_proj input) and the (post-ablation)
    activation. A single backward pass yields the gradient for EVERY head at
    once -- that is what makes S-EAP O(candidates), not O(candidates*heads)."""
    head_dim = adapter.get_head_dim(handle)
    out_projs = {l: adapter._find_attn_out_proj(handle.layers[l]) for l in layers}
    leaves: dict[int, torch.Tensor] = {}
    batch = handle.tokenizer([prompt], return_tensors="pt").to(handle.device)

    def make_hook(l):
        def pre_hook(module, args, kwargs):
            x = (args[0] if args else kwargs["input"]).clone()
            for (al, ah) in ablate_heads:
                if al != l:
                    continue
                lo, hi = ah * head_dim, (ah + 1) * head_dim
                if ablate_mode == "zero":
                    x[:, :, lo:hi] = 0.0
                else:
                    x[:, :, lo:hi] = x[:, :, lo:hi].mean(dim=(0, 1), keepdim=True)
            # fresh leaf: d(metric)/d(this activation) at THIS run, same pattern as run_eap
            x = x.detach().requires_grad_(True)
            leaves[l] = x
            if args:
                return (x,) + args[1:], kwargs
            kwargs["input"] = x
            return args, kwargs
        return pre_hook

    with adapter.MODEL_LOCK:
        hs = [out_projs[l].register_forward_pre_hook(make_hook(l), with_kwargs=True) for l in layers]
        try:
            logits = handle.model(**batch).logits
            metric = logits[0, -1, io_id] - logits[0, -1, s_id]
            grads = torch.autograd.grad(metric, [leaves[l] for l in layers], allow_unused=True)
        finally:
            for h in hs:
                h.remove()

    gmap, amap = {}, {}
    for l, g in zip(layers, grads):
        amap[l] = leaves[l].detach()
        gmap[l] = g.detach() if g is not None else torch.zeros_like(amap[l])
    return gmap, amap


def run_synergy_eap(handle, clean_prompt, corrupted_prompt, io_token, s_token, layer_range,
                    n_heads, ablate_candidates=None, ablate_mode="zero", k=10):
    from .colab_causal import synergy
    if ablate_mode != "zero":
        raise ValueError("Use the explicit zero-ablation S-EAP contract")
    return synergy(handle, clean_prompt, corrupted_prompt, io_token, s_token, layer_range,
                   n_heads, ablate_candidates, k)


def get_attention_pattern(handle: adapter.ModelHandle, layer_idx: int, head_idx: int,
                           prompt: str, target_word_idx: int, source_word_idx: int) -> str:
    """Tool: get_attention_pattern(). Full attention matrix for one head;
    reports mass from a source token position to a target token position
    (e.g. IO-token -> final-token attention, indicating a name-mover head)."""
    if getattr(handle, "attn_impl", "eager") != "eager":
        raise RuntimeError(
            f"get_attention_pattern needs eager attention, which is numerically broken for "
            f"{handle.model_id} on this transformers build (adapter fell back to "
            f"{handle.attn_impl!r}); skip attention-pattern analysis for this model")
    batch = handle.tokenizer([prompt], return_tensors="pt").to(handle.device)
    with adapter.MODEL_LOCK, torch.no_grad():
        out = handle.model(**batch, output_attentions=True)
    if not out.attentions:
        raise RuntimeError(f"{handle.model_id} returned no attention weights "
                           f"(attn_impl={getattr(handle, 'attn_impl', '?')})")
    attn = out.attentions[layer_idx][0, head_idx]  # [seq, seq]
    mass = attn[target_word_idx, source_word_idx].item()
    from .digest import attention_digest
    return attention_digest(f"L{layer_idx}H{head_idx}", mass,
                             f"token[{source_word_idx}] as seen from token[{target_word_idx}]")


def get_max_activating_examples(handle: adapter.ModelHandle, layer_idx: int, head_idx: int,
                                 dataset: list[str], k: int = 3) -> str:
    """Tool: get_max_activating_examples(). Ranks a prompt set by this head's
    output-projection-input norm at the last token -- top-k inputs that drive
    the head hardest."""
    out_proj = adapter._find_attn_out_proj(handle.layers[layer_idx])
    head_dim = adapter.get_head_dim(handle)
    lo, hi = head_idx * head_dim, (head_idx + 1) * head_dim
    scores = []
    for text in dataset:
        batch = handle.tokenizer([text], return_tensors="pt").to(handle.device)
        store = {}

        def hook(module, args, kwargs):
            x = args[0] if args else kwargs["input"]
            store["norm"] = x[:, -1, lo:hi].norm().item()
        with adapter.MODEL_LOCK:
            h = out_proj.register_forward_pre_hook(hook, with_kwargs=True)
            with torch.no_grad():
                handle.model(**batch)
            h.remove()
        scores.append((text, store["norm"]))
    scores.sort(key=lambda t: -t[1])
    top = scores[:k]
    return f"max_activating_examples L{layer_idx}H{head_idx}: " + "; ".join(f"{t!r}={v:.2f}" for t, v in top)


def activation_patch(handle: adapter.ModelHandle, layer_idx: int, head_idx: int,
                      clean_prompt: str, corrupted_prompt: str, io_token: str, s_token: str) -> str:
    """Tool: activation_patch(). Fine-grained path patching for one head --
    thin wrapper over adapter.run_with_head_patch for standalone use outside
    run_acdc's sweep."""
    metric_fn = _ioi_metric_fn(handle, io_token, s_token)
    ci = handle.tokenizer([clean_prompt], return_tensors="pt").to(handle.device)
    xi = handle.tokenizer([corrupted_prompt], return_tensors="pt").to(handle.device)
    with adapter.MODEL_LOCK, torch.no_grad():
        clean_m = metric_fn(handle.model(**ci).logits)
        corr_m = metric_fn(handle.model(**xi).logits)
    patched_m = adapter.run_with_head_patch(handle, layer_idx, head_idx, ci["input_ids"],
                                             ci["attention_mask"], xi["input_ids"], xi["attention_mask"], metric_fn)
    denom = clean_m - corr_m
    frac = (patched_m - corr_m) / denom if abs(denom) > 1e-8 else float("nan")
    return f"activation_patch L{layer_idx}H{head_idx}: fraction_recovered={frac:.2f}"
