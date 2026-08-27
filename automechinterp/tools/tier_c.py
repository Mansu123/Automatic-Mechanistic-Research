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
    io_id = handle.tokenizer.encode(" " + io_token.strip())[0]
    s_id = handle.tokenizer.encode(" " + s_token.strip())[0]

    def metric_fn(logits: torch.Tensor) -> float:
        last = logits[0, -1]
        return float((last[io_id] - last[s_id]).item())
    return metric_fn


def run_acdc(handle: adapter.ModelHandle, clean_prompt: str, corrupted_prompt: str,
             io_token: str, s_token: str, layer_range: range, n_heads: int,
             threshold: float = 0.10) -> tuple[str, list[tuple[int, int]]]:
    """Tool: run_acdc(). Simplified ACDC: patch each (layer, head) from the
    clean run into the corrupted run one at a time; keep heads whose patch
    recovers > `threshold` of the clean-minus-corrupted metric gap. Returns
    the minimal circuit as a (layer, head) list plus a text digest."""
    metric_fn = _ioi_metric_fn(handle, io_token, s_token)
    ci = handle.tokenizer([clean_prompt], return_tensors="pt").to(handle.device)
    xi = handle.tokenizer([corrupted_prompt], return_tensors="pt").to(handle.device)
    with adapter.MODEL_LOCK, torch.no_grad():
        clean_m = metric_fn(handle.model(**ci).logits)
        corr_m = metric_fn(handle.model(**xi).logits)
    denom = clean_m - corr_m

    names, scores = [], []
    circuit = []
    for layer_idx in layer_range:
        for head_idx in range(n_heads):
            patched_m = adapter.run_with_head_patch(
                handle, layer_idx, head_idx,
                ci["input_ids"], ci["attention_mask"], xi["input_ids"], xi["attention_mask"], metric_fn)
            frac = (patched_m - corr_m) / denom if abs(denom) > 1e-8 else 0.0
            names.append(f"L{layer_idx}H{head_idx}")
            scores.append(frac)
            if frac >= threshold:
                circuit.append((layer_idx, head_idx))

    import numpy as np
    digest = topk_digest(names, np.array(scores), k=8, label="head (fraction of gap recovered)")
    digest += f" | circuit(threshold={threshold})={circuit}"
    return digest, circuit


def run_eap(handle: adapter.ModelHandle, clean_prompt: str, corrupted_prompt: str,
            io_token: str, s_token: str, layer_range: range, n_heads: int) -> str:
    """Tool: run_eap(). Edge Attribution Patching approximation: gradient of
    the metric w.r.t. each head's out-projection input slice, evaluated at
    the corrupted activation, times (clean - corrupted) activation diff --
    the standard linear-approximation EAP score, avoiding one forward pass
    per head (unlike run_acdc, which is exact but O(layers*heads))."""
    metric_fn = _ioi_metric_fn(handle, io_token, s_token)
    ci = handle.tokenizer([clean_prompt], return_tensors="pt").to(handle.device)
    xi = handle.tokenizer([corrupted_prompt], return_tensors="pt").to(handle.device)
    head_dim = adapter.get_head_dim(handle)

    names, scores = [], []
    for layer_idx in layer_range:
        out_proj = adapter._find_attn_out_proj(handle.layers[layer_idx])
        captured = {}

        def cap_clean(module, args, kwargs):
            captured["clean"] = (args[0] if args else kwargs["input"]).detach().clone()

        grads = {}

        def cap_corr_grad(module, args, kwargs):
            # detach into a fresh leaf tensor: we want d(metric)/d(this activation)
            # evaluated at the corrupted run, not a gradient w.r.t. earlier params
            x = (args[0] if args else kwargs["input"]).detach().clone().requires_grad_(True)
            grads["x"] = x
            if args:
                return (x,) + args[1:], kwargs
            kwargs["input"] = x
            return args, kwargs

        with adapter.MODEL_LOCK:
            h = out_proj.register_forward_pre_hook(cap_clean, with_kwargs=True)
            with torch.no_grad():
                handle.model(**ci)
            h.remove()

            h = out_proj.register_forward_pre_hook(cap_corr_grad, with_kwargs=True)
            logits = handle.model(**xi).logits
            io_id = handle.tokenizer.encode(" " + io_token.strip())[0]
            s_id = handle.tokenizer.encode(" " + s_token.strip())[0]
            metric_t = logits[0, -1, io_id] - logits[0, -1, s_id]
            # torch.autograd.grad (rather than .backward() + reading .grad) gets the
            # gradient w.r.t. this specific intermediate tensor directly -- it doesn't
            # depend on leaf-tensor bookkeeping, which is the more fragile path here.
            (grad,) = torch.autograd.grad(metric_t, grads["x"], retain_graph=False, allow_unused=False)
            grad = grad.detach()
            h.remove()

        min_len = min(grad.shape[1], captured["clean"].shape[1], grads["x"].shape[1])
        clean_slice = captured["clean"][:, -min_len:, :]
        corr_slice = grads["x"].detach()[:, -min_len:, :]
        g_slice = grad[:, -min_len:, :]
        diff = clean_slice - corr_slice
        for head_idx in range(n_heads):
            lo, hi = head_idx * head_dim, (head_idx + 1) * head_dim
            score = (g_slice[:, :, lo:hi] * diff[:, :, lo:hi]).sum().item()
            names.append(f"L{layer_idx}H{head_idx}")
            scores.append(score)

    import numpy as np
    return topk_digest(names, np.array(scores), k=8, label="head (EAP linear-approx score)")


def get_attention_pattern(handle: adapter.ModelHandle, layer_idx: int, head_idx: int,
                           prompt: str, target_word_idx: int, source_word_idx: int) -> str:
    """Tool: get_attention_pattern(). Full attention matrix for one head;
    reports mass from a source token position to a target token position
    (e.g. IO-token -> final-token attention, indicating a name-mover head)."""
    batch = handle.tokenizer([prompt], return_tensors="pt").to(handle.device)
    with adapter.MODEL_LOCK, torch.no_grad():
        out = handle.model(**batch, output_attentions=True)
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
