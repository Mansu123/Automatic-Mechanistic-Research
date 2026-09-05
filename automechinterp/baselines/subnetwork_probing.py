"""Subnetwork Probing baseline for circuit discovery.

Method: Subnetwork Probing (Cao et al. 2021 / de Cao et al. 2022).

Optimizes a continuous mask m ∈ [0,1]^(L×H) applied to each attention
head's output-projection input slice. The circuit is the set of heads
where m > threshold at convergence.

Objective:
    L = -metric(masked_forward) + λ * ||m||_1

Uses Adam gradient descent through the model via the standard pre-hook
infrastructure in adapter.py. No external library required.
"""
from __future__ import annotations

import time
from typing import Callable

import torch
import torch.nn.functional as F

from ..tools import adapter
from . import MethodResult

_DEFAULT_N_STEPS   = 200
_DEFAULT_LR        = 0.03
_DEFAULT_LAMBDA_L1 = 0.01
_DEFAULT_THRESHOLD = 0.5


def _build_mask_metric(
    handle: adapter.ModelHandle,
    mask: torch.Tensor,          # [n_layers, n_heads], requires_grad
    clean_input_ids: torch.Tensor,
    clean_attention_mask: torch.Tensor,
    io_id: int,
    s_id: int,
    head_dim: int,
) -> torch.Tensor:
    """One differentiable forward pass with soft head-masking applied.

    For each head (l, h): scales the out-projection input slice by sigmoid(mask[l,h]).
    A mask value >> 0 keeps the head; << 0 suppresses it.
    Returns the logit-diff metric as a differentiable scalar.
    """
    n_layers, n_heads = mask.shape
    sigmoid_mask = torch.sigmoid(mask)   # [n_layers, n_heads]

    hooks = []

    def make_hook(layer_idx: int):
        def pre_hook(module, args, kwargs):
            x = (args[0] if args else kwargs["input"])
            # Build a per-head scale vector [n_heads*head_dim] that is
            # differentiable w.r.t. mask.  Use repeat_interleave (no in-place
            # ops) so the autograd graph stays intact through GPT-2's Conv1D
            # (which uses as_strided internally — any in-place write on its
            # output causes "modified by an inplace operation" errors).
            scale = sigmoid_mask[layer_idx].repeat_interleave(head_dim)  # [n_heads*head_dim]
            # Broadcast over batch and sequence: [1, 1, n_heads*head_dim]
            x_scaled = x * scale.view(1, 1, -1)
            if args:
                return (x_scaled,) + args[1:], kwargs
            kwargs["input"] = x_scaled
            return args, kwargs
        return pre_hook

    # Do NOT hold MODEL_LOCK across a backward() call — that deadlocks when
    # other threads try to acquire it.  Subnetwork Probing trains a single
    # mask sequentially, so concurrent access is not a concern here.
    for l in range(n_layers):
        proj = adapter._find_attn_out_proj(handle.layers[l])
        h = proj.register_forward_pre_hook(make_hook(l), with_kwargs=True)
        hooks.append(h)
    try:
        logits = handle.model(input_ids=clean_input_ids,
                               attention_mask=clean_attention_mask).logits
    finally:
        for h in hooks:
            h.remove()

    return logits[0, -1, io_id] - logits[0, -1, s_id]


def run_subnetwork_probing(
    handle: adapter.ModelHandle,
    task: dict,
    layer_range: range | None = None,
    n_heads: int | None = None,
    n_steps: int = _DEFAULT_N_STEPS,
    lr: float = _DEFAULT_LR,
    lambda_l1: float = _DEFAULT_LAMBDA_L1,
    threshold: float = _DEFAULT_THRESHOLD,
    **kwargs,
) -> MethodResult:
    """Run Subnetwork Probing circuit discovery on `task`.

    Args:
        handle:      ModelHandle for the target model.
        task:        Task dict (same schema as behaviors.py / task_suites).
        layer_range: Layers to search over (default: all layers).
        n_heads:     Number of attention heads (default: from task["n_heads"]).
        n_steps:     Gradient-descent iterations.
        lr:          Adam learning rate.
        lambda_l1:   L1 sparsity coefficient.
        threshold:   sigmoid(mask) > threshold → head is in circuit.

    Returns:
        MethodResult with discovered circuit, metric recovery, runtime, etc.
    """
    t0 = time.time()

    n_heads     = n_heads or task["n_heads"]
    layer_range = layer_range or range(handle.n_layers)
    layers      = list(layer_range)
    n_layers    = len(layers)

    cp  = task["clean_prompt"]
    xp  = task["corrupted_prompt"]
    io_t, s_t = task["io_token"], task["s_token"]

    # [-1] not [0]: cross-tokenizer safety (see behaviors.py _make_task)
    io_id = handle.tokenizer.encode(" " + io_t.strip())[-1]
    s_id  = handle.tokenizer.encode(" " + s_t.strip())[-1]

    ci = handle.tokenizer([cp], return_tensors="pt").to(handle.device)
    xi = handle.tokenizer([xp], return_tensors="pt").to(handle.device)
    head_dim = adapter.get_head_dim(handle)

    # Measure baseline metric gap (no masking)
    with adapter.MODEL_LOCK, torch.no_grad():
        clean_logits = handle.model(**ci).logits[0, -1]
        corr_logits  = handle.model(**xi).logits[0, -1]
    clean_m = float((clean_logits[io_id] - clean_logits[s_id]).item())
    corr_m  = float((corr_logits[io_id]  - corr_logits[s_id]).item())
    denom   = clean_m - corr_m

    # Initialise mask at 0 (sigmoid → 0.5, all heads equally weighted)
    # We only mask the layers we're searching; others pass through unchanged.
    # Use a full [n_layers_model, n_heads] mask indexed by actual layer index.
    mask = torch.zeros(handle.n_layers, n_heads, dtype=torch.float32,
                        device=handle.device, requires_grad=False)
    # Only the layers in layer_range are optimized; fix others at large positive
    # (sigmoid ≈ 1, i.e. full pass-through) so they don't interfere.
    mask_data = torch.full((handle.n_layers, n_heads), fill_value=5.0,
                            device=handle.device)
    for l in layers:
        mask_data[l, :] = 0.0
    mask_param = mask_data.clone().requires_grad_(True)

    optimizer = torch.optim.Adam([mask_param], lr=lr)
    step_count = 0

    # Precompute a frozen tensor: large positive for non-search layers (sigmoid≈1),
    # zero for search layers. Each step: effective = frozen_offset + mask_param * search_mask
    # This keeps gradients flowing only through mask_param entries for search layers.
    search_mask = torch.zeros(handle.n_layers, n_heads,
                               dtype=torch.float32, device=handle.device)
    frozen_offset = torch.full((handle.n_layers, n_heads), fill_value=5.0,
                                device=handle.device)
    for l in layers:
        search_mask[l, :] = 1.0
        frozen_offset[l, :] = 0.0
    # frozen_offset and search_mask are non-leaf tensors with no grad

    for step in range(n_steps):
        optimizer.zero_grad()

        # effective_mask[l,h] = mask_param[l,h] for search layers (grad flows),
        #                      = 5.0            for others (frozen, sigmoid≈1)
        effective_mask = frozen_offset + mask_param * search_mask

        metric_t = _build_mask_metric(handle, effective_mask,
                                       ci["input_ids"], ci["attention_mask"],
                                       io_id, s_id, head_dim)

        # Maximize metric (keep important heads) + L1 on sigmoid values (sparsity)
        sig_vals = torch.sigmoid(effective_mask[layers, :])
        loss = -metric_t + lambda_l1 * sig_vals.sum()
        loss.backward()
        optimizer.step()
        step_count += 1

    # Extract circuit: heads where sigmoid(mask) > threshold
    with torch.no_grad():
        final_sig = torch.sigmoid(mask_param)
    circuit = []
    for l in layers:
        for h in range(n_heads):
            if float(final_sig[l, h].item()) > threshold:
                circuit.append((l, h))

    # Measure metric recovery with discovered circuit only (ablate all others)
    all_heads = [(l, h) for l in range(handle.n_layers) for h in range(n_heads)]
    complement = [lh for lh in all_heads if lh not in set(circuit)]

    if circuit and abs(denom) > 1e-8:
        ablated_m = float(adapter.ablate_head_set(
            handle, complement,
            ci["input_ids"], ci["attention_mask"],
            lambda logits: float((logits[0, -1, io_id] - logits[0, -1, s_id]).item()),
        ))
        metric_recovery = (ablated_m - corr_m) / denom
    else:
        metric_recovery = 0.0

    runtime = time.time() - t0

    return MethodResult(
        method="subnetwork_probing",
        circuit=circuit,
        circuit_score=float(final_sig[layers, :].max().item()),
        metric_recovery=metric_recovery,
        tool_calls=step_count,
        runtime_s=runtime,
        metadata={
            "n_steps": n_steps, "lr": lr, "lambda_l1": lambda_l1,
            "threshold": threshold, "clean_m": clean_m, "corr_m": corr_m,
            "final_mask_max": float(final_sig[layers, :].max().item()),
            "final_mask_min": float(final_sig[layers, :].min().item()),
            "sparsity": float((final_sig[layers, :] > threshold).float().mean().item()),
        },
    )
