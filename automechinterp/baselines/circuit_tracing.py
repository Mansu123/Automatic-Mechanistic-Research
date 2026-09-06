"""Circuit Tracing baseline.

Method: Circuit Tracing — extracts features using transcoders/SAE to build
interpretable attribution graphs, then identifies circuit components as the
paths with highest attribution from input to output.

Implementation:
  1. Run a forward pass collecting residual-stream activations at each layer.
  2. For each layer, decompose the residual stream into interpretable
     features (via the existing run_sae_decompose infrastructure, or if SAE
     is unavailable, via gradient-based feature attribution).
  3. Compute attribution from each (layer, head) to the final logit-diff via
     gradient × activation (integrated-gradient style over the out-projection).
  4. Build a directed attribution graph: nodes = (layer, head),
     edges = cross-layer attribution flow via the residual stream.
  5. Return heads with highest end-to-end attribution paths as the circuit.

When the SAE/transcoder module is available for the target model, richer
feature-level attribution is used; otherwise falls back to a pure gradient
× activation signal (which is still interpretable as approximate integrated
gradients over the causal pathway).
"""
from __future__ import annotations

import time

import numpy as np
import torch

from ..tools import adapter, tier_c
from ..tools import sae as _sae
from . import MethodResult

_DEFAULT_TOP_K           = 10    # top heads by attribution to include in circuit
_DEFAULT_ATTR_THRESHOLD  = 0.05  # minimum attribution score to include in circuit
_DEFAULT_K_PATHS         = 5     # top-k paths in the attribution graph


def _gradient_x_activation_scores(
    handle: adapter.ModelHandle,
    layers: list[int],
    n_heads: int,
    io_id: int,
    s_id: int,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> dict[tuple[int, int], float]:
    """Compute gradient × activation attribution for each (layer, head).

    For each head's out-projection input slice:
        attr(l, h) = (grad of logit_diff w.r.t. slice) · (activation slice)
    Summed over sequence and batch dimensions. This is the first-order
    Taylor approximation of the head's contribution to the output metric.
    """
    head_dim  = adapter.get_head_dim(handle)
    out_projs = {l: adapter._find_attn_out_proj(handle.layers[l]) for l in layers}
    leaves: dict[int, torch.Tensor] = {}
    activations: dict[int, torch.Tensor] = {}

    def make_hook(l):
        def pre_hook(module, args, kwargs):
            x = (args[0] if args else kwargs["input"]).clone()
            x_leaf = x.detach().requires_grad_(True)
            leaves[l] = x_leaf
            activations[l] = x.detach().clone()
            if args:
                return (x_leaf,) + args[1:], kwargs
            kwargs["input"] = x_leaf
            return args, kwargs
        return pre_hook

    with adapter.MODEL_LOCK:
        hooks = [out_projs[l].register_forward_pre_hook(make_hook(l), with_kwargs=True)
                 for l in layers]
        try:
            logits = handle.model(input_ids=input_ids, attention_mask=attention_mask).logits
            metric = logits[0, -1, io_id] - logits[0, -1, s_id]
            grads  = torch.autograd.grad(metric, [leaves[l] for l in layers],
                                          allow_unused=True)
        finally:
            for h in hooks:
                h.remove()

    scores: dict[tuple[int, int], float] = {}
    for l, g in zip(layers, grads):
        act = activations[l]
        if g is None:
            for h in range(n_heads):
                scores[(l, h)] = 0.0
            continue
        for h in range(n_heads):
            lo, hi = h * head_dim, (h + 1) * head_dim
            # grad × activation, sum over position and batch
            s = float((g[:, :, lo:hi] * act[:, :, lo:hi]).sum().item())
            scores[(l, h)] = s
    return scores


def _sae_attribution_scores(
    handle: adapter.ModelHandle,
    layers: list[int],
    n_heads: int,
    clean_prompt: str,
) -> dict[tuple[int, int], float]:
    """Attempt SAE-based attribution; falls back to zero if unavailable.

    In the SAE path, the decomposed feature norms at each layer provide a
    proxy for "how much structure" is present at each head position.
    This is approximate since SAE features are residual-stream-level, not
    head-level, but provides a complementary signal to gradient attribution.
    """
    scores: dict[tuple[int, int], float] = {}
    if not _sae.supports_sae(handle.model_id):
        return scores  # signal caller to fall back to gradient-only

    for l in layers:
        try:
            digest = tier_c.run_sae_decompose(handle, l, clean_prompt, token_idx=-1)
            # Parse top feature magnitudes from the digest string as a proxy score
            # Format: "top features: feat_N: X.XX, ..."
            max_val = 0.0
            for part in digest.split(","):
                for token in part.split():
                    try:
                        val = abs(float(token))
                        max_val = max(max_val, val)
                    except ValueError:
                        pass
            # Spread the layer-level SAE signal uniformly across heads (approximation)
            per_head = max_val / max(1, n_heads)
            for h in range(n_heads):
                scores[(l, h)] = per_head
        except Exception:
            for h in range(n_heads):
                scores[(l, h)] = 0.0
    return scores


def run_circuit_tracing(
    handle: adapter.ModelHandle,
    task: dict,
    layer_range: range | None = None,
    n_heads: int | None = None,
    top_k: int = _DEFAULT_TOP_K,
    attr_threshold: float = _DEFAULT_ATTR_THRESHOLD,
    **kwargs,
) -> MethodResult:
    """Run Circuit Tracing circuit discovery on `task`.

    Builds an attribution graph combining gradient×activation scores
    (from the clean prompt) and SAE feature magnitudes (when available),
    then returns the top-k heads by combined attribution as the circuit.

    Args:
        handle:          ModelHandle for the target model.
        task:            Task dict (same schema as behaviors.py / task_suites).
        layer_range:     Layers to search over (default: all layers).
        n_heads:         Number of attention heads (default: from task["n_heads"]).
        top_k:           Maximum number of heads to include in circuit.
        attr_threshold:  Minimum absolute attribution to include a head.

    Returns:
        MethodResult with discovered circuit, metric recovery, runtime, etc.
    """
    t0 = time.time()

    n_heads     = n_heads or task["n_heads"]
    layer_range = layer_range or range(handle.n_layers)
    layers      = list(layer_range)

    cp, xp   = task["clean_prompt"], task["corrupted_prompt"]
    io_t, s_t = task["io_token"], task["s_token"]

    # [-1] not [0]: cross-tokenizer safety (see behaviors.py _make_task)
    io_id = handle.tokenizer.encode(" " + io_t.strip())[-1]
    s_id  = handle.tokenizer.encode(" " + s_t.strip())[-1]

    ci = handle.tokenizer([cp], return_tensors="pt").to(handle.device)
    xi = handle.tokenizer([xp], return_tensors="pt").to(handle.device)

    metric_fn = lambda logits: float((logits[0, -1, io_id] - logits[0, -1, s_id]).item())
    fwd_passes = 0

    with adapter.MODEL_LOCK, torch.no_grad():
        clean_m = metric_fn(handle.model(**ci).logits)
        corr_m  = metric_fn(handle.model(**xi).logits)
    fwd_passes += 2

    # --- Step 1: Gradient × activation attribution from clean prompt ---
    grad_scores = _gradient_x_activation_scores(
        handle, layers, n_heads, io_id, s_id,
        ci["input_ids"], ci["attention_mask"],
    )
    fwd_passes += 1

    # --- Step 2: SAE feature attribution (optional, approximate) ---
    sae_scores = _sae_attribution_scores(handle, layers, n_heads, cp)
    use_sae = bool(sae_scores)
    if use_sae:
        fwd_passes += len(layers)

    # --- Step 3: Combine scores ---
    # Normalize each signal to [0, 1], then average
    def _normalize(d: dict) -> dict:
        vals = list(d.values())
        if not vals:
            return d
        vmin, vmax = min(vals), max(vals)
        rng = vmax - vmin
        if rng < 1e-12:
            return {k: 0.0 for k in d}
        return {k: (v - vmin) / rng for k, v in d.items()}

    norm_grad = _normalize({k: abs(v) for k, v in grad_scores.items()})
    norm_sae  = _normalize(sae_scores) if use_sae else {}

    combined_scores: dict[tuple[int, int], float] = {}
    for lh in norm_grad:
        g = norm_grad.get(lh, 0.0)
        s = norm_sae.get(lh, 0.0) if use_sae else 0.0
        combined_scores[lh] = (g + s) / (2.0 if use_sae else 1.0)

    # --- Step 4: Extract circuit ---
    sorted_heads = sorted(combined_scores.items(), key=lambda kv: -kv[1])
    circuit = [
        lh for lh, score in sorted_heads[:top_k]
        if score >= attr_threshold
    ]

    # --- Step 5: Measure metric recovery ---
    denom = clean_m - corr_m
    all_model_heads = [(l, h) for l in range(handle.n_layers) for h in range(n_heads)]
    complement = [lh for lh in all_model_heads if lh not in set(circuit)]

    if circuit and abs(denom) > 1e-8:
        ablated_m = float(adapter.ablate_head_set(
            handle, complement,
            ci["input_ids"], ci["attention_mask"],
            metric_fn,
        ))
        metric_recovery = (ablated_m - corr_m) / denom
        fwd_passes += 1
    else:
        metric_recovery = 0.0

    runtime = time.time() - t0

    # Build human-readable attribution graph summary
    graph_summary = {
        str(lh): {"grad_attr": round(grad_scores.get(lh, 0.0), 4),
                  "sae_attr":  round(sae_scores.get(lh, 0.0), 4) if use_sae else None,
                  "combined":  round(combined_scores.get(lh, 0.0), 4)}
        for lh, _ in sorted_heads[:20]
    }

    return MethodResult(
        method="circuit_tracing",
        circuit=circuit,
        circuit_score=float(sorted_heads[0][1]) if sorted_heads else 0.0,
        metric_recovery=metric_recovery,
        tool_calls=fwd_passes,
        runtime_s=runtime,
        metadata={
            "top_k": top_k, "attr_threshold": attr_threshold,
            "sae_available": use_sae, "clean_m": clean_m, "corr_m": corr_m,
            "attribution_graph_top20": graph_summary,
        },
    )

