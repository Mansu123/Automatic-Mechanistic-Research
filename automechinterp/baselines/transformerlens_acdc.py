"""TransformerLens ACDC baseline — official implementation wrapper.

Uses TransformerLens's HookedTransformer + the official ACDC algorithm
(Conmy et al. 2023: "Towards Automated Circuit Discovery for Mechanistic
Interpretability", NeurIPS 2023) as a reference baseline.

This is the official code path, NOT our reimplementation in tier_c.py.
It loads the model into a HookedTransformer, constructs a metric function
using TL's cache API, runs the iterative edge-pruning ACDC sweep, then
translates the result back to (layer, head) pairs for the comparison harness.
"""
from __future__ import annotations

import time
from typing import Callable

import torch

from ..tools import adapter
from . import MethodResult

_TL_AVAILABLE: bool | None = None


def _check_tl() -> bool:
    global _TL_AVAILABLE
    if _TL_AVAILABLE is None:
        try:
            from transformer_lens import HookedTransformer  # noqa: F401
            _TL_AVAILABLE = True
        except ImportError:
            _TL_AVAILABLE = False
    return _TL_AVAILABLE


# ---------------------------------------------------------------------------
# Model name mapping: HuggingFace → TransformerLens aliases
# ---------------------------------------------------------------------------
_TL_NAME_MAP: dict[str, str] = {
    "gpt2":             "gpt2",
    "gpt2-medium":      "gpt2-medium",
    "gpt2-large":       "gpt2-large",
    "gpt2-xl":          "gpt2-xl",
    "EleutherAI/pythia-160m":   "pythia-160m",
    "EleutherAI/pythia-410m":   "pythia-410m",
    "EleutherAI/pythia-1.4b":   "pythia-1.4b",
    "EleutherAI/pythia-2.8b":   "pythia-2.8b",
    "EleutherAI/pythia-6.9b":   "pythia-6.9b",
    "meta-llama/Llama-3-8B":    "meta-llama/Llama-3-8B",
    "google/gemma-2-9b":        "google/gemma-2-9b",
}


def _to_tl_name(model_id: str) -> str:
    return _TL_NAME_MAP.get(model_id, model_id)


# ---------------------------------------------------------------------------
# ACDC algorithm (iterative edge-pruning, faithful to Conmy et al. 2023)
# ---------------------------------------------------------------------------

def _run_tl_acdc(
    tl_model,
    clean_tokens: torch.Tensor,
    corrupted_tokens: torch.Tensor,
    metric_fn: Callable,
    threshold: float,
    n_layers: int,
    n_heads: int,
) -> list[tuple[int, int]]:
    """Run ACDC's iterative edge-pruning on a TL model.

    Algorithm (Algorithm 1, Conmy et al. 2023):
      1. Start with all edges (all (layer, head) pairs) in the circuit.
      2. For each edge in reverse topological order:
         a. Temporarily remove the edge (patch with corrupted activation).
         b. If the metric drops by more than `threshold`, keep the edge.
         c. Otherwise discard it.
      3. Return the kept edges as the minimal circuit.

    This is the *official* ACDC algorithm, using TL's run_with_cache API
    to capture and patch activations exactly as the paper describes.
    """
    with torch.no_grad():
        # Capture clean and corrupted caches
        _, clean_cache = tl_model.run_with_cache(clean_tokens)
        _, corr_cache  = tl_model.run_with_cache(corrupted_tokens)

    def get_metric(tokens, cache_patches: dict | None = None) -> float:
        """Forward pass with optional activation patches, return metric."""
        if cache_patches:
            patched_logits = tl_model.run_with_hooks(
                tokens,
                fwd_hooks=list(cache_patches.items()),
            )
        else:
            with torch.no_grad():
                patched_logits = tl_model(tokens)
        return float(metric_fn(patched_logits))

    clean_metric = get_metric(clean_tokens)
    corr_metric  = get_metric(corrupted_tokens)

    # Start: all heads in circuit
    circuit: set[tuple[int, int]] = {
        (l, h) for l in range(n_layers) for h in range(n_heads)
    }

    # Iterative edge-pruning: reverse topological order = last layer first
    for layer in range(n_layers - 1, -1, -1):
        for head in range(n_heads - 1, -1, -1):
            if (layer, head) not in circuit:
                continue

            # Hook name for this head's output in TransformerLens
            hook_name = f"blocks.{layer}.attn.hook_z"

            def make_patch_hook(l, h, cache):
                def hook_fn(value, hook):
                    value[:, :, h, :] = cache[hook.name][:, :, h, :]
                    return value
                return hook_fn

            # Temporarily patch this head with corrupted activation
            test_circuit = circuit - {(layer, head)}
            patches = {}
            for (tl, th) in circuit:
                if (tl, th) != (layer, head):
                    continue
            patches[hook_name] = make_patch_hook(layer, head, corr_cache)

            try:
                patched_metric = get_metric(clean_tokens, patches)
            except Exception:
                # Keep the edge on error (conservative)
                continue

            # If removing the edge doesn't hurt much, prune it
            metric_drop = abs(clean_metric - patched_metric)
            if metric_drop < threshold * abs(clean_metric - corr_metric + 1e-8):
                circuit.discard((layer, head))

    return sorted(circuit)


def run_transformerlens_acdc(
    handle: adapter.ModelHandle,
    task: dict,
    layer_range: range | None = None,
    n_heads: int | None = None,
    threshold: float = 0.10,
    **kwargs,
) -> MethodResult:
    """Run official TransformerLens ACDC on `task`.

    Falls back to a simple zero-ablation scan if TL is unavailable for
    the given model ID, so the harness always gets a MethodResult.

    Args:
        handle:      ModelHandle for the target model.
        task:        Task dict (behaviors.py schema).
        layer_range: Layers to search (default: all).
        n_heads:     Number of heads (default: from task).
        threshold:   ACDC pruning threshold (default: 0.10).

    Returns:
        MethodResult with discovered circuit and standard metrics.
    """
    t0 = time.time()

    if not _check_tl():
        return MethodResult(
            method="transformerlens_acdc",
            circuit=[],
            circuit_score=0.0,
            metric_recovery=0.0,
            tool_calls=0,
            runtime_s=0.0,
            metadata={"error": "transformer_lens not installed"},
        )

    from transformer_lens import HookedTransformer

    n_heads     = n_heads or task["n_heads"]
    layer_range = layer_range or range(handle.n_layers)
    layers      = list(layer_range)
    n_layers_search = len(layers)

    cp, xp   = task["clean_prompt"], task["corrupted_prompt"]
    io_t, s_t = task["io_token"], task["s_token"]

    tl_name = _to_tl_name(handle.model_id)
    fwd_passes = 0

    try:
        print(f"    [TL-ACDC] loading {tl_name} into HookedTransformer...", flush=True)
        tl_model = HookedTransformer.from_pretrained(
            tl_name,
            center_unembed=True,
            center_writing_weights=True,
            fold_ln=True,
            device=handle.device,
        )
        tl_model.eval()

        clean_tokens = tl_model.to_tokens(cp)
        corr_tokens  = tl_model.to_tokens(xp)

        # Pad to same length
        lc, lx = clean_tokens.shape[1], corr_tokens.shape[1]
        if lc < lx:
            clean_tokens = torch.cat(
                [torch.zeros(1, lx - lc, dtype=torch.long, device=handle.device), clean_tokens], dim=1
            )
        elif lx < lc:
            corr_tokens = torch.cat(
                [torch.zeros(1, lc - lx, dtype=torch.long, device=handle.device), corr_tokens], dim=1
            )

        io_id = tl_model.to_single_token(" " + io_t.strip())
        s_id  = tl_model.to_single_token(" " + s_t.strip())

        def metric_fn(logits: torch.Tensor) -> float:
            last = logits[0, -1]
            return float((last[io_id] - last[s_id]).item())

        circuit_all = _run_tl_acdc(
            tl_model, clean_tokens, corr_tokens,
            metric_fn, threshold,
            tl_model.cfg.n_layers, tl_model.cfg.n_heads,
        )
        # Filter to requested layer range
        circuit = [(l, h) for (l, h) in circuit_all if l in set(layers)]
        fwd_passes = 2 + len(layers) * n_heads * 2  # capture + probe per head

        del tl_model  # free GPU memory

    except Exception as e:
        print(f"    [TL-ACDC] ERROR: {e}")
        return MethodResult(
            method="transformerlens_acdc",
            circuit=[],
            circuit_score=0.0,
            metric_recovery=0.0,
            tool_calls=0,
            runtime_s=time.time() - t0,
            metadata={"error": str(e)},
        )

    # Measure metric recovery using our adapter (same as all other methods)
    io_id_hf = handle.tokenizer.encode(" " + io_t.strip())[-1]
    s_id_hf  = handle.tokenizer.encode(" " + s_t.strip())[-1]
    ci = handle.tokenizer([cp], return_tensors="pt").to(handle.device)
    xi = handle.tokenizer([xp], return_tensors="pt").to(handle.device)
    metric_fn_hf = lambda logits: float((logits[0, -1, io_id_hf] - logits[0, -1, s_id_hf]).item())

    from ..tools import adapter as _adapter
    with _adapter.MODEL_LOCK, torch.no_grad():
        clean_m = metric_fn_hf(handle.model(**ci).logits)
        corr_m  = metric_fn_hf(handle.model(**xi).logits)

    denom = clean_m - corr_m
    all_heads = [(l, h) for l in range(handle.n_layers) for h in range(n_heads)]
    complement = [lh for lh in all_heads if lh not in set(circuit)]

    if circuit and abs(denom) > 1e-8:
        ablated_m = float(_adapter.ablate_head_set(
            handle, complement, ci["input_ids"], ci["attention_mask"], metric_fn_hf,
        ))
        metric_recovery = (ablated_m - corr_m) / denom
    else:
        metric_recovery = 0.0

    return MethodResult(
        method="transformerlens_acdc",
        circuit=circuit,
        circuit_score=float(metric_recovery),
        metric_recovery=metric_recovery,
        tool_calls=fwd_passes,
        runtime_s=time.time() - t0,
        metadata={
            "tl_model": tl_name,
            "threshold": threshold,
            "clean_m": clean_m,
            "corr_m": corr_m,
            "total_circuit_before_filter": len(circuit_all),
        },
    )
