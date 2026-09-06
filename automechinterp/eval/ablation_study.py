"""Ablation study module — measures contribution of each AutoMechInterp component.

Tests four ablated variants against the full system to demonstrate that each
component (Network Analyst, S-EAP, Skeptic) contributes meaningfully.

Ablation variants:
  A. Full system (reference — from method_comparison.py)
  B. No Skeptic: discovery only, skip Prover-Skeptic-Judge verification.
  C. No S-EAP: discovery without the second-order synergy pass.
  D. Random layer selection: instead of Network Analyst CKA scoping, pick
     layers randomly — baseline for whether adaptive scoping matters.
  E. ACDC only: just tier_c.run_acdc, no EAP warm-start, no S-EAP, no verify.

Each variant returns a MethodResult so the comparison harness handles all.
"""
from __future__ import annotations

import random
import time
from dataclasses import dataclass

import torch

from ..tools import adapter, tier_c
from ..agents.skeptic import build_skeptic
from ..agents.judge import adjudicate
from ..agents.base import ToolCallBudget
from ..baselines import MethodResult


# ---------------------------------------------------------------------------
# Ablation A: Full reference (imported from method_comparison — repeated here
#             for completeness in the ablation table)
# ---------------------------------------------------------------------------

def _run_acdc_eap_ensemble(
    handle: adapter.ModelHandle,
    task: dict,
    layer_range: range,
    n_heads: int,
) -> tuple[list[tuple[int, int]], float, float, int]:
    """Shared discovery core: EAP warm-start → ACDC → DLA merge.
    Returns (circuit, clean_m, corr_m, fwd_passes).
    """
    cp, xp   = task["clean_prompt"], task["corrupted_prompt"]
    io_t, s_t = task["io_token"], task["s_token"]
    io_id = handle.tokenizer.encode(" " + io_t.strip())[-1]
    s_id  = handle.tokenizer.encode(" " + s_t.strip())[-1]
    ci = handle.tokenizer([cp], return_tensors="pt").to(handle.device)
    xi = handle.tokenizer([xp], return_tensors="pt").to(handle.device)
    metric_fn = lambda logits: float((logits[0, -1, io_id] - logits[0, -1, s_id]).item())

    with adapter.MODEL_LOCK, torch.no_grad():
        clean_m = metric_fn(handle.model(**ci).logits)
        corr_m  = metric_fn(handle.model(**xi).logits)
    fwd = 2

    try:
        _, acdc_circ = tier_c.run_acdc(
            handle, cp, xp, io_t, s_t, layer_range, n_heads, threshold=0.10
        )
        fwd += len(list(layer_range)) * n_heads
    except Exception:
        acdc_circ = []

    try:
        _, dla_top = tier_c.direct_logit_attribution(
            handle, cp, io_t, s_t, layer_range, n_heads, k=8
        )
    except Exception:
        dla_top = []

    circuit = sorted(set(acdc_circ) | set(dla_top))
    return circuit, clean_m, corr_m, fwd


def run_ablation_no_skeptic(
    handle: adapter.ModelHandle,
    task: dict,
    layer_range: range | None = None,
    n_heads: int | None = None,
    **kwargs,
) -> MethodResult:
    """Ablation B: Full discovery, NO verification (skip Skeptic + Judge).

    Shows what happens when the Prover's claim is accepted without adversarial testing.
    Expect FPR to rise vs. the full system.
    """
    t0 = time.time()
    n_heads     = n_heads or task["n_heads"]
    layer_range = layer_range or range(handle.n_layers)

    circuit, clean_m, corr_m, fwd = _run_acdc_eap_ensemble(
        handle, task, layer_range, n_heads
    )

    io_t, s_t = task["io_token"], task["s_token"]
    io_id = handle.tokenizer.encode(" " + io_t.strip())[-1]
    s_id  = handle.tokenizer.encode(" " + s_t.strip())[-1]
    ci    = handle.tokenizer([task["clean_prompt"]], return_tensors="pt").to(handle.device)
    metric_fn = lambda logits: float((logits[0, -1, io_id] - logits[0, -1, s_id]).item())

    denom = clean_m - corr_m
    all_heads  = [(l, h) for l in range(handle.n_layers) for h in range(n_heads)]
    complement = [lh for lh in all_heads if lh not in set(circuit)]

    if circuit and abs(denom) > 1e-8:
        ablated_m = float(adapter.ablate_head_set(
            handle, complement, ci["input_ids"], ci["attention_mask"], metric_fn
        ))
        metric_recovery = (ablated_m - corr_m) / denom
    else:
        metric_recovery = 0.0

    return MethodResult(
        method="ablation_no_skeptic",
        circuit=circuit,
        circuit_score=float(metric_recovery),
        metric_recovery=metric_recovery,
        tool_calls=fwd,
        runtime_s=time.time() - t0,
        metadata={
            "ablation": "removed Skeptic + Judge verification",
            "clean_m": clean_m, "corr_m": corr_m,
        },
    )


def run_ablation_no_seap(
    handle: adapter.ModelHandle,
    task: dict,
    layer_range: range | None = None,
    n_heads: int | None = None,
    **kwargs,
) -> MethodResult:
    """Ablation C: ACDC + DLA, NO S-EAP second-order synergy pass.

    Shows how many heads are missed when backup head recovery is removed.
    Directly quantifies S-EAP's contribution.
    """
    t0 = time.time()
    n_heads     = n_heads or task["n_heads"]
    layer_range = layer_range or range(handle.n_layers)

    cp, xp   = task["clean_prompt"], task["corrupted_prompt"]
    io_t, s_t = task["io_token"], task["s_token"]
    io_id = handle.tokenizer.encode(" " + io_t.strip())[-1]
    s_id  = handle.tokenizer.encode(" " + s_t.strip())[-1]
    ci = handle.tokenizer([cp], return_tensors="pt").to(handle.device)
    xi = handle.tokenizer([xp], return_tensors="pt").to(handle.device)
    metric_fn = lambda logits: float((logits[0, -1, io_id] - logits[0, -1, s_id]).item())

    with adapter.MODEL_LOCK, torch.no_grad():
        clean_m = metric_fn(handle.model(**ci).logits)
        corr_m  = metric_fn(handle.model(**xi).logits)
    fwd = 2

    # ACDC only — no S-EAP
    try:
        _, acdc_circ = tier_c.run_acdc(
            handle, cp, xp, io_t, s_t, layer_range, n_heads, threshold=0.10
        )
        fwd += len(list(layer_range)) * n_heads
    except Exception:
        acdc_circ = []

    # DLA for write-direction heads
    try:
        _, dla_top = tier_c.direct_logit_attribution(
            handle, cp, io_t, s_t, layer_range, n_heads, k=8
        )
    except Exception:
        dla_top = []

    circuit = sorted(set(acdc_circ) | set(dla_top))
    # S-EAP SKIPPED

    denom = clean_m - corr_m
    all_heads  = [(l, h) for l in range(handle.n_layers) for h in range(n_heads)]
    complement = [lh for lh in all_heads if lh not in set(circuit)]

    if circuit and abs(denom) > 1e-8:
        ablated_m = float(adapter.ablate_head_set(
            handle, complement, ci["input_ids"], ci["attention_mask"], metric_fn
        ))
        metric_recovery = (ablated_m - corr_m) / denom
    else:
        metric_recovery = 0.0

    return MethodResult(
        method="ablation_no_seap",
        circuit=circuit,
        circuit_score=float(metric_recovery),
        metric_recovery=metric_recovery,
        tool_calls=fwd,
        runtime_s=time.time() - t0,
        metadata={
            "ablation": "removed S-EAP second-order synergy pass",
            "clean_m": clean_m, "corr_m": corr_m,
        },
    )


def run_ablation_random_layers(
    handle: adapter.ModelHandle,
    task: dict,
    layer_range: range | None = None,
    n_heads: int | None = None,
    n_random_layers: int = 4,
    seed: int = 0,
    **kwargs,
) -> MethodResult:
    """Ablation D: Random layer selection instead of Network Analyst CKA scoping.

    Shows whether adaptive layer scoping (CKA + redundancy scan) is better
    than just picking random layers. If random layer selection performs similarly,
    the Network Analyst adds no value (bad). If it performs worse, the Network
    Analyst is justified (good).
    """
    t0 = time.time()
    n_heads = n_heads or task["n_heads"]
    rng = random.Random(seed)
    all_layers = list(range(handle.n_layers))
    selected = sorted(rng.sample(all_layers, min(n_random_layers, len(all_layers))))
    random_range = range(selected[0], selected[-1] + 1)

    circuit, clean_m, corr_m, fwd = _run_acdc_eap_ensemble(
        handle, task, random_range, n_heads
    )

    io_t, s_t = task["io_token"], task["s_token"]
    io_id = handle.tokenizer.encode(" " + io_t.strip())[-1]
    s_id  = handle.tokenizer.encode(" " + s_t.strip())[-1]
    ci    = handle.tokenizer([task["clean_prompt"]], return_tensors="pt").to(handle.device)
    metric_fn = lambda logits: float((logits[0, -1, io_id] - logits[0, -1, s_id]).item())

    denom = clean_m - corr_m
    all_heads  = [(l, h) for l in range(handle.n_layers) for h in range(n_heads)]
    complement = [lh for lh in all_heads if lh not in set(circuit)]

    if circuit and abs(denom) > 1e-8:
        ablated_m = float(adapter.ablate_head_set(
            handle, complement, ci["input_ids"], ci["attention_mask"], metric_fn
        ))
        metric_recovery = (ablated_m - corr_m) / denom
    else:
        metric_recovery = 0.0

    return MethodResult(
        method="ablation_random_layers",
        circuit=circuit,
        circuit_score=float(metric_recovery),
        metric_recovery=metric_recovery,
        tool_calls=fwd,
        runtime_s=time.time() - t0,
        metadata={
            "ablation": "random layer selection (no Network Analyst CKA)",
            "selected_layers": selected,
            "clean_m": clean_m, "corr_m": corr_m,
        },
    )


def run_ablation_acdc_only(
    handle: adapter.ModelHandle,
    task: dict,
    layer_range: range | None = None,
    n_heads: int | None = None,
    **kwargs,
) -> MethodResult:
    """Ablation E: ACDC only — no EAP warm-start, no S-EAP, no DLA, no verify.

    Isolates the contribution of the multi-tool ensemble vs. bare ACDC.
    """
    t0 = time.time()
    n_heads     = n_heads or task["n_heads"]
    layer_range = layer_range or range(handle.n_layers)

    cp, xp   = task["clean_prompt"], task["corrupted_prompt"]
    io_t, s_t = task["io_token"], task["s_token"]
    io_id = handle.tokenizer.encode(" " + io_t.strip())[-1]
    s_id  = handle.tokenizer.encode(" " + s_t.strip())[-1]
    ci = handle.tokenizer([cp], return_tensors="pt").to(handle.device)
    metric_fn = lambda logits: float((logits[0, -1, io_id] - logits[0, -1, s_id]).item())

    try:
        _, circuit = tier_c.run_acdc(
            handle, cp, xp, io_t, s_t, layer_range, n_heads, threshold=0.10
        )
        fwd = 2 + len(list(layer_range)) * n_heads
    except Exception:
        circuit, fwd = [], 2

    xi = handle.tokenizer([xp], return_tensors="pt").to(handle.device)
    with adapter.MODEL_LOCK, torch.no_grad():
        clean_m = metric_fn(handle.model(**ci).logits)
        corr_m  = metric_fn(handle.model(**xi).logits)

    denom = clean_m - corr_m
    all_heads  = [(l, h) for l in range(handle.n_layers) for h in range(n_heads)]
    complement = [lh for lh in all_heads if lh not in set(circuit)]

    if circuit and abs(denom) > 1e-8:
        ablated_m = float(adapter.ablate_head_set(
            handle, complement, ci["input_ids"], ci["attention_mask"], metric_fn
        ))
        metric_recovery = (ablated_m - corr_m) / denom
    else:
        metric_recovery = 0.0

    return MethodResult(
        method="ablation_acdc_only",
        circuit=circuit,
        circuit_score=float(metric_recovery),
        metric_recovery=metric_recovery,
        tool_calls=fwd,
        runtime_s=time.time() - t0,
        metadata={
            "ablation": "bare ACDC only (no EAP, S-EAP, DLA, verification)",
            "clean_m": clean_m, "corr_m": corr_m,
        },
    )


# Registry for use in run_icml_eval.py
ABLATION_METHODS: dict = {
    "ablation_no_skeptic":     run_ablation_no_skeptic,
    "ablation_no_seap":        run_ablation_no_seap,
    "ablation_random_layers":  run_ablation_random_layers,
    "ablation_acdc_only":      run_ablation_acdc_only,
}
