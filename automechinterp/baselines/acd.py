"""Active Circuit Discovery (ACD) baseline.

Method: Active Circuit Discovery (ACD) — formulates circuit discovery as a
Partially Observable Markov Decision Process (POMDP).

POMDP formulation:
  - Hidden state:   true binary circuit membership of each head
  - Observations:   ablation/patching metric responses (real-valued)
  - Actions:        which (layer, head) to probe next
  - Belief:         Bernoulli probability b(l,h) that head (l,h) is in-circuit
  - Policy:         maximum expected information gain (entropy reduction)
  - Termination:    all beliefs < ε_entropy, or budget exhausted

Belief update (Bayesian):
  After probing head (l,h) and observing patch_fraction f:
    - If f > high_threshold: likelihood that head is in-circuit increases
    - If f < low_threshold:  likelihood that head is NOT in-circuit increases
    - Otherwise:             mild update

This makes ACD efficient: it only probes heads where uncertainty is high,
avoiding the exhaustive O(L*H) sweep of ACDC.
"""
from __future__ import annotations

import math
import time

import torch

from ..tools import adapter
from . import MethodResult

_DEFAULT_BUDGET          = 40       # max probes (forward passes)
_DEFAULT_HIGH_THRESHOLD  = 0.15     # patch fraction → confident "in circuit"
_DEFAULT_LOW_THRESHOLD   = 0.05     # patch fraction → confident "not in circuit"
_DEFAULT_BELIEF_INIT     = 0.3      # prior: 30% chance any head is in-circuit
_DEFAULT_CIRCUIT_BELIEF  = 0.5      # belief > this → include head in circuit
_DEFAULT_ENTROPY_STOP    = 0.05     # stop when max per-head entropy < this


def _binary_entropy(p: float) -> float:
    """H(Bernoulli(p))."""
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return -(p * math.log2(p) + (1 - p) * math.log2(1 - p))


def _bayesian_update(prior: float, observation: float,
                     high_threshold: float, low_threshold: float) -> float:
    """Update Bernoulli belief via approximate likelihood.

    Likelihood model:
      P(obs | in_circuit=1) = Gaussian-like: high if obs > high_threshold
      P(obs | in_circuit=0) = Gaussian-like: high if obs < low_threshold
    """
    if observation > high_threshold:
        # Evidence FOR being in-circuit
        likelihood_in  = 1.0 - math.exp(-(observation - high_threshold) * 10)
        likelihood_out = math.exp(-(observation - high_threshold) * 10)
    elif observation < low_threshold:
        # Evidence AGAINST being in-circuit
        likelihood_in  = math.exp(-(low_threshold - observation) * 10)
        likelihood_out = 1.0 - math.exp(-(low_threshold - observation) * 10)
    else:
        # Ambiguous observation — mild update toward in-circuit
        likelihood_in  = 0.6
        likelihood_out = 0.4

    # Bayes: posterior ∝ likelihood * prior
    unnorm_in  = likelihood_in  * prior
    unnorm_out = likelihood_out * (1.0 - prior)
    total = unnorm_in + unnorm_out
    if total < 1e-12:
        return prior
    return unnorm_in / total


def _probe_head(handle: adapter.ModelHandle, layer_idx: int, head_idx: int,
                clean_input_ids: torch.Tensor, clean_attention_mask: torch.Tensor,
                corrupted_input_ids: torch.Tensor, corrupted_attention_mask: torch.Tensor,
                io_id: int, s_id: int, clean_m: float, corr_m: float) -> float:
    """Probe head (l,h) via activation patching; return fraction of gap recovered."""
    metric_fn = lambda logits: float((logits[0, -1, io_id] - logits[0, -1, s_id]).item())
    patched_m = adapter.run_with_head_patch(
        handle, layer_idx, head_idx,
        clean_input_ids, clean_attention_mask,
        corrupted_input_ids, corrupted_attention_mask,
        metric_fn,
    )
    denom = clean_m - corr_m
    return (patched_m - corr_m) / denom if abs(denom) > 1e-8 else 0.0


def run_acd(
    handle: adapter.ModelHandle,
    task: dict,
    layer_range: range | None = None,
    n_heads: int | None = None,
    budget: int = _DEFAULT_BUDGET,
    high_threshold: float = _DEFAULT_HIGH_THRESHOLD,
    low_threshold: float = _DEFAULT_LOW_THRESHOLD,
    belief_init: float = _DEFAULT_BELIEF_INIT,
    circuit_belief: float = _DEFAULT_CIRCUIT_BELIEF,
    entropy_stop: float = _DEFAULT_ENTROPY_STOP,
    **kwargs,
) -> MethodResult:
    """Run Active Circuit Discovery (ACD) on `task`.

    Args:
        handle:          ModelHandle for the target model.
        task:            Task dict (same schema as behaviors.py / task_suites).
        layer_range:     Layers to search over (default: all layers).
        n_heads:         Number of attention heads (default: from task["n_heads"]).
        budget:          Maximum number of probes (each probe = 2 forward passes).
        high_threshold:  Patch fraction > this → confident in-circuit.
        low_threshold:   Patch fraction < this → confident not-in-circuit.
        belief_init:     Prior probability any head is in-circuit.
        circuit_belief:  Posterior threshold to include head in final circuit.
        entropy_stop:    Stop when all beliefs have entropy below this.

    Returns:
        MethodResult with discovered circuit, metric recovery, runtime, etc.
    """
    t0 = time.time()

    n_heads     = n_heads or task["n_heads"]
    layer_range = layer_range or range(handle.n_layers)
    layers      = list(layer_range)

    cp, xp  = task["clean_prompt"], task["corrupted_prompt"]
    io_t, s_t = task["io_token"], task["s_token"]

    # [-1] not [0]: cross-tokenizer safety (see behaviors.py _make_task)
    io_id = handle.tokenizer.encode(" " + io_t.strip())[-1]
    s_id  = handle.tokenizer.encode(" " + s_t.strip())[-1]

    ci = handle.tokenizer([cp], return_tensors="pt").to(handle.device)
    xi = handle.tokenizer([xp], return_tensors="pt").to(handle.device)

    # Baseline metrics
    metric_fn = lambda logits: float((logits[0, -1, io_id] - logits[0, -1, s_id]).item())
    with adapter.MODEL_LOCK, torch.no_grad():
        clean_m = metric_fn(handle.model(**ci).logits)
        corr_m  = metric_fn(handle.model(**xi).logits)

    # Initialize belief state: b[l][h] = P(head (l,h) in circuit)
    beliefs: dict[tuple[int, int], float] = {
        (l, h): belief_init for l in layers for h in range(n_heads)
    }
    observations: dict[tuple[int, int], float] = {}
    probe_count = 0

    # POMDP loop: always probe the head with highest belief entropy
    for _ in range(budget):
        # Select action: highest-entropy head not yet probed (or re-probe if all done)
        candidates = [lh for lh in beliefs if lh not in observations]
        if not candidates:
            break

        # Sort by entropy descending (max information gain)
        candidates.sort(key=lambda lh: -_binary_entropy(beliefs[lh]))
        target_lh = candidates[0]

        # Check stopping condition: all remaining candidates have low entropy
        max_entropy = max(_binary_entropy(beliefs[lh]) for lh in candidates)
        if max_entropy < entropy_stop:
            break

        # Take action: probe the selected head
        l, h = target_lh
        frac = _probe_head(
            handle, l, h,
            ci["input_ids"], ci["attention_mask"],
            xi["input_ids"], xi["attention_mask"],
            io_id, s_id, clean_m, corr_m,
        )
        probe_count += 1
        observations[target_lh] = frac

        # Belief update
        beliefs[target_lh] = _bayesian_update(
            beliefs[target_lh], frac, high_threshold, low_threshold
        )

    # Extract circuit: heads where belief > circuit_belief
    circuit = sorted(
        [lh for lh, b in beliefs.items() if b >= circuit_belief],
        key=lambda lh: -beliefs[lh],
    )

    # Measure metric recovery: ablate complement, measure clean-prompt metric
    denom = clean_m - corr_m
    all_heads = [(l, h) for l in range(handle.n_layers) for h in range(n_heads)]
    complement = [lh for lh in all_heads if lh not in set(circuit)]

    if circuit and abs(denom) > 1e-8:
        ablated_m = float(adapter.ablate_head_set(
            handle, complement,
            ci["input_ids"], ci["attention_mask"],
            metric_fn,
        ))
        metric_recovery = (ablated_m - corr_m) / denom
    else:
        metric_recovery = 0.0

    runtime = time.time() - t0

    return MethodResult(
        method="acd",
        circuit=circuit,
        circuit_score=max(beliefs.values()) if beliefs else 0.0,
        metric_recovery=metric_recovery,
        tool_calls=probe_count,
        runtime_s=runtime,
        metadata={
            "budget": budget, "probes_used": probe_count,
            "high_threshold": high_threshold, "low_threshold": low_threshold,
            "belief_init": belief_init, "circuit_belief": circuit_belief,
            "clean_m": clean_m, "corr_m": corr_m,
            "final_beliefs": {str(k): round(v, 3) for k, v in beliefs.items()},
            "observations": {str(k): round(v, 3) for k, v in observations.items()},
            "n_probed": len(observations),
            "n_total_heads": len(layers) * n_heads,
            "probe_efficiency": round(len(observations) / max(1, len(layers) * n_heads), 3),
        },
    )

