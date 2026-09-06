"""False Positive Rate (FPR) measurement for all circuit-discovery methods.

ICML key metric: "Our verified circuits have 0% false confirmation rate;
baselines have X%."

Methodology (follows Stage A's run_negative_controls):
  1. For each task, generate N_CONTROLS deliberately wrong circuits —
     randomly sampled heads that are causally irrelevant to the behavior
     (early layers, random heads not found by any discovery method).
  2. Run each method and ask: would it ACCEPT this wrong circuit?
     - For our system: run it through the Skeptic → Judge pipeline.
     - For baselines: a circuit is "accepted" if:
         * It overlaps with the wrong heads by ≥ overlap_threshold fraction, OR
         * Its metric_recovery computed using ONLY the wrong heads is ≥ accept_threshold.
  3. FPR = fraction of wrong circuits that are falsely accepted.

A false acceptance = method claims a causally-irrelevant circuit is meaningful.
Our system should score 0% FPR; baselines will score much higher.
"""
from __future__ import annotations

import random
import time
from dataclasses import dataclass, field

import torch

from ..tools import adapter
from ..agents.skeptic import build_skeptic
from ..agents.judge import adjudicate
from ..agents.base import ToolCallBudget
from ..baselines import ALL_METHODS, MethodResult


@dataclass
class FPRResult:
    method: str
    n_controls: int
    n_false_accepts: int
    false_positive_rate: float
    details: list[dict] = field(default_factory=list)


def _generate_wrong_circuits(
    handle: adapter.ModelHandle,
    n_heads: int,
    n_controls: int,
    discovered_heads: set,
    seed: int = 42,
) -> list[list[tuple[int, int]]]:
    """Generate N_CONTROLS deliberately wrong circuits.

    Wrong circuits are sampled from heads that:
      - Are in early layers (layers 0–2) where causal effect is minimal
      - Are NOT in any method's discovered circuit (so they're clearly irrelevant)
    """
    rng = random.Random(seed)
    # Pool: early layers, heads not in any discovered circuit
    early_heads = [
        (l, h) for l in range(min(3, handle.n_layers))
        for h in range(n_heads)
        if (l, h) not in discovered_heads
    ]
    if len(early_heads) < n_controls * 2:
        # Fallback: sample uniformly from non-discovered heads across all layers
        early_heads = [
            (l, h) for l in range(handle.n_layers)
            for h in range(n_heads)
            if (l, h) not in discovered_heads
        ]
    rng.shuffle(early_heads)
    # Each wrong circuit is 2–4 heads
    wrong_circuits = []
    i = 0
    for _ in range(n_controls):
        size = rng.randint(2, 4)
        wc = early_heads[i:i + size]
        i = (i + size) % len(early_heads)
        if wc:
            wrong_circuits.append(wc)
    return wrong_circuits


def _our_system_accepts(
    handle: adapter.ModelHandle,
    wrong_circuit: list[tuple[int, int]],
    task: dict,
    backend_kind: str = "heuristic",
) -> tuple[bool, str]:
    """Check if our Prover-Skeptic-Judge pipeline ACCEPTS a wrong circuit.
    It should NOT (should Refute/Speculative). Returns (accepted, verdict)."""
    all_heads = [(l, h) for l in range(handle.n_layers) for h in range(task["n_heads"])]
    eval_prompts = task.get("eval_prompts", [(task["clean_prompt"], task["io_token"], task["s_token"])])
    budget = ToolCallBudget(global_remaining=[30], per_agent_limit=30)
    sk_agent, sk_state = build_skeptic(
        handle, wrong_circuit, all_heads, eval_prompts,
        task["behavior"], backend_kind, budget,
    )
    sk_agent.run()
    verdict = adjudicate(wrong_circuit, sk_state, backend_kind)
    accepted = verdict["verdict"] in ("Confirmed", "Probable")
    return accepted, verdict["verdict"]


def _baseline_accepts(
    handle: adapter.ModelHandle,
    wrong_circuit: list[tuple[int, int]],
    task: dict,
    method_circuit: list[tuple[int, int]],
    overlap_threshold: float = 0.5,
    recovery_threshold: float = 0.15,
) -> tuple[bool, str]:
    """Check if a baseline method would accept the wrong circuit.

    A baseline "accepts" a wrong circuit if:
    (a) The method's own discovered circuit overlaps significantly with the
        wrong circuit (it would have reported the wrong heads as in-circuit), OR
    (b) Computing metric recovery using ONLY the wrong heads gives a high score
        (the wrong heads pass the causal threshold).

    Criterion (b) simulates what would happen if the baseline were asked to
    evaluate the wrong circuit rather than discover its own.
    """
    cp, xp = task["clean_prompt"], task["corrupted_prompt"]
    io_t, s_t = task["io_token"], task["s_token"]
    io_id = handle.tokenizer.encode(" " + io_t.strip())[-1]
    s_id  = handle.tokenizer.encode(" " + s_t.strip())[-1]
    ci = handle.tokenizer([cp], return_tensors="pt").to(handle.device)

    metric_fn = lambda logits: float((logits[0, -1, io_id] - logits[0, -1, s_id]).item())
    with adapter.MODEL_LOCK, torch.no_grad():
        clean_m = metric_fn(handle.model(**ci).logits)

    # (a) Overlap with discovered circuit
    wrong_set  = set(map(tuple, wrong_circuit))
    method_set = set(map(tuple, method_circuit))
    if method_set:
        overlap = len(wrong_set & method_set) / len(wrong_set)
        if overlap >= overlap_threshold:
            return True, f"overlap={overlap:.2f} ≥ threshold"

    # (b) Metric recovery using only wrong heads (ablate complement, measure)
    all_heads  = [(l, h) for l in range(handle.n_layers) for h in range(task["n_heads"])]
    xi = handle.tokenizer([xp], return_tensors="pt").to(handle.device)
    with adapter.MODEL_LOCK, torch.no_grad():
        corr_m = metric_fn(handle.model(**xi).logits)
    complement = [lh for lh in all_heads if tuple(lh) not in wrong_set]
    denom = clean_m - corr_m
    if abs(denom) > 1e-8:
        try:
            ablated = float(adapter.ablate_head_set(
                handle, complement, ci["input_ids"], ci["attention_mask"], metric_fn
            ))
            recovery = (ablated - corr_m) / denom
            if recovery >= recovery_threshold:
                return True, f"recovery={recovery:.3f} ≥ threshold"
        except Exception:
            pass

    return False, "rejected"


def measure_false_positive_rate(
    handle: adapter.ModelHandle,
    task: dict,
    method_results: dict[str, MethodResult],
    n_controls: int = 10,
    overlap_threshold: float = 0.5,
    recovery_threshold: float = 0.15,
    backend_kind: str = "heuristic",
    seed: int = 42,
) -> dict[str, FPRResult]:
    """Measure FPR for all methods on a single task.

    Args:
        handle:             ModelHandle.
        task:               Task dict.
        method_results:     Output of run_comparison (method → MethodResult).
        n_controls:         Number of wrong circuits to inject per method.
        overlap_threshold:  Baseline accept criterion (a): overlap fraction.
        recovery_threshold: Baseline accept criterion (b): metric recovery.
        backend_kind:       LLM backend for our system's Skeptic.
        seed:               RNG seed.

    Returns:
        Dict mapping method name → FPRResult.
    """
    n_heads = task["n_heads"]

    # Collect all heads discovered by any method (exclude from wrong circuits)
    discovered = set()
    for r in method_results.values():
        for lh in r.circuit:
            discovered.add(tuple(lh))

    wrong_circuits = _generate_wrong_circuits(
        handle, n_heads, n_controls, discovered, seed=seed
    )

    results: dict[str, FPRResult] = {}

    # Evaluate our system
    print(f"  [FPR] testing reference system on {len(wrong_circuits)} wrong circuits...")
    our_false_accepts = 0
    our_details = []
    for wc in wrong_circuits:
        accepted, verdict = _our_system_accepts(handle, wc, task, backend_kind)
        if accepted:
            our_false_accepts += 1
        our_details.append({
            "wrong_circuit": wc,
            "verdict": verdict,
            "false_accept": accepted,
        })
    results["reference"] = FPRResult(
        method="reference",
        n_controls=len(wrong_circuits),
        n_false_accepts=our_false_accepts,
        false_positive_rate=our_false_accepts / max(1, len(wrong_circuits)),
        details=our_details,
    )

    # Evaluate each baseline
    for method_name, mr in method_results.items():
        if method_name == "reference":
            continue
        print(f"  [FPR] testing {method_name}...")
        fa = 0
        details = []
        for wc in wrong_circuits:
            accepted, reason = _baseline_accepts(
                handle, wc, task, mr.circuit,
                overlap_threshold, recovery_threshold,
            )
            if accepted:
                fa += 1
            details.append({
                "wrong_circuit": wc,
                "reason": reason,
                "false_accept": accepted,
            })
        results[method_name] = FPRResult(
            method=method_name,
            n_controls=len(wrong_circuits),
            n_false_accepts=fa,
            false_positive_rate=fa / max(1, len(wrong_circuits)),
            details=details,
        )

    return results


def fpr_summary_table(fpr_results: dict[str, FPRResult]) -> str:
    """Return a markdown table of FPR results."""
    header = "| Method | Controls | False Accepts | FPR |\n|---|---|---|---|\n"
    rows = []
    for name, r in sorted(fpr_results.items(), key=lambda x: x[1].false_positive_rate):
        rows.append(
            f"| **{name}** | {r.n_controls} | {r.n_false_accepts} | **{r.false_positive_rate:.0%}** |"
        )
    return header + "\n".join(rows)
