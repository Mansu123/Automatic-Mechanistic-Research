"""Method Comparison Harness.

Runs all circuit-discovery methods (baselines + reference system) on a
single task and computes a unified set of metrics for head-to-head comparison.

Metrics (computed per method per task):
  - precision:         fraction of reported heads that are true positives
  - recall:            fraction of true circuit heads captured
  - F1:                harmonic mean of precision and recall
  - budget_used:       method-specific budget units consumed
  - metric_recovery:   fraction of clean–corrupted gap recovered by circuit alone
  - runtime_s:         wall-clock seconds
  - circuit_size:      number of heads in discovered circuit

"True positive" definition (in priority order):
  1. oracle_circuit in task dict (known ground truth for induction / greater-than)
  2. Cross-method consensus: heads found by ≥ consensus_k methods are treated as TP

Reference system: ACDC + EAP ensemble (your existing component_agent pipeline).
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Callable

import torch

from ..tools import adapter
from ..tools import tier_c
from ..baselines import MethodResult, ALL_METHODS


# ---------------------------------------------------------------------------
# Reference system: wraps existing ACDC + EAP + DLA ensemble
# ---------------------------------------------------------------------------

def _run_reference(handle: adapter.ModelHandle, task: dict,
                   layer_range: range | None = None,
                   n_heads: int | None = None) -> MethodResult:
    """Run the existing ACDC + DLA reference system.

    Mirrors what component_agent.py does but exposed as a MethodResult
    so the comparison harness can treat it identically to baselines.
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

    # ACDC sweep (simplified, same as tier_c.run_acdc)
    try:
        acdc_digest, acdc_circuit = tier_c.run_acdc(
            handle, cp, xp, io_t, s_t, layer_range, n_heads, threshold=0.10
        )
    except Exception as e:
        acdc_digest, acdc_circuit = f"ERROR: {e}", []

    # DLA for complementary write-direction signal
    try:
        dla_digest, dla_top = tier_c.direct_logit_attribution(
            handle, cp, io_t, s_t, layer_range, n_heads, k=8
        )
    except Exception as e:
        dla_digest, dla_top = f"ERROR: {e}", []

    circuit = sorted(set(acdc_circuit) | set(dla_top))

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
    else:
        metric_recovery = 0.0

    return MethodResult(
        method="reference_acdc_dla",
        circuit=circuit,
        circuit_score=float(metric_recovery),
        metric_recovery=metric_recovery,
        tool_calls=len(layers := list(layer_range)) * n_heads + 1,
        runtime_s=time.time() - t0,
        metadata={
            "acdc_digest": acdc_digest, "dla_digest": dla_digest,
            "clean_m": clean_m, "corr_m": corr_m,
        },
    )


# ---------------------------------------------------------------------------
# Precision / Recall / F1 computation
# ---------------------------------------------------------------------------

def _prf1(predicted: set, gold: set) -> tuple[float, float, float]:
    """Compute precision, recall, F1 over sets of (layer, head) tuples."""
    if not predicted and not gold:
        return 1.0, 1.0, 1.0
    if not predicted:
        return 0.0, 0.0, 0.0
    if not gold:
        return 0.0, 0.0, 0.0
    tp = len(predicted & gold)
    p  = tp / len(predicted)
    r  = tp / len(gold)
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return p, r, f1


# ---------------------------------------------------------------------------
# Main comparison entry point
# ---------------------------------------------------------------------------

@dataclass
class MethodComparison:
    task_name:        str
    model_id:         str
    oracle_available: bool
    oracle_circuit:   list   # empty if no oracle
    results: list[dict] = field(default_factory=list)  # one per method

    def summary_table(self) -> str:
        """Return a markdown table of all method results."""
        header = (
            "| Method | Circuit | Prec | Rec | F1 | Recovery | Budget | Time (s) |\n"
            "|---|---|---|---|---|---|---|---|\n"
        )
        rows = []
        for r in self.results:
            circ_str = str(r["circuit"])[:50] + ("..." if len(str(r["circuit"])) > 50 else "")
            rows.append(
                f"| {r['method']} | `{circ_str}` "
                f"| {r['precision']:.2f} | {r['recall']:.2f} | {r['f1']:.2f} "
                f"| {r['metric_recovery']:.2f} | {r['budget_used']} | {r['runtime_s']:.1f} |"
            )
        oracle_note = (
            f"\n_Oracle circuit: {self.oracle_circuit}_\n"
            if self.oracle_available
            else "\n_No oracle — precision/recall vs. cross-method consensus_\n"
        )
        return header + "\n".join(rows) + oracle_note


def run_comparison(
    handle: adapter.ModelHandle,
    task: dict,
    methods: list[str] | None = None,
    layer_range: range | None = None,
    consensus_k: int = 2,
    **method_kwargs,
) -> MethodComparison:
    """Run all requested methods on `task` and collect comparison metrics.

    Args:
        handle:       ModelHandle for the target model.
        task:         Task dict (same schema as behaviors.py / task_suites).
        methods:      List of method names to run (default: all + reference).
                      Valid names: "subnetwork_probing", "acd", "mechrl",
                      "circuit_tracing", "reference".
        layer_range:  Layers to pass to each method (default: all layers).
        consensus_k:  Number of methods that must agree to count as TP
                      (only used when no oracle is available).
        **method_kwargs: Passed through to each method's run_* function.

    Returns:
        MethodComparison dataclass with per-method results and a summary table.
    """
    if methods is None:
        methods = list(ALL_METHODS.keys()) + ["reference"]

    layer_range = layer_range or range(handle.n_layers)
    n_heads     = task["n_heads"]

    # Determine ground truth (oracle or cross-method consensus)
    oracle = task.get("oracle_circuit")
    oracle_set = set(map(tuple, oracle)) if oracle else set()

    # --- Run all methods ---
    method_results: dict[str, MethodResult] = {}
    for method_name in methods:
        print(f"  [{method_name}] running...", flush=True)
        try:
            if method_name == "reference":
                result = _run_reference(handle, task, layer_range, n_heads)
            elif method_name in ALL_METHODS:
                result = ALL_METHODS[method_name](
                    handle, task, layer_range=layer_range, n_heads=n_heads,
                    **method_kwargs,
                )
            else:
                print(f"  [{method_name}] unknown method, skipping")
                continue
            method_results[method_name] = result
            print(f"  [{method_name}] circuit={result.circuit}, "
                  f"recovery={result.metric_recovery:.3f}, "
                  f"t={result.runtime_s:.1f}s")
        except Exception as e:
            print(f"  [{method_name}] ERROR: {e}")

    # --- Build cross-method consensus gold if no oracle ---
    if not oracle_set and len(method_results) >= consensus_k:
        head_counts: dict[tuple, int] = {}
        for r in method_results.values():
            for lh in r.circuit:
                head_counts[tuple(lh)] = head_counts.get(tuple(lh), 0) + 1
        oracle_set = {lh for lh, cnt in head_counts.items() if cnt >= consensus_k}

    # --- Compute precision / recall / F1 per method ---
    comparison_results = []
    for method_name, result in method_results.items():
        predicted = {tuple(lh) for lh in result.circuit}
        p, r, f1  = _prf1(predicted, oracle_set) if oracle_set else (float("nan"),) * 3
        comparison_results.append({
            "method":          method_name,
            "circuit":         result.circuit,
            "circuit_size":    len(result.circuit),
            "circuit_score":   result.circuit_score,
            "precision":       p,
            "recall":          r,
            "f1":              f1,
            "metric_recovery": result.metric_recovery,
            "budget_used":     result.tool_calls,
            "runtime_s":       result.runtime_s,
            "metadata":        result.metadata,
        })

    return MethodComparison(
        task_name=task["behavior"],
        model_id=handle.model_id,
        oracle_available=bool(oracle),
        oracle_circuit=list(oracle) if oracle else list(oracle_set),
        results=comparison_results,
    )


def run_task_suite_comparison(
    handle: adapter.ModelHandle,
    tasks: list[dict],
    methods: list[str] | None = None,
    out_dir: str | Path = "output/method_comparison",
    layer_range: range | None = None,
    **method_kwargs,
) -> list[MethodComparison]:
    """Run comparison across a list of tasks and write results to disk."""
    from datetime import datetime
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    all_comparisons: list[MethodComparison] = []

    for task in tasks:
        print(f"\n{'='*70}")
        print(f"Task: {task['behavior']}  |  model: {handle.model_id}")
        print("="*70)
        comp = run_comparison(
            handle, task,
            methods=methods,
            layer_range=layer_range,
            **method_kwargs,
        )
        all_comparisons.append(comp)

        # Write per-task markdown report
        slug = task["behavior"].replace(" ", "_")[:50]
        report_path = out_dir / f"{slug}_{handle.model_id.replace('/', '_')}_{ts}.md"
        with open(report_path, "w") as f:
            f.write(f"# Circuit Discovery Comparison\n\n")
            f.write(f"**Task:** {task['behavior']}  \n")
            f.write(f"**Model:** {handle.model_id}  \n")
            f.write(f"**Oracle:** {'Yes' if comp.oracle_available else 'Cross-method consensus'}  \n\n")
            f.write(comp.summary_table())
            f.write("\n\n## Per-method metadata\n\n")
            for r in comp.results:
                f.write(f"### {r['method']}\n\n")
                f.write(f"```json\n{json.dumps(r['metadata'], indent=2, default=str)}\n```\n\n")
        print(f"  → Report: {report_path}")

    # Write summary JSON
    summary_path = out_dir / f"comparison_summary_{ts}.json"
    summary = []
    for comp in all_comparisons:
        summary.append({
            "task":     comp.task_name,
            "model":    comp.model_id,
            "oracle":   comp.oracle_available,
            "results": [
                {k: v for k, v in r.items() if k != "metadata"}
                for r in comp.results
            ],
        })
    summary_path.write_text(json.dumps(summary, indent=2, default=str))
    print(f"\nSummary JSON: {summary_path}")

    return all_comparisons

