#!/usr/bin/env python3
"""ICML Evaluation Master Script.

Runs the complete ICML-ready evaluation pipeline:

  1. Multi-seed comparison (n=50) with mean ± std per task
  2. False Positive Rate (FPR) measurement via negative controls
  3. Ablation study (remove Skeptic / S-EAP / Network Analyst / ACDC-only)
  4. TransformerLens official ACDC as additional reference baseline
  5. Statistical significance tests (Wilcoxon signed-rank)
  6. LaTeX + Markdown output tables

Usage:
  # Full ICML eval (GPT-2, all tasks, n=50)
  python run_icml_eval.py --task induction --n-seeds 50 --model gpt2

  # Fast smoke test (n=5 seeds, 1 behavior per task)
  python run_icml_eval.py --task induction --n-seeds 5 --n-behaviors 1 --model gpt2

  # Large-model sweep
  python run_icml_eval.py --task all --n-seeds 20 --model EleutherAI/pythia-1.4b

  # Ablation-only (skip FPR, faster)
  python run_icml_eval.py --task induction --n-seeds 10 --model gpt2 --ablation-only
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from automechinterp.tools import adapter
from automechinterp import config
from automechinterp.tasks.induction_tasks import build_induction_tasks
from automechinterp.tasks.greater_than_tasks import build_greater_than_tasks
from automechinterp.tasks.agentic_tracing_tasks import build_agentic_tracing_tasks
from automechinterp.baselines import ALL_METHODS
from automechinterp.baselines.transformerlens_acdc import run_transformerlens_acdc
from automechinterp.eval.method_comparison import run_comparison
from automechinterp.eval.false_positive_rate import measure_false_positive_rate, fpr_summary_table
from automechinterp.eval.ablation_study import ABLATION_METHODS
from automechinterp.eval.statistical_summary import (
    aggregate_results, add_fpr_to_aggregated,
    markdown_table, latex_table, save_aggregated,
)

_TASK_BUILDERS = {
    "induction":    build_induction_tasks,
    "greater_than": build_greater_than_tasks,
    "agentic":      build_agentic_tracing_tasks,
}

# All methods for comparison (incl. TL-ACDC and ablations)
_ALL_METHODS_EXTENDED = {
    **ALL_METHODS,
    "transformerlens_acdc": run_transformerlens_acdc,
    **ABLATION_METHODS,
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="ICML-ready evaluation with n=50 seeds, FPR, ablations, stats.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--task",      default="induction",
                   choices=list(_TASK_BUILDERS.keys()) + ["all"])
    p.add_argument("--model",     default=None,
                   help="Model ID (default: config.TARGET_MODEL_ID)")
    p.add_argument("--device",    default=None)
    p.add_argument("--n-seeds",   type=int, default=50,
                   help="Number of random seeds (= sample size for stats, default: 50)")
    p.add_argument("--n-behaviors", type=int, default=None,
                   help="Max task variants per suite (default: all)")
    p.add_argument("--methods",   default="all",
                   help="Comma-sep method list or 'all'. Includes TL-ACDC and ablations.")
    p.add_argument("--out-dir",   default="output/icml_eval")
    p.add_argument("--layer-range", default=None,
                   help="Layers to search, e.g. '0-11'")
    p.add_argument("--n-controls", type=int, default=10,
                   help="Wrong circuits per task for FPR measurement (default: 10)")
    p.add_argument("--ablation-only", action="store_true",
                   help="Skip FPR measurement (faster)")
    p.add_argument("--skip-tl",   action="store_true",
                   help="Skip TransformerLens ACDC (saves memory on small GPUs)")
    p.add_argument("--skip-ablations", action="store_true",
                   help="Skip ablation variants")
    p.add_argument("--sp-n-steps",  type=int,   default=200)
    p.add_argument("--acd-budget",  type=int,   default=40)
    p.add_argument("--rl-episodes", type=int,   default=60)
    p.add_argument("--consensus-k", type=int,   default=2)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    model_id = args.model or config.TARGET_MODEL_ID
    device   = args.device or config.DEVICE
    out_dir  = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Parse methods
    base_methods = list(ALL_METHODS.keys()) + ["reference"]
    if not args.skip_tl:
        base_methods.append("transformerlens_acdc")
    if not args.skip_ablations:
        base_methods += list(ABLATION_METHODS.keys())

    if args.methods.strip().lower() == "all":
        methods = base_methods
    else:
        methods = [m.strip() for m in args.methods.split(",")]

    # Parse layer range
    layer_range = None
    if args.layer_range:
        s, e = args.layer_range.split("-")
        layer_range = range(int(s), int(e) + 1)

    method_kwargs = {
        "n_steps": args.sp_n_steps,
        "lr": 0.03, "lambda_l1": 0.01,
        "budget": args.acd_budget,
        "n_episodes": args.rl_episodes,
        "top_k": 10, "seed": 0,
        "consensus_k": args.consensus_k,
    }

    task_keys = list(_TASK_BUILDERS.keys()) if args.task == "all" else [args.task]

    print("=" * 70)
    print("ICML Evaluation Pipeline")
    print(f"  Model:   {model_id}  |  Device: {device}")
    print(f"  Tasks:   {task_keys}  |  Seeds: {args.n_seeds}")
    print(f"  Methods: {methods}")
    print("=" * 70)

    print(f"\nLoading {model_id}...")
    handle = adapter.register_model(model_id, device=device)
    print(f"  {handle.n_layers} layers, {handle.n_layers * handle.model.config.num_attention_heads} heads")

    for task_key in task_keys:
        print(f"\n{'='*70}")
        print(f"Task Suite: {task_key}")
        print("=" * 70)

        tasks = _TASK_BUILDERS[task_key](handle, seed=0)
        if args.n_behaviors:
            tasks = tasks[:args.n_behaviors]
        print(f"  {len(tasks)} task variant(s)")

        # Seed loop: each seed builds tasks with a different RNG seed → different
        # prompt instances → genuine distribution over tasks
        all_seed_results: list[list[dict]] = []
        all_fpr_results:  list[dict]       = []

        for seed in range(args.n_seeds):
            print(f"\n  --- Seed {seed + 1}/{args.n_seeds} ---")
            seed_tasks = _TASK_BUILDERS[task_key](handle, seed=seed)
            if args.n_behaviors:
                seed_tasks = seed_tasks[:args.n_behaviors]

            seed_comparison_results: list[dict] = []
            seed_fpr: dict = {}

            for task in seed_tasks:
                print(f"    Task: {task['behavior']}")
                # Run comparison (all methods)
                comp = run_comparison(
                    handle, task,
                    methods=[m for m in methods if m not in ABLATION_METHODS],
                    layer_range=layer_range,
                    consensus_k=args.consensus_k,
                    **{k: v for k, v in method_kwargs.items()
                       if k not in ("consensus_k",)},
                )
                seed_comparison_results.extend(comp.results)

                # Run ablations
                if not args.skip_ablations:
                    for abl_name, abl_fn in ABLATION_METHODS.items():
                        if abl_name in methods:
                            from automechinterp.baselines import MethodResult
                            print(f"    [{abl_name}] running...", flush=True)
                            try:
                                abl_result: MethodResult = abl_fn(
                                    handle, task,
                                    layer_range=layer_range,
                                    n_heads=task["n_heads"],
                                )
                                oracle = task.get("oracle_circuit")
                                oracle_set = set(map(tuple, oracle)) if oracle else set()
                                predicted = {tuple(lh) for lh in abl_result.circuit}
                                if oracle_set:
                                    from automechinterp.eval.method_comparison import _prf1
                                    p, r, f1 = _prf1(predicted, oracle_set)
                                else:
                                    p = r = f1 = float("nan")
                                seed_comparison_results.append({
                                    "method": abl_name,
                                    "circuit": abl_result.circuit,
                                    "circuit_size": len(abl_result.circuit),
                                    "precision": p, "recall": r, "f1": f1,
                                    "metric_recovery": abl_result.metric_recovery,
                                    "budget_used": abl_result.tool_calls,
                                    "runtime_s": abl_result.runtime_s,
                                    "metadata": abl_result.metadata,
                                })
                                print(f"    [{abl_name}] circuit={abl_result.circuit}, "
                                      f"recovery={abl_result.metric_recovery:.3f}")
                            except Exception as e:
                                print(f"    [{abl_name}] ERROR: {e}")

                # FPR measurement
                if not args.ablation_only:
                    print(f"    [FPR] measuring false positive rate...")
                    method_results_dict = {
                        r["method"]: type("R", (), {
                            "circuit": r["circuit"],
                            "metric_recovery": r["metric_recovery"],
                        })()
                        for r in seed_comparison_results
                    }
                    try:
                        from automechinterp.baselines import MethodResult as MR
                        method_results_typed = {}
                        for r in seed_comparison_results:
                            method_results_typed[r["method"]] = MR(
                                method=r["method"],
                                circuit=r["circuit"],
                                circuit_score=0.0,
                                metric_recovery=r["metric_recovery"],
                                tool_calls=r["budget_used"],
                                runtime_s=r["runtime_s"],
                            )
                        fpr_results = measure_false_positive_rate(
                            handle, task, method_results_typed,
                            n_controls=args.n_controls,
                            seed=seed * 100,
                        )
                        for m, fpr in fpr_results.items():
                            seed_fpr.setdefault(m, []).append(fpr.false_positive_rate)
                            print(f"    [FPR:{m}] {fpr.n_false_accepts}/{fpr.n_controls} "
                                  f"= {fpr.false_positive_rate:.0%}")
                    except Exception as e:
                        print(f"    [FPR] ERROR: {e}")

            all_seed_results.append(seed_comparison_results)
            all_fpr_results.append(seed_fpr)

        # Aggregate across seeds
        print(f"\n  Aggregating {args.n_seeds} seeds...")
        agg = aggregate_results(all_seed_results)

        # Add FPR to aggregated results
        if not args.ablation_only and all_fpr_results:
            from automechinterp.eval.false_positive_rate import FPRResult as FPRR
            fpr_aggregated = []
            for seed_fpr in all_fpr_results:
                seed_dict = {}
                for m, fprs in seed_fpr.items():
                    for fpr_val in fprs:
                        seed_dict[m] = FPRR(
                            method=m, n_controls=args.n_controls,
                            n_false_accepts=round(fpr_val * args.n_controls),
                            false_positive_rate=fpr_val,
                        )
                fpr_aggregated.append(seed_dict)
            add_fpr_to_aggregated(agg, fpr_aggregated)

        # Print and save results
        print(f"\n{'='*70}")
        print(f"RESULTS: {task_key} | {model_id} | n={args.n_seeds}")
        print("=" * 70)
        print(markdown_table(agg))

        save_aggregated(agg, out_dir, task_key, model_id)

        # Save raw results for reproducibility
        raw_path = out_dir / f"{task_key}_{model_id.replace('/', '_')}_raw.json"
        raw_path.write_text(json.dumps(all_seed_results, indent=2, default=str))
        print(f"  Raw results: {raw_path}")

    print(f"\nAll results saved to {out_dir}/")
    print("Done.")


if __name__ == "__main__":
    main()
