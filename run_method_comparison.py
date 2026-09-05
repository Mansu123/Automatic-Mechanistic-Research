#!/usr/bin/env python3
"""Circuit Discovery Method Comparison CLI.

Evaluates four circuit-discovery methodologies against the existing
ACDC+DLA reference system across three canonical task suites:

  - induction:     Repeated-token pattern completion (Olsson et al. 2022 oracle)
  - greater_than:  Numerical year comparison (Hanna et al. 2023 oracle)
  - agentic:       Multi-step reasoning vs dummy-pass circuit tracing (no oracle)

Usage:
  python run_method_comparison.py --task induction --methods all --model gpt2
  python run_method_comparison.py --task greater_than --methods acd,subnetwork_probing
  python run_method_comparison.py --task all --model gpt2 --n-behaviors 3

Output:
  output/method_comparison/<task>_<model>_<timestamp>.md   (per-task reports)
  output/method_comparison/comparison_summary_<timestamp>.json
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make sure the project root is on sys.path when invoked directly
sys.path.insert(0, str(Path(__file__).resolve().parent))

from automechinterp.tools import adapter
from automechinterp import config
from automechinterp.tasks.induction_tasks import build_induction_tasks
from automechinterp.tasks.greater_than_tasks import build_greater_than_tasks
from automechinterp.tasks.agentic_tracing_tasks import build_agentic_tracing_tasks
from automechinterp.eval.method_comparison import run_task_suite_comparison
from automechinterp.baselines import ALL_METHODS

_ALL_METHOD_NAMES = list(ALL_METHODS.keys()) + ["reference"]

_TASK_BUILDERS = {
    "induction":    build_induction_tasks,
    "greater_than": build_greater_than_tasks,
    "agentic":      build_agentic_tracing_tasks,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run circuit-discovery method comparison across task suites.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--task",
        default="induction",
        choices=list(_TASK_BUILDERS.keys()) + ["all"],
        help="Task suite to run (default: induction).",
    )
    parser.add_argument(
        "--methods",
        default="all",
        help=(
            "Comma-separated list of methods to run, or 'all'. "
            f"Valid: {', '.join(_ALL_METHOD_NAMES)}. Default: all."
        ),
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Target model ID (default: config.TARGET_MODEL_ID).",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Device (cuda/cpu, default: auto-detect from config).",
    )
    parser.add_argument(
        "--n-behaviors",
        type=int,
        default=None,
        help="Only run the first N task variants per suite (smoke test).",
    )
    parser.add_argument(
        "--out-dir",
        default="output/method_comparison",
        help="Directory to write reports and summary JSON (default: output/method_comparison).",
    )
    parser.add_argument(
        "--layer-range",
        default=None,
        help="Layers to search, e.g. '0-11'. Default: all layers.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for task generation and methods (default: 0).",
    )
    # Per-method overrides
    parser.add_argument("--sp-n-steps",   type=int,   default=200,  help="Subnetwork Probing: gradient steps.")
    parser.add_argument("--sp-lr",        type=float, default=0.03, help="Subnetwork Probing: learning rate.")
    parser.add_argument("--sp-lambda",    type=float, default=0.01, help="Subnetwork Probing: L1 coefficient.")
    parser.add_argument("--acd-budget",   type=int,   default=40,   help="ACD: max probes.")
    parser.add_argument("--rl-episodes",  type=int,   default=60,   help="MechRL: training episodes.")
    parser.add_argument("--ct-topk",      type=int,   default=10,   help="Circuit Tracing: top-k heads.")
    parser.add_argument(
        "--consensus-k",
        type=int,
        default=2,
        help="Cross-method consensus: heads found by >= K methods treated as TP (default: 2).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    model_id = args.model or config.TARGET_MODEL_ID
    device   = args.device or config.DEVICE

    # Parse methods
    if args.methods.strip().lower() == "all":
        methods = _ALL_METHOD_NAMES
    else:
        methods = [m.strip() for m in args.methods.split(",")]
        invalid = [m for m in methods if m not in _ALL_METHOD_NAMES]
        if invalid:
            print(f"Unknown method(s): {invalid}. Valid: {_ALL_METHOD_NAMES}")
            sys.exit(1)

    # Parse layer range
    layer_range = None
    if args.layer_range:
        parts = args.layer_range.split("-")
        if len(parts) == 2:
            layer_range = range(int(parts[0]), int(parts[1]) + 1)
        else:
            print(f"Invalid --layer-range: {args.layer_range}. Use format 'start-end', e.g. '0-11'.")
            sys.exit(1)

    # Build per-method kwargs
    method_kwargs: dict = {
        "n_steps":       args.sp_n_steps,
        "lr":            args.sp_lr,
        "lambda_l1":     args.sp_lambda,
        "budget":        args.acd_budget,
        "n_episodes":    args.rl_episodes,
        "top_k":         args.ct_topk,
        "seed":          args.seed,
    }

    # Determine task suites to run
    task_keys = list(_TASK_BUILDERS.keys()) if args.task == "all" else [args.task]

    print("=" * 70)
    print("Circuit Discovery Method Comparison")
    print(f"  Model:   {model_id}  ({device})")
    print(f"  Tasks:   {task_keys}")
    print(f"  Methods: {methods}")
    print("=" * 70)

    print(f"\nLoading model: {model_id}...")
    handle = adapter.register_model(model_id, device=device)
    print(f"  {handle.n_layers} layers, {handle.n_layers * handle.model.config.num_attention_heads} heads total")

    all_tasks: list[dict] = []
    for key in task_keys:
        builder = _TASK_BUILDERS[key]
        tasks = builder(handle, seed=args.seed)
        if args.n_behaviors is not None:
            tasks = tasks[:args.n_behaviors]
        print(f"\n  Suite '{key}': {len(tasks)} task variant(s)")
        all_tasks.extend(tasks)

    if not all_tasks:
        print("No tasks to run.")
        sys.exit(0)

    comparisons = run_task_suite_comparison(
        handle,
        all_tasks,
        methods=methods,
        out_dir=args.out_dir,
        layer_range=layer_range,
        consensus_k=args.consensus_k,
        **method_kwargs,
    )

    # Final summary to stdout
    print("\n" + "=" * 70)
    print("COMPARISON SUMMARY")
    print("=" * 70)
    for comp in comparisons:
        print(f"\n## {comp.task_name} ({comp.model_id})")
        print(comp.summary_table())

    print(f"\nDone. Reports written to: {args.out_dir}/")


if __name__ == "__main__":
    main()

