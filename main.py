#!/usr/bin/env python3
"""AutoMechInterp entrypoint.

Runs Stage A (Sec. 4.3): the full hierarchical pipeline --
Network Analyst -> Layer Agents -> Component Agents -> Skeptic -> Judge --
on GPT-2 small against the Indirect Object Identification (IOI) task, scored
against the paper's published ground-truth circuit, plus negative controls.

Usage:
    python3 main.py                          # heuristic backend (no LLM needed, default)
    AMI_LLM_BACKEND=hf_local python3 main.py  # open-source LLM agent brain (see below)
    python3 main.py --gap-matrix              # just print the related-work gap table and exit
    python3 main.py --smoke-test-qwen         # exercise the HFLocalBackend code path with a
                                               # small Qwen2.5-0.5B-Instruct model (not the real
                                               # heavy tier -- see automechinterp/config.py)
    python3 main.py --graph                   # also save a PNG + JSON of the results to output/
    python3 main.py --target-model gpt2-medium --graph   # interpret a different target model
                                               # (NOTE: published IOI ground truth only exists for
                                               # GPT-2 small -- other models still run, but
                                               # recall/precision are reported as N/A, not guessed)

Model roles (automechinterp/config.py), overridable via env vars OR the
--target-model / --backend flags below:
    AMI_TARGET_MODEL   the network being interpreted            (default: gpt2)
    AMI_HEAVY_MODEL     open-source LLM behind the agents' brain  (default: Qwen/Qwen2.5-7B-Instruct)
    AMI_LLM_BACKEND     heuristic | hf_local | openai | anthropic (default: heuristic)

The heuristic backend needs no GPU and no downloads beyond the target model,
and encodes the exact decision rules the proposal describes in prose, so the
whole pipeline is reproducible on a laptop. Point AMI_LLM_BACKEND=hf_local
with AMI_HEAVY_MODEL=Qwen/Qwen2.5-7B-Instruct at real hardware (the
proposal's own target: a Colab A100) to run the agents on an actual 7B
open-source reasoning model instead -- no other code changes needed, per
automechinterp/llm_backends.py.
"""
import argparse
import sys

from automechinterp import config, gap_work
from automechinterp.stage_a import run_stage_a


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gap-matrix", action="store_true",
                         help="print the related-work gap-closure matrix and exit")
    parser.add_argument("--smoke-test-qwen", action="store_true",
                         help="run one agent step through HFLocalBackend with a small Qwen model")
    parser.add_argument("--backend", default=None, choices=["heuristic", "hf_local", "openai", "anthropic"],
                         help="override AMI_LLM_BACKEND for this run (which model drives the AGENTS)")
    parser.add_argument("--target-model", default=None,
                         help="override AMI_TARGET_MODEL for this run (which model is being "
                              "INTERPRETED, e.g. gpt2, gpt2-medium, or any HF causal LM id)")
    parser.add_argument("--heavy-model", default=None,
                         help="override AMI_HEAVY_MODEL (only used when --backend hf_local)")
    parser.add_argument("--graph", action="store_true",
                         help="save a PNG + JSON of the results to --output-dir")
    parser.add_argument("--output-dir", default="output",
                         help="directory for --graph output (default: output)")
    parser.add_argument("--ablations", action="store_true",
                         help="run Sec. 4.7 ablation studies (hierarchy, verification, Network "
                              "Analyst flagging strategy) instead of Stage A, and exit")
    parser.add_argument("--stage-b", action="store_true",
                         help="run Stage B (Sec. 4.4): multi-behavior Layer Atlas, instead of "
                              "Stage A, and exit")
    parser.add_argument("--stage-c", action="store_true",
                         help="run Stage C (Sec. 4.5): cross-model layer diff against a locally "
                              "fine-tuned biomedical stand-in, instead of Stage A, and exit")
    args = parser.parse_args()

    if args.gap_matrix:
        gap_work.print_matrix()
        return

    if args.smoke_test_qwen:
        _smoke_test_qwen()
        return

    if args.ablations:
        _run_ablations(args.graph, args.output_dir)
        return

    if args.stage_b:
        _run_stage_b(args.backend or config.LLM_BACKEND, args.graph, args.output_dir)
        return

    if args.stage_c:
        _run_stage_c(args.graph, args.output_dir)
        return

    print("Related-work gaps this run is designed to close (Sec. 2 of the proposal):\n")
    gap_work.print_matrix()

    backend_kind = args.backend or config.LLM_BACKEND
    target_model_id = args.target_model or config.TARGET_MODEL_ID
    backend_kwargs = None
    if backend_kind == "hf_local":
        heavy_model_id = args.heavy_model or config.HEAVY_MODEL_ID
        backend_kwargs = {"model_id": heavy_model_id, "device": config.DEVICE}
        print(f"Reasoning backend: hf_local -> {heavy_model_id} (device={config.DEVICE})\n")

    output = run_stage_a(backend_kind=backend_kind, backend_kwargs=backend_kwargs,
                          target_model_id=target_model_id)

    if args.graph:
        from automechinterp.visualize import plot_stage_a_results
        png_path, json_path = plot_stage_a_results(output, backend_kind, out_dir=args.output_dir)
        print(f"\nSaved graph -> {png_path}")
        print(f"Saved raw results -> {json_path}")


def _run_stage_c(graph: bool, output_dir: str):
    from automechinterp.stage_c import run_stage_c
    result = run_stage_c()
    if graph:
        from automechinterp.visualize import plot_stage_c_results
        png_path, json_path = plot_stage_c_results(result, out_dir=output_dir)
        print(f"\nSaved graph -> {png_path}")
        print(f"Saved raw results -> {json_path}")


def _run_stage_b(backend_kind: str, graph: bool, output_dir: str):
    from automechinterp.stage_b import run_stage_b
    result = run_stage_b(backend_kind=backend_kind)
    if graph:
        from automechinterp.visualize import plot_stage_b_results
        png_path, json_path = plot_stage_b_results(result, out_dir=output_dir)
        print(f"\nSaved graph -> {png_path}")
        print(f"Saved raw results -> {json_path}")


def _run_ablations(graph: bool, output_dir: str):
    from automechinterp.ablations import run_all_ablations
    print("Running Sec. 4.7 ablation studies on GPT-2...\n")
    result = run_all_ablations()

    h, v, n = result["hierarchy_ablation"], result["verification_ablation"], result["network_analyst_ablation"]
    print("\n" + "-" * 78)
    print("ABLATION RESULTS")
    print("-" * 78)
    print(f"Hierarchy ablation        : {h['context_reduction_factor']:.2f}x context reduction "
          f"(flat={h['flat_total_context_chars']} chars vs hierarchical max="
          f"{h['hierarchical_max_agent_context_chars']} chars, over {h['n_layers_checked']} layers)")
    print(f"Verification ablation     : self-confirmation false-confirmation rate="
          f"{v['self_confirmation_false_confirmation_rate']:.0%} vs Prover-Skeptic-Judge="
          f"{v['prover_skeptic_judge_false_confirmation_rate']:.0%} (n={v['n_claims']} known-wrong claims)")
    print(f"Network Analyst ablation  : layer-localization recall -- guided="
          f"{n['layer_localization_recall']['guided']:.0%}, uniform="
          f"{n['layer_localization_recall']['uniform']:.0%}, random_avg="
          f"{n['layer_localization_recall']['random_avg']:.0%}")

    if graph:
        from automechinterp.visualize import plot_ablations
        png_path, json_path = plot_ablations(result, out_dir=output_dir)
        print(f"\nSaved graph -> {png_path}")
        print(f"Saved raw results -> {json_path}")


def _smoke_test_qwen():
    """Proves the open-source-LLM agent brain actually works, without
    requiring the 7B model's RAM footprint: swaps in
    config.SMOKETEST_MODEL_ID (Qwen2.5-0.5B-Instruct) for one Network
    Analyst step. Production runs should use AMI_HEAVY_MODEL instead."""
    from automechinterp.tools import adapter
    from automechinterp.agents.base import ToolCallBudget
    from automechinterp.agents.network_analyst import build_network_analyst
    from automechinterp.stage_a import build_ioi_task

    print(f"Loading target model '{config.TARGET_MODEL_ID}' and reasoning model "
          f"'{config.SMOKETEST_MODEL_ID}' (smoke test only -- production default is "
          f"{config.HEAVY_MODEL_ID}, see automechinterp/config.py)...")
    handle = adapter.register_model(config.TARGET_MODEL_ID, device="cpu")
    task = build_ioi_task(handle)
    budget = ToolCallBudget(global_remaining=[3], per_agent_limit=3)
    agent, state = build_network_analyst(
        handle, task["probe_texts"], task["task_metric_fn"], "hf_local", budget,
        backend_kwargs={"model_id": config.SMOKETEST_MODEL_ID, "device": "cpu", "max_new_tokens": 128})
    agent.run()
    print("\nSmoke test complete -- HFLocalBackend produced real tool-call decisions above.")
    print(f"state so far: {list(state.keys())}")


if __name__ == "__main__":
    sys.exit(main())
