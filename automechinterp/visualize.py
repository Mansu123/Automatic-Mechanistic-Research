"""Plots Stage A results (Sec. 4.6 metrics) to a PNG + JSON in an output
directory. Kept separate from stage_a.py so plotting is optional -- importing
matplotlib and rendering a figure is not something the core pipeline should
ever depend on.
"""
from __future__ import annotations

import json
import os
from datetime import datetime

import matplotlib
matplotlib.use("Agg")  # headless: never try to open a GUI window from a terminal run
import matplotlib.pyplot as plt

from . import stage_a


def plot_stage_a_results(output: dict, backend_kind: str, out_dir: str = "output") -> tuple[str, str]:
    """`output` is exactly what stage_a.run_stage_a() returns. Saves a PNG and
    a JSON (same numbers, machine-readable) into out_dir and returns both paths."""
    os.makedirs(out_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_id = output["target_model_id"]
    safe_model_name = model_id.replace("/", "_")

    result = output["result"]
    scored = output["scored"]
    negative = output["negative_controls"]
    layer_states = result["layer_states"]

    layers = sorted(layer_states.keys())
    fractions = [layer_states[l].get("fraction_recovered", 0.0) for l in layers]
    gt_layers = set(stage_a.GT_LAYERS) if scored["ground_truth_valid"] else set()
    bar_colors = ["#2ca02c" if l in gt_layers else "#4c72b0" for l in layers]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(f"AutoMechInterp -- Stage A -- {model_id} (backend={backend_kind})", fontsize=13)

    # Panel 1: per-layer causal effect (patch_layer fraction_recovered)
    ax = axes[0]
    if layers:
        ax.bar([str(l) for l in layers], fractions, color=bar_colors)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title("Per-layer causal effect (patch_layer)")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Fraction of clean-corrupted gap recovered")
    if gt_layers:
        ax.legend(handles=[
            plt.Rectangle((0, 0), 1, 1, color="#2ca02c", label="ground-truth layer"),
            plt.Rectangle((0, 0), 1, 1, color="#4c72b0", label="flagged layer"),
        ], fontsize=8, loc="best")

    # Panel 2: headline metrics as percentages
    ax = axes[1]
    metric_names, metric_values = [], []
    if scored["ground_truth_valid"]:
        metric_names += ["Layer recall", "Circuit recall", "Circuit precision"]
        metric_values += [scored["layer_localization_recall"], scored["circuit_recall"],
                           scored["circuit_precision"]]
    metric_names.append("Negative-control\nrefutation rate")
    metric_values.append(negative["n_refuted"] / negative["n_controls"] if negative["n_controls"] else 0.0)
    bars = ax.bar(metric_names, [v * 100 for v in metric_values], color="#dd8452")
    ax.set_ylim(0, 112)
    ax.set_ylabel("%")
    ax.set_title("Headline metrics")
    ax.tick_params(axis="x", labelsize=8.5)
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    for b, v in zip(bars, metric_values):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 2, f"{v:.0%}", ha="center", fontsize=9)

    # Panel 3: text summary
    ax = axes[2]
    ax.axis("off")
    verdict = result["verdict"]["verdict"] if result["verdict"] else "N/A (no claim survived triage)"
    lines = [
        f"Target model: {model_id}",
        f"Reasoning backend: {backend_kind}",
        f"Flagged layers: {scored['flagged_layers']}",
        f"Claimed circuit: {scored['claimed_heads']}",
    ]
    if scored["ground_truth_valid"]:
        lines.append(f"Ground-truth circuit: {sorted(stage_a.GT_HEADS)}")
        lines.append(f"True positives: {scored['true_positive_heads']}")
    else:
        lines.append("Ground truth: N/A for this model (GPT-2-only)")
    lines += [
        f"Judge verdict: {verdict}",
        f"Tool calls spent: {result['tool_calls_spent']}",
        f"Cache hit rate: {result['cache_stats']['hit_rate']:.0%}",
    ]
    ax.text(0, 1, "\n".join(lines), fontsize=9.5, va="top", family="monospace", wrap=True)
    ax.set_title("Run summary")

    fig.tight_layout(rect=[0, 0, 1, 0.94])
    png_path = os.path.join(out_dir, f"stage_a_{safe_model_name}_{timestamp}.png")
    fig.savefig(png_path, dpi=150)
    plt.close(fig)

    json_path = os.path.join(out_dir, f"stage_a_{safe_model_name}_{timestamp}.json")
    with open(json_path, "w") as f:
        json.dump({
            "target_model_id": model_id,
            "backend_kind": backend_kind,
            "flagged_layers": scored["flagged_layers"],
            "claimed_heads": scored["claimed_heads"],
            "ground_truth_valid": scored["ground_truth_valid"],
            "layer_localization_recall": scored["layer_localization_recall"],
            "circuit_recall": scored["circuit_recall"],
            "circuit_precision": scored["circuit_precision"],
            "judge_verdict": verdict,
            "negative_controls_refuted": negative["n_refuted"],
            "negative_controls_total": negative["n_controls"],
            "tool_calls_spent": result["tool_calls_spent"],
            "cache_hit_rate": result["cache_stats"]["hit_rate"],
            "per_layer_fraction_recovered": {str(l): layer_states[l].get("fraction_recovered") for l in layers},
        }, f, indent=2)

    return png_path, json_path
