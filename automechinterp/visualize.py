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


def plot_ablations(result: dict, out_dir: str = "output") -> tuple[str, str]:
    """`result` is exactly what ablations.run_all_ablations() returns."""
    os.makedirs(out_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_id = result["target_model_id"]
    h = result["hierarchy_ablation"]
    v = result["verification_ablation"]
    n = result["network_analyst_ablation"]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(f"AutoMechInterp -- Ablation Studies (Sec. 4.7) -- {model_id}", fontsize=13)

    # Panel 1: hierarchy ablation -- context growth curve
    ax = axes[0]
    xs = list(range(1, h["n_layers_checked"] + 1))
    ax.plot(xs, h["flat_context_after_n_layers"], marker="o", label="flat single agent", color="#c44e52")
    ax.plot(xs, h["hierarchical_max_context_after_n_layers"], marker="s",
            label="hierarchical (max per-agent)", color="#4c72b0")
    ax.set_xlabel("Layers analyzed so far")
    ax.set_ylabel("Evidence-log size (characters)")
    ax.set_title(f"Hierarchy ablation ({h['context_reduction_factor']:.1f}x reduction)")
    ax.legend(fontsize=8.5)

    # Panel 2: verification ablation -- false-confirmation rate
    ax = axes[1]
    names = ["Self-confirmation\n(no Skeptic)", "Prover-Skeptic-Judge\n(full pipeline)"]
    values = [v["self_confirmation_false_confirmation_rate"] * 100,
              v["prover_skeptic_judge_false_confirmation_rate"] * 100]
    bars = ax.bar(names, values, color=["#c44e52", "#55a868"])
    ax.set_ylim(0, 112)
    ax.set_ylabel("False-confirmation rate (%)")
    ax.set_title(f"Verification ablation (n={v['n_claims']} known-wrong claims)")
    for b, val in zip(bars, values):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 2, f"{val:.0f}%", ha="center", fontsize=9)

    # Panel 3: network analyst ablation -- layer-flagging strategy recall
    ax = axes[2]
    strat_names = ["Guided\n(CKA+redundancy)", "Uniform\n(all layers)", "Random\n(avg of trials)"]
    strat_values = [n["layer_localization_recall"]["guided"] * 100,
                    n["layer_localization_recall"]["uniform"] * 100,
                    n["layer_localization_recall"]["random_avg"] * 100]
    strat_n = [n["n_layers_flagged"]["guided"], n["n_layers_flagged"]["uniform"], n["n_layers_flagged"]["random"]]
    bars = ax.bar(strat_names, strat_values, color="#8172b2")
    ax.set_ylim(0, 112)
    ax.set_ylabel("Layer-localization recall (%)")
    ax.set_title("Network Analyst flagging-strategy ablation")
    for b, val, cnt in zip(bars, strat_values, strat_n):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 2, f"{val:.0f}%\n(n={cnt})",
                ha="center", fontsize=8.5)

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    png_path = os.path.join(out_dir, f"ablations_{model_id.replace('/', '_')}_{timestamp}.png")
    fig.savefig(png_path, dpi=150)
    plt.close(fig)

    json_path = os.path.join(out_dir, f"ablations_{model_id.replace('/', '_')}_{timestamp}.json")
    with open(json_path, "w") as f:
        json.dump(result, f, indent=2)

    return png_path, json_path


def plot_stage_b_results(result: dict, out_dir: str = "output") -> tuple[str, str]:
    """`result` is exactly what stage_b.run_stage_b() returns."""
    import numpy as np

    os.makedirs(out_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_id = result["target_model_id"]
    n_layers = result["n_layers"]
    behaviors = result["behaviors"]
    per_behavior = result["per_behavior_results"]

    # matrix[i, layer] = 1 if behavior i has a CLAIMED head at that layer, else 0
    matrix = np.zeros((len(behaviors), n_layers))
    for i, r in enumerate(per_behavior):
        for (l, h) in r["claimed_heads"]:
            matrix[i, l] = 1

    fig, axes = plt.subplots(1, 2, figsize=(15, 5), gridspec_kw={"width_ratios": [1.4, 1]})
    fig.suptitle(f"AutoMechInterp -- Stage B Layer Atlas -- {model_id} ({len(behaviors)} behaviors)",
                 fontsize=13)

    ax = axes[0]
    im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(range(n_layers))
    ax.set_xticklabels(range(n_layers))
    ax.set_yticks(range(len(behaviors)))
    ax.set_yticklabels([b.split(" (")[0] for b in behaviors], fontsize=9)
    ax.set_xlabel("Layer")
    ax.set_title("Claimed circuit present at layer, by behavior")
    for i in range(len(behaviors)):
        for l in range(n_layers):
            if matrix[i, l] > 0:
                heads = [f"H{h}" for (ll, h) in per_behavior[i]["claimed_heads"] if ll == l]
                ax.text(l, i, ",".join(heads), ha="center", va="center", fontsize=7)

    ax = axes[1]
    n_with_circuit = matrix.sum(axis=0)
    colors = ["#c44e52" if n == 0 else ("#8172b2" if n == 1 else "#55a868") for n in n_with_circuit]
    ax.bar(range(n_layers), n_with_circuit, color=colors)
    ax.set_xlabel("Layer")
    ax.set_ylabel("# behaviors with a claimed circuit here")
    ax.set_title("Cross-behavior circuit density")
    ax.set_xticks(range(n_layers))
    ax.legend(handles=[
        plt.Rectangle((0, 0), 1, 1, color="#c44e52", label="no circuit"),
        plt.Rectangle((0, 0), 1, 1, color="#8172b2", label="specialized (1)"),
        plt.Rectangle((0, 0), 1, 1, color="#55a868", label="shared/hub (2+)"),
    ], fontsize=8)

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    png_path = os.path.join(out_dir, f"stage_b_atlas_{model_id.replace('/', '_')}_{timestamp}.png")
    fig.savefig(png_path, dpi=150)
    plt.close(fig)

    json_path = os.path.join(out_dir, f"stage_b_atlas_{model_id.replace('/', '_')}_{timestamp}.json")
    with open(json_path, "w") as f:
        json.dump(result, f, indent=2)

    return png_path, json_path


def plot_stage_c_results(result: dict, out_dir: str = "output") -> tuple[str, str]:
    """`result` is exactly what stage_c.run_stage_c() returns."""
    os.makedirs(out_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_id = result["base_model_id"]
    n_layers = result["n_layers"]
    change_loci = set(result["change_loci"])

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(f"AutoMechInterp -- Stage C Cross-Model Layer Diff -- {model_id} vs "
                 f"biomedical fine-tuned stand-in", fontsize=13)

    # Panel 1: CKA per layer
    ax = axes[0]
    colors = ["#c44e52" if l in change_loci else "#4c72b0" for l in range(n_layers)]
    ax.bar(range(n_layers), result["layer_ckas"], color=colors)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Layer")
    ax.set_ylabel("CKA(base, fine-tuned)")
    ax.set_title("Representational alignment per layer")
    ax.axhline(1.0, color="black", linewidth=0.5, linestyle="--")
    ax.legend(handles=[
        plt.Rectangle((0, 0), 1, 1, color="#c44e52", label="change locus (lowest CKA)"),
        plt.Rectangle((0, 0), 1, 1, color="#4c72b0", label="stable layer"),
    ], fontsize=8, loc="lower left")

    # Panel 2: causal transplant confirmation
    ax = axes[1]
    names = [f"L{tr['layer']}" for tr in result["transplant_results"]]
    values = [tr["fraction_of_gap_transferred"] * 100 for tr in result["transplant_results"]]
    names.append(f"L{result['control_layer']}\n(control)")
    values.append(result["control_fraction_transferred"] * 100)
    colors2 = ["#c44e52"] * len(result["transplant_results"]) + ["#8c8c8c"]
    bars = ax.bar(names, values, color=colors2)
    ax.set_ylabel("% of capability gap transferred")
    ax.set_title("Causal transplant confirmation")
    for b, v in zip(bars, values):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.3, f"{v:.0f}%", ha="center", fontsize=9)

    # Panel 3: text summary
    ax = axes[2]
    ax.axis("off")
    lines = [
        f"Base: {result['base_model_id']}",
        f"Fine-tuned: {os.path.basename(result['finetuned_model_path'])}",
        f"Capability gap: {result['capability_gap']:+.2f}",
        f"Change loci: {result['change_loci']}",
        "",
        "Transplant results:",
    ]
    for tr in result["transplant_results"]:
        lines.append(f"  L{tr['layer']}: CKA={tr['cka']:.3f}, "
                      f"{tr['fraction_of_gap_transferred']:.0%} of gap")
    lines.append(f"  Control L{result['control_layer']}: {result['control_fraction_transferred']:.0%} of gap")
    ax.text(0, 1, "\n".join(lines), fontsize=9.5, va="top", family="monospace")
    ax.set_title("Run summary")

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    png_path = os.path.join(out_dir, f"stage_c_diff_{model_id.replace('/', '_')}_{timestamp}.png")
    fig.savefig(png_path, dpi=150)
    plt.close(fig)

    json_path = os.path.join(out_dir, f"stage_c_diff_{model_id.replace('/', '_')}_{timestamp}.json")
    with open(json_path, "w") as f:
        json.dump(result, f, indent=2)

    return png_path, json_path
