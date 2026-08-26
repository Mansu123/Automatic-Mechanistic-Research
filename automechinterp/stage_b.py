"""Stage B -- Autonomous Layer Atlas (Sec. 4.4), run on GPT-2.

The proposal's Stage B target was Gemma 2 9B across 24 behaviors; this runs
the identical methodology -- same hierarchy, same tools, same "which layers
does the Network Analyst flag, which circuits do Component Agents confirm"
pipeline -- on GPT-2 across 4 behaviors (one per category named in Sec. 4.4:
coreference, syntax, factual recall, lexical semantics), which is what fits
this machine. Every layer gets a role card built from real per-behavior
results, not a template.
"""
from __future__ import annotations

from . import config
from .behaviors import ALL_BEHAVIORS
from .hierarchy import run_hierarchy
from .tools import adapter
from .agents.base import LOG


def _role_hypothesis(layer_idx: int, flagged_by: list[str], circuits: dict[str, list]) -> str:
    """Based on CLAIMED circuits (component-level, precise), not the coarse
    "flagged" layer set (Network-Analyst-level triage, deliberately permissive
    -- Sec. 3.1's whole point is that Network Analyst casts a wide net and
    Component Agents narrow it down). A layer flagged by every behavior but
    claimed by none just means it was investigated everywhere and never
    panned out at the circuit level -- the atlas should say that, not call it
    "general-purpose"."""
    n_total = len(ALL_BEHAVIORS)
    behaviors_with_circuit = sorted(circuits.keys())
    n_with_circuit = len(behaviors_with_circuit)
    if n_with_circuit == 0:
        investigated = f" (investigated by {len(flagged_by)}/{n_total}, no circuit survived)" if flagged_by else ""
        return f"no confirmed circuit at this layer{investigated}"
    if n_with_circuit == n_total:
        return f"hub layer -- circuit found for all {n_total} tested behaviors"
    if n_with_circuit == 1:
        return f"specialized -- circuit found only for '{behaviors_with_circuit[0]}'"
    return f"shared circuit layer -- found for {n_with_circuit}/{n_total} behaviors: {behaviors_with_circuit}"


def run_stage_b(backend_kind: str = "heuristic", target_model_id: str | None = None) -> dict:
    backend_kind = backend_kind or config.LLM_BACKEND
    target_model_id = target_model_id or config.TARGET_MODEL_ID
    print("=" * 78)
    print(f"STAGE B -- Autonomous Layer Atlas ({target_model_id}, "
          f"{len(ALL_BEHAVIORS)} behaviors)")
    print("=" * 78)

    handle = adapter.register_model(target_model_id, device=config.DEVICE)
    LOG.emit("System", f"registered {target_model_id}: {handle.n_layers} layers (device={config.DEVICE})")

    per_behavior_results = []
    layer_flagged_by: dict[int, list[str]] = {l: [] for l in range(handle.n_layers)}
    layer_circuits: dict[int, dict[str, list]] = {l: {} for l in range(handle.n_layers)}

    for build_behavior in ALL_BEHAVIORS:
        task = build_behavior(handle)
        LOG.emit("System", f"--- behavior: {task['behavior']} ---")
        LOG.emit("System", f"probe: '{task['clean_prompt']}' -> expect '{task['io_token']}' "
                             f"over '{task['s_token']}'")
        result = run_hierarchy(handle, task, backend_kind=backend_kind)
        per_behavior_results.append({
            "behavior": task["behavior"], "category": task["category"],
            "flagged_layers": result["flagged_layers"], "claimed_heads": result["claimed_heads"],
            "verdict": result["verdict"]["verdict"] if result["verdict"] else None,
            "tool_calls_spent": result["tool_calls_spent"],
        })
        for l in result["flagged_layers"]:
            layer_flagged_by[l].append(task["behavior"])
        for (l, h) in result["claimed_heads"]:
            layer_circuits[l].setdefault(task["behavior"], []).append((l, h))

    atlas = {}
    for l in range(handle.n_layers):
        atlas[l] = {
            "flagged_by_behaviors": layer_flagged_by[l],
            "n_behaviors_flagged": len(layer_flagged_by[l]),
            "circuits_by_behavior": layer_circuits[l],
            "role_hypothesis": _role_hypothesis(l, layer_flagged_by[l], layer_circuits[l]),
        }

    # Cross-behavior consistency (Jaccard/IoU of flagged-layer sets between
    # every pair of behaviors) -- a repurposing of Sec. 4.4's "do independent
    # runs converge" check: here it's "do DIFFERENT behaviors converge on the
    # same layers," a genuinely different (and arguably more informative)
    # question, not the literal same-category-multiple-seeds version.
    behaviors_list = [r["behavior"] for r in per_behavior_results]
    pairwise_iou = {}
    for i in range(len(behaviors_list)):
        for j in range(i + 1, len(behaviors_list)):
            si = set(per_behavior_results[i]["flagged_layers"])
            sj = set(per_behavior_results[j]["flagged_layers"])
            union = si | sj
            iou = len(si & sj) / len(union) if union else 0.0
            pairwise_iou[f"{behaviors_list[i]} vs {behaviors_list[j]}"] = iou

    print("\n" + "-" * 78)
    print("LAYER ATLAS")
    print("-" * 78)
    for l in range(handle.n_layers):
        print(f"L{l:2d}: {atlas[l]['role_hypothesis']}")

    print("\nPer-behavior summary:")
    for r in per_behavior_results:
        print(f"  {r['behavior']}: flagged={r['flagged_layers']}, claimed={r['claimed_heads']}, "
              f"verdict={r['verdict']}")

    return {
        "target_model_id": config.TARGET_MODEL_ID,
        "n_layers": handle.n_layers,
        "behaviors": behaviors_list,
        "per_behavior_results": per_behavior_results,
        "layer_atlas": {str(l): v for l, v in atlas.items()},
        "pairwise_consistency_iou": pairwise_iou,
    }
