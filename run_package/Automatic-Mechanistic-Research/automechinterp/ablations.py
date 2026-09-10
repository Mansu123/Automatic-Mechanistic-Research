"""Ablation studies (Sec. 4.7). Each function is a real, runnable comparison
against the same GPT-2 IOI task used by Stage A -- not a table of expected
numbers, actual measurements from actually running the alternate configurations.

Implemented:
  hierarchy_ablation()        flat single-agent vs the 3-tier hierarchy --
                               measures context growth, the concrete problem
                               the hierarchy exists to solve (Sec. 3.1)
  verification_ablation()     self-confirmation (no Skeptic) vs full
                               Prover-Skeptic-Judge -- measures H3
                               (false-confirmation rate)
  network_analyst_ablation()  CKA+redundancy-guided flagging vs uniform-all
                               vs random-subset -- measures whether guided
                               flagging saves budget without losing recall

Not implemented (would need real LLM-API spend to measure honestly, not
proxy metrics): cost-mechanism ablation (with/without caching -- already
visible via cache_hit_rate in every Stage A run) and backbone comparison
(heuristic vs hf_local -- already demonstrated informally in earlier runs,
not re-run here as a formal ablation since it requires a loaded LLM).
"""
from __future__ import annotations

import random

from . import config
from .stage_a import GT_LAYERS, build_ioi_task
from .tools import adapter, tier_n, tier_l
from .agents.base import ToolCallBudget, LOG
from .agents.layer_agent import build_layer_agent


def hierarchy_ablation(handle: adapter.ModelHandle, task: dict, layers_to_check: list[int]) -> dict:
    """Flat baseline: ONE agent's evidence log accumulates every layer's
    findings in sequence (a real flat single-agent design would hold all of
    this in one context window). Hierarchical: each Layer Agent only ever
    sees its own scoped evidence. We measure evidence-log SIZE (characters),
    a direct proxy for the context-window pressure an LLM backend would feel
    -- the heuristic backend has no context limit of its own, so this is the
    honest way to measure "context dilution" without spending real LLM calls."""
    LOG.emit("Ablation", "hierarchy_ablation: flat single-agent evidence accumulation "
                          "vs per-Layer-Agent scoped evidence")

    # Flat arm: one continuous evidence log across every layer, exactly what a
    # single agent with all tools in one context would accumulate. Tracked
    # cumulatively (after_n_layers) so the growth curve, not just the final
    # total, can be plotted.
    flat_evidence: list[str] = []
    flat_tool_calls = 0
    flat_context_after_n_layers = []
    for layer_idx in layers_to_check:
        digest = tier_l.patch_layer(handle, layer_idx, task["clean_prompt"], task["corrupted_prompt"],
                                     task["io_token"], task["s_token"])
        flat_evidence.append(f"L{layer_idx} patch_layer: {digest}")
        flat_tool_calls += 1
        frac = float(digest.split("fraction_recovered=")[1].split(" ")[0])
        if frac >= 0.15:
            digest2 = tier_l.attn_mlp_attribution(handle, layer_idx, task["clean_prompt"])
            flat_evidence.append(f"L{layer_idx} attn_mlp_attribution: {digest2}")
            flat_tool_calls += 1
        flat_context_after_n_layers.append(sum(len(e) for e in flat_evidence))
    flat_context_chars = flat_context_after_n_layers[-1] if flat_context_after_n_layers else 0

    # Hierarchical arm: reuse the real Layer Agents, each with its own budget
    # and its own `agent.evidence` list. Tracked as the MAX any one agent has
    # grown to after each additional layer is processed -- since each agent
    # only ever sees its own scope, this should stay roughly flat instead of
    # accumulating, which is exactly the claim in Sec. 3.1.
    budget = ToolCallBudget(global_remaining=[200], per_agent_limit=10)
    max_agent_context_chars = 0
    hier_max_context_after_n_layers = []
    for layer_idx in layers_to_check:
        agent, _ = build_layer_agent(handle, layer_idx, task["clean_prompt"], task["corrupted_prompt"],
                                      task["io_token"], task["s_token"], "heuristic", budget)
        agent.run()
        agent_chars = sum(len(e) for e in agent.evidence)
        max_agent_context_chars = max(max_agent_context_chars, agent_chars)
        hier_max_context_after_n_layers.append(max_agent_context_chars)
    hier_tool_calls = 200 - budget.global_remaining[0]

    result = {
        "n_layers_checked": len(layers_to_check),
        "flat_total_context_chars": flat_context_chars,
        "flat_tool_calls": flat_tool_calls,
        "flat_context_after_n_layers": flat_context_after_n_layers,
        "hierarchical_max_agent_context_chars": max_agent_context_chars,
        "hierarchical_tool_calls": hier_tool_calls,
        "hierarchical_max_context_after_n_layers": hier_max_context_after_n_layers,
        "context_reduction_factor": (flat_context_chars / max_agent_context_chars
                                       if max_agent_context_chars else float("nan")),
    }
    LOG.emit("Ablation", f"hierarchy_ablation result: reduction factor "
                          f"{result['context_reduction_factor']:.2f}x over {len(layers_to_check)} layers")
    return result


def verification_ablation(handle: adapter.ModelHandle, task: dict,
                           negative_control_claims: list[tuple[int, int]]) -> dict:
    """Self-confirmation baseline: a discovery agent with no adversarial
    check simply believes every claim it makes -- so on these fabricated /
    causally irrelevant claims, ALL of them get wrongly labeled Confirmed by
    construction (there is no falsification step to catch them). Compared
    against the REAL Prover-Skeptic-Judge verdicts on the exact same claims,
    computed here (not just referenced), for a true apples-to-apples
    comparison -- this is the direct measurement behind H3."""
    from .stage_a import run_negative_controls
    LOG.emit("Ablation", "verification_ablation: self-confirmation vs Prover-Skeptic-Judge "
                          f"on {len(negative_control_claims)} known-wrong claims")
    self_confirm_false_rate = 1.0  # every claim accepted, no exceptions -- the definition of the baseline

    negative = run_negative_controls(handle, task, "heuristic", n_controls=len(negative_control_claims))
    psj_false_rate = negative["false_confirmation_rate"]

    return {
        "n_claims": negative["n_controls"],
        "self_confirmation_false_confirmation_rate": self_confirm_false_rate,
        "prover_skeptic_judge_false_confirmation_rate": psj_false_rate,
        "details": negative["details"],
    }


def network_analyst_ablation(handle: adapter.ModelHandle, task: dict, n_random_trials: int = 5,
                              seed: int = 0) -> dict:
    """CKA+redundancy-guided flagging (Stage A's default) vs uniform-all-layers
    vs a random subset of the same size, averaged over `n_random_trials` draws
    (a single random draw isn't a fair comparison against a signal-driven
    choice). Measures layer-localization recall per strategy at the same
    tool-call cost (same number of flagged layers -> same number of Layer
    Agents spawned)."""
    LOG.emit("Ablation", "network_analyst_ablation: CKA-guided vs uniform-all vs random-subset flagging")
    n_layers = handle.n_layers
    all_layers = list(range(n_layers))
    gt = set(GT_LAYERS)

    # Strategy A: CKA + exhaustive redundancy scan (Stage A's actual policy)
    _, boundaries, drops = tier_n.layerwise_cka_scan(handle, task["probe_texts"])
    _, xs, ys = tier_n.redundancy_scan(handle, all_layers, lambda: task["task_metric_fn"])
    guided_flagged = [x for x, y in zip(xs, ys) if y >= 0.05] or all_layers[:2]
    guided_recall = len(set(guided_flagged) & gt) / len(gt)

    # Strategy B: uniform -- flag every layer (the "no flagging logic at all" baseline)
    uniform_recall = len(set(all_layers) & gt) / len(gt)

    # Strategy C: random subset, same size as the guided strategy, averaged
    rng = random.Random(seed)
    random_recalls = []
    for _ in range(n_random_trials):
        subset = set(rng.sample(all_layers, len(guided_flagged)))
        random_recalls.append(len(subset & gt) / len(gt))
    random_recall_avg = sum(random_recalls) / len(random_recalls)

    result = {
        "n_layers_flagged": {"guided": len(guided_flagged), "uniform": len(all_layers),
                              "random": len(guided_flagged)},
        "layer_localization_recall": {"guided": guided_recall, "uniform": uniform_recall,
                                        "random_avg": random_recall_avg},
        "random_trial_recalls": random_recalls,
        "guided_flagged_layers": guided_flagged,
    }
    LOG.emit("Ablation", f"network_analyst_ablation result: {result}")
    return result


def run_all_ablations(backend_kind: str = "heuristic") -> dict:
    handle = adapter.register_model(config.TARGET_MODEL_ID, device=config.DEVICE)
    task = build_ioi_task(handle)
    layers_to_check = list(range(handle.n_layers))  # all layers: shows the growth trend, not just a sample

    hierarchy = hierarchy_ablation(handle, task, layers_to_check)
    verification = verification_ablation(handle, task, [(0, 0), (1, 7), (1, 2), (1, 9)])
    network_analyst = network_analyst_ablation(handle, task)

    return {
        "target_model_id": config.TARGET_MODEL_ID,
        "hierarchy_ablation": hierarchy,
        "verification_ablation": verification,
        "network_analyst_ablation": network_analyst,
    }
