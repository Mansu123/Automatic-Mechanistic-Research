"""Recursive Analysis Flow (Sec. 3.2, 3.5):

  Network Analyst profiles model & flags interesting layers
  -> Weight Agent + Safety Agent run a whole-model pass (parallel)
  -> spawns Layer Agents (parallel, one per flagged layer)
  -> for each load-bearing layer: Lens / Probe / Feature Agents (parallel)
  -> each Layer Agent diagnoses its layer & spawns Component Agents
  -> Component Agents run circuit discovery (incl. S-EAP second-order pass)
  -> Steering Agent constructively validates each component-level direction
  -> every claim passes Prover-Skeptic-Judge verification before inclusion

This module is the generic wiring; task-specific prompts and metrics live in
stage_a.py / behaviors.py.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed

from . import config
from .tools import adapter
from .tools.cache import CACHE
from .agents.base import ToolCallBudget, LOG
from .agents.network_analyst import build_network_analyst
from .agents.layer_agent import build_layer_agent
from .agents.component_agent import build_component_agent
from .agents.skeptic import build_skeptic
from .agents.judge import adjudicate
from .agents.orchestrator import HypothesisLedger, decide_layers_to_spawn
from .agents.lens_agent import build_lens_agent
from .agents.probe_agent import build_probe_agent
from .agents.feature_agent import build_feature_agent
from .agents.steering_agent import build_steering_agent
from .agents.weight_agent import build_weight_agent
from .agents.safety_agent import build_safety_agent

LOAD_BEARING_FRAC = 0.15


def run_hierarchy(handle: adapter.ModelHandle, task: dict,
                   backend_kind: str | None = None, backend_kwargs: dict | None = None,
                   deep_techniques: bool | None = None) -> dict:
    backend_kind = backend_kind or config.LLM_BACKEND
    backend_kwargs = backend_kwargs or {}
    if deep_techniques is None:
        deep_techniques = config.DEEP_TECHNIQUES
    base_budget = config.GLOBAL_TOOL_CALL_BUDGET
    if deep_techniques:
        base_budget = max(base_budget, 500)
    budget = ToolCallBudget(global_remaining=[base_budget], per_agent_limit=config.PER_LAYER_AGENT_BUDGET)
    ledger = HypothesisLedger()

    cp, xp = task["clean_prompt"], task["corrupted_prompt"]
    io_t, s_t = task["io_token"], task["s_token"]
    pos_texts = task.get("contrastive_pos", [cp])
    neg_texts = task.get("contrastive_neg", [xp])
    ref_texts = task.get("probe_texts", [cp])

    LOG.emit("Orchestrator", f"delegating stage-0 profiling to Network Analyst (backend={backend_kind})")
    na_agent, na_state = build_network_analyst(handle, task["probe_texts"], task["task_metric_fn"],
                                                backend_kind, budget, backend_kwargs)
    na_agent.run()
    flagged_layers = na_state.get("flagged_layers", [])

    # ---- whole-model technique pass: Weight + Safety agents ----
    weight_state, safety_state = {}, {}
    if deep_techniques:
        def _weight():
            a, s = build_weight_agent(handle, backend_kind, budget, backend_kwargs)
            a.run(); return s

        def _safety():
            a, s = build_safety_agent(handle, ref_texts, cp, backend_kind, budget, backend_kwargs)
            a.run(); return s
        with ThreadPoolExecutor(max_workers=2) as ex:
            fw, fs = ex.submit(_weight), ex.submit(_safety)
            weight_state, safety_state = fw.result(), fs.result()
        LOG.emit("Orchestrator", "whole-model Weight + Safety technique pass complete")

    layers_to_run = decide_layers_to_spawn(flagged_layers, budget.global_remaining[0])
    layer_states: dict[int, dict] = {}

    def run_one_layer(layer_idx):
        agent, state = build_layer_agent(handle, layer_idx, cp, xp, io_t, s_t,
                                          backend_kind, budget, backend_kwargs)
        agent.run()
        return layer_idx, state

    if layers_to_run:
        with ThreadPoolExecutor(max_workers=len(layers_to_run)) as ex:
            futures = [ex.submit(run_one_layer, l) for l in layers_to_run]
            for f in as_completed(futures):
                layer_idx, state = f.result()
                layer_states[layer_idx] = state

    # ---- per load-bearing layer: Lens / Probe / Feature agents ----
    load_bearing = [l for l, s in layer_states.items()
                    if s.get("fraction_recovered", 0.0) >= LOAD_BEARING_FRAC]
    lens_states: dict[int, dict] = {}
    probe_states: dict[int, dict] = {}
    feature_states: dict[int, dict] = {}
    if deep_techniques and load_bearing:
        LOG.emit("Orchestrator", f"spawning Lens/Probe/Feature agents for load-bearing layers {load_bearing}")

        def run_layer_techniques(layer_idx):
            la, ls = build_lens_agent(handle, layer_idx, cp, xp, io_t, s_t, backend_kind, budget, backend_kwargs)
            pa, ps = build_probe_agent(handle, layer_idx, pos_texts, neg_texts, cp, io_t, s_t,
                                       backend_kind, budget, backend_kwargs)
            fa, fs = build_feature_agent(handle, layer_idx, pos_texts + neg_texts, cp,
                                         backend_kind, budget, backend_kwargs)
            la.run(); pa.run(); fa.run()
            return layer_idx, ls, ps, fs
        with ThreadPoolExecutor(max_workers=min(4, len(load_bearing))) as ex:
            futures = [ex.submit(run_layer_techniques, l) for l in load_bearing]
            for f in as_completed(futures):
                li, ls, ps, fs = f.result()
                lens_states[li], probe_states[li], feature_states[li] = ls, ps, fs

    component_layers = [l for l, s in layer_states.items() if s.get("spawn_component_agent")]

    def run_one_component(layer_idx):
        agent, state = build_component_agent(handle, layer_idx, cp, xp, io_t, s_t, task["n_heads"],
                                              backend_kind, budget, backend_kwargs)
        agent.run()
        return layer_idx, state

    steering_states: dict[int, dict] = {}
    if component_layers:
        LOG.emit("Orchestrator", f"attention-driven effect found at layers {component_layers}; "
                                   "spawning Component Agents")
        with ThreadPoolExecutor(max_workers=len(component_layers)) as ex:
            futures = [ex.submit(run_one_component, l) for l in component_layers]
            for f in as_completed(futures):
                layer_idx, state = f.result()
                for (l, h) in state.get("circuit", []):
                    ledger.add(l, h, f"ComponentAgent{layer_idx}")

        if deep_techniques:
            def run_steering(layer_idx):
                a, s = build_steering_agent(handle, layer_idx, pos_texts, neg_texts, cp, io_t, s_t,
                                            backend_kind, budget, backend_kwargs)
                a.run()
                return layer_idx, s
            with ThreadPoolExecutor(max_workers=len(component_layers)) as ex:
                for f in as_completed([ex.submit(run_steering, l) for l in component_layers]):
                    li, s = f.result()
                    steering_states[li] = s
            LOG.emit("Orchestrator", "Steering Agent constructive-validation pass complete")

    claimed_heads = ledger.merged_circuit()
    all_heads = [(l, h) for l in range(handle.n_layers) for h in range(task["n_heads"])]

    verdict_report = None
    if claimed_heads:
        LOG.emit("Orchestrator", f"sending claim {claimed_heads} to Skeptic")
        sk_agent, sk_state = build_skeptic(handle, claimed_heads, all_heads, task["eval_prompts"],
                                            task["behavior"], backend_kind, budget, backend_kwargs)
        sk_agent.run()
        verdict_report = adjudicate(claimed_heads, sk_state, backend_kind, backend_kwargs)
        LOG.emit("Judge", f"Verdict [{verdict_report['verdict']}]: {verdict_report['reasoning']}")
    else:
        LOG.emit("Orchestrator", "no component-level claim survived Layer Agent triage; nothing to verify")

    return {
        "flagged_layers": flagged_layers,
        "layer_states": layer_states,
        "claimed_heads": claimed_heads,
        "verdict": verdict_report,
        "weight_findings": weight_state,
        "safety_findings": safety_state,
        "lens_findings": lens_states,
        "probe_findings": probe_states,
        "feature_findings": feature_states,
        "steering_findings": steering_states,
        "tool_calls_spent": base_budget - budget.global_remaining[0],
        "cache_stats": CACHE.stats(),
    }
