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
import time
import threading

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
                   deep_techniques: bool | None = None,
                   tool_budget: int | None = None, verification_fraction: float = .30,
                   max_workers: int = 1, max_component_layers: int | None = None,
                   max_specialist_layers: int | None = None) -> dict:
    backend_kind = backend_kind or config.LLM_BACKEND
    backend_kwargs = backend_kwargs or {}
    if deep_techniques is None:
        deep_techniques = config.DEEP_TECHNIQUES
    base_budget = tool_budget if tool_budget is not None else config.GLOBAL_TOOL_CALL_BUDGET
    if base_budget < 4 or not 0 < verification_fraction < 1 or max_workers < 1:
        raise ValueError("Invalid explicit budget or worker configuration")
    reserved = max(4, int(base_budget * verification_fraction))
    budget = ToolCallBudget(global_remaining=[base_budget-reserved], per_agent_limit=config.PER_LAYER_AGENT_BUDGET)
    trace_start, started = len(LOG.entries), time.perf_counter()
    agent_runs, trace_lock = [], threading.Lock()
    def execute(agent):
        agent.run()
        with trace_lock:
            agent_runs.append({"agent":agent.name,"telemetry":agent.telemetry,
                               "errors":sorted(agent._failed_tools),"evidence":agent.evidence,
                               "available_tools":list(agent.tools),"termination":getattr(agent,"termination","unknown")})
    ledger = HypothesisLedger()

    cp, xp = task["clean_prompt"], task["corrupted_prompt"]
    io_t, s_t = task["io_token"], task["s_token"]
    pos_texts = task.get("contrastive_pos", [cp])
    neg_texts = task.get("contrastive_neg", [xp])
    ref_texts = task.get("probe_texts", [cp])

    LOG.emit("Orchestrator", f"delegating stage-0 profiling to Network Analyst (backend={backend_kind})")
    na_agent, na_state = build_network_analyst(handle, task["probe_texts"], task["task_metric_fn"],
                                                backend_kind, budget, backend_kwargs)
    execute(na_agent)
    flagged_layers = na_state.get("flagged_layers", [])

    # ---- whole-model technique pass: Weight + Safety agents ----
    weight_state, safety_state = {}, {}
    if deep_techniques:
        def _weight():
            a, s = build_weight_agent(handle, backend_kind, budget, backend_kwargs)
            execute(a); return s

        def _safety():
            a, s = build_safety_agent(handle, ref_texts, cp, backend_kind, budget, backend_kwargs)
            execute(a); return s
        with ThreadPoolExecutor(max_workers=min(max_workers,2)) as ex:
            fw, fs = ex.submit(_weight), ex.submit(_safety)
            weight_state, safety_state = fw.result(), fs.result()
        LOG.emit("Orchestrator", "whole-model Weight + Safety technique pass complete")

    import os
    layers_to_run = (list(range(handle.n_layers)) if os.environ.get("AMI_LAYER_SCOPE") == "all"
                     else decide_layers_to_spawn(flagged_layers, budget.global_remaining[0]))
    layer_states: dict[int, dict] = {}

    def run_one_layer(layer_idx):
        agent, state = build_layer_agent(handle, layer_idx, cp, xp, io_t, s_t,
                                          backend_kind, budget, backend_kwargs)
        execute(agent)
        return layer_idx, state

    if layers_to_run:
        with ThreadPoolExecutor(max_workers=min(max_workers,len(layers_to_run))) as ex:
            futures = [ex.submit(run_one_layer, l) for l in layers_to_run]
            for f in as_completed(futures):
                layer_idx, state = f.result()
                layer_states[layer_idx] = state

    component_layers = sorted([l for l, s in layer_states.items() if s.get("spawn_component_agent")],
                              key=lambda l: (-layer_states[l].get("fraction_recovered", 0), l))
    deferred_component_layers = component_layers[max_component_layers:] if max_component_layers is not None else []
    component_layers = component_layers[:max_component_layers] if max_component_layers is not None else component_layers
    component_states = {}

    def run_one_component(layer_idx):
        agent, state = build_component_agent(handle, layer_idx, cp, xp, io_t, s_t, task["n_heads"],
                                              backend_kind, budget, backend_kwargs)
        execute(agent)
        return layer_idx, state

    steering_states: dict[int, dict] = {}
    if component_layers:
        LOG.emit("Orchestrator", f"attention-driven effect found at layers {component_layers}; "
                                   "spawning Component Agents")
        with ThreadPoolExecutor(max_workers=min(max_workers,len(component_layers))) as ex:
            futures = [ex.submit(run_one_component, l) for l in component_layers]
            for f in as_completed(futures):
                layer_idx, state = f.result()
                component_states[layer_idx] = state
                for (l, h) in state.get("circuit", []):
                    ledger.add(l, h, f"ComponentAgent{layer_idx}")

        if deep_techniques:
            def run_steering(layer_idx):
                a, s = build_steering_agent(handle, layer_idx, pos_texts, neg_texts, cp, io_t, s_t,
                                            backend_kind, budget, backend_kwargs)
                execute(a)
                return layer_idx, s
            with ThreadPoolExecutor(max_workers=min(max_workers,len(component_layers))) as ex:
                for f in as_completed([ex.submit(run_steering, l) for l in component_layers]):
                    li, s = f.result()
                    steering_states[li] = s
            LOG.emit("Orchestrator", "Steering Agent constructive-validation pass complete")

    # ---- per load-bearing layer: Lens / Probe / Feature agents ----
    load_bearing = [l for l, s in layer_states.items()
                    if s.get("fraction_recovered", 0.0) >= LOAD_BEARING_FRAC]
    load_bearing = sorted(load_bearing, key=lambda l: (-layer_states[l].get("fraction_recovered", 0), l))
    deferred_specialist_layers = load_bearing[max_specialist_layers:] if max_specialist_layers is not None else []
    load_bearing = load_bearing[:max_specialist_layers] if max_specialist_layers is not None else load_bearing
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
            execute(la); execute(pa); execute(fa)
            return layer_idx, ls, ps, fs
        with ThreadPoolExecutor(max_workers=min(max_workers, len(load_bearing))) as ex:
            futures = [ex.submit(run_layer_techniques, l) for l in load_bearing]
            for f in as_completed(futures):
                li, ls, ps, fs = f.result()
                lens_states[li], probe_states[li], feature_states[li] = ls, ps, fs

    claimed_heads = ledger.merged_circuit()
    all_heads = [(l, h) for l in range(handle.n_layers) for h in range(task["n_heads"])]

    verdict_report = None
    sk_state = {}
    verification_budget = ToolCallBudget([reserved + budget.global_remaining[0]],
                                        config.PER_LAYER_AGENT_BUDGET)
    if claimed_heads:
        LOG.emit("Orchestrator", f"sending claim {claimed_heads} to Skeptic")
        sk_agent, sk_state = build_skeptic(handle, claimed_heads, all_heads, task["eval_prompts"],
                                            task["behavior"], backend_kind, verification_budget, backend_kwargs,
                                            stress_prompts=task.get("stress_prompts"),
                                            stress_data_status=task.get("stress_data_status","unvalidated"))
        execute(sk_agent)
        verdict_report = adjudicate(claimed_heads, sk_state, backend_kind, backend_kwargs)
        LOG.emit("Judge", f"Verdict [{verdict_report['verdict']}]: {verdict_report['reasoning']}")
    else:
        LOG.emit("Orchestrator", "no component-level claim survived Layer Agent triage; nothing to verify")

    return {
        "n_layers": handle.n_layers,
        "network_findings": na_state,
        "flagged_layers": flagged_layers,
        "layer_states": layer_states,
        "component_states": component_states,
        "search_coverage": {"layers_scanned": sorted(layer_states),
                            "component_layers": component_layers,
                            "deferred_component_layers": deferred_component_layers,
                            "specialist_layers": load_bearing,
                            "deferred_specialist_layers": deferred_specialist_layers},
        "claimed_heads": claimed_heads,
        "verdict": verdict_report,
        "weight_findings": weight_state,
        "safety_findings": safety_state,
        "lens_findings": lens_states,
        "probe_findings": probe_states,
        "feature_findings": feature_states,
        "steering_findings": steering_states,
        "tool_calls_spent": base_budget - verification_budget.global_remaining[0],
        "verification_state": sk_state,
        "agent_runs": agent_runs,
        "trace": LOG.entries[trace_start:],
        "execution_status": "tool_errors" if any(a["errors"] for a in agent_runs) else "completed",
        "run_configuration": {"backend":backend_kind,"backend_kwargs":backend_kwargs,
                              "deep_techniques":deep_techniques,"tool_budget":base_budget,
                              "verification_reserved":reserved,"max_workers":max_workers,
                              "model_revision":handle.revision,"attention_backend":handle.attn_impl,
                              "max_component_layers":max_component_layers,
                              "max_specialist_layers":max_specialist_layers,
                              "discovery_metric":"full_continuation_log_probability_margin",
                              "layer_intervention":"last_prompt_position_block_update"},
        "wall_seconds": time.perf_counter()-started,
        "cache_stats": CACHE.stats(),
    }
