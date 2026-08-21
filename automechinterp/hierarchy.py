"""Recursive Analysis Flow (Sec. 3.2, 3.5):

  Network Analyst profiles model & flags interesting layers
  -> spawns Layer Agents (parallel, one per flagged layer)
  -> each Layer Agent diagnoses its layer & spawns Component Agents
  -> Component Agents run circuit discovery + interventions
  -> findings bubble up: component evidence -> layer report -> claim
  -> every claim passes Prover-Skeptic-Judge verification before inclusion

This module is the generic wiring (Phase 2 of the build, per the proposal's
Weeks 3-8 timeline); task-specific prompts and metrics live in stage_a.py.
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


def run_hierarchy(handle: adapter.ModelHandle, task: dict,
                   backend_kind: str | None = None, backend_kwargs: dict | None = None) -> dict:
    backend_kind = backend_kind or config.LLM_BACKEND
    backend_kwargs = backend_kwargs or {}
    budget = ToolCallBudget(global_remaining=[config.GLOBAL_TOOL_CALL_BUDGET],
                             per_agent_limit=config.PER_LAYER_AGENT_BUDGET)
    ledger = HypothesisLedger()

    LOG.emit("Orchestrator", f"delegating stage-0 profiling to Network Analyst (backend={backend_kind})")
    na_agent, na_state = build_network_analyst(handle, task["probe_texts"], task["task_metric_fn"],
                                                backend_kind, budget, backend_kwargs)
    na_agent.run()
    flagged_layers = na_state.get("flagged_layers", [])

    layers_to_run = decide_layers_to_spawn(flagged_layers, budget.global_remaining[0])
    layer_states: dict[int, dict] = {}

    def run_one_layer(layer_idx):
        agent, state = build_layer_agent(handle, layer_idx, task["clean_prompt"], task["corrupted_prompt"],
                                          task["io_token"], task["s_token"], backend_kind, budget, backend_kwargs)
        agent.run()
        return layer_idx, state

    if layers_to_run:
        with ThreadPoolExecutor(max_workers=len(layers_to_run)) as ex:
            futures = [ex.submit(run_one_layer, l) for l in layers_to_run]
            for f in as_completed(futures):
                layer_idx, state = f.result()
                layer_states[layer_idx] = state

    component_layers = [l for l, s in layer_states.items() if s.get("spawn_component_agent")]

    def run_one_component(layer_idx):
        agent, state = build_component_agent(handle, layer_idx, task["clean_prompt"], task["corrupted_prompt"],
                                              task["io_token"], task["s_token"], task["n_heads"],
                                              backend_kind, budget, backend_kwargs)
        agent.run()
        return layer_idx, state

    if component_layers:
        LOG.emit("Orchestrator", f"attention-driven effect found at layers {component_layers}; "
                                   "spawning Component Agents")
        with ThreadPoolExecutor(max_workers=len(component_layers)) as ex:
            futures = [ex.submit(run_one_component, l) for l in component_layers]
            for f in as_completed(futures):
                layer_idx, state = f.result()
                for (l, h) in state.get("circuit", []):
                    ledger.add(l, h, f"ComponentAgent{layer_idx}")

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
        "tool_calls_spent": config.GLOBAL_TOOL_CALL_BUDGET - budget.global_remaining[0],
        "cache_stats": CACHE.stats(),
    }
