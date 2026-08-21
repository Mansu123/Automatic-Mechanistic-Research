"""Network Analyst (Sec. 3.2): Stage-0 whole-model profiling. Adaptively
scopes redundancy_scan to the layers CKA already flagged instead of sweeping
every layer unconditionally -- the concrete behavior that answers
gap_work.py's "dalvi-fixed-pipeline" gap: the next tool call is chosen
from evidence, not fixed in advance.
"""
from __future__ import annotations

from ..llm_backends import make_backend
from ..tools import adapter, tier_n
from .base import Agent, ToolCallBudget, LOG


def build_network_analyst(handle: adapter.ModelHandle, probe_texts: list[str],
                           task_metric_fn, backend_kind: str, budget: ToolCallBudget,
                           backend_kwargs: dict | None = None):
    state: dict = {}
    order = ["profile_network", "layerwise_cka_scan", "redundancy_scan_flagged"]

    def do_profile():
        digest = tier_n.profile_network(handle)
        state["profile"] = digest
        return digest

    def do_cka_scan():
        digest, boundaries, drops = tier_n.layerwise_cka_scan(handle, probe_texts)
        state["cka_digest"] = digest
        state["cka_boundaries"] = boundaries
        state["cka_drops"] = drops
        return digest

    def do_redundancy_flagged():
        boundaries = state.get("cka_boundaries", [])
        drops = state.get("cka_drops", [])
        cka_flagged = [b for b, d in zip(boundaries, drops) if d >= 0.15]

        # Sec. 4.7 "Network Analyst ablation" trade-off, decided at run time:
        # small/cheap models get an exhaustive redundancy scan (CKA is only a
        # coarse, task-agnostic signal -- Sec. 5.2 risk); on deeper models,
        # scope to the CKA-flagged boundaries to keep the tool-call budget sane.
        if handle.n_layers <= 16:
            candidates = list(range(handle.n_layers))
            LOG.emit("NetworkAnalyst", f"{handle.n_layers} layers is cheap enough to redundancy-scan "
                                         "exhaustively rather than trust the task-agnostic CKA signal alone")
        else:
            candidates = cka_flagged or list(range(min(4, handle.n_layers)))
            LOG.emit("NetworkAnalyst", f"adaptively scoping redundancy_scan to CKA-flagged layers "
                                         f"{candidates} (not all {handle.n_layers} layers)",
                     gap_id="dalvi-fixed-pipeline")

        digest, xs, ys = tier_n.redundancy_scan(handle, candidates, lambda: task_metric_fn)
        state["redundancy_digest"] = digest
        state["flagged_layers"] = [x for x, y in zip(xs, ys) if y >= 0.05] or cka_flagged or candidates[:2]
        return digest

    tools = {
        "profile_network": do_profile,
        "layerwise_cka_scan": do_cka_scan,
        "redundancy_scan_flagged": do_redundancy_flagged,
    }

    def policy_fn(evidence_text: str, tool_menu: list[str]) -> dict:
        if "profile" not in state:
            return {"action": "profile_network", "args": {}, "reasoning": "heuristic Network Analyst step 1/3"}
        if "cka_digest" not in state:
            return {"action": "layerwise_cka_scan", "args": {}, "reasoning": "heuristic Network Analyst step 2/3"}
        if "redundancy_digest" not in state:
            return {"action": "redundancy_scan_flagged", "args": {}, "reasoning": "heuristic Network Analyst step 3/3"}
        return {"action": "stop", "args": {}, "reasoning": "stage-0 profiling complete"}

    kwargs = backend_kwargs or {}
    backend = make_backend(backend_kind, policy_fn=policy_fn, **kwargs)
    agent = Agent(backend, tools, budget, max_steps=len(order) + 1)
    agent.name = "NetworkAnalyst"
    agent.system_prompt = (
        "You are the Network Analyst in a hierarchical mechanistic-interpretability system. "
        "Profile the target model, scan for representation shifts across depth, and flag the "
        "layers most likely to matter for the task, so Layer Agents only need to be spawned "
        "for those layers."
    )
    return agent, state
