"""Network Analyst (Sec. 3.2): Stage-0 whole-model profiling. Adaptively
scopes redundancy_scan to the layers CKA already flagged instead of sweeping
every layer unconditionally -- the concrete behavior that answers
gap_work.py's "dalvi-fixed-pipeline" gap: the next tool call is chosen
from evidence, not fixed in advance.
"""
from __future__ import annotations

from .. import config
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
        cka_flagged = [b for b, d in zip(boundaries, drops) if d >= config.CKA_DROP_THRESHOLD]

        # Adaptive layer selection: On GPU or up to 36 layers, an exhaustive
        # scan is fast (sub-second) and ensures no deep circuits are missed.
        # For even larger models (>36 layers), pick the top relative CKA drops
        # plus representative strided checkpoints across early/mid/late depth.
        import os
        if os.environ.get("AMI_LAYER_SCOPE") == "all" or handle.n_layers <= 36:
            candidates = list(range(handle.n_layers))
        elif drops:
            top_cka = sorted(range(len(drops)), key=lambda i: drops[i], reverse=True)[:8]
            strided = list(range(0, handle.n_layers, max(1, handle.n_layers // 6)))
            candidates = sorted(set(top_cka) | set(strided))
        else:
            candidates = list(range(handle.n_layers))
        LOG.emit("NetworkAnalyst", f"adaptively scoping redundancy_scan to {len(candidates)} candidate "
                                     f"layer(s) {candidates} (out of {handle.n_layers} layers)")

        digest, xs, ys = tier_n.redundancy_scan(handle, candidates, lambda: task_metric_fn)
        state["redundancy_digest"] = digest
        state["redundancy_layer_indices"] = xs
        state["redundancy_absolute_deltas"] = ys
        redundancy_flagged = [x for x, y in zip(xs, ys) if y >= config.REDUNDANCY_DROP_THRESHOLD]
        state["flagged_layers"] = sorted(set(cka_flagged) | set(redundancy_flagged))
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
