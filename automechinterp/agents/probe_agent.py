"""Probe Agent (new): per load-bearing layer, test whether the task-relevant
distinction is linearly encoded here -- linear probe accuracy, k-sparse
probe (how distributed), MDL (how accessible), and a causal test of the probe
direction. Uses the task's own clean vs corrupted prompt sets as the
contrastive pair (they differ exactly in the variable the task turns on).
"""
from __future__ import annotations

from ..llm_backends import make_backend
from ..tools import adapter
from ..techniques import probing
from .base import Agent, ToolCallBudget, LOG, digest_dict


def build_probe_agent(handle: adapter.ModelHandle, layer_idx: int,
                       pos_texts: list[str], neg_texts: list[str],
                       test_prompt: str, io_token: str, s_token: str,
                       backend_kind: str, budget: ToolCallBudget, backend_kwargs: dict | None = None):
    state: dict = {}

    def do_linear():
        r = probing.linear_probe(handle, layer_idx, pos_texts, neg_texts, concept="task-variable")
        state["linear_probe"] = r
        return digest_dict(r)

    def do_sparse():
        r = probing.sparse_probe(handle, layer_idx, pos_texts, neg_texts)
        state["sparse_probe"] = r
        return digest_dict(r)

    def do_mdl():
        r = probing.mdl_probe(handle, layer_idx, pos_texts, neg_texts)
        state["mdl_probe"] = r
        return digest_dict(r)

    def do_causal():
        r = probing.probe_direction_causal_test(handle, layer_idx, pos_texts, neg_texts,
                                                test_prompt, io_token, s_token)
        state["probe_causal"] = r
        return digest_dict(r)

    tools = {"linear_probe": do_linear, "sparse_probe": do_sparse,
             "mdl_probe": do_mdl, "probe_direction_causal_test": do_causal}
    order = list(tools.keys())

    def policy_fn(evidence_text: str, tool_menu: list[str]) -> dict:
        if "linear_probe" not in state:
            return {"action": "linear_probe", "args": {}, "reasoning": "is the task variable linearly readable here?"}
        acc = state["linear_probe"].get("cv_accuracy", 0)
        if acc < 0.6:
            return {"action": "stop", "args": {}, "reasoning": f"probe acc {acc} ~ chance; concept not encoded here"}
        for t in order[1:]:
            if t not in state:
                return {"action": t, "args": {}, "reasoning": f"probe step: {t}"}
        return {"action": "stop", "args": {}, "reasoning": "probing complete"}

    backend = make_backend(backend_kind, policy_fn=policy_fn, **(backend_kwargs or {}))
    agent = Agent(backend, tools, budget, max_steps=5)
    agent.name = f"ProbeAgent{layer_idx}"
    agent.system_prompt = (f"You are the Probe Agent for layer {layer_idx}. Test whether the "
                           "task-relevant distinction is linearly encoded, how distributed it is, "
                           "and whether its direction is causal.")
    return agent, state
