"""Steering Agent (new): a second causal-validation agent that runs AFTER a
circuit claim, feeding the Judge. Where the Skeptic ablates, the Steering
Agent *intervenes constructively*: does adding / removing / erasing the
task-relevant direction move the behaviour the way the claim predicts? A
claim that survives ablation but whose direction has no steering effect is
weaker than one that passes both.
"""
from __future__ import annotations

from ..llm_backends import make_backend
from ..tools import adapter
from ..techniques import steering, editing
from .base import Agent, ToolCallBudget, LOG, digest_dict


def build_steering_agent(handle: adapter.ModelHandle, layer_idx: int,
                          pos_texts: list[str], neg_texts: list[str],
                          test_prompt: str, io_token: str, s_token: str,
                          backend_kind: str, budget: ToolCallBudget, backend_kwargs: dict | None = None):
    state: dict = {}

    def do_add():
        r = steering.activation_addition(handle, layer_idx, pos_texts, neg_texts, test_prompt, strength=5.0)
        state["addition"] = r
        return digest_dict(r)

    def do_ablate():
        r = steering.ablation_steering(handle, layer_idx, pos_texts, neg_texts, test_prompt)
        state["ablation"] = r
        return digest_dict(r)

    def do_leace():
        r = editing.leace(handle, layer_idx, pos_texts, neg_texts, test_prompt, io_token, s_token)
        state["leace"] = r
        return digest_dict(r)

    tools = {"activation_addition": do_add, "ablation_steering": do_ablate, "leace": do_leace}
    _key = {"activation_addition": "addition", "ablation_steering": "ablation", "leace": "leace"}

    def policy_fn(evidence_text: str, tool_menu: list[str]) -> dict:
        for t in tools:
            if _key[t] not in state:
                return {"action": t, "args": {}, "reasoning": f"steering validation: {t}"}
        return {"action": "stop", "args": {}, "reasoning": "steering validation complete"}

    backend = make_backend(backend_kind, policy_fn=policy_fn, **(backend_kwargs or {}))
    agent = Agent(backend, tools, budget, max_steps=4)
    agent.name = f"SteeringAgent{layer_idx}"
    agent.system_prompt = (f"You are the Steering Agent for layer {layer_idx}. Constructively "
                           "intervene on the task-relevant direction and check the behaviour moves "
                           "as the circuit claim predicts.")
    return agent, state
