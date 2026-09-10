"""Lens Agent (new): per load-bearing layer, decode what the residual stream
at this layer represents -- logit lens, Jacobian lens, logit-lens trajectory
for the answer token, and a Patchscopes read-back. Observational; its output
tells the Component/Judge agents *what* a causally-important layer is holding,
not just that it matters.
"""
from __future__ import annotations

from ..llm_backends import make_backend
from ..tools import adapter
from ..techniques import basic, hidden_state
from .base import Agent, ToolCallBudget, LOG, digest_dict


def build_lens_agent(handle: adapter.ModelHandle, layer_idx: int,
                      clean_prompt: str, corrupted_prompt: str, io_token: str, s_token: str,
                      backend_kind: str, budget: ToolCallBudget, backend_kwargs: dict | None = None):
    state: dict = {}

    def do_logit_lens():
        r = basic.logit_lens(handle, clean_prompt)
        state["logit_lens"] = r
        return digest_dict({"emergence_layer": r["emergence_layer"],
                            "this_layer_top": next((t for i, t, _ in r["per_layer_top"] if i == layer_idx), None),
                            "final": r["final_top_token"]})

    def do_jacobian_lens():
        r = basic.jacobian_lens(handle, clean_prompt, layer_idx, io_token, s_token)
        state["jacobian_lens"] = r
        return digest_dict(r)

    def do_trajectory():
        r = hidden_state.logit_lens_trajectory(handle, clean_prompt, io_token)
        state["logit_lens_trajectory"] = r
        return digest_dict({"target": io_token, "here": next(
            (rp for i, rk, rp in r["per_layer_(rank,prob)"] if i == layer_idx), None),
            "curve": r["per_layer_(rank,prob)"]})

    def do_patchscopes():
        r = hidden_state.patchscopes(handle, clean_prompt, -1, layer_idx=layer_idx)
        state["patchscopes"] = r
        return digest_dict(r)

    tools = {"logit_lens": do_logit_lens, "jacobian_lens": do_jacobian_lens,
             "logit_lens_trajectory": do_trajectory, "patchscopes": do_patchscopes}
    order = list(tools.keys())

    def policy_fn(evidence_text: str, tool_menu: list[str]) -> dict:
        for t in order:
            if t not in state:
                return {"action": t, "args": {}, "reasoning": f"lens step: {t}"}
        return {"action": "stop", "args": {}, "reasoning": "layer decoded"}

    backend = make_backend(backend_kind, policy_fn=policy_fn, **(backend_kwargs or {}))
    agent = Agent(backend, tools, budget, max_steps=5)
    agent.name = f"LensAgent{layer_idx}"
    agent.system_prompt = (f"You are the Lens Agent for layer {layer_idx}. Decode what this "
                           "layer's residual stream represents via vocabulary lenses.")
    return agent, state
