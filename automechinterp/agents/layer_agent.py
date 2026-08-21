"""Layer Agent (Sec. 3.2, 3.4): one per flagged layer, scoped context (just
this layer + the task's clean/corrupted prompt pair). Escalates to
attn_mlp_attribution only if patch_layer already showed the layer is
causally load-bearing -- another instance of choosing the next call from
evidence rather than always running the full tool suite on every layer.
"""
from __future__ import annotations

from ..llm_backends import make_backend
from ..tools import adapter, tier_l
from .base import Agent, ToolCallBudget, LOG


def build_layer_agent(handle: adapter.ModelHandle, layer_idx: int,
                       clean_prompt: str, corrupted_prompt: str,
                       io_token: str, s_token: str,
                       backend_kind: str, budget: ToolCallBudget,
                       backend_kwargs: dict | None = None):
    state: dict = {}

    def do_patch_layer():
        digest = tier_l.patch_layer(handle, layer_idx, clean_prompt, corrupted_prompt, io_token, s_token)
        state["patch_digest"] = digest
        frac = float(digest.split("fraction_recovered=")[1].split(" ")[0])
        state["fraction_recovered"] = frac
        return digest

    def do_attribution():
        digest = tier_l.attn_mlp_attribution(handle, layer_idx, clean_prompt)
        state["attribution_digest"] = digest
        # norm-share is only a coarse pre-filter -- a head can be causally decisive
        # (e.g. a name-mover head) while contributing a modest share of the
        # layer's total activation norm, so this threshold is deliberately low;
        # run_eap/run_acdc do the real per-head causal test downstream.
        state["spawn_component_agent"] = "attn=" in digest and float(
            digest.split("attn=")[1].split("%")[0]) > 20.0
        return digest

    def do_logit_lens():
        digest = tier_l.logit_lens(handle, layer_idx, clean_prompt)
        state["logit_lens_digest"] = digest
        return digest

    tools = {
        "patch_layer": do_patch_layer,
        "attn_mlp_attribution": do_attribution,
        "logit_lens": do_logit_lens,
    }

    def policy_fn(evidence_text: str, tool_menu: list[str]) -> dict:
        if "patch_digest" not in state:
            return {"action": "patch_layer", "args": {}, "reasoning": "localize causal effect first"}
        frac = state.get("fraction_recovered", 0.0)
        if frac < 0.15:
            LOG.emit("LayerAgent", f"L{layer_idx}: patch recovered only {frac:.2f} of the gap -> "
                                     "not load-bearing, skipping deeper analysis", gap_id="single-agent-context-dilution")
            return {"action": "stop", "args": {}, "reasoning": "low causal effect, no deeper analysis needed"}
        if "attribution_digest" not in state:
            return {"action": "attn_mlp_attribution", "args": {},
                     "reasoning": "layer is causally load-bearing; decompose attn vs mlp"}
        if "logit_lens_digest" not in state:
            return {"action": "logit_lens", "args": {}, "reasoning": "check whether this layer is a write-out layer"}
        return {"action": "stop", "args": {}, "reasoning": "layer diagnosis complete"}

    kwargs = backend_kwargs or {}
    backend = make_backend(backend_kind, policy_fn=policy_fn, **kwargs)
    agent = Agent(backend, tools, budget, max_steps=4)
    agent.name = f"LayerAgent{layer_idx}"
    agent.system_prompt = (
        f"You are the Layer Agent for layer {layer_idx}. Diagnose whether this layer causally "
        "matters for the task, and if so, whether the effect is attention- or MLP-driven, so the "
        "Orchestrator knows whether to spawn a Component Agent."
    )
    return agent, state
