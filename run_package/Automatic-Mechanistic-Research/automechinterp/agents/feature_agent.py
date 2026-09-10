"""Feature Agent (new): per load-bearing layer, decompose the residual stream
into interpretable features -- a fast toy SAE (any model), the public SAE
where one exists (GPT-2), feature-direction geometry (superposition check),
and the effective dimensionality of the layer.
"""
from __future__ import annotations

from ..llm_backends import make_backend
from ..tools import adapter
from ..techniques import superposition, feature_geometry
from .base import Agent, ToolCallBudget, LOG, digest_dict


def build_feature_agent(handle: adapter.ModelHandle, layer_idx: int, texts: list[str],
                         clean_prompt: str,
                         backend_kind: str, budget: ToolCallBudget, backend_kwargs: dict | None = None):
    state: dict = {}

    def do_toy_sae():
        import os
        r = superposition.train_toy_sae(handle, layer_idx, texts,
                expansion=int(os.environ.get("AMI_SAE_EXPANSION", "1")),
                steps=int(os.environ.get("AMI_SAE_STEPS", "100")))
        state["toy_sae"] = r
        return digest_dict(r)

    def do_public_sae():
        r = superposition.public_sae_decompose(handle, layer_idx, clean_prompt)
        state["public_sae"] = r
        return digest_dict(r)

    def do_geometry():
        import os
        r = feature_geometry.feature_direction_geometry(handle, layer_idx, texts,
                expansion=int(os.environ.get("AMI_SAE_EXPANSION", "1")),
                steps=int(os.environ.get("AMI_SAE_STEPS", "100")))
        state["geometry"] = r
        return digest_dict(r)

    def do_dim():
        r = feature_geometry.activation_dimensionality(handle, texts, layers=[layer_idx])
        state["dimensionality"] = r
        return digest_dict(r)

    tools = {"activation_dimensionality": do_dim, "train_toy_sae": do_toy_sae,
             "feature_direction_geometry": do_geometry, "public_sae_decompose": do_public_sae}
    _key = {"activation_dimensionality": "dimensionality", "train_toy_sae": "toy_sae",
            "feature_direction_geometry": "geometry", "public_sae_decompose": "public_sae"}

    def policy_fn(evidence_text: str, tool_menu: list[str]) -> dict:
        for t in tools:
            if _key[t] not in state:
                return {"action": t, "args": {}, "reasoning": f"feature step: {t}"}
        return {"action": "stop", "args": {}, "reasoning": "feature decomposition complete"}

    backend = make_backend(backend_kind, policy_fn=policy_fn, **(backend_kwargs or {}))
    agent = Agent(backend, tools, budget, max_steps=5)
    agent.name = f"FeatureAgent{layer_idx}"
    agent.system_prompt = (f"You are the Feature Agent for layer {layer_idx}. Decompose this layer "
                           "into sparse interpretable features and characterise its superposition.")
    return agent, state
