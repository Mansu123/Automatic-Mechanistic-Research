"""Weight Agent (new): whole-model, static (no forward passes needed for most
of it). Runs alongside the Network Analyst -- singular-value / effective-rank
spectrum of the attention-out and MLP-out matrices, per-layer parameter-norm
profile, and whether the embedding and unembedding are tied.
"""
from __future__ import annotations

from ..llm_backends import make_backend
from ..tools import adapter
from ..techniques import weight_space
from .base import Agent, ToolCallBudget, LOG, digest_dict


def build_weight_agent(handle: adapter.ModelHandle, backend_kind: str,
                        budget: ToolCallBudget, backend_kwargs: dict | None = None):
    state: dict = {}

    def do_attn_svd():
        r = weight_space.parameter_svd(handle, which="attn_out")
        state["attn_svd"] = r
        return digest_dict(r)

    def do_mlp_svd():
        r = weight_space.parameter_svd(handle, which="mlp_out")
        state["mlp_svd"] = r
        return digest_dict(r)

    def do_norms():
        r = weight_space.weight_norm_profile(handle)
        state["norms"] = r
        return digest_dict(r)

    def do_tie():
        r = weight_space.embedding_unembedding_alignment(handle)
        state["tie"] = r
        return digest_dict(r)

    tools = {"attn_out_svd": do_attn_svd, "mlp_out_svd": do_mlp_svd,
             "weight_norm_profile": do_norms, "embedding_unembedding_alignment": do_tie}
    _key = {"attn_out_svd": "attn_svd", "mlp_out_svd": "mlp_svd",
            "weight_norm_profile": "norms", "embedding_unembedding_alignment": "tie"}

    def policy_fn(evidence_text: str, tool_menu: list[str]) -> dict:
        for t in tools:
            if _key[t] not in state:
                return {"action": t, "args": {}, "reasoning": f"weight analysis: {t}"}
        return {"action": "stop", "args": {}, "reasoning": "weight analysis complete"}

    backend = make_backend(backend_kind, policy_fn=policy_fn, **(backend_kwargs or {}))
    agent = Agent(backend, tools, budget, max_steps=5)
    agent.name = "WeightAgent"
    agent.system_prompt = ("You are the Weight Agent. Analyse the model's parameter structure "
                           "directly -- rank, norm distribution, embedding tying.")
    return agent, state
