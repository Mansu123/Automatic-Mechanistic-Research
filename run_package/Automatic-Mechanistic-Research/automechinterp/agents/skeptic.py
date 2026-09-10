"""Skeptic (Sec. 3.2, 3.4): structurally independent of the discovery agents
-- it only ever receives a claimed circuit (list of (layer, head) tuples)
and the behavior it is supposed to explain, never the Component Agent's own
reasoning, and its job is to try to break the claim via four falsification
tests. This is the direct implementation of gap_work.py's
"maia-confirmation-bias" gap: MAIA sometimes accepted a hypothesis after one
matching example with no adversarial follow-up; here nothing reaches the
Judge without surviving exclusion_ablation, minimality_check and
counterexample_search.
"""
from __future__ import annotations

from ..llm_backends import make_backend
from ..tools import adapter, verification
from .base import Agent, ToolCallBudget, LOG


def build_skeptic(handle: adapter.ModelHandle, claimed_heads: list[tuple[int, int]],
                   all_heads: list[tuple[int, int]], eval_prompts: list[tuple[str, str, str]],
                   behavior: str, backend_kind: str, budget: ToolCallBudget,
                   backend_kwargs: dict | None = None,
                   stress_prompts: list | None = None,
                   stress_data_status: str = "unvalidated"):
    import json
    state: dict = {"verification_records": {}}

    def save(record, key, flag):
        state["verification_records"][record["check"]] = record
        digest = json.dumps({k:v for k,v in record.items() if not k.startswith("raw_") and k not in ("records","complement")}, allow_nan=False)
        state[key] = digest
        state[flag] = record["passed"]
        return digest

    def do_ablate():
        return save(verification.necessity(handle, claimed_heads, eval_prompts),
                    "ablate_digest", "ablate_specific")

    def do_exclusion():
        return save(verification.completeness(handle, claimed_heads, all_heads, eval_prompts),
                    "exclusion_digest", "exclusion_complete")

    def do_minimality():
        return save(verification.minimality(handle, claimed_heads, all_heads, eval_prompts),
                    "minimality_digest", "minimal")

    def do_counterexample():
        return save(verification.counterexamples(handle, claimed_heads, stress_prompts or [],
                                                 data_status=stress_data_status),
                    "counterexample_digest", "no_counterexample")

    tools = {
        "ablate_component": do_ablate,
        "exclusion_ablation": do_exclusion,
        "minimality_check": do_minimality,
        "counterexample_search": do_counterexample,
    }

    def policy_fn(evidence_text: str, tool_menu: list[str]) -> dict:
        if "ablate_digest" not in state:
            return {"action": "ablate_component", "args": {}, "reasoning": "does removing the claim hurt the behavior?"}
        if not state["ablate_specific"]:
            LOG.emit("Skeptic", "ablation showed no specific effect -- refuting early instead of "
                                  "running the full falsification suite on a claim already dead",
                     gap_id="maia-confirmation-bias")
            return {"action": "stop", "args": {}, "reasoning": "claim already fails the weakest test; refute now"}
        if "exclusion_digest" not in state:
            return {"action": "exclusion_ablation", "args": {}, "reasoning": "does anything outside the claim also matter?"}
        if "minimality_digest" not in state:
            return {"action": "minimality_check", "args": {}, "reasoning": "is every claimed head load-bearing?"}
        if "counterexample_digest" not in state:
            return {"action": "counterexample_search", "args": {}, "reasoning": "search for an input that breaks the claim"}
        return {"action": "stop", "args": {}, "reasoning": "verification suite complete"}

    kwargs = backend_kwargs or {}
    backend = make_backend(backend_kind, policy_fn=policy_fn, **kwargs)
    agent = Agent(backend, tools, budget, max_steps=4)
    agent.name = "Skeptic"
    agent.system_prompt = (
        f"You are the Skeptic. A Prover claims heads {claimed_heads} explain '{behavior}'. "
        "Your job is to try to falsify this claim via ablation, exclusion, minimality and "
        "counterexample search -- do not accept it on the first passing test."
    )
    return agent, state
