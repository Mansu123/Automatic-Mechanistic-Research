"""Judge (Sec. 3.2): no tools, reads only the Prover's claim and the
Skeptic's evidence transcript, and assigns Confirmed / Probable /
Speculative / Refuted (Sec. 4.6 evaluation metrics). Because it shares the
same backend interface as every other agent, swapping in an LLM backend
(Qwen2.5-7B-Instruct, Claude, GPT-4o) here is the same one-line change as
everywhere else -- the heuristic rule below is just one implementation of
`decide`, written out because the proposal's own eval-metric thresholds
(Sec. 4.6) already specify it precisely enough to encode directly.
"""
from __future__ import annotations

from ..llm_backends import make_backend


def adjudicate(claimed_heads: list[tuple[int, int]], skeptic_state: dict,
                backend_kind: str, backend_kwargs: dict | None = None) -> dict:
    evidence = "\n".join([
        f"claimed circuit: {claimed_heads}",
        f"ablate_component: {skeptic_state.get('ablate_digest', 'not run')}",
        f"exclusion_ablation: {skeptic_state.get('exclusion_digest', 'not run')}",
        f"minimality_check: {skeptic_state.get('minimality_digest', 'not run')}",
        f"counterexample_search: {skeptic_state.get('counterexample_digest', 'not run')}",
    ])

    def policy_fn(evidence_text: str, tool_menu: list[str]) -> dict:
        specific = skeptic_state.get("ablate_specific", False)
        complete = skeptic_state.get("exclusion_complete", None)
        minimal = skeptic_state.get("minimal", None)
        no_counter = skeptic_state.get("no_counterexample", None)

        if not specific:
            return {"action": "Refuted", "args": {},
                     "reasoning": "ablating the claimed circuit did not produce a specific effect"}
        if complete is False:
            return {"action": "Speculative", "args": {},
                     "reasoning": "exclusion ablation found unexplained effects outside the claim"}
        if minimal is False:
            return {"action": "Probable", "args": {},
                     "reasoning": "circuit works but includes non-load-bearing components"}
        if complete and minimal and no_counter:
            return {"action": "Confirmed", "args": {},
                     "reasoning": "specific, complete, minimal, and no counterexample found"}
        return {"action": "Probable", "args": {},
                 "reasoning": "passes core tests but verification suite was not fully run"}

    kwargs = backend_kwargs or {}
    backend = make_backend(backend_kind, policy_fn=policy_fn, **kwargs)
    action = backend.decide(
        "You are the Judge. Read the Prover's claim and the Skeptic's evidence only -- you have "
        "no tools of your own. Assign exactly one of: Confirmed, Probable, Speculative, Refuted.",
        evidence, ["Confirmed", "Probable", "Speculative", "Refuted"])
    return {"verdict": action.get("action", "Speculative"), "reasoning": action.get("reasoning", ""),
            "evidence": evidence}
