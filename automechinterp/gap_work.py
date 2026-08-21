"""Section 2 of the proposal, as data instead of prose.

Each entry is a specific limitation of a prior system and the concrete
AutoMechInterp component that closes it. Agents tag their log lines with a
`gap_id` (see agents/base.py:AgentLog.emit) so a live run shows, action by
action, which prior-work limitation is being addressed -- not just a claim
in a related-work paragraph but something exercised in the trace.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class Gap:
    gap_id: str
    prior_work: str
    limitation: str
    closed_by: str


GAPS: list[Gap] = [
    Gap(
        gap_id="dalvi-fixed-pipeline",
        prior_work="Dalvi et al. 2019 (neuron-level correlation analysis)",
        limitation="Static statistical procedure fixed in advance; cannot adaptively "
                    "choose what to test next; needs labeled property data or multiple "
                    "trained models; neuron granularity only.",
        closed_by="NetworkAnalyst/LayerAgent ReAct loop picks the next tool call from "
                   "the evidence gathered so far (hierarchy.py); operates on circuits "
                   "and layer roles, not just neurons; needs no labeled data.",
    ),
    Gap(
        gap_id="sasc-noninteractive",
        prior_work="SASC (Singh et al. 2023, non-interactive NL explanation)",
        limitation="One-pass, single-shot: explains one scalar module from its top-"
                    "activating ngrams and never follows up on a weak explanation; no "
                    "module-to-module (circuit) analysis; ngram-bound.",
        closed_by="ComponentAgent iterates (run_eap -> activation_patch -> "
                   "get_max_activating_examples) until effect size is resolved, and "
                   "reports circuits (sets of heads/MLPs), not isolated modules.",
    ),
    Gap(
        gap_id="find-synthetic-only",
        prior_work="FIND / AIA (Schwettmann et al. 2023)",
        limitation="Evaluates a single agent on synthetic black-box functions with no "
                    "causal access to real network internals; best AIA failed on 48% of "
                    "functions, limited by breadth of search.",
        closed_by="Tool layer (tools/adapter.py) runs against real model internals via "
                   "PyTorch hooks; the hierarchy localizes breadth-of-search failures to "
                   "one layer/component instead of the whole model.",
    ),
    Gap(
        gap_id="maia-confirmation-bias",
        prior_work="MAIA (Shaham et al. 2024)",
        limitation="Single GPT-4V agent; documented confirmation bias (accepts a "
                    "hypothesis after one high-activation exemplar); correlational only; "
                    "vision-only; needs human supervision to catch mistakes.",
        closed_by="Prover-Skeptic-Judge triad (agents/skeptic.py, agents/judge.py): a "
                   "structurally independent Skeptic runs exclusion_ablation, "
                   "minimality_check and counterexample_search; only claims that "
                   "survive get a Confirmed verdict from the Judge.",
    ),
    Gap(
        gap_id="single-agent-context-dilution",
        prior_work="Generic single-agent automated interpretability",
        limitation="18+ tool schemas plus a deep model's accumulated evidence overwhelm "
                    "one context window; does not scale in depth.",
        closed_by="Hierarchical decomposition (Network Analyst -> Layer Agents -> "
                   "Component Agents), each holding only evidence in its own scope "
                   "(agents/network_analyst.py, agents/layer_agent.py, "
                   "agents/component_agent.py).",
    ),
    Gap(
        gap_id="transformer-only-tooling",
        prior_work="Most MI tooling (TransformerLens-centric pipelines)",
        limitation="Assumes a transformer decoder stack; cannot be pointed at an "
                    "arbitrary PyTorch network.",
        closed_by="tools/adapter.py discovers the layer stack via named_modules() with "
                   "no transformer-specific import required; verified against both a "
                   "GPT-2 (transformer.h) and a Qwen2/Llama-family (model.layers) module "
                   "layout in this codebase.",
    ),
]


def print_matrix() -> None:
    for g in GAPS:
        print(f"[{g.gap_id}]")
        print(f"  prior work : {g.prior_work}")
        print(f"  limitation : {g.limitation}")
        print(f"  closed by  : {g.closed_by}")
        print()


def gap_by_id(gap_id: str) -> Gap:
    for g in GAPS:
        if g.gap_id == gap_id:
            return g
    raise KeyError(gap_id)
