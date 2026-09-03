"""Evaluation rubrics for scoring generated interpretability reports.

Direct implementation of the mentor-feedback evaluation plan: 7 core rubrics
(scored on every report) + 8 angle-specific rubrics (scored only on the
report's own angle), 1-5 scale throughout -- the mentor's explicit choice
over 1-10, made specifically to protect inter-rater agreement across only 3
human experts (too many choice points on a 10-point scale with 3 raters
tends to produce low agreement, which would undermine the whole evaluation).

Used by two consumers that must stay in lockstep:
  - automechinterp/agents/rubric_judge.py  (AI-based / automatic evaluation,
    Claude or GPT only, per the mentor's "top-tier judges only" instruction)
  - human_review/                          (human expert evaluation, same
    rubric ids and scale so the two evaluation arms are directly comparable)
"""
from __future__ import annotations

from dataclasses import dataclass

SCALE_MIN = 1
SCALE_MAX = 5


@dataclass(frozen=True)
class Rubric:
    id: str
    name: str
    description: str


CORE_RUBRICS: list[Rubric] = [
    Rubric("localization_accuracy", "Localization accuracy",
           "Did the report correctly identify the causally load-bearing layer(s)?"),
    Rubric("circuit_precision_recall", "Circuit precision/recall",
           "Of the components claimed, how many are real load-bearing components vs. padding, "
           "and how many real ones were missed?"),
    Rubric("verification_rigor", "Verification rigor",
           "Did the claim actually survive the Skeptic's four falsification checks "
           "(ablate/exclusion/minimality/counterexample), or is it asserted without adversarial pressure?"),
    Rubric("explanatory_depth", "Explanatory depth",
           "Does the report explain the mechanism (why this head/layer does this), not just report a "
           "coordinate?"),
    Rubric("faithfulness", "Faithfulness / evidence-grounding",
           "Is every claim traceable to an actual tool-call result, with no unsupported leap?"),
    Rubric("clarity_structure", "Clarity & structure",
           "Organized, readable, appropriately uses figures/tables where relevant?"),
    Rubric("calibration", "Calibration",
           "Does the stated confidence (Confirmed/Probable/Speculative/Refuted) match the actual "
           "evidence, including honestly reporting null results rather than overclaiming?"),
]

# Angle numbers match the "Angle N: ..." prefix each behaviors.py category string
# starts with (see behaviors.py's ANGLE_*_BEHAVIORS lists / _make_task category arg).
ANGLE_RUBRICS: dict[int, list[Rubric]] = {
    1: [],  # Linguistic Structure -- no angle-specific add-on defined by the mentor
    2: [
        Rubric("multi_hop_robustness", "Multi-hop robustness",
               "For multi-hop facts, is each hop's contribution separately verified, not just the "
               "end-to-end effect?"),
    ],
    3: [
        Rubric("cross_instance_generalization", "Cross-instance generalization",
               "Was the circuit tested across multiple operand values, not just one instance?"),
        Rubric("null_result_quality", "Null-result quality",
               "When no circuit is found (an expected outcome for MLP-heavy arithmetic behaviors at "
               "small scale), is the reasoning for why still substantive?"),
    ],
    4: [
        Rubric("induction_consistency", "Induction consistency",
               "Does the same mechanism reappear across different induction templates, or is it a "
               "one-off?"),
    ],
    5: [
        Rubric("harm_awareness", "Harm-awareness",
               "Does the report flag downstream fairness implications, not just the raw mechanism?"),
        Rubric("overclaim_guard", "Overclaim guard",
               "Does it avoid generalizing a 'bias circuit' from one template/prompt pair?"),
    ],
    6: [
        Rubric("sense_disambiguation_specificity", "Sense-disambiguation specificity",
               "Does the report distinguish the claimed circuit from a simple lexical-frequency "
               "confound?"),
    ],
    7: [
        Rubric("valence_vs_lexical_confound", "Valence vs. lexical confound",
               "Does the report separate a genuine sentiment/valence representation from a "
               "bag-of-positive-words lexical shortcut?"),
    ],
    8: [
        Rubric("symbolic_execution_fidelity", "Symbolic-execution fidelity",
               "Is the claimed mechanism shown to actually track operator/operand semantics rather "
               "than memorized surface strings of common expressions?"),
    ],
    9: [
        Rubric("world_model_vs_association", "World-model vs. association",
               "Does the report argue the circuit encodes a physical/causal relation rather than a "
               "co-occurrence prior between the entities named?"),
    ],
    10: [
        Rubric("cross_lingual_consistency", "Cross-lingual consistency",
               "Does the same mechanism appear across more than one language pair, or is it a "
               "single-language artifact?"),
    ],
    11: [
        Rubric("entity_binding_verification", "Entity-binding verification",
               "Is the claimed binding/tracking mechanism tested under a swap of the entities or "
               "their order, not just one assignment?"),
    ],
    12: [
        Rubric("negation_scope_specificity", "Negation-scope specificity",
               "Does the report show the circuit flips with the logical operator (negation, "
               "quantifier, conditional) rather than keying on surface cue words?"),
    ],
    13: [
        Rubric("comparison_direction_control", "Comparison-direction control",
               "Is the claimed circuit shown to encode the ordering relation itself (larger/smaller, "
               "more/fewer), verified by flipping the comparison word, not just the operand pair?"),
    ],
    14: [
        Rubric("ordering_vs_memorized_list", "Ordering vs. memorized list",
               "Does the report distinguish a genuine successor/predecessor computation from "
               "recall of a fixed sequence (months, days) as a frozen string?"),
    ],
    15: [
        Rubric("relation_transfer", "Relation transfer",
               "Is the analogical relation shown to transfer across different A:B pairs, rather than "
               "the circuit memorizing one specific pair?"),
    ],
    16: [
        Rubric("productive_vs_stored_form", "Productive vs. stored form",
               "Does the report test whether the morphological transformation generalizes to held-out "
               "stems, versus retrieving a stored irregular form?"),
    ],
    17: [
        Rubric("implicature_vs_literal", "Implicature vs. literal content",
               "Does the report show the circuit tracks the implied/pragmatic meaning rather than the "
               "literal lexical content of the cue phrase?"),
    ],
    18: [
        Rubric("knowledge_hop_isolation", "Knowledge-hop isolation",
               "For multi-step scientific facts, is each inferential step separately probed rather "
               "than only the final answer token?"),
    ],
    19: [
        Rubric("geographic_relation_specificity", "Geographic-relation specificity",
               "Is the claimed circuit distinguished from a raw place-name co-occurrence prior "
               "(e.g. 'Nile' near 'Egypt') by a controlled contrast?"),
    ],
    20: [
        Rubric("attribution_vs_frequency", "Attribution vs. frequency",
               "Does the report rule out that the creator-to-work mapping is just the most frequent "
               "name completion, using a matched-frequency counterexample?"),
    ],
    21: [
        Rubric("definition_vs_context", "Definition vs. context cue",
               "Is the claimed concept representation shown to be the term's meaning rather than a "
               "shortcut from a single give-away context word?"),
    ],
    22: [
        Rubric("figurative_completion_grounding", "Figurative-completion grounding",
               "Does the report show the circuit completes the idiom as a fixed multi-word unit, not "
               "via literal word-by-word association?"),
    ],
    23: [
        Rubric("count_fact_vs_arithmetic", "Count-fact vs. arithmetic",
               "Does the report clarify whether the answer is a retrieved numeric fact or an actual "
               "counting computation, and localize accordingly?"),
    ],
    24: [
        Rubric("unit_dimension_control", "Unit-dimension control",
               "Is the claimed circuit shown to track the physical dimension (length, mass, time) "
               "rather than surface unit-word co-occurrence?"),
    ],
    25: [
        Rubric("temporal_fact_vs_name_prior", "Temporal-fact vs. name prior",
               "For 'first name ___' history completions, does the report rule out a plain name-"
               "frequency prior with a matched distractor?"),
    ],
}

# Applies once a report is part of a cross-model-size sweep (mentor's scaling ask,
# point 3): not tied to one angle, so it is appended for every report whenever the
# caller says this report is part of a multi-tier run.
SCALING_RUBRIC = Rubric(
    "scaling_coherence", "Scaling coherence",
    "Is the report explicit about whether/how the finding changes across model scale, rather than "
    "treating each model size in isolation?",
)


def angle_number_from_category(category: str) -> int | None:
    """category looks like 'Angle 3: Arithmetic' -- see behaviors.py."""
    if not category.startswith("Angle "):
        return None
    try:
        return int(category.split(":", 1)[0].removeprefix("Angle ").strip())
    except ValueError:
        return None


def rubrics_for_report(category: str, is_part_of_scaling_sweep: bool = False) -> list[Rubric]:
    """The 7 core rubrics + this report's angle-specific add-ons (+ scaling
    coherence when this report is one of several sizes of the same task)."""
    angle = angle_number_from_category(category)
    rubrics = list(CORE_RUBRICS) + list(ANGLE_RUBRICS.get(angle, []))
    if is_part_of_scaling_sweep:
        rubrics = rubrics + [SCALING_RUBRIC]
    return rubrics


def all_rubrics() -> list[Rubric]:
    """Every rubric that exists, core + all angle-specific + scaling -- the
    flat ~15-item list the mentor's plan targets, used to build the human
    review sheet (which needs every possible column up front)."""
    seen: dict[str, Rubric] = {r.id: r for r in CORE_RUBRICS}
    for group in ANGLE_RUBRICS.values():
        for r in group:
            seen[r.id] = r
    seen[SCALING_RUBRIC.id] = SCALING_RUBRIC
    return list(seen.values())
