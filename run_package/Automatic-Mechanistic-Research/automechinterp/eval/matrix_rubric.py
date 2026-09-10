"""Identical eight-metric human and LLM report-quality rubric, version 1."""
import math

VERSION = "mir_shared8_1to10_v2"
METRICS = [
    ("localization", "Localization and scope", "Identifies the layers/components/positions investigated, with evidence for the stated localization or a carefully bounded null."),
    ("causal_validity", "Causal validity", "Supports causal claims with correctly defined interventions, signed effects, baselines and uncertainty; separates decodability from causation."),
    ("verification", "Independent verification", "Tests claims with necessity/sufficiency, joint controls, counterexamples and independent evidence; declares checks not performed."),
    ("generalization", "Behavior and angle controls", "Tests the stated behavior beyond the discovery example, using the angle-specific controls and independent data."),
    ("faithfulness", "Evidence faithfulness", "Every material claim is traceable to actual tool evidence; errors, missing measurements and approximations are accurately represented."),
    ("explanation", "Mechanistic explanatory depth", "Explains the computation or precisely diagnoses why it remains unresolved, distinguishing a functional account from a coordinate or correlation."),
    ("calibration", "Uncertainty and limitations", "Confidence matches evidence; distinguishes a scientific null, low capability, failed tools, limited coverage and insufficient data."),
    ("reproducibility", "Clarity and reproducibility", "Makes task, model revision, prompts, intervention scope, agent/tool coverage, metrics and run configuration clear and reproducible."),
]
IDS = [m[0] for m in METRICS]
CRITICAL = ["causal_validity", "verification", "faithfulness"]
ANCHORS = {
    1: "No usable evidence, or a fundamental contradiction of this criterion.",
    2: "Criterion is mentioned but nearly all required evidence is absent or incorrect.",
    3: "One relevant observation; major flaws prevent the claimed conclusion.",
    4: "Partial evidence with substantial unresolved methodological weaknesses.",
    5: "Basic descriptive support; important controls or details remain missing.",
    6: "Mostly sound, with explicit limits, but one material evidence gap remains.",
    7: "Sound and sufficiently supported for its stated scope; minor limitations are disclosed.",
    8: "Strong independent support and appropriate controls; conclusions are well bounded.",
    9: "Comprehensive, reproducible support including robust alternative-explanation tests.",
    10: "Exceptional evidence meeting all requirements, with independent replication and no material unresolved issue.",
}
ANGLE_CONTROLS = [
    "Entity/order swaps, distractors, long-distance dependencies, agreement and coreference contrasts.",
    "Verified factual answers, held-out entities, matched-frequency distractors; isolate hops only for actual multi-hop tasks.",
    "New operands and operators, direction changes, algorithmic versus memorized solutions.",
    "New sequences and mappings, repeated-pattern controls, context and dependency-distance changes.",
    "Balanced demographic substitutions and matched contexts; association is not normative correctness.",
    "Sense and context changes, frequency controls, lexical versus relational effects.",
    "Negation, mixed valence, lexical controls and context-dependent sentiment.",
    "Executable expected results, operator/variable changes, syntax versus semantic controls.",
    "Physical/causal relation changes and matched association controls.",
    "Multiple languages and tokenizations, parallel meanings, language-specific uncertainty.",
    "Entity identity and ordering swaps, distractors and discourse distance.",
    "Operator, quantifier and negation-scope flips with matched lexical content.",
    "Reverse comparison direction, swap operands and hold out numeric values.",
    "Successor/predecessor reversal and new sequences rather than fixed-list recall.",
    "Transfer the relation to new A:B pairs and matched unrelated distractors.",
    "Held-out stems and productive forms; separate irregular-form retrieval.",
    "Literal versus implied meaning with matched words and varied speaker context.",
    "Source-checked scientific relations; isolate intermediate hops where relevant.",
    "Geographic relation changes and place-frequency/co-occurrence controls.",
    "Creator/work attribution with source-checked labels and matched-frequency distractors.",
    "Definition versus giveaway context; explicit domain and temporal scope.",
    "Figurative versus literal readings and full multi-word continuation scoring.",
    "Retrieved count facts versus actual counting, with changed set members and sizes.",
    "Unit and physical-dimension changes; test conversions and distractor units.",
    "Source-checked chronology and events with name-frequency and temporal distractors.",
]

# Metric-specific milestones complement the shared ten-point scale. The lower
# score in each band applies when the milestone is only partially satisfied.
METRIC_BANDS = {
    "localization": ["No identifiable scope or wrong coordinates", "Some coordinates but selection/scope is unclear", "Reproducible localization with missing alternative-location controls", "All investigated layers and selected components are evidenced; omissions explicit", "Localization replicated with alternate positions/components and negative controls"],
    "causal_validity": ["Unsupported causal assertion or invalid intervention", "Intervention reported without reliable baseline/scope", "Signed effects and baselines supplied; uncertainty or controls incomplete", "Correctly scoped interventions, controls and uncertainty support the bounded claim", "Causal account independently replicated against plausible competing mechanisms"],
    "verification": ["Claim accepted without a usable check", "One weak or circular check", "Some necessity/sufficiency/minimality checks; independent testing incomplete", "Independent checks, joint controls and counterexamples support the stated scope", "Independent replication and adversarial alternatives converge on the same conclusion"],
    "generalization": ["Only a discovery example or leaked evaluation", "Repeated near-identical examples called independent", "Disjoint prompts with acknowledged template dependence; angle controls incomplete", "Held-out task variants and relevant angle controls substantiate transfer", "Independent data and adversarial variants replicate across the claimed distribution"],
    "faithfulness": ["Fabricated evidence or material contradiction", "Several material claims lack evidence or distort results", "Most claims traceable; some important references/qualifiers absent", "All material claims traceable, including errors, nulls and approximation limits", "Independent evidence audit reproduces every material claim with no unresolved mismatch"],
    "explanation": ["No coherent mechanistic account", "Coordinates/correlations presented as an explanation", "Plausible computation described with unresolved causal links", "Evidence supports computation steps or a precise diagnosis of what remains unresolved", "Mechanistic account predicts novel interventions and rejects competing explanations"],
    "calibration": ["Certainty despite absent or contradictory evidence", "Major uncertainty hidden or conflated with null effects", "Limitations acknowledged but confidence only partly justified", "Confidence matches controls; failures, low capability and scientific nulls distinguished", "Independent replication and sensitivity analysis justify all confidence statements"],
    "reproducibility": ["Cannot identify task/model or reproduce setup", "Some setup details but essential prompts/configuration missing", "Main setup reproducible with missing provenance or intervention details", "Pinned model, prompts, configuration, evidence and tool scope are reproducible", "Independent rerun reproduces results, with environment and sensitivity checks"],
}


def rubric_text(angle):
    angle = int(angle)
    try:  # lazy import keeps matrix_rubric free of a load-time dependency on the catalog
        from .catalog import ANGLE_NAMES
        angle_label = f"Angle {angle} ({ANGLE_NAMES[angle-1]})"
    except Exception:
        angle_label = f"Angle {angle}"
    lines = [f"Rubric {VERSION}. Human and LLM use the SAME eight metrics and integer 1–10 scale.",
             "Score evidence actually supplied, not imagined work. A null result can score well when rigorously established. Missing evidence is not a high score merely because it is disclosed.",
             "All eight scores are required for a finished review. Empty human cells mean unscored, not zero.",
             f"{angle_label} generalization checks: {ANGLE_CONTROLS[angle-1]}"]
    for key, title, definition in METRICS:
        lines.append(f"{key} — {title}: {definition}")
        lines.extend(f"  {2*i+1}–{2*i+2}: {criterion}." for i,criterion in enumerate(METRIC_BANDS[key]))
    lines.append("Use the lower score in a band for partial satisfaction, the upper for full satisfaction; apply the shared score anchors below. Score the specified element within context; do not attribute work by other agents to it.")
    lines += [f"{n}: {text}" for n, text in ANCHORS.items()]
    return "\n".join(lines)


def validate_scores(value):
    if not isinstance(value, dict) or set(value) != set(IDS):
        raise ValueError("Exactly the eight shared rubric scores are required")
    if any(type(value[k]) is not int or not 1 <= value[k] <= 10 for k in IDS):
        raise ValueError("Scores must be integers from 1 through 10; blanks remain incomplete")
    return value


def health(scores, execution_ok=True):
    validate_scores(scores)
    mean = sum(scores.values())/8
    passed = execution_ok and mean >= 7.5 and min(scores.values()) >= 5 and all(scores[k] >= 7 for k in CRITICAL)
    return {"mean_1to10": mean, "percent_of_maximum": 10*mean,
            "status": "pass" if passed else "needs_review", "execution_ok": execution_ok,
            "rule": "mean >=7.5, every metric >=5, causal/verification/faithfulness >=7, execution complete",
            "scope": "report quality, not proof that a discovered circuit is true"}


def judge_schema():
    evidence = {"type": "object", "properties": {
        "score": {"type": "integer", "minimum": 1, "maximum": 10},
        "reason": {"type": "string"},
        "evidence_ids": {"type": "array", "items": {"type": "string"}, "minItems": 1}},
        "required": ["score", "reason", "evidence_ids"], "additionalProperties": False}
    return {"type": "object", "properties": {"metrics": {"type": "object",
            "properties": {k: evidence for k in IDS}, "required": IDS, "additionalProperties": False}},
            "required": ["metrics"], "additionalProperties": False}


def validate_judgment(result, evidence_ids):
    if not isinstance(result, dict) or set(result) != {"metrics"}:
        raise ValueError("Invalid judgment envelope")
    metrics = result["metrics"]
    if not isinstance(metrics, dict) or set(metrics) != set(IDS):
        raise ValueError("Judge did not score every metric")
    scores = {}
    for k, row in metrics.items():
        if not isinstance(row, dict) or set(row) != {"score", "reason", "evidence_ids"}:
            raise ValueError("Malformed rubric evidence")
        if not isinstance(row["reason"], str) or not row["reason"].strip():
            raise ValueError("Each score needs a reason")
        if not isinstance(row["evidence_ids"], list) or not row["evidence_ids"] or not all(isinstance(x,str) and x in evidence_ids for x in row["evidence_ids"]):
            raise ValueError("Unknown or missing evidence reference")
        scores[k] = row["score"]
    return validate_scores(scores)
