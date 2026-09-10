# Shared rubric -- 8 metrics, integer 1-10

**Source of truth: `automechinterp/eval/matrix_rubric.py` (`mir_shared8_1to10_v2`).** If this
file and the code disagree, the code wins -- regenerate this file.

The human reviewers (this folder) and the GLM judge (`llm_review/`) score the
**same evidence** on the **same eight metrics** with the **same integer 1-10
scale**, so the two arms are directly comparable and the judge can be validated
against the humans (`run_review.py` / `automechinterp/eval/agreement.py`).

All eight scores are required for a finished review. An empty cell means
unscored, never zero. A rigorously established null result can score high;
merely disclosing that evidence is missing does not earn a high score. Score the
specified element's own work in its supplied context -- do not credit it with
another agent's achievements.

Critical metrics (a report cannot pass below 7 on any of these): **causal_validity, verification, faithfulness**.

## The eight metrics

### localization -- Localization and scope

Identifies the layers/components/positions investigated, with evidence for the stated localization or a carefully bounded null.

- **1-2:** No identifiable scope or wrong coordinates.
- **3-4:** Some coordinates but selection/scope is unclear.
- **5-6:** Reproducible localization with missing alternative-location controls.
- **7-8:** All investigated layers and selected components are evidenced; omissions explicit.
- **9-10:** Localization replicated with alternate positions/components and negative controls.

Use the lower score in a band for partial satisfaction, the upper for full.

### causal_validity -- Causal validity

Supports causal claims with correctly defined interventions, signed effects, baselines and uncertainty; separates decodability from causation.

- **1-2:** Unsupported causal assertion or invalid intervention.
- **3-4:** Intervention reported without reliable baseline/scope.
- **5-6:** Signed effects and baselines supplied; uncertainty or controls incomplete.
- **7-8:** Correctly scoped interventions, controls and uncertainty support the bounded claim.
- **9-10:** Causal account independently replicated against plausible competing mechanisms.

Use the lower score in a band for partial satisfaction, the upper for full.

### verification -- Independent verification

Tests claims with necessity/sufficiency, joint controls, counterexamples and independent evidence; declares checks not performed.

- **1-2:** Claim accepted without a usable check.
- **3-4:** One weak or circular check.
- **5-6:** Some necessity/sufficiency/minimality checks; independent testing incomplete.
- **7-8:** Independent checks, joint controls and counterexamples support the stated scope.
- **9-10:** Independent replication and adversarial alternatives converge on the same conclusion.

Use the lower score in a band for partial satisfaction, the upper for full.

### generalization -- Behavior and angle controls

Tests the stated behavior beyond the discovery example, using the angle-specific controls and independent data.

- **1-2:** Only a discovery example or leaked evaluation.
- **3-4:** Repeated near-identical examples called independent.
- **5-6:** Disjoint prompts with acknowledged template dependence; angle controls incomplete.
- **7-8:** Held-out task variants and relevant angle controls substantiate transfer.
- **9-10:** Independent data and adversarial variants replicate across the claimed distribution.

Use the lower score in a band for partial satisfaction, the upper for full.

### faithfulness -- Evidence faithfulness

Every material claim is traceable to actual tool evidence; errors, missing measurements and approximations are accurately represented.

- **1-2:** Fabricated evidence or material contradiction.
- **3-4:** Several material claims lack evidence or distort results.
- **5-6:** Most claims traceable; some important references/qualifiers absent.
- **7-8:** All material claims traceable, including errors, nulls and approximation limits.
- **9-10:** Independent evidence audit reproduces every material claim with no unresolved mismatch.

Use the lower score in a band for partial satisfaction, the upper for full.

### explanation -- Mechanistic explanatory depth

Explains the computation or precisely diagnoses why it remains unresolved, distinguishing a functional account from a coordinate or correlation.

- **1-2:** No coherent mechanistic account.
- **3-4:** Coordinates/correlations presented as an explanation.
- **5-6:** Plausible computation described with unresolved causal links.
- **7-8:** Evidence supports computation steps or a precise diagnosis of what remains unresolved.
- **9-10:** Mechanistic account predicts novel interventions and rejects competing explanations.

Use the lower score in a band for partial satisfaction, the upper for full.

### calibration -- Uncertainty and limitations

Confidence matches evidence; distinguishes a scientific null, low capability, failed tools, limited coverage and insufficient data.

- **1-2:** Certainty despite absent or contradictory evidence.
- **3-4:** Major uncertainty hidden or conflated with null effects.
- **5-6:** Limitations acknowledged but confidence only partly justified.
- **7-8:** Confidence matches controls; failures, low capability and scientific nulls distinguished.
- **9-10:** Independent replication and sensitivity analysis justify all confidence statements.

Use the lower score in a band for partial satisfaction, the upper for full.

### reproducibility -- Clarity and reproducibility

Makes task, model revision, prompts, intervention scope, agent/tool coverage, metrics and run configuration clear and reproducible.

- **1-2:** Cannot identify task/model or reproduce setup.
- **3-4:** Some setup details but essential prompts/configuration missing.
- **5-6:** Main setup reproducible with missing provenance or intervention details.
- **7-8:** Pinned model, prompts, configuration, evidence and tool scope are reproducible.
- **9-10:** Independent rerun reproduces results, with environment and sensitivity checks.

Use the lower score in a band for partial satisfaction, the upper for full.

## Shared 1-10 anchors (apply per metric, alongside its bands)

| Score | Meaning |
|---|---|
| 1 | No usable evidence, or a fundamental contradiction of this criterion. |
| 2 | Criterion is mentioned but nearly all required evidence is absent or incorrect. |
| 3 | One relevant observation; major flaws prevent the claimed conclusion. |
| 4 | Partial evidence with substantial unresolved methodological weaknesses. |
| 5 | Basic descriptive support; important controls or details remain missing. |
| 6 | Mostly sound, with explicit limits, but one material evidence gap remains. |
| 7 | Sound and sufficiently supported for its stated scope; minor limitations are disclosed. |
| 8 | Strong independent support and appropriate controls; conclusions are well bounded. |
| 9 | Comprehensive, reproducible support including robust alternative-explanation tests. |
| 10 | Exceptional evidence meeting all requirements, with independent replication and no material unresolved issue. |

## Angle-specific generalization checks

These feed the **generalization** metric: the report must test the stated
behavior beyond its discovery example using the controls for its own angle.

| Angle | Name | Controls the report should use |
|---:|---|---|
| 1 | Linguistic Structure (Syntax & Coreference) | Entity/order swaps, distractors, long-distance dependencies, agreement and coreference contrasts. |
| 2 | Factual & World Knowledge | Verified factual answers, held-out entities, matched-frequency distractors; isolate hops only for actual multi-hop tasks. |
| 3 | Reasoning & Arithmetic | New operands and operators, direction changes, algorithmic versus memorized solutions. |
| 4 | In-Context Learning / Induction | New sequences and mappings, repeated-pattern controls, context and dependency-distance changes. |
| 5 | Social Bias & Fairness | Balanced demographic substitutions and matched contexts; association is not normative correctness. |
| 6 | Lexical Semantics & Word Sense | Sense and context changes, frequency controls, lexical versus relational effects. |
| 7 | Sentiment & Emotion | Negation, mixed valence, lexical controls and context-dependent sentiment. |
| 8 | Code & Formal Reasoning | Executable expected results, operator/variable changes, syntax versus semantic controls. |
| 9 | Commonsense & Physical Reasoning | Physical/causal relation changes and matched association controls. |
| 10 | Multilingual / Cross-lingual | Multiple languages and tokenizations, parallel meanings, language-specific uncertainty. |
| 11 | Entity Tracking & Discourse | Entity identity and ordering swaps, distractors and discourse distance. |
| 12 | Negation & Logic | Operator, quantifier and negation-scope flips with matched lexical content. |
| 13 | Quantitative Comparison | Reverse comparison direction, swap operands and hold out numeric values. |
| 14 | Temporal & Sequential Ordering | Successor/predecessor reversal and new sequences rather than fixed-list recall. |
| 15 | Analogical Reasoning | Transfer the relation to new A:B pairs and matched unrelated distractors. |
| 16 | Morphology & Word Formation | Held-out stems and productive forms; separate irregular-form retrieval. |
| 17 | Pragmatics & Implicature | Literal versus implied meaning with matched words and varied speaker context. |
| 18 | Scientific & Technical Knowledge | Source-checked scientific relations; isolate intermediate hops where relevant. |
| 19 | Geography & Spatial Knowledge | Geographic relation changes and place-frequency/co-occurrence controls. |
| 20 | Arts, Literature & Culture | Creator/work attribution with source-checked labels and matched-frequency distractors. |
| 21 | Economics, Politics & Law | Definition versus giveaway context; explicit domain and temporal scope. |
| 22 | Idioms & Figurative Language | Figurative versus literal readings and full multi-word continuation scoring. |
| 23 | Counting & Set Facts | Retrieved count facts versus actual counting, with changed set members and sizes. |
| 24 | Units & Measurement | Unit and physical-dimension changes; test conversions and distractor units. |
| 25 | History & Chronology | Source-checked chronology and events with name-frequency and temporal distractors. |

## Pass rules

**Report pass:** mean >= 7.5, every metric >= 5, each critical metric >= 7, and
completed execution. **Whole-run health (`human_health.json`):** full execution
and report-scoring coverage with >= 80% of reports passing. The LLM arm's health
also requires a validated judge. These are proposed preregistered operating
thresholds, not universal scientific cutoffs.
