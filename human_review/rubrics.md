# Rubric Dimensions

Source of truth: `automechinterp/eval/rubrics.py` (the AI judge in
`automechinterp/agents/rubric_judge.py` scores against the exact same
definitions, so AI and human scores are directly comparable). If this file
and the code ever disagree, the code wins -- update this file to match.

**Scale: 1-5 for every dimension**, chosen over a finer 1-10 scale
specifically to protect agreement across only 3 human raters (see the
mentor's explicit reasoning in `README.md`). General anchors, apply per
dimension using its own description below:

| Score | Meaning |
|---|---|
| 1 | Fails badly / actively wrong or misleading on this dimension |
| 2 | Weak -- present but with clear, specific problems |
| 3 | Adequate -- does the job, nothing notable either way |
| 4 | Strong -- clearly good, a minor gap at most |
| 5 | Excellent -- no reasonable improvement comes to mind |

---

## Core rubrics (score on every report)

1. **Localization accuracy** -- did the report correctly identify the
   causally load-bearing layer(s)? *Check against the Skeptic's ablation
   evidence in the report, not the report's own claim.*
2. **Circuit precision/recall** -- of the components claimed, how many are
   real load-bearing components vs. padding, and how many real ones were
   missed?
3. **Verification rigor** -- did the claim actually survive the Skeptic's
   four falsification checks (ablate / exclusion / minimality /
   counterexample), or is it asserted without adversarial pressure? *A
   report with an empty or partial evidence transcript should score low
   here regardless of what the verdict line says.*
4. **Explanatory depth** -- does the report explain the *mechanism* (why
   this head/layer does this), not just report a coordinate?
5. **Faithfulness / evidence-grounding** -- is every claim traceable to an
   actual tool-call result in the report, with no unsupported leap?
6. **Clarity & structure** -- organized, readable, uses tables/structure
   where it helps?
7. **Calibration** -- does the stated verdict (Confirmed / Probable /
   Speculative / Refuted) match the actual evidence, including honestly
   reporting null results rather than overclaiming? *A confident claim that
   the evidence doesn't support should score low even if the mechanism
   described sounds plausible.*

## Angle-specific rubrics (score only when the report's angle matches)

8. **Multi-hop robustness** *(Factual & World Knowledge only)* -- for
   multi-hop facts, is each hop's contribution separately verified, not
   just the end-to-end effect?
9. **Cross-instance generalization** *(Reasoning & Arithmetic only)* -- was
   the circuit tested across multiple operand values, not just one
   instance?
10. **Null-result quality** *(Reasoning & Arithmetic only)* -- when no
    circuit is found (expected for MLP-heavy arithmetic at small scale), is
    the reasoning for why still substantive, not just "nothing found"?
11. **Induction consistency** *(In-Context Learning / Induction only)* --
    does the same mechanism reappear across different induction templates,
    or is it a one-off?
12. **Harm-awareness** *(Social Bias & Fairness only)* -- does the report
    flag downstream fairness implications, not just the raw mechanism?
13. **Overclaim guard** *(Social Bias & Fairness only)* -- does it avoid
    generalizing a "bias circuit" from one template/prompt pair?
14. **Sense-disambiguation specificity** *(Lexical Semantics only)* -- does
    the report distinguish the claimed circuit from a simple
    lexical-frequency confound?

## Scaling rubric (score only on reports that are part of a multi-model-size sweep)

15. **Scaling coherence** -- is the report explicit about whether/how the
    finding changes across model scale, rather than treating each model
    size in isolation? *Not applicable to a single-model run -- leave blank
    unless the report is explicitly part of a scaling comparison.*
