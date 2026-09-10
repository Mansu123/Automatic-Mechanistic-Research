# Human expert review

The **human half** of the two-arm evaluation. Human reviewers and the GLM judge
(`../llm_review/`) score the **same evidence** against the **same eight-metric,
integer 1-10 rubric** (`rubrics.md`; source of truth
`automechinterp/eval/matrix_rubric.py`), so the two sets of scores compare
head-to-head and the judge can be validated against the humans.

## Folder layout

```
human_review/
  README.md              this file
  rubrics.md             the 8 metrics + bands + anchors + all 25 angle controls
  scoring_template.csv   column reference for a completed reviewer CSV
  aggregate_scores.py    combine the reviewers' CSVs for a run -> human_health.json
  agent_layer_matrix.py  reading aid: agent x layer grid per report
```

The authoritative workflow scores a **run directory** produced by `run_colab.py`
(22 models x 203 behaviors). The single-model `main.py --stage-d` /
`run_eval_matrix.py` paths still write reports to `human_review/reports/`
(created on demand) using the same rubric, but do not produce the run-directory
`review/` export the aggregators consume.

## Workflow

1. **Export the review material** from a finished GPU run:

   ```
   python run_review.py --run-dir <results-root>/<run_id> --export
   ```

   This writes, under `<run_id>/review/`:
   `review_items.jsonl` (every report / layer / agent / tool / S-EAP item),
   `human_scores_BLANK_TEMPLATE.csv` (every item),
   `human_calibration_BLANK_TEMPLATE.csv` (100 reports),
   `human_validation_BLANK_TEMPLATE.csv` (550 reports, one per model-angle cell),
   `agent_layer_tool_matrix.csv`, `evaluation_matrix.csv`,
   `seap_exact_vs_approximation.csv`, and the coverage graphs.

2. **Calibrate then freeze.** Score the 100-report calibration set first, discuss
   and sharpen any metric wording in `matrix_rubric.py` that reviewers keep
   splitting on, then freeze the rubric.

3. **Double-score the 550-report validation set.** **Two reviewers score
   independently** (distinct `reviewer_id` values) before seeing any judge
   result. Keep both independent CSVs. Then produce an explicit adjudicated
   reference with `reviewer_id=consensus`.

4. Copy the BLANK_TEMPLATE files to working files under
   `<run_id>/review/completed_human_scores/` and fill in `reviewer_id` plus the
   eight scores per row. **Preserve** `item_id`, `rubric_version` and
   `job_sha256` -- stale evidence or a different rubric version is rejected.
   Never edit a BLANK_TEMPLATE in place; exporting regenerates it.

5. **Aggregate:**

   ```
   python human_review/aggregate_scores.py --run-dir <results-root>/<run_id>
   ```

   Writes `review/human_health.json` (pass rule: mean >= 7.5, every metric >= 5,
   critical metrics >= 7, full coverage, >= 80% of reports passing),
   `review/human_review_aggregate.csv` (per-item mean/stdev), and prints
   pairwise inter-reviewer agreement per metric. Low agreement on a metric means
   that metric's wording needs sharpening -- **do not widen the 1-10 scale.**

6. **Validate the judge** (cross-arm, not part of this folder):

   ```
   python run_review.py --run-dir <results-root>/<run_id> \
     --human-scores <run_id>/review/completed_human_scores/*.csv
   ```

   `review/judge_validation.json` reports macro quadratic weighted kappa,
   report-cluster bootstrap CI, within-one agreement, exact agreement, MAE and
   signed judge bias. Official gate: 550 paired reports, macro QWK >= 0.80,
   bootstrap lower bound >= 0.75, within-one >= 0.80, each critical metric's
   kappa >= 0.75.

## Why a null result still gets scored

"No confirmed circuit, here is why" is a valid report. Score it on all eight
metrics normally -- `calibration` and `explanation` exist precisely to reward a
well-reasoned null over a confident but unsupported claim.
