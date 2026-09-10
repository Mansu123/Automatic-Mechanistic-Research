# LLM judge review

The **LLM half** of the two-arm evaluation, mirroring `../human_review/`. An
independent LLM judge scores the **same evidence** humans see against the **same
eight-metric, integer 1-10 rubric** (`../human_review/rubrics.md`; source of
truth `automechinterp/eval/matrix_rubric.py`).

## Judge model

`z-ai/glm-5.2:free` (GLM 5.2) via **OpenRouter**. Availability, zero pricing and
structured-output support are re-checked at preflight; there is **no automatic
paid or model fallback**.

The API key is read only from the environment variable **`OPENROUTER_API_KEY`**
(Colab: add it to *Secrets* and grant the notebook access). It is never written
into code, notebooks, the release ZIP, request logs, reports or score files. If
a key was ever pasted into a chat or a file, rotate it.

## Folder layout

```
llm_review/
  README.md            this file
  run_judge.py         the resumable GLM judge runner
  aggregate_scores.py  aggregate the judge's scores for a run -> judge_health.json
```

Judge score files land under the **run directory** (`<run_id>/review/judge_scores/`),
next to the humans' `completed_human_scores/` -- they are per-run data and belong
with the run.

## Workflow

1. **Export the review material** (same step the human arm uses):

   ```
   python run_review.py --run-dir <results-root>/<run_id> --export
   ```

2. **Preflight** the judge endpoint:

   ```
   python -m llm_review.run_judge --preflight
   ```

3. **Score.** Resumable; each score is checksummed against its evidence and the
   rubric version, and re-scoring only happens if the evidence changed.

   ```
   python -m llm_review.run_judge --run-dir <results-root>/<run_id> \
     --sample all --item-types report,layer,agent,tool,seap --max-items 50
   ```

   For report-level judge validation use `--sample validation --item-types report`.
   The free tier imposes daily/request limits; run on CPU and resume later. This
   is far more than 4,466 calls -- every layer, agent and tool is a separate item.

4. **Aggregate:**

   ```
   python llm_review/aggregate_scores.py --run-dir <results-root>/<run_id>
   ```

   Writes `review/judge_health.json`, `review/judge_score_summary.csv`,
   `review/judge_aggregate.json` and the `05_judge_*` / `09_judge_*` graphs.
   The judge arm is always reported **provisional** here.

## Validation gate (why the judge's scores can be trusted)

Report-level agreement with the humans is established only by `run_review.py`
(`automechinterp/eval/agreement.py`), on **550 held-out paired reports**, one per
model-angle cell: macro quadratic weighted kappa >= 0.80, 95% report-cluster
bootstrap lower bound >= 0.75, within-one-point agreement >= 0.80, and each
critical metric's kappa >= 0.75. Exact agreement, MAE and signed judge bias are
also reported. Constant or insufficient ratings cannot validate the judge. A
validated report-level judge does **not** license element-level (layer / agent /
tool / S-EAP) substitution for a human -- those item types need their own study.
