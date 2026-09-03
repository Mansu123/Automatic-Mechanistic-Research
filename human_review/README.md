# Human Expert Review

This folder is the human half of the two-arm evaluation the mentor feedback
asked for (see `rubrics.md` for the full rubric text, and
`automechinterp/eval/rubrics.py` for the source of truth both the AI judge
and this folder score against).

## Who this is for

~3 expert scorers, each working independently (no discussion before all 3
have submitted scores -- that's what makes an inter-rater agreement check
meaningful afterward).

## Folder layout

```
human_review/
  README.md              this file
  rubrics.md             the 15 rubric dimensions, written out for humans
  scoring_template.csv   blank scoring sheet -- copy this, don't edit in place
  reports/                one .md report per (model, behavior) -- what you score
  scores/                 drop your filled-in copy here
  aggregate_scores.py     combines all reviewers' CSVs once everyone is done
  agent_layer_matrix.py   optional helper: agent x layer matrix per report
```

`agent_layer_matrix.py` is a reading aid, not part of scoring. Run
`python3 human_review/agent_layer_matrix.py` to get one matrix per report
(agents as rows, layers as columns) showing where each agent -- Network
Analyst, Layer Agent, Component Agent, Skeptic/Judge -- actually did work.
Useful for rubric 1 (localization accuracy): check whether the claimed
circuit's layer is one the Skeptic's ablations actually support. Output is
written to `output/eval/agent_layer_matrix.md`.

## What to do

1. Run `python3 main.py --stage-d` (already done if `reports/` isn't empty)
   to generate the reports you'll be scoring. Each file in `reports/` is one
   report: which behavior, which model, what circuit was claimed, and what
   the Skeptic/Judge found. For a cross-model / scaling sweep (several model
   sizes of the same behaviors, which is when `scaling_coherence` applies),
   use `python3 run_eval_matrix.py` instead -- see the repo README. Reports
   are named `<model>__<behavior>.md`, so one behavior can appear several
   times, once per model size.
2. Copy `scoring_template.csv` to `scores/<your_name>.csv`.
3. For every report in `reports/`, read it and fill in a score from 1-5 for
   each rubric that applies to that report. **Not every rubric applies to
   every report** -- each report's `applicable_rubrics` list is in the Stage
   D JSON output (`output/eval/stage_d_*.json`) if you need to check; angle-
   specific rubrics only apply to that report's angle (e.g.
   `harm_awareness` only applies to Social Bias reports). Leave inapplicable
   cells blank rather than guessing a score.
4. Use the full 1-5 scale. See `rubrics.md` for what each number should mean
   -- don't default to the middle just because a report is "fine."
5. Score independently. Don't compare notes with the other 2 reviewers until
   everyone has submitted.
6. Once all 3 CSVs are in `scores/`, run:
   ```
   python3 human_review/aggregate_scores.py
   ```
   This averages each rubric item across the 3 reviewers per report and
   reports pairwise agreement (fraction of rater pairs within 1 point of
   each other, per rubric). **This is the mentor's explicit check**: a 1-10
   scale was rejected specifically because too many choice points with only
   3 raters tends to produce low agreement. If agreement comes back low on
   the 1-5 scale here too, don't widen the scale to fix it -- that makes
   agreement worse, not better. Instead, that rubric's description probably
   needs to be made more concrete (an ambiguous rubric produces disagreement
   regardless of how many buckets the scale has).

## Why a null result still gets scored

A report that says "no confirmed circuit, here's why" is a valid report, not
a failure to review. Score it normally -- `null_result_quality` and
`calibration` exist specifically to reward a well-reasoned null result over
a confident but unsupported claim. Don't penalize a report just because it
didn't find something.
