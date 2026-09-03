#!/usr/bin/env python3
"""Combines the 3 human reviewers' CSVs in human_review/scores/ into one
per-report, per-rubric average, and reports pairwise inter-rater agreement.

Usage:
    python3 human_review/aggregate_scores.py

Agreement is computed as the fraction of reviewer PAIRS whose scores on the
same (report_id, rubric) are within 1 point of each other -- deliberately
simple (not Cohen's/Fleiss' kappa) because the goal here is exactly what the
mentor asked for: "check agreement/variance across the 3 experts" as a gate
on whether the 1-5 scale is holding up, not a publication-grade statistic.
Low agreement on the *rubric that's causing it* is the actionable signal --
per the mentor's guidance, the fix is sharpening that rubric's wording, not
widening the scale.
"""
from __future__ import annotations

import csv
import statistics
from collections import defaultdict
from itertools import combinations
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCORES_DIR = HERE / "scores"
OUT_CSV = HERE.parent / "output" / "eval" / "human_review_aggregate.csv"

NON_RUBRIC_COLS = {"reviewer_name", "report_id", "notes"}


def load_scores() -> dict[str, dict[tuple[str, str], int]]:
    """reviewer_name -> {(report_id, rubric_id): score}"""
    per_reviewer: dict[str, dict[tuple[str, str], int]] = {}
    csv_files = sorted(p for p in SCORES_DIR.glob("*.csv") if p.name != "scoring_template.csv")
    if not csv_files:
        raise SystemExit(f"no reviewer CSVs found in {SCORES_DIR} -- copy scoring_template.csv there "
                          "first (see human_review/README.md)")
    for path in csv_files:
        with path.open(newline="") as f:
            reader = csv.DictReader(f)
            rubric_cols = [c for c in (reader.fieldnames or []) if c not in NON_RUBRIC_COLS]
            for row in reader:
                reviewer = row.get("reviewer_name", "").strip() or path.stem
                report_id = row.get("report_id", "").strip()
                if not report_id or report_id.startswith("<"):
                    continue
                bucket = per_reviewer.setdefault(reviewer, {})
                for rubric_id in rubric_cols:
                    raw = (row.get(rubric_id) or "").strip()
                    if not raw:
                        continue
                    try:
                        bucket[(report_id, rubric_id)] = int(raw)
                    except ValueError:
                        print(f"WARNING: {path.name} row {report_id}/{rubric_id}: "
                              f"non-integer score {raw!r}, skipping")
    return per_reviewer


def main() -> None:
    per_reviewer = load_scores()
    reviewers = sorted(per_reviewer)
    print(f"Loaded scores from {len(reviewers)} reviewer(s): {reviewers}\n")

    # (report_id, rubric_id) -> [scores from each reviewer that scored it]
    by_item: dict[tuple[str, str], list[int]] = defaultdict(list)
    for reviewer in reviewers:
        for key, score in per_reviewer[reviewer].items():
            by_item[key].append(score)

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    agreement_by_rubric: dict[str, list[bool]] = defaultdict(list)
    for (report_id, rubric_id), scores in sorted(by_item.items()):
        mean = statistics.fmean(scores)
        stdev = statistics.pstdev(scores) if len(scores) > 1 else 0.0
        pair_agree = [abs(a - b) <= 1 for a, b in combinations(scores, 2)]
        agreement_by_rubric[rubric_id].extend(pair_agree)
        rows.append({
            "report_id": report_id, "rubric_id": rubric_id, "n_raters": len(scores),
            "mean": round(mean, 2), "stdev": round(stdev, 2), "scores": scores,
        })

    with OUT_CSV.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["report_id", "rubric_id", "n_raters", "mean", "stdev", "scores"])
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"Wrote per-item averages -> {OUT_CSV}\n")

    print("Pairwise agreement by rubric (fraction of rater-pairs within 1 point):")
    print("-" * 60)
    overall_pairs: list[bool] = []
    for rubric_id in sorted(agreement_by_rubric):
        pairs = agreement_by_rubric[rubric_id]
        overall_pairs.extend(pairs)
        rate = sum(pairs) / len(pairs) if pairs else float("nan")
        flag = "  <-- LOW, consider sharpening this rubric's wording" if pairs and rate < 0.7 else ""
        print(f"  {rubric_id:35s} {rate:5.0%}  (n={len(pairs)} pairs){flag}")
    if overall_pairs:
        overall_rate = sum(overall_pairs) / len(overall_pairs)
        print("-" * 60)
        print(f"  {'OVERALL':35s} {overall_rate:5.0%}  (n={len(overall_pairs)} pairs)")


if __name__ == "__main__":
    main()
