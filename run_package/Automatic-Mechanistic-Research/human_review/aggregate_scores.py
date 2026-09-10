#!/usr/bin/env python3
"""Aggregate the independent human reviewers' rubric scores for one run.

The human half of the two-arm evaluation, mirroring
``llm_review/aggregate_scores.py``. Reads every completed reviewer CSV under
``<run-dir>/review/completed_human_scores/`` (copies of the exported
``human_*_BLANK_TEMPLATE.csv`` with ``reviewer_id`` filled in), validates
every row against the current rubric version and evidence hash, then:

  * builds the consensus scores -- the row a reviewer filed as
    ``reviewer_id=consensus`` if present, otherwise the per-item mean across
    the independent reviewers, rounded to an integer
  * draws the per-element and per-report rubric-mean graphs
    -> ``<run-dir>/review/graphs/05_human_*.png`` / ``09_human_*``
  * writes the human health verdict -> ``<run-dir>/review/human_health.json``
  * writes per-item mean/stdev -> ``<run-dir>/review/human_review_aggregate.csv``
  * reports pairwise inter-reviewer agreement per metric (fraction of
    reviewer pairs within 1 point) -- the gate on whether the shared 1-10
    scale is holding up. Low agreement on a metric is a signal to sharpen
    that metric's wording in ``automechinterp/eval/matrix_rubric.py``, not to
    widen the scale.

The head-to-head human-vs-judge comparison and Cohen's quadratic kappa live
in ``run_review.py`` / ``automechinterp/eval/agreement.py``.

    python -m human_review.aggregate_scores --run-dir <results-root>/<run_id>
    python human_review/aggregate_scores.py --run-dir <results-root>/<run_id>
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root
from automechinterp.eval.matrix_rubric import IDS
from automechinterp.eval.review_export import load_review_items, load_human_ratings, summarize_source, write_csv


def _consensus(ratings: dict[str, dict[str, dict]]) -> dict[str, dict[str, int]]:
    explicit = ratings.get("consensus", {})
    independent = {r: s for r, s in ratings.items() if r != "consensus"}
    items = {i for scores in ratings.values() for i in scores}
    out: dict[str, dict[str, int]] = {}
    for i in items:
        if i in explicit:
            out[i] = explicit[i]
            continue
        filed = [independent[r][i] for r in independent if i in independent[r]]
        if filed:
            out[i] = {k: round(statistics.fmean(s[k] for s in filed)) for k in IDS}
    return out


def aggregate(run_dir: Path, files: list[Path]) -> dict:
    run_dir = Path(run_dir)
    review = run_dir / "review"
    if not (review / "review_items.jsonl").exists():
        raise SystemExit(f"{review}/review_items.jsonl missing -- run `python run_review.py --run-dir {run_dir} --export` first")
    if not files:
        files = sorted((review / "completed_human_scores").glob("*.csv"))
    if not files:
        raise SystemExit(f"No completed reviewer CSVs. Copy {review}/human_*_BLANK_TEMPLATE.csv into "
                         f"{review}/completed_human_scores/ and fill in reviewer_id + the eight scores.")
    items = load_review_items(run_dir)
    ratings = load_human_ratings(run_dir, files, items)
    reviewers = sorted(ratings)
    consensus = _consensus(ratings)
    health_out = summarize_source(run_dir, "human", consensus, items, judge_validated=True)

    # per-item mean/stdev across everyone who scored it (independent + consensus)
    by_item: dict[tuple[str, str], list[int]] = defaultdict(list)
    for scores in ratings.values():
        for i, s in scores.items():
            for k in IDS:
                by_item[(i, k)].append(s[k])
    agg_rows = [{"item_id": i, "metric": k, "n_raters": len(v),
                 "mean": round(statistics.fmean(v), 2),
                 "stdev": round(statistics.pstdev(v), 2) if len(v) > 1 else 0.0,
                 "scores": v}
                for (i, k), v in sorted(by_item.items())]
    write_csv(review / "human_review_aggregate.csv", agg_rows)

    # pairwise within-1 agreement per metric, across the independent reviewers
    independent = [r for r in reviewers if r != "consensus"]
    agreement = {}
    for k in IDS:
        pairs = []
        common_items = {i for r in independent for i in ratings[r]}
        for i in common_items:
            filed = [ratings[r][i][k] for r in independent if i in ratings[r]]
            pairs += [abs(a - b) <= 1 for a, b in combinations(filed, 2)]
        agreement[k] = {"within_one_rate": round(sum(pairs) / len(pairs), 3) if pairs else None,
                        "n_pairs": len(pairs),
                        "flag": "LOW -- sharpen this metric's wording" if pairs and sum(pairs) / len(pairs) < 0.7 else ""}

    summary = {
        "run_id": json.loads((run_dir / "run_manifest.json").read_text())["run_id"],
        "reviewers": reviewers,
        "consensus_source": "explicit consensus rows" if "consensus" in ratings else "per-item mean of independent reviewers",
        "scored_items": len(consensus),
        "inter_reviewer_agreement": agreement,
        "health": health_out,
    }
    (review / "human_aggregate.json").write_text(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--human-scores", type=Path, nargs="*", default=[],
                    help="explicit reviewer CSV paths (default: <run-dir>/review/completed_human_scores/*.csv)")
    args = ap.parse_args()
    print(json.dumps(aggregate(args.run_dir, args.human_scores), indent=2))


if __name__ == "__main__":
    main()
