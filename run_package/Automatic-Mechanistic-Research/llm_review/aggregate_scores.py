#!/usr/bin/env python3
"""Aggregate the GLM judge's rubric scores for one run into a health readout.

The LLM half of the two-arm evaluation, mirroring
``human_review/aggregate_scores.py``. Reads every scored file under
``<run-dir>/review/judge_scores/`` (written by ``llm_review/run_judge.py``),
checks each stored score still matches its evidence-backed judgment for the
current rubric version, then:

  * draws the per-element (layer/agent/tool/S-EAP) and per-report rubric-mean
    graphs -> ``<run-dir>/review/graphs/05_judge_*.png`` / ``09_judge_*``
  * writes the judge health verdict -> ``<run-dir>/review/judge_health.json``
  * writes a per-metric / per-angle / per-model judge summary CSV
    -> ``<run-dir>/review/judge_score_summary.csv``

The judge arm is always reported as **provisional** here: report-level
agreement with humans is established only by ``run_review.py`` (see
``automechinterp/eval/agreement.py``), and it never licenses element-level
substitution for a human reviewer.

    python -m llm_review.aggregate_scores --run-dir <results-root>/<run_id>
    python llm_review/aggregate_scores.py --run-dir <results-root>/<run_id>
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root
import numpy as np
from automechinterp.eval.matrix_rubric import IDS, health
from automechinterp.eval.review_export import load_review_items, load_judge_scores, summarize_source, write_csv


def aggregate(run_dir: Path) -> dict:
    run_dir = Path(run_dir)
    review = run_dir / "review"
    if not (review / "review_items.jsonl").exists():
        raise SystemExit(f"{review}/review_items.jsonl missing -- run `python run_review.py --run-dir {run_dir} --export` first")
    items = load_review_items(run_dir)
    judge, versions = load_judge_scores(run_dir, items)
    if not judge:
        print(f"No scored judge files under {review}/judge_scores/ yet -- run `python -m llm_review.run_judge --run-dir {run_dir}`.")
    health_out = summarize_source(run_dir, "judge", judge, items, judge_validated=False)

    rows = []
    per_metric = defaultdict(list)
    for i, scores in judge.items():
        item = items[i]
        h = health(scores, item["execution_status"] == "completed")
        rows.append({"item_id": i, "item_type": item["item_type"], "model": item["model"],
                     "angle": item["angle"], "angle_name": item["angle_name"],
                     "behavior_id": item["behavior_id"], "element": item["element"],
                     **scores, "mean_1to10": round(h["mean_1to10"], 3),
                     "health": "provisional_" + h["status"]})
        for k in IDS:
            per_metric[k].append(scores[k])
    write_csv(review / "judge_score_summary.csv", sorted(rows, key=lambda r: (r["model"], r["angle"], r["item_type"], r["element"])))

    summary = {
        "run_id": json.loads((run_dir / "run_manifest.json").read_text())["run_id"],
        "judge_versions": [list(v) for v in sorted(versions)],
        "scored_items": len(judge),
        "scored_by_type": dict(sorted({t: sum(items[i]["item_type"] == t for i in judge)
                                       for t in {items[i]["item_type"] for i in judge}}.items())) if judge else {},
        "per_metric_mean_1to10": {k: round(float(np.mean(per_metric[k])), 3) if per_metric[k] else None for k in IDS},
        "health": health_out,
        "judge_validation": "provisional until run_review.py validates report-level agreement with humans",
    }
    (review / "judge_aggregate.json").write_text(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", type=Path, required=True)
    print(json.dumps(aggregate(ap.parse_args().run_dir), indent=2))


if __name__ == "__main__":
    main()
