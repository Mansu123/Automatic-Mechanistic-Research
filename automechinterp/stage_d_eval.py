"""Stage D -- Evaluation pipeline (mentor-feedback point 5).

Runs the existing hierarchy across every registered behavior (behaviors.py),
writes a human-readable report per (model, behavior) via eval/report_writer.py,
optionally scores each report with the new RubricJudge agent (Claude/GPT
only), and drops every report into human_review/reports/ so the 3 human
experts have something to score against the same rubrics.

This does NOT replace stage_b.py's Layer Atlas -- it reuses the same
run_hierarchy() call per behavior and adds the reporting + scoring layer the
mentor's evaluation plan needs on top, so Stage B's atlas-building and Stage
D's report/score-generation can be run independently or together without
duplicating the (expensive) hierarchy runs' logic.
"""
from __future__ import annotations

import json
from pathlib import Path

from . import config
from .behaviors import ALL_BEHAVIORS
from .eval.report_writer import write_report, save_report, report_id
from .eval.rubrics import rubrics_for_report
from .hierarchy import run_hierarchy
from .tools import adapter
from .agents.base import LOG

HUMAN_REVIEW_REPORTS_DIR = Path(__file__).resolve().parent.parent / "human_review" / "reports"


def run_stage_d(backend_kind: str = "heuristic", target_model_id: str | None = None,
                 ai_judge_backend: str | None = None,
                 is_part_of_scaling_sweep: bool = False,
                 reports_dir: str | Path = HUMAN_REVIEW_REPORTS_DIR,
                 behaviors: list | None = None,
                 target_dtype=None) -> dict:
    """ai_judge_backend: None to skip automatic scoring (e.g. no API key
    configured yet), or 'anthropic'/'openai' to run RubricJudge on every
    report -- see agents/rubric_judge.py's top-tier-only requirement.

    behaviors: subset of behaviors.ALL_BEHAVIORS to run (default: all) -- lets
    the eval-matrix runner do a quick smoke pass before a full sweep.
    target_dtype: optional torch dtype for the target model (e.g. torch.float16
    on a GPU for the 3B-7B entries); None keeps transformers' default."""
    backend_kind = backend_kind or config.LLM_BACKEND
    target_model_id = target_model_id or config.TARGET_MODEL_ID
    behaviors = list(behaviors) if behaviors is not None else ALL_BEHAVIORS
    print("=" * 78)
    print(f"STAGE D -- Evaluation ({target_model_id}, {len(behaviors)} behaviors, "
          f"ai_judge={ai_judge_backend or 'skipped'})")
    print("=" * 78)

    handle = adapter.register_model(target_model_id, device=config.DEVICE, torch_dtype=target_dtype)
    LOG.emit("System", f"registered {target_model_id}: {handle.n_layers} layers (device={config.DEVICE})")

    rubric_judge = None
    if ai_judge_backend:
        from .agents.rubric_judge import RubricJudge
        rubric_judge = RubricJudge(ai_judge_backend)

    reports_dir = Path(reports_dir)
    per_report: list[dict] = []

    for build_behavior in behaviors:
        task = build_behavior(handle)
        LOG.emit("System", f"--- Stage D report: {task['behavior']} ---")
        result = run_hierarchy(handle, task, backend_kind=backend_kind, deep_techniques=False)

        rid = report_id(task, target_model_id)
        report_text = write_report(task, result, target_model_id)
        report_path = save_report(report_text, reports_dir, rid)

        applicable_rubrics = [r.id for r in rubrics_for_report(task["category"], is_part_of_scaling_sweep)]
        entry = {
            "report_id": rid,
            "behavior": task["behavior"],
            "category": task["category"],
            "model": target_model_id,
            "report_path": str(report_path),
            "applicable_rubrics": applicable_rubrics,
            "verdict": result["verdict"]["verdict"] if result["verdict"] else None,
            "ai_scores": None,
        }

        if rubric_judge is not None:
            entry["ai_scores"] = rubric_judge.score(report_text, task["category"], is_part_of_scaling_sweep)
            LOG.emit("RubricJudge", f"{rid}: {entry['ai_scores'].get('scores', entry['ai_scores'])}")

        per_report.append(entry)

    print("\n" + "-" * 78)
    print(f"STAGE D: wrote {len(per_report)} report(s) to {reports_dir}")
    print("-" * 78)
    for e in per_report:
        print(f"  {e['report_id']}: verdict={e['verdict']}, "
              f"ai_scored={'yes' if e['ai_scores'] else 'no'}")

    return {
        "target_model_id": target_model_id,
        "reports_dir": str(reports_dir),
        "reports": per_report,
    }


def save_stage_d_json(result: dict, out_dir: str = "output/eval") -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    from datetime import datetime
    from .eval.report_writer import _slug
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    # model ids like "Qwen/Qwen2.5-0.5B-Instruct" contain a literal "/" -- slug it
    # so this doesn't try to write into a (nonexistent) subdirectory named "Qwen".
    path = out_dir / f"stage_d_{_slug(result['target_model_id'])}_{ts}.json"
    path.write_text(json.dumps(result, indent=2, default=str))
    return path
