"""Stage D -- Evaluation pipeline.

Runs the existing hierarchy across every registered behavior (behaviors.py)
and writes a human-readable report per (model, behavior) via
eval/report_writer.py. Every report is scored against the single shared
rubric in eval/matrix_rubric.py: eight metrics, integer 1-10, identical for
the human reviewers (human_review/) and the GLM judge (llm_review/). This
stage only produces the reports and their evidence; scoring happens in
run_review.py / run_judge.py.

This does NOT replace stage_b.py's Layer Atlas -- it reuses the same
run_hierarchy() call per behavior and adds the reporting layer on top, so
Stage B's atlas-building and Stage D's report generation can be run
independently or together without duplicating the (expensive) hierarchy
runs' logic.
"""
from __future__ import annotations

import json
from pathlib import Path

from . import config
from .behaviors import ALL_BEHAVIORS
from .eval.report_writer import write_report, save_report, report_id
from .hierarchy import run_hierarchy
from .tools import adapter
from .agents.base import LOG

HUMAN_REVIEW_REPORTS_DIR = Path(__file__).resolve().parent.parent / "human_review" / "reports"


def run_stage_d(backend_kind: str = "heuristic", target_model_id: str | None = None,
                 reports_dir: str | Path = HUMAN_REVIEW_REPORTS_DIR,
                 behaviors: list | None = None,
                 target_dtype=None, seed: int = 0, deep_techniques: bool | None = None,
                 target_revision: str | None = None, target_cache_dir: str | None = None,
                 backend_kwargs: dict | None = None, tool_budget: int | None = None,
                 target_handle=None,
                 max_component_layers: int | None = None,
                 max_specialist_layers: int | None = None) -> dict:
    """behaviors: subset of behaviors.ALL_BEHAVIORS to run (default: all) -- lets
    the eval-matrix runner do a quick smoke pass before a full sweep.
    target_dtype: optional torch dtype for the target model (e.g. torch.float16
    on a GPU for the 3B-7B entries); None keeps transformers' default.

    Reports are scored later against eval/matrix_rubric.py by run_review.py
    (humans) and run_judge.py (GLM); this function does not score."""
    backend_kind = backend_kind or config.LLM_BACKEND
    target_model_id = target_model_id or config.TARGET_MODEL_ID
    behaviors = list(behaviors) if behaviors is not None else ALL_BEHAVIORS
    print("=" * 78)
    print(f"STAGE D -- Evaluation ({target_model_id}, {len(behaviors)} behaviors)")
    print("=" * 78)

    handle = target_handle or adapter.register_model(target_model_id, device=config.DEVICE, torch_dtype=target_dtype,
                                    revision=target_revision, cache_dir=target_cache_dir)
    LOG.emit("System", f"registered {target_model_id}: {handle.n_layers} layers (device={config.DEVICE})")

    reports_dir = Path(reports_dir)
    per_report: list[dict] = []

    for build_behavior in behaviors:
        handle._causal_reference_cache = {}
        task = build_behavior(handle, seed=seed)
        LOG.emit("System", f"--- Stage D report: {task['behavior']} ---")
        result = run_hierarchy(handle, task, backend_kind=backend_kind, deep_techniques=deep_techniques,
                               backend_kwargs=backend_kwargs, tool_budget=tool_budget,
                               max_component_layers=max_component_layers,
                               max_specialist_layers=max_specialist_layers)

        identity = {"model":target_model_id,"revision":handle.revision,"seed":seed,
                    "pipeline":result["run_configuration"],"behavior":build_behavior.__name__,
                    "dataset_hash":task.get("dataset_hash"),
                    "dtype":str(next(handle.model.parameters()).dtype)}
        rid = report_id(task, target_model_id, identity)
        report_text = write_report(task, result, target_model_id)
        report_path = save_report(report_text, reports_dir, rid)

        from .eval.matrix_rubric import IDS
        entry = {
            "report_id": rid,
            "behavior": task["behavior"],
            "category": task["category"],
            "model": target_model_id,
            "report_path": str(report_path),
            "applicable_rubrics": IDS.copy(),
            "verdict": result["verdict"]["verdict"] if result["verdict"] else None,
            "run_identity": identity,
            "raw_hierarchy_result": result,
            "execution_status": result["execution_status"],
            "data_audit": task.get("data_audit", {}),
            "task_definition": {k:v for k,v in task.items() if k != "task_metric_fn"},
            "task_examples": {"discovery_pairs":task.get("discovery_pairs",[]),
                              "evaluation_pairs":task.get("eval_pairs",[]),
                              "stress_status":task.get("stress_data_status","unvalidated")},
        }

        per_report.append(entry)

    print("\n" + "-" * 78)
    print(f"STAGE D: wrote {len(per_report)} report(s) to {reports_dir}")
    print("-" * 78)
    for e in per_report:
        print(f"  {e['report_id']}: verdict={e['verdict']}, "
              f"execution={e['execution_status']}")

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
