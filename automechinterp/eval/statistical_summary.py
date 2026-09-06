"""Statistical summary module for n=50 multi-seed evaluation.

Aggregates per-task results across seeds and computes:
  - mean ± std for every metric (F1, precision, recall, recovery, budget, FPR)
  - per-method 95% confidence intervals (normal approximation, n≥30)
  - Wilcoxon signed-rank test p-values vs. reference system
  - Generates the full results table in LaTeX and Markdown formats
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class AggregatedResult:
    method: str
    n_seeds: int
    metrics: dict[str, list[float]]  # metric_name → list of per-seed values

    def mean(self, key: str) -> float:
        vals = self.metrics.get(key, [])
        return float(np.mean(vals)) if vals else float("nan")

    def std(self, key: str) -> float:
        vals = self.metrics.get(key, [])
        return float(np.std(vals, ddof=1)) if len(vals) > 1 else float("nan")

    def ci95(self, key: str) -> tuple[float, float]:
        """95% CI via normal approximation: mean ± 1.96 * std / sqrt(n)."""
        vals = self.metrics.get(key, [])
        n = len(vals)
        if n < 2:
            return float("nan"), float("nan")
        m = float(np.mean(vals))
        s = float(np.std(vals, ddof=1))
        margin = 1.96 * s / math.sqrt(n)
        return m - margin, m + margin

    def format(self, key: str, decimals: int = 3) -> str:
        m, s = self.mean(key), self.std(key)
        if math.isnan(m):
            return "—"
        return f"{m:.{decimals}f} ± {s:.{decimals}f}"


def wilcoxon_p(vals_a: list[float], vals_b: list[float]) -> float:
    """Wilcoxon signed-rank test p-value comparing two paired samples.
    Returns 1.0 if scipy not available or insufficient data.
    """
    try:
        from scipy.stats import wilcoxon
        if len(vals_a) < 5 or len(vals_a) != len(vals_b):
            return float("nan")
        diffs = [a - b for a, b in zip(vals_a, vals_b)]
        if all(d == 0 for d in diffs):
            return 1.0
        _, p = wilcoxon(diffs)
        return float(p)
    except Exception:
        return float("nan")


def aggregate_results(
    all_run_results: list[list[dict]],
) -> dict[str, AggregatedResult]:
    """Aggregate a list of per-seed comparison results into AggregatedResult per method.

    Args:
        all_run_results: List of per-seed results. Each element is the list of
            `comp.results` dicts from `run_comparison()` (one per method).

    Returns:
        Dict: method → AggregatedResult with all metric distributions.
    """
    method_metrics: dict[str, dict[str, list[float]]] = {}

    for seed_results in all_run_results:
        for r in seed_results:
            m = r["method"]
            if m not in method_metrics:
                method_metrics[m] = {
                    k: [] for k in [
                        "precision", "recall", "f1",
                        "metric_recovery", "circuit_size",
                        "budget_used", "runtime_s",
                    ]
                }
            for key in method_metrics[m]:
                val = r.get(key, float("nan"))
                if not math.isnan(val):
                    method_metrics[m][key].append(float(val))

    return {
        m: AggregatedResult(method=m, n_seeds=len(v.get("f1", [])), metrics=v)
        for m, v in method_metrics.items()
    }


def add_fpr_to_aggregated(
    agg: dict[str, AggregatedResult],
    fpr_per_seed: list[dict],  # list of {method: FPRResult} per seed
) -> None:
    """Add false positive rate data to existing AggregatedResult objects."""
    for seed_fpr in fpr_per_seed:
        for method, fpr_result in seed_fpr.items():
            if method in agg:
                agg[method].metrics.setdefault("fpr", []).append(
                    fpr_result.false_positive_rate
                )


def markdown_table(
    agg: dict[str, AggregatedResult],
    reference_method: str = "reference",
    metrics: list[str] | None = None,
) -> str:
    """Generate a markdown table with mean ± std for each metric.

    Adds Wilcoxon p-values vs. the reference system in the last column.
    Bold the best value per metric column.
    """
    if metrics is None:
        metrics = ["f1", "metric_recovery", "fpr", "budget_used"]

    metric_labels = {
        "f1": "F1",
        "precision": "Precision",
        "recall": "Recall",
        "metric_recovery": "Recovery",
        "fpr": "FPR ↓",
        "circuit_size": "Circuit Size",
        "budget_used": "Budget",
        "runtime_s": "Time (s)",
    }

    header_cols = ["Method"] + [metric_labels.get(m, m) for m in metrics] + ["p-val vs. ours"]
    header = "| " + " | ".join(header_cols) + " |\n"
    header += "| " + " | ".join(["---"] * len(header_cols)) + " |\n"

    # Find best value per metric for bolding
    best: dict[str, float] = {}
    for m in metrics:
        vals = {name: agg[name].mean(m) for name in agg if not math.isnan(agg[name].mean(m))}
        if vals:
            # Higher is better for all metrics except fpr, budget_used, runtime_s
            if m in ("fpr", "budget_used", "runtime_s"):
                best[m] = min(vals.values())
            else:
                best[m] = max(vals.values())

    ref_vals = {m: agg.get(reference_method, AggregatedResult("", 0, {})).metrics.get(m, [])
                for m in metrics}

    rows = []
    for name, result in sorted(agg.items(), key=lambda x: -x[1].mean("f1") if not math.isnan(x[1].mean("f1")) else 0):
        cells = [f"**{name}**" if name == reference_method else name]
        for m in metrics:
            fmt = result.format(m, decimals=3)
            mean_val = result.mean(m)
            is_best = not math.isnan(mean_val) and m in best and abs(mean_val - best[m]) < 1e-6
            cells.append(f"**{fmt}**" if is_best else fmt)

        # Wilcoxon p-value vs. reference
        if name != reference_method and "f1" in result.metrics:
            p = wilcoxon_p(result.metrics.get("f1", []), ref_vals.get("f1", []))
            p_str = f"{p:.3f}" if not math.isnan(p) else "—"
        else:
            p_str = "—"
        cells.append(p_str)
        rows.append("| " + " | ".join(cells) + " |")

    return header + "\n".join(rows)


def latex_table(
    agg: dict[str, AggregatedResult],
    metrics: list[str] | None = None,
    caption: str = "Circuit discovery method comparison (mean ± std, n=50 seeds).",
    label: str = "tab:comparison",
) -> str:
    """Generate a LaTeX tabular environment for the paper."""
    if metrics is None:
        metrics = ["f1", "metric_recovery", "fpr", "budget_used"]

    metric_labels = {
        "f1": "F1",
        "metric_recovery": "Recovery",
        "fpr": "FPR $\\downarrow$",
        "budget_used": "Budget",
    }

    col_spec = "l" + "c" * len(metrics)
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\begin{tabular}{" + col_spec + "}",
        r"\toprule",
        "Method & " + " & ".join(metric_labels.get(m, m) for m in metrics) + r" \\",
        r"\midrule",
    ]

    for name, result in sorted(agg.items(), key=lambda x: -x[1].mean("f1") if not math.isnan(x[1].mean("f1")) else 0):
        cells = [name.replace("_", r"\_")]
        for m in metrics:
            m_val = result.mean(m)
            s_val = result.std(m)
            if math.isnan(m_val):
                cells.append("—")
            else:
                cells.append(f"${m_val:.3f} \\pm {s_val:.3f}$")
        lines.append(" & ".join(cells) + r" \\")

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        r"\end{table}",
    ]
    return "\n".join(lines)


def save_aggregated(
    agg: dict[str, AggregatedResult],
    out_dir: Path,
    task_name: str,
    model_id: str,
) -> None:
    """Write JSON + markdown + LaTeX results to disk."""
    from datetime import datetime
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = f"{task_name}_{model_id.replace('/', '_')}_{ts}"

    # JSON
    json_data = {
        name: {
            "n_seeds": r.n_seeds,
            "metrics": {
                k: {
                    "mean": round(r.mean(k), 4),
                    "std": round(r.std(k), 4),
                    "ci95_lo": round(r.ci95(k)[0], 4),
                    "ci95_hi": round(r.ci95(k)[1], 4),
                    "values": [round(v, 4) for v in r.metrics.get(k, [])],
                }
                for k in r.metrics
            }
        }
        for name, r in agg.items()
    }
    (out_dir / f"{stem}_stats.json").write_text(json.dumps(json_data, indent=2))

    # Markdown
    md = f"# {task_name} ({model_id}) — n={next(iter(agg.values())).n_seeds if agg else 0} seeds\n\n"
    md += markdown_table(agg) + "\n"
    (out_dir / f"{stem}_stats.md").write_text(md)

    # LaTeX
    lt = latex_table(agg, caption=f"{task_name} on {model_id}.", label=f"tab:{task_name}")
    (out_dir / f"{stem}_stats.tex").write_text(lt)

    print(f"  Saved stats to {out_dir}/{stem}_stats.{{json,md,tex}}")
