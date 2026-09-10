#!/usr/bin/env python3
"""Publication figures rendered from a real evaluation run -- no hardcoded numbers.

`run_colab.py` -> `automechinterp/eval/review_export.py:export_run` already
writes the full set of analysis graphs under `<run_id>/review/graphs/` (model x
angle coverage, agent x layer x tool, S-EAP, rubric/agreement). This script
re-renders a curated three-figure subset at publication DPI, driven entirely by
that run's exported CSVs. It fabricates nothing: if the run directory or a
required CSV is missing it errors out rather than falling back to invented data.

    python generate_figures.py --run-dir <results-root>/<run_id>
    python generate_figures.py --run-dir <run> --out output/figures
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ANGLES = list(range(1, 26))


def _read_csv(path: Path) -> list[dict]:
    if not path.exists():
        raise SystemExit(f"missing {path} -- run `python run_review.py --run-dir <run> --export` first")
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def _num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _heatmap(ax, data, rows, cols, title, vmin, vmax, cbar_label):
    arr = np.ma.masked_invalid(np.asarray(data, float))
    cmap = plt.colormaps["viridis"].copy()
    cmap.set_bad("#d9d9d9")
    im = ax.imshow(arr, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xticks(range(len(cols)), cols, fontsize=7, rotation=90)
    ax.set_yticks(range(len(rows)), rows, fontsize=7)
    ax.set_title(title, fontweight="bold")
    cb = ax.figure.colorbar(im, ax=ax, shrink=0.8)
    cb.set_label(cbar_label)


def figures(run_dir: Path, out: Path) -> list[Path]:
    run_dir = Path(run_dir)
    review = run_dir / "review"
    manifest = json.loads((run_dir / "run_manifest.json").read_text())
    models = [m["model_id"] for m in manifest["config"]["models"]]
    out.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 9, "axes.titleweight": "bold", "savefig.dpi": 300,
                         "axes.spines.top": False, "axes.spines.right": False})
    written: list[Path] = []

    # -- Figure 1: model x angle completed-behavior coverage -------------------
    coverage = _read_csv(run_dir / "coverage.csv")
    done = defaultdict(int)
    total = defaultdict(int)
    for r in coverage:
        key = (r["model"], int(r["angle"]))
        total[key] += 1
        done[key] += r["status"] == "completed"
    data = [[100 * done[(m, a)] / total[(m, a)] if total[(m, a)] else np.nan for a in ANGLES] for m in models]
    fig, ax = plt.subplots(figsize=(max(11, len(ANGLES) * 0.45), max(4, len(models) * 0.32)))
    _heatmap(ax, data, models, [str(a) for a in ANGLES],
             f"Completed behavior coverage (%) — {len(models)} models × 25 angles × "
             f"{len(manifest['config']['seeds'])} seed(s)", 0, 100, "% of jobs completed")
    fig.tight_layout()
    p = out / "figure1_model_angle_coverage.png"
    fig.savefig(p); fig.savefig(p.with_suffix(".pdf")); plt.close(fig); written.append(p)

    # -- Figure 2: model x angle rubric mean, else S-EAP score ----------------
    element_path = review / "all_element_scores_and_health.csv"
    cells = defaultdict(list)
    if element_path.exists():
        for r in _read_csv(element_path):
            if r["item_type"] == "report" and r["source"].startswith(("human", "judge")):
                v = _num(r.get("mean"))
                if v is not None:
                    cells[(r["model"], int(r["angle"]))].append(v)
        label, unit = "Report rubric mean (1–10) — scored reports", "rubric mean (1–10)"
    if not cells:
        for r in _read_csv(review / "evaluation_matrix.csv"):
            v = _num(r.get("seap_score"))
            if v is not None:
                cells[(r["model"], int(r["angle"]))].append(v)
        label = "S-EAP descriptive accuracy 100/(1+normalized MAE) — no rubric scores yet"
        unit = "S-EAP score (0–100)"
    vmax = 10 if unit.startswith("rubric") else 100
    data = [[float(np.mean(cells[(m, a)])) if cells[(m, a)] else np.nan for a in ANGLES] for m in models]
    fig, ax = plt.subplots(figsize=(max(11, len(ANGLES) * 0.45), max(4, len(models) * 0.32)))
    _heatmap(ax, data, models, [str(a) for a in ANGLES], label, 0, vmax, unit)
    fig.tight_layout()
    p = out / "figure2_model_angle_evaluation.png"
    fig.savefig(p); fig.savefig(p.with_suffix(".pdf")); plt.close(fig); written.append(p)

    # -- Figure 3: S-EAP exact vs signed approximation + per-model nMAE -------
    seap = _read_csv(review / "seap_exact_vs_approximation.csv")
    x = np.array([_num(r["exact"]) for r in seap if _num(r["exact"]) is not None])
    y = np.array([_num(r["approximation"]) for r in seap if _num(r["approximation"]) is not None])
    nmae = {}
    for r in _read_csv(review / "evaluation_matrix.csv"):
        v = _num(r.get("seap_normalized_mae"))
        if v is not None:
            nmae.setdefault(r["model"], []).append(v)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    if len(x):
        lo, hi = float(min(x.min(), y.min())), float(max(x.max(), y.max()))
        ax1.plot([lo, hi], [lo, hi], "k--", lw=1)
        ax1.scatter(x, y, s=8, alpha=0.25, color="#235789")
    ax1.set(xlabel="Exact signed joint interaction I(i,j)", ylabel="S-EAP signed approximation",
            title="Paired interaction validation (pairs within a prompt are dependent)")
    present = [m for m in models if m in nmae]
    ax2.bar(range(len(present)), [float(np.mean(nmae[m])) for m in present], color="#C8772D",
            edgecolor="black", lw=0.8)
    ax2.set_xticks(range(len(present)), present, rotation=40, ha="right", fontsize=7)
    ax2.set(ylabel="mean normalized MAE (lower is better)", title="S-EAP approximation error by model")
    ax2.axhline(1.0, color="#888", lw=1, ls=":")
    fig.tight_layout()
    p = out / "figure3_seap_exact_vs_approximation.png"
    fig.savefig(p); fig.savefig(p.with_suffix(".pdf")); plt.close(fig); written.append(p)

    return written


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("output/figures"))
    args = ap.parse_args()
    if not (args.run_dir / "run_manifest.json").exists():
        raise SystemExit(f"{args.run_dir} is not a completed run directory (no run_manifest.json)")
    for p in figures(args.run_dir, args.out):
        print(f"wrote {p} (+ .pdf)")


if __name__ == "__main__":
    main()
