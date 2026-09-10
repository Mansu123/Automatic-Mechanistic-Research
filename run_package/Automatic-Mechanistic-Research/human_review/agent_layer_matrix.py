#!/usr/bin/env python3
"""Agent x layer matrix per report, pivoted from the exported coverage CSV.

Reading aid for the human reviewers, not part of scoring. ``run_review.py
--export`` writes ``<run-dir>/review/agent_layer_tool_matrix.csv`` -- the full
report x agent x layer x tool execution inventory, every row straight out of
the deterministic ``report_complete.coverage_records`` (nothing inferred).
This collapses it to one agent x layer grid per report so a reviewer can see
at a glance where the analysis concentrated and whether the claimed circuit
sits on a layer the Layer/Component agents and the Skeptic actually worked.

Each cell shows ``ok/total`` tool calls that succeeded at that (agent, layer),
with ``!`` if any tool errored and ``-`` if every tool there was not_run.

    python -m human_review.agent_layer_matrix --run-dir <results-root>/<run_id>
    python human_review/agent_layer_matrix.py --run-dir <run> --report-id <rid>
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path


def _cell(rows: list[dict]) -> str:
    if not rows:
        return ""
    total = len(rows)
    ok = sum(r["status"] == "ok" for r in rows)
    errored = any(r["status"] == "error" for r in rows)
    if ok == 0 and all(r["status"] == "not_run" for r in rows):
        return "-"
    return f"{ok}/{total}" + ("!" if errored else "")


def build(csv_path: Path, only: set[str] | None = None) -> str:
    by_report: dict[str, list[dict]] = defaultdict(list)
    with csv_path.open(newline="") as f:
        for row in csv.DictReader(f):
            by_report[row["report_id"]].append(row)

    out = ["# Agent x layer matrix",
           "",
           f"One grid per report, pivoted from `{csv_path.name}`. Cells: successful / "
           "total tool calls at that (agent, layer); `!` = a tool errored, `-` = all "
           "not_run, blank = that agent never operates there.",
           ""]
    for rid in sorted(by_report):
        if only and rid not in only:
            continue
        rows = by_report[rid]
        model = rows[0]["model"]
        angle = rows[0]["angle"]
        agents = sorted({r["agent"] for r in rows})
        layers = sorted({int(r["layer"]) for r in rows if r["layer"] not in ("", "global", "None")})
        cols = ["global"] + [str(l) for l in layers]
        grid: dict[tuple[str, str], list[dict]] = defaultdict(list)
        for r in rows:
            key = "global" if r["layer"] in ("", "global", "None") else r["layer"]
            grid[(r["agent"], key)].append(r)

        out.append(f"## {rid}")
        out.append("")
        out.append(f"- **Model:** {model}  **Angle:** {angle}")
        out.append("")
        out.append("| Agent | " + " | ".join(f"L{c}" if c != "global" else "global" for c in cols) + " |")
        out.append("|---|" + "---|" * len(cols))
        for agent in agents:
            out.append(f"| {agent} | " + " | ".join(_cell(grid.get((agent, c), [])) or " " for c in cols) + " |")
        out.append("")
    return "\n".join(out) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--report-id", action="append", default=[], help="limit to these report ids (repeatable)")
    ap.add_argument("--out", type=Path, default=None, help="default: <run-dir>/review/agent_layer_matrix.md")
    args = ap.parse_args()
    csv_path = args.run_dir / "review" / "agent_layer_tool_matrix.csv"
    if not csv_path.exists():
        raise SystemExit(f"{csv_path} missing -- run `python run_review.py --run-dir {args.run_dir} --export` first")
    text = build(csv_path, set(args.report_id) or None)
    out = args.out or (args.run_dir / "review" / "agent_layer_matrix.md")
    out.write_text(text)
    print(text)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
