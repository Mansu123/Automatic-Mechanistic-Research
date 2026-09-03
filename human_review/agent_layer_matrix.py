#!/usr/bin/env python3
"""Builds an agent x layer matrix for every report in human_review/reports/.

Each Stage D report (see automechinterp/eval/report_writer.py) records, in
prose and tables, which layer each agent in the hierarchy actually touched:

  Network Analyst   -> "flagged layers"
  Layer Agent       -> the "Layer Agent findings" table (one row per layer)
  Component Agent   -> "Spawned Component Agent" column + the "Claimed circuit"
  Skeptic -> Judge  -> the per-head ablations in the evidence transcript

This script re-reads those sections and lays them out as one matrix per
report -- rows are agents, columns are layers -- so a reviewer can see at a
glance where the analysis concentrated and whether the claimed circuit sits
on a layer the Layer Agent actually found load-bearing (rubric 1,
"localization accuracy": check against the Skeptic's evidence, not the
report's own claim).

Deterministic and LLM-free, exactly like report_writer.py: every cell is
copied straight out of the report text, nothing is inferred.

Usage:
    python3 human_review/agent_layer_matrix.py                # all reports
    python3 human_review/agent_layer_matrix.py gpt2__arithmetic_single_digit_addition
    python3 human_review/agent_layer_matrix.py --out some/where.md
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPORTS_DIR = HERE / "reports"
DEFAULT_OUT = HERE.parent / "output" / "eval" / "agent_layer_matrix.md"

AGENTS = ["Network Analyst", "Layer Agent", "Component Agent", "Skeptic -> Judge"]


class ReportMatrix:
    def __init__(self, report_id: str, text: str) -> None:
        self.report_id = report_id
        self.text = text
        self.behavior = _first_line_title(text)
        self.model = _field(text, "Target model")
        self.verdict = _verdict(text)
        self.flagged = _flagged_layers(text)
        self.layer_rows = _layer_agent_rows(text)          # layer -> {frac, spawned, high_superpos}
        self.claimed = _claimed_heads(text)                # list[(layer, head)]
        self.skeptic_heads = _skeptic_head_results(text)   # (layer, head) -> "SPECIFIC" | "weak"

    # --- per-agent, per-layer cell text -------------------------------------
    def _layers(self) -> list[int]:
        seen = set(self.flagged) | set(self.layer_rows)
        seen |= {l for l, _ in self.claimed}
        seen |= {l for (l, _h) in self.skeptic_heads}
        if not seen:
            return []
        return list(range(0, max(seen) + 1))

    def _cell(self, agent: str, layer: int) -> str:
        if agent == "Network Analyst":
            return "flag" if layer in self.flagged else ""
        if agent == "Layer Agent":
            row = self.layer_rows.get(layer)
            if row is None:
                return ""
            cell = f"{row['frac']:+.2f}"
            if row["high_superpos"]:
                cell += " hi-sp"
            return cell
        if agent == "Component Agent":
            bits = []
            row = self.layer_rows.get(layer)
            if row is not None and row["spawned"]:
                bits.append("spawn")
            heads = sorted(h for l, h in self.claimed if l == layer)
            if heads:
                bits.append("claim " + ",".join(f"H{h}" for h in heads))
            return " / ".join(bits)
        if agent == "Skeptic -> Judge":
            results = sorted((h, v) for (l, h), v in self.skeptic_heads.items() if l == layer)
            return " ".join(f"H{h}:{'SPECIFIC' if v == 'SPECIFIC' else 'weak'}" for h, v in results)
        return ""

    def to_markdown(self) -> str:
        layers = self._layers()
        out = [f"## {self.report_id}", ""]
        out.append(f"- **Behavior:** {self.behavior}")
        out.append(f"- **Model:** {self.model}")
        out.append(f"- **Verdict:** {self.verdict or 'n/a'}")
        out.append(f"- **Claimed circuit:** "
                   + (", ".join(f"L{l}H{h}" for l, h in self.claimed) if self.claimed
                      else "none (null result)"))
        out.append("")
        if not layers:
            out.append("_No layer-level agent activity recorded in this report._")
            out.append("")
            return "\n".join(out)

        header = "| Agent | " + " | ".join(f"L{l}" for l in layers) + " |"
        sep = "|---|" + "---|" * len(layers)
        out.append(header)
        out.append(sep)
        for agent in AGENTS:
            cells = [self._cell(agent, l) or " " for l in layers]
            out.append(f"| {agent} | " + " | ".join(cells) + " |")
        out.append("")
        out.append("_Layer Agent cells show fraction recovered (activation patching); "
                   "`flag` = Network Analyst shortlisted the layer; `spawn` = a Component "
                   "Agent was launched there; Skeptic row is per-head ablation outcome._")
        out.append("")
        return "\n".join(out)


# --- parsing helpers -------------------------------------------------------
def _first_line_title(text: str) -> str:
    m = re.search(r"^#\s+(.+)$", text, re.M)
    return m.group(1).strip() if m else "?"


def _field(text: str, label: str) -> str:
    m = re.search(rf"\*\*{re.escape(label)}:\*\*\s*(.+?)\s*$", text, re.M)
    return m.group(1).strip() if m else "?"


def _verdict(text: str) -> str | None:
    m = re.search(r"\*\*Verdict:\s*(.+?)\*\*", text)
    return m.group(1).strip() if m else None


def _flagged_layers(text: str) -> list[int]:
    m = re.search(r"Flagged\s+\d+\s+layer\(s\):\s*\[([0-9,\s]*)\]", text)
    if not m:
        return []
    return [int(x) for x in re.findall(r"\d+", m.group(1))]


def _layer_agent_rows(text: str) -> dict[int, dict]:
    sec = _section(text, "Layer Agent findings")
    rows: dict[int, dict] = {}
    for m in re.finditer(
        r"^\|\s*L(\d+)\s*\|\s*([-+]?[0-9.]+|n/a)\s*\|\s*(\w+)\s*\|\s*(\w+)\s*\|",
        sec, re.M,
    ):
        layer = int(m.group(1))
        frac_raw = m.group(2)
        rows[layer] = {
            "frac": float(frac_raw) if frac_raw != "n/a" else float("nan"),
            "spawned": m.group(3).lower() == "true",
            "high_superpos": m.group(4).lower() == "true",
        }
    return rows


def _claimed_heads(text: str) -> list[tuple[int, int]]:
    sec = _section(text, "Claimed circuit")
    # first backtick-quoted list on the line, e.g. `[(9, 1), (11, 8)]`
    m = re.search(r"`\[(.*?)\]`", sec, re.S)
    if not m:
        return []
    return [(int(a), int(b)) for a, b in re.findall(r"\(\s*(\d+)\s*,\s*(\d+)\s*\)", m.group(1))]


def _skeptic_head_results(text: str) -> dict[tuple[int, int], str]:
    sec = _section(text, "Verification (Skeptic -> Judge)")
    results: dict[tuple[int, int], str] = {}
    for m in re.finditer(
        r"ablate_component\s+L(\d+)H(\d+)\b.*?->\s*(SPECIFIC EFFECT|weak/no effect)",
        sec,
    ):
        key = (int(m.group(1)), int(m.group(2)))
        verdict = "SPECIFIC" if m.group(3) == "SPECIFIC EFFECT" else "weak"
        # a SPECIFIC result wins if the same head is mentioned twice
        if results.get(key) != "SPECIFIC":
            results[key] = verdict
    return results


def _section(text: str, heading: str) -> str:
    """Body of a `## heading` section, up to the next `## ` or EOF."""
    m = re.search(rf"^##\s+{re.escape(heading)}\s*$(.*?)(?=^##\s|\Z)", text, re.M | re.S)
    return m.group(1) if m else ""


# --- driver --------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("report_ids", nargs="*",
                    help="report ids (filenames without .md) to include; default: all")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT,
                    help=f"combined markdown output path (default: {DEFAULT_OUT})")
    args = ap.parse_args()

    paths = sorted(REPORTS_DIR.glob("*.md"))
    if args.report_ids:
        wanted = {rid.removesuffix(".md") for rid in args.report_ids}
        paths = [p for p in paths if p.stem in wanted]
        missing = wanted - {p.stem for p in paths}
        if missing:
            raise SystemExit(f"no report(s) found for: {sorted(missing)}")
    if not paths:
        raise SystemExit(f"no reports in {REPORTS_DIR} -- run `python3 main.py --stage-d` first")

    docs = ["# Agent x layer matrix",
            "",
            "One matrix per report: which agent in the hierarchy touched which "
            "layer, pulled straight from the Stage D report text. Generated by "
            "`human_review/agent_layer_matrix.py`.",
            ""]
    for path in paths:
        rm = ReportMatrix(path.stem, path.read_text())
        block = rm.to_markdown()
        docs.append(block)
        print(block)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(docs))
    print(f"\nWrote combined matrix for {len(paths)} report(s) -> {args.out}")


if __name__ == "__main__":
    main()
