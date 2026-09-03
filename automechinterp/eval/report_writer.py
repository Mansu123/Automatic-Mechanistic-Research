"""Turns one run_hierarchy() result into a standalone markdown report.

Addresses the mentor's "Depth of Findings" point directly: the raw pipeline
output (hierarchy.py's return dict) is structured JSON, not something a
human or an AI judge can read for clarity/explanatory-depth against the
rubrics in eval/rubrics.py. This module is the missing step in between --
every report it writes is what actually gets scored, by both
agents/rubric_judge.py and the human reviewers under human_review/.

Deliberately NOT an LLM agent: every field below is copied straight out of
the evidence the hierarchy already produced (Sec. 5's "faithfulness"
requirement -- nothing here is invented prose), so there is nothing for a
policy to decide. Free-form synthesis, if wanted later, belongs in a real
agent (with its own rubric score for whether it stayed grounded), not in
this deterministic formatter.
"""
from __future__ import annotations

import re
from pathlib import Path


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def report_id(task: dict, model_id: str) -> str:
    return f"{_slug(model_id)}__{_slug(task['behavior'])}"


def write_report(task: dict, result: dict, model_id: str) -> str:
    verdict_report = result.get("verdict")
    lines: list[str] = []

    lines.append(f"# {task['behavior']}")
    lines.append("")
    lines.append(f"**Angle / category:** {task['category']}  ")
    lines.append(f"**Target model:** {model_id}  ")
    lines.append(f"**Tool calls spent:** {result.get('tool_calls_spent', 'n/a')}")
    lines.append("")

    lines.append("## Task definition")
    lines.append("")
    lines.append(f"- Clean prompt: `{task['clean_prompt']}`")
    lines.append(f"- Corrupted prompt: `{task['corrupted_prompt']}`")
    lines.append(f"- Expected token: `{task['io_token']}`  vs. contrast token: `{task['s_token']}`")
    lines.append(f"- Held-out eval prompts: {len(task.get('eval_prompts', []))}")
    lines.append("")

    lines.append("## Network Analyst: flagged layers")
    lines.append("")
    flagged = result.get("flagged_layers", [])
    lines.append(f"Flagged {len(flagged)} layer(s): {flagged}" if flagged else "No layers flagged.")
    lines.append("")

    layer_states = result.get("layer_states", {})
    if layer_states:
        lines.append("## Layer Agent findings")
        lines.append("")
        lines.append("| Layer | Fraction recovered | Spawned Component Agent | High superposition |")
        lines.append("|---|---|---|---|")
        for layer_idx in sorted(layer_states):
            s = layer_states[layer_idx]
            frac = s.get("fraction_recovered")
            frac_str = f"{frac:.3f}" if isinstance(frac, (int, float)) else "n/a"
            lines.append(f"| L{layer_idx} | {frac_str} | "
                         f"{s.get('spawn_component_agent', False)} | {s.get('high_superposition', False)} |")
        lines.append("")

    claimed = result.get("claimed_heads", [])
    lines.append("## Claimed circuit")
    lines.append("")
    lines.append(f"`{claimed}`" if claimed else "No component-level claim survived Layer Agent triage "
                                                  "(null result).")
    lines.append("")

    lines.append("## Verification (Skeptic -> Judge)")
    lines.append("")
    if verdict_report:
        lines.append(f"**Verdict: {verdict_report['verdict']}**")
        lines.append("")
        lines.append(f"Reasoning: {verdict_report['reasoning']}")
        lines.append("")
        lines.append("Evidence transcript:")
        lines.append("```")
        lines.append(verdict_report.get("evidence", ""))
        lines.append("```")
    else:
        lines.append("Not run -- no claim reached the Skeptic.")
    lines.append("")

    return "\n".join(lines)


def save_report(report_text: str, out_dir: str | Path, rid: str) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{rid}.md"
    path.write_text(report_text)
    return path
