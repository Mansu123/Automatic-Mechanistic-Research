"""RubricJudge -- new agent added for the mentor-feedback evaluation plan
("AI-based (automatic) evaluation: use only top-tier models as judges --
Claude or GPT, not smaller models. Score every generated report against the
rubrics."). Scores an already-written report (eval/report_writer.py output)
against the rubrics in eval/rubrics.py.

Deliberately refuses to run on heuristic/hf_local backends rather than
silently downgrading judge quality -- the mentor was explicit on this point.

Distinct from agents/judge.py: that Judge adjudicates one circuit CLAIM
(Confirmed/Probable/Speculative/Refuted) from a fixed 4-item menu through the
same LLMBackend.decide() every other agent uses. This one scores a whole
REPORT against ~7-9 rubric dimensions at once, each a 1-5 integer + written
justification -- free-form structured output that doesn't fit decide()'s
single-action-from-a-menu contract, so it talks to the Anthropic/OpenAI
client directly instead (same pattern llm_backends.py's OpenAIBackend /
AnthropicBackend already use).
"""
from __future__ import annotations

import json
import re
from typing import Any

from ..eval.rubrics import rubrics_for_report

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)

TOP_TIER_BACKENDS = {"openai", "anthropic"}


class RubricJudge:
    name = "rubric_judge"

    def __init__(self, backend_kind: str, model: str | None = None):
        if backend_kind not in TOP_TIER_BACKENDS:
            raise ValueError(
                f"RubricJudge requires a top-tier backend ({sorted(TOP_TIER_BACKENDS)}), got "
                f"'{backend_kind}' -- the mentor's evaluation plan is explicit that automatic "
                "judging must use Claude or GPT, not a smaller model.")
        self.backend_kind = backend_kind
        self.model = model or ("claude-sonnet-5" if backend_kind == "anthropic" else "gpt-4o")

    def _call(self, prompt: str) -> str:
        if self.backend_kind == "anthropic":
            import anthropic
            client = anthropic.Anthropic()
            resp = client.messages.create(
                model=self.model, max_tokens=2048,
                messages=[{"role": "user", "content": prompt}],
            )
            return resp.content[0].text
        from openai import OpenAI
        client = OpenAI()
        resp = client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
        )
        return resp.choices[0].message.content

    def score(self, report_text: str, category: str,
              is_part_of_scaling_sweep: bool = False) -> dict[str, Any]:
        rubrics = rubrics_for_report(category, is_part_of_scaling_sweep)
        rubric_block = "\n".join(f"- {r.id} ({r.name}): {r.description}" for r in rubrics)
        prompt = (
            "You are scoring a mechanistic-interpretability report against a fixed rubric. "
            f"Score each of the following {len(rubrics)} dimensions on a 1-5 integer scale "
            "(1=fails badly, 5=excellent), each with a one-sentence justification grounded in "
            "specifics from the report -- do not invent details not present in the report.\n\n"
            f"Rubric dimensions:\n{rubric_block}\n\n"
            f"Report:\n{report_text}\n\n"
            "Respond with ONLY a JSON object mapping each rubric id to "
            '{"score": <1-5 int>, "justification": "<one sentence>"}.'
        )
        raw = self._call(prompt)
        m = _JSON_RE.search(raw)
        if not m:
            return {"error": f"unparseable judge output: {raw[:300]!r}", "raw": raw}
        try:
            scores = json.loads(m.group(0))
        except json.JSONDecodeError:
            return {"error": f"invalid JSON from judge: {raw[:300]!r}", "raw": raw}
        return {"rubric_ids": [r.id for r in rubrics], "scores": scores, "model": self.model}


def score_report(report_text: str, category: str, backend_kind: str, model: str | None = None,
                  is_part_of_scaling_sweep: bool = False) -> dict[str, Any]:
    return RubricJudge(backend_kind, model=model).score(report_text, category, is_part_of_scaling_sweep)
