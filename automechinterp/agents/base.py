"""Shared ReAct loop scaffold + budget tracking + cost telemetry (Sec. 3.1, 3.6).

Every concrete agent (NetworkAnalyst, LayerAgent, ComponentAgent, Skeptic)
exposes a `tools: dict[str, Callable[[], str]]` -- each value a zero-arg
closure already bound to its context (a model handle, a prompt pair, a
candidate layer index, ...). The agent doesn't know or care whether the
brain picking the next key is a hand-written rule table (HeuristicBackend)
or Qwen2.5-7B-Instruct (HFLocalBackend): both implement
`LLMBackend.decide(system, evidence, tool_menu) -> {"action": tool_name}`.
That single seam is what makes the Sec. 4.7 "backbone comparison" ablation a
one-line swap instead of a rewrite.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Callable

from ..llm_backends import LLMBackend


@dataclass
class ToolCallBudget:
    """Global cap + per-agent sub-budget (Sec. 3.6 'Budget-aware termination').
    `global_remaining` is a shared one-element list so every agent spawned
    during a run draws from the same pool; a lock makes spend() safe when
    Layer Agents run concurrently (Sec. 3.1: 'Layer Agents run in parallel')."""
    global_remaining: list  # list[int], shared mutable cell
    per_agent_limit: int
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def spend(self) -> bool:
        with self._lock:
            if self.global_remaining[0] <= 0:
                return False
            self.global_remaining[0] -= 1
            return True


class AgentLog:
    def __init__(self):
        self.entries: list[dict] = []

    def emit(self, agent: str, message: str, gap_id: str | None = None) -> None:
        self.entries.append({"agent": agent, "message": message, "gap_id": gap_id})
        tag = f" [{gap_id}]" if gap_id else ""
        print(f"[{agent}]{tag} {message}")

    def dump(self) -> str:
        return "\n".join(f"[{e['agent']}] {e['message']}" for e in self.entries)


LOG = AgentLog()


class Agent:
    name = "agent"
    system_prompt = ""

    def __init__(self, backend: LLMBackend, tools: dict[str, Callable[[], str]],
                 budget: ToolCallBudget, max_steps: int = 8):
        self.backend = backend
        self.tools = tools
        self.budget = budget
        self.max_steps = min(max_steps, budget.per_agent_limit)
        self.evidence: list[str] = []
        self._failed_tools: set[str] = set()

    def run(self, initial_evidence: str = "") -> list[str]:
        if initial_evidence:
            self.evidence.append(initial_evidence)
        for _ in range(self.max_steps):
            if not self.budget.spend():
                LOG.emit(self.name, "tool-call budget exhausted; stopping with best-effort evidence")
                break
            evidence_text = "\n".join(f"{i + 1}. {e}" for i, e in enumerate(self.evidence)) or "(none yet)"
            action = self.backend.decide(self.system_prompt, evidence_text, list(self.tools.keys()))
            tool_name = action.get("action", "stop")
            if tool_name == "stop":
                LOG.emit(self.name, f"stop ({action.get('reasoning', '')})")
                break
            if tool_name not in self.tools:
                LOG.emit(self.name, f"requested unknown tool '{tool_name}'; stopping")
                break
            if tool_name in self._failed_tools:
                # Sec. 5.2 "Agent loops without convergence": a policy that keeps asking
                # for a tool that already errored (e.g. a heuristic whose success-state
                # key never got set) would otherwise burn its whole budget retrying it.
                LOG.emit(self.name, f"'{tool_name}' already failed once this run; stopping "
                                      "instead of retrying it to convergence")
                break
            try:
                observation = self.tools[tool_name]()
            except Exception as e:  # a failed tool call becomes evidence, not a crash
                observation = f"ERROR: {type(e).__name__}: {e}"
                self._failed_tools.add(tool_name)
            LOG.emit(self.name, f"{tool_name} -> {observation}")
            self.evidence.append(f"{tool_name}: {observation}")
        return self.evidence
