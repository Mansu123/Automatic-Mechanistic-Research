"""Orchestrator (Sec. 3.2): owns the task, the global hypothesis ledger, and
the compute/token budget. No tools of its own -- it decides which layers get
agents, deduplicates/arbitrates claims that different Layer/Component Agents
report for the same component, and enforces budget-aware termination
(Sec. 3.6). The actual wiring of Network Analyst -> Layer Agents ->
Component Agents -> Skeptic -> Judge lives in hierarchy.py; this module is
just the ledger + policy the Orchestrator's log lines in that trace refer to.
"""
from __future__ import annotations

from .base import LOG


class HypothesisLedger:
    """Deduplicates claims across parallel Layer/Component Agents (Sec. 5.2:
    'Parallel Layer Agents duplicate work or conflict' -> mitigation)."""

    def __init__(self):
        self.claims: dict[tuple[int, int], list[str]] = {}

    def add(self, layer_idx: int, head_idx: int, source_agent: str) -> bool:
        """Returns True if this is a new claim, False if it's a duplicate
        already reported by another agent (arbitration = keep first, log rest)."""
        key = (layer_idx, head_idx)
        is_new = key not in self.claims
        self.claims.setdefault(key, []).append(source_agent)
        if not is_new:
            LOG.emit("Orchestrator", f"L{layer_idx}H{head_idx} independently reported by "
                                       f"{self.claims[key]} -- deduplicating in the ledger")
        return is_new

    def merged_circuit(self) -> list[tuple[int, int]]:
        return sorted(self.claims.keys())


def decide_layers_to_spawn(flagged_layers: list[int], global_remaining: int,
                            per_layer_cost_estimate: int = 4) -> list[int]:
    """Budget-aware termination (Sec. 3.6): only spawn as many parallel Layer
    Agents as the remaining budget can afford; log a best-effort note if some
    flagged layers have to be dropped."""
    affordable = max(0, global_remaining // per_layer_cost_estimate)
    chosen = flagged_layers[:affordable]
    if len(chosen) < len(flagged_layers):
        LOG.emit("Orchestrator", f"budget allows {affordable} Layer Agents; spawning {chosen}, "
                                   f"deferring {flagged_layers[len(chosen):]} (best-effort atlas)")
    else:
        LOG.emit("Orchestrator", f"spawning Layer Agents in parallel for flagged layers {chosen}")
    return chosen
