"""Baselines package: four external circuit-discovery methodologies.

Each method exposes:
    run_<method>(handle, task, layer_range, n_heads, **kwargs) -> MethodResult

MethodResult is a dataclass with a common schema so the comparison harness
can swap methods interchangeably.
"""
from dataclasses import dataclass, field

@dataclass
class MethodResult:
    """Unified output schema for all circuit-discovery baseline methods."""
    method: str                          # method name, e.g. "subnetwork_probing"
    circuit: list                        # discovered (layer, head) pairs
    circuit_score: float                 # method-specific confidence scalar
    metric_recovery: float               # fraction of clean–corrupted gap recovered
    tool_calls: int                      # budget units consumed (steps / fwd passes)
    runtime_s: float                     # wall-clock seconds
    metadata: dict = field(default_factory=dict)  # method-specific diagnostics

from .subnetwork_probing import run_subnetwork_probing
from .acd import run_acd
from .mechrl import run_mechrl
from .circuit_tracing import run_circuit_tracing
from .transformerlens_acdc import run_transformerlens_acdc

ALL_METHODS: dict = {
    "subnetwork_probing":   run_subnetwork_probing,
    "acd":                  run_acd,
    "mechrl":               run_mechrl,
    "circuit_tracing":      run_circuit_tracing,
    "transformerlens_acdc": run_transformerlens_acdc,
}

__all__ = [
    "MethodResult",
    "ALL_METHODS",
    "run_subnetwork_probing",
    "run_acd",
    "run_mechrl",
    "run_circuit_tracing",
    "run_transformerlens_acdc",
]


