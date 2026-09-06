"""Tasks package for canonical circuit-discovery evaluation task suites."""
from .induction_tasks import INDUCTION_TASKS, induction_oracle_circuit
from .greater_than_tasks import GREATER_THAN_TASKS, greater_than_oracle_circuit
from .agentic_tracing_tasks import AGENTIC_TRACING_TASKS

__all__ = [
    "INDUCTION_TASKS",
    "induction_oracle_circuit",
    "GREATER_THAN_TASKS",
    "greater_than_oracle_circuit",
    "AGENTIC_TRACING_TASKS",
]

