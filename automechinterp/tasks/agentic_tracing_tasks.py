"""Agentic tracing task suite for circuit-discovery evaluation.

Tests the pipeline's ability to trace heuristics in a model doing structured
multi-step agentic behavior (chain-of-thought style). The key contrast is:

  REAL agentic pass:
    Prompt follows a coherent 2-3 step reasoning pattern.
    The model should complete the final "therefore" conclusion correctly.

  DUMMY pass:
    Prompt has the same surface structure (step markers, formatting) but the
    logic is incoherent. The conclusion the model predicts is a "heuristic
    shortcut" driven by surface cues rather than reasoning.

Circuit = the components that distinguish real vs dummy agentic completion.
This tests whether circuit-discovery methods can find the "reasoning circuit"
rather than just latching onto surface-level pattern completion heads.

No external oracle: ground truth is determined by cross-method consensus
and ablation causal verification during the comparison sweep.
"""
from __future__ import annotations

import random
import torch
from ..tools import adapter

# ---------------------------------------------------------------------------
# Task instances: (real_prompt, dummy_prompt, real_answer, dummy_answer)
#
# real_prompt:   logically coherent agentic step sequence
# dummy_prompt:  same surface structure, broken logic (steps don't connect)
# real_answer:   the correct logical conclusion token
# dummy_answer:  the surface-heuristic completion (wrong answer, surface-right)
# ---------------------------------------------------------------------------

_AGENTIC_PAIRS: list[tuple[str, str, str, str]] = [
    # Step-by-step deduction
    (
        "Step 1: All birds can fly. Step 2: A penguin is a bird. Step 3: Therefore a penguin can",
        "Step 1: All cars have wheels. Step 2: A penguin is a bird. Step 3: Therefore a penguin can",
        "fly", "drive",
    ),
    (
        "Step 1: If it rains, the ground gets wet. Step 2: It is raining. Step 3: Therefore the ground is",
        "Step 1: If it snows, the sky turns orange. Step 2: It is raining. Step 3: Therefore the ground is",
        "wet", "orange",
    ),
    (
        "Step 1: All metals conduct electricity. Step 2: Copper is a metal. Step 3: Therefore copper",
        "Step 1: All fruits are sweet. Step 2: Copper is a metal. Step 3: Therefore copper",
        "conducts", "tastes",
    ),
    (
        "Step 1: If X > Y and Y > Z then X > Z. Step 2: A > B and B > C. Step 3: Therefore A is greater than",
        "Step 1: If X > Y and Y > Z then X > Z. Step 2: A > B and C > B. Step 3: Therefore A is greater than",
        "C", "B",
    ),
    # Chain-of-thought arithmetic
    (
        "Observation: There are 3 apples. Action: Add 2 more. Result: Now there are",
        "Observation: There are 3 apples. Action: Add 2 more. Result: Now there are",
        "5", "3",
    ),
    (
        "Goal: reach the store. Step 1: Go north 2 blocks. Step 2: Go east 1 block. Step 3: You have arrived",
        "Goal: reach the store. Step 1: Go north 2 blocks. Step 2: Go west 1 block. Step 3: You have arrived",
        "north", "south",
    ),
    # Heuristic-trap: surface completion vs logical answer
    (
        "Premise: Every doctor is a person. Premise: Alice is a doctor. Conclusion: Alice is a",
        "Premise: Every dog is an animal. Premise: Alice is a doctor. Conclusion: Alice is a",
        "person", "dog",
    ),
    (
        "Rule: Winners get gold. Rule: Bob won the race. Conclusion: Bob receives",
        "Rule: Winners get silver. Rule: Bob won the race. Conclusion: Bob receives",
        "gold", "silver",
    ),
]

# Dummy-pass variants where the model relies on surface heuristics
_DUMMY_ONLY_PAIRS: list[tuple[str, str, str, str]] = [
    (
        "Step 1: A. Step 2: B. Step 3: Therefore",
        "Step 1: P. Step 2: Q. Step 3: Therefore",
        "C", "R",  # neither makes sense -- tests if heads fire on "Therefore X" pattern
    ),
]


def _make_agentic_task(handle: adapter.ModelHandle,
                        name: str,
                        real_prompt: str,
                        dummy_prompt: str,
                        real_answer: str,
                        dummy_answer: str,
                        seed: int = 0) -> dict:
    """Build an agentic tracing task dict (same schema as behaviors.py tasks)."""
    rng = random.Random(seed)

    def task_metric_fn() -> float:
        """Logit diff: real completion token vs surface-heuristic dummy token."""
        batch = handle.tokenizer([real_prompt], return_tensors="pt").to(handle.device)
        # [-1] not [0]: cross-tokenizer safety (see behaviors.py _make_task)
        real_id  = handle.tokenizer.encode(" " + real_answer.strip())[-1]
        dummy_id = handle.tokenizer.encode(" " + dummy_answer.strip())[-1]
        with torch.no_grad():
            logits = handle.model(**batch).logits[0, -1]
        return (logits[real_id] - logits[dummy_id]).item()

    n_heads = getattr(handle.model.config, "num_attention_heads", None) \
              or handle.model.config.n_head

    return {
        "behavior": name,
        "category": "Agentic: Heuristic Tracing",
        "clean_prompt": real_prompt,
        "corrupted_prompt": dummy_prompt,
        "io_token": real_answer,
        "s_token": dummy_answer,
        "eval_prompts": [(real_prompt, real_answer, dummy_answer)],
        "probe_texts": [real_prompt, dummy_prompt],
        "contrastive_pos": [real_prompt],
        "contrastive_neg": [dummy_prompt],
        "task_metric_fn": task_metric_fn,
        "n_heads": n_heads,
        "oracle_circuit": None,   # no external oracle; cross-method consensus used
        # Agentic-specific
        "real_prompt": real_prompt,
        "dummy_prompt": dummy_prompt,
        "real_answer": real_answer,
        "dummy_answer": dummy_answer,
    }


def build_agentic_tracing_tasks(handle: adapter.ModelHandle, seed: int = 0) -> list[dict]:
    """Return one task dict per agentic tracing scenario."""
    tasks = []
    for i, (rp, dp, ra, da) in enumerate(_AGENTIC_PAIRS):
        name = f"agentic_trace_{i}"
        tasks.append(_make_agentic_task(handle, name, rp, dp, ra, da, seed=seed + i))
    return tasks


# Flat list of callables (handle) -> task for the comparison harness
AGENTIC_TRACING_TASKS = [
    (lambda h, _i=i, _rp=rp, _dp=dp, _ra=ra, _da=da:
        _make_agentic_task(h, f"agentic_trace_{_i}", _rp, _dp, _ra, _da))
    for i, (rp, dp, ra, da) in enumerate(_AGENTIC_PAIRS)
]

