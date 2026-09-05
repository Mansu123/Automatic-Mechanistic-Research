"""Induction task suite for circuit-discovery evaluation.

Induction (Olsson et al. 2022): the model must complete a repeated-token
sequence, e.g. "A B C D ... A B C D ... A" -> predict "B".

Three difficulty tiers:
  - token:   single token repetitions (base case)
  - bigram:  two-token patterns
  - trigram: three-token patterns

Oracle circuit for GPT-2 (from Olsson et al. 2022):
  - Previous-token heads:  L2H2, L2H11, L3H0 (create "shifted" key signal)
  - Induction heads:       L5H1, L5H5, L6H9, L7H10  (attend to prev occurrence + attend next)

The oracle is used as a reference for precision/recall scoring in the
comparison harness; for models other than GPT-2 no oracle exists and
metrics fall back to cross-method consensus.
"""
from __future__ import annotations

import random
import torch
from ..tools import adapter

# ---------------------------------------------------------------------------
# Oracle circuit (GPT-2 only)
# ---------------------------------------------------------------------------

_GPT2_PREV_TOKEN_HEADS = [(2, 2), (2, 11), (3, 0)]
_GPT2_INDUCTION_HEADS  = [(5, 1), (5, 5), (6, 9), (7, 10)]
_GPT2_ORACLE            = _GPT2_PREV_TOKEN_HEADS + _GPT2_INDUCTION_HEADS


def induction_oracle_circuit(model_id: str) -> list[tuple[int, int]] | None:
    """Return the known induction circuit for `model_id`, or None if unknown."""
    if "gpt2" in model_id.lower() and "medium" not in model_id.lower() \
            and "large" not in model_id.lower() and "xl" not in model_id.lower():
        return _GPT2_ORACLE
    return None


# ---------------------------------------------------------------------------
# Task variants
# ---------------------------------------------------------------------------

# (clean_prompt, corrupted_prompt, io_token, s_token)
# All prompts end with the first element of the repeated pattern so the
# model must predict the next element.  corrupted_prompt either breaks the
# repetition or replaces the query element with a different pattern.

_TOKEN_PAIRS: list[tuple[str, str, str, str]] = [
    # simple single-token repetition
    ("red blue green red blue green red",
     "red blue green orange purple yellow red",
     "blue", "orange"),
    ("alpha beta gamma alpha beta gamma alpha",
     "alpha beta gamma delta epsilon zeta alpha",
     "beta", "delta"),
    ("cat dog fox cat dog fox cat",
     "cat dog fox bear wolf elk cat",
     "dog", "bear"),
    ("one two three one two three one",
     "one two three four five six one",
     "two", "four"),
]

_BIGRAM_PAIRS: list[tuple[str, str, str, str]] = [
    ("AB CD EF AB CD EF AB CD",
     "AB CD EF GH IJ KL AB CD",
     "EF", "GH"),
    ("XY ZW UV XY ZW UV XY ZW",
     "XY ZW UV RS TQ NM XY ZW",
     "UV", "RS"),
]

_TRIGRAM_PAIRS: list[tuple[str, str, str, str]] = [
    ("A1 B2 C3 D4 A1 B2 C3 D4 A1 B2 C3",
     "A1 B2 C3 D4 E5 F6 G7 H8 A1 B2 C3",
     "D4", "E5"),
    ("X1 X2 X3 X4 X1 X2 X3 X4 X1 X2 X3",
     "X1 X2 X3 X4 Y1 Y2 Y3 Y4 X1 X2 X3",
     "X4", "Y4"),
]

# Each entry: (name, pairs_list)
INDUCTION_TASK_VARIANTS: list[tuple[str, list]] = [
    ("induction_token",   _TOKEN_PAIRS),
    ("induction_bigram",  _BIGRAM_PAIRS),
    ("induction_trigram", _TRIGRAM_PAIRS),
]


def _make_induction_task(handle: adapter.ModelHandle,
                          name: str,
                          pairs: list[tuple[str, str, str, str]],
                          seed: int = 0) -> dict:
    """Build a task dict (same schema as behaviors.py tasks) from a pair list."""
    rng = random.Random(seed)
    clean_prompt, corrupted_prompt, io_token, s_token = rng.choice(pairs)

    # [-1] not [0]: see behaviors.py _make_task for cross-tokenizer safety note.
    def task_metric_fn() -> float:
        total = 0.0
        for cp, _, io_t, s_t in [(clean_prompt, corrupted_prompt, io_token, s_token)] \
                + [rng.choice(pairs) for _ in range(3)]:
            batch = handle.tokenizer([cp], return_tensors="pt").to(handle.device)
            io_id = handle.tokenizer.encode(" " + io_t.strip())[-1]
            s_id  = handle.tokenizer.encode(" " + s_t.strip())[-1]
            with torch.no_grad():
                logits = handle.model(**batch).logits[0, -1]
            total += (logits[io_id] - logits[s_id]).item()
        return total / 4

    n_heads = getattr(handle.model.config, "num_attention_heads", None) \
              or handle.model.config.n_head

    return {
        "behavior": name,
        "category": "Angle 4: Induction",
        "clean_prompt": clean_prompt,
        "corrupted_prompt": corrupted_prompt,
        "io_token": io_token,
        "s_token": s_token,
        "eval_prompts": [(clean_prompt, io_token, s_token)],
        "probe_texts": [clean_prompt],
        "contrastive_pos": [clean_prompt],
        "contrastive_neg": [corrupted_prompt],
        "task_metric_fn": task_metric_fn,
        "n_heads": n_heads,
        "oracle_circuit": None,  # filled in by the suite builder below
    }


def build_induction_tasks(handle: adapter.ModelHandle, seed: int = 0) -> list[dict]:
    """Return one task dict per variant, with oracle_circuit populated."""
    oracle = induction_oracle_circuit(handle.model_id)
    tasks = []
    for name, pairs in INDUCTION_TASK_VARIANTS:
        t = _make_induction_task(handle, name, pairs, seed)
        t["oracle_circuit"] = oracle
        tasks.append(t)
    return tasks


# Flat list for use by the comparison harness: callables (handle) -> task
INDUCTION_TASKS = [
    (lambda h, _name=name, _pairs=pairs: _make_induction_task(h, _name, _pairs))
    for name, pairs in INDUCTION_TASK_VARIANTS
]

