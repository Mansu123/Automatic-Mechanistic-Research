"""Greater-Than task suite for circuit-discovery evaluation.

Greater-Than (Hanna et al. 2023): the model must compare two years and
predict whether a given year is greater or smaller than a reference.

Prompt template:
    "The year [X] is [greater/less] than the year"  -> predict a year token

Oracle circuit for GPT-2 (from Hanna et al. 2023, Table 1):
  - MLP layers 7-11 (numerical comparison circuitry lives in MLPs, not heads)
  - Attention heads primarily involved: L5H5 (position), L7H3, L8H11
  - Note: majority of the mechanism is MLP-based; head oracle is approximate.

The task metric is the sum of logit differences across a range of valid
"greater than" year tokens vs. "less than" year tokens.
"""
from __future__ import annotations

import random
import torch
from ..tools import adapter

# ---------------------------------------------------------------------------
# Oracle circuit (GPT-2 only, approximate from Hanna et al. 2023)
# ---------------------------------------------------------------------------

_GPT2_GT_HEADS = [(5, 5), (7, 3), (8, 11)]
_GPT2_GT_MLPS  = list(range(7, 12))  # MLP layers 7-11


def greater_than_oracle_circuit(model_id: str) -> dict | None:
    """Return the known greater-than circuit for `model_id`, or None if unknown.

    Returns a dict with 'heads' and 'mlp_layers' keys because the greater-than
    mechanism is split across attention heads and MLP layers.
    """
    if "gpt2" in model_id.lower() and "medium" not in model_id.lower() \
            and "large" not in model_id.lower() and "xl" not in model_id.lower():
        return {"heads": _GPT2_GT_HEADS, "mlp_layers": _GPT2_GT_MLPS}
    return None


# ---------------------------------------------------------------------------
# Year ranges and prompt templates
# ---------------------------------------------------------------------------

# Year pairs: (small_year, large_year) where large_year > small_year
_YEAR_PAIRS: list[tuple[int, int]] = [
    (45, 62), (32, 78), (51, 89), (23, 67), (41, 93),
    (17, 55), (38, 72), (60, 85), (29, 74), (44, 91),
]

# Tokens for "less than" years (years < small_year in the pair)
# and "greater than" years (years > large_year in the pair)
_YEAR_TOKENS_SMALL = [str(y) for y in range(10, 45)]  # clearly small years
_YEAR_TOKENS_LARGE = [str(y) for y in range(70, 99)]  # clearly large years

_GT_TEMPLATE = "The year {X} is greater than the year"
_LT_TEMPLATE = "The year {X} is less than the year"


def _make_greater_than_task(handle: adapter.ModelHandle,
                             pair: tuple[int, int],
                             seed: int = 0) -> dict:
    """One task: model predicts a year value consistent with the comparison."""
    small_year, large_year = pair
    rng = random.Random(seed)

    # clean: "Year X is GREATER than the year" -> should predict a year < X
    clean_prompt = _GT_TEMPLATE.format(X=large_year)
    # corrupted: "Year X is LESS than the year" -> same surface, opposite direction
    corrupted_prompt = _LT_TEMPLATE.format(X=large_year)

    # io_token: a year that IS less than large_year (correct completion for clean)
    io_token = str(small_year)
    # s_token: a year that is NOT less than large_year (incorrect for clean, correct for corrupted)
    s_token = str(min(99, large_year + rng.randint(5, 15)))

    def task_metric_fn() -> float:
        """Average logit diff across several small vs large year token pairs."""
        total = 0.0
        count = 0
        batch = handle.tokenizer([clean_prompt], return_tensors="pt").to(handle.device)
        with torch.no_grad():
            logits = handle.model(**batch).logits[0, -1]

        # Average over a set of "should be predicted" small years vs "should not" large years
        small_sample = rng.sample(_YEAR_TOKENS_SMALL, min(5, len(_YEAR_TOKENS_SMALL)))
        large_sample = rng.sample(_YEAR_TOKENS_LARGE, min(5, len(_YEAR_TOKENS_LARGE)))
        for sy, ly in zip(small_sample, large_sample):
            # [-1] not [0]: cross-tokenizer safety (see behaviors.py _make_task)
            s_id = handle.tokenizer.encode(" " + sy.strip())[-1]
            l_id = handle.tokenizer.encode(" " + ly.strip())[-1]
            total += (logits[s_id] - logits[l_id]).item()
            count += 1
        return total / count if count else 0.0

    n_heads = getattr(handle.model.config, "num_attention_heads", None) \
              or handle.model.config.n_head

    return {
        "behavior": f"greater_than_{small_year}_vs_{large_year}",
        "category": "Angle 13: Quantitative Comparison",
        "clean_prompt": clean_prompt,
        "corrupted_prompt": corrupted_prompt,
        "io_token": io_token,
        "s_token": s_token,
        "eval_prompts": [(clean_prompt, io_token, s_token)],
        "probe_texts": [clean_prompt, corrupted_prompt],
        "contrastive_pos": [clean_prompt],
        "contrastive_neg": [corrupted_prompt],
        "task_metric_fn": task_metric_fn,
        "n_heads": n_heads,
        "oracle_circuit": None,  # filled in by builder
        # Greater-than specific
        "small_year": small_year,
        "large_year": large_year,
    }


def build_greater_than_tasks(handle: adapter.ModelHandle,
                              seed: int = 0,
                              n_pairs: int | None = None) -> list[dict]:
    """Return one task dict per year pair, with oracle_circuit populated."""
    oracle = greater_than_oracle_circuit(handle.model_id)
    pairs = _YEAR_PAIRS[:n_pairs] if n_pairs else _YEAR_PAIRS
    tasks = []
    for i, pair in enumerate(pairs):
        t = _make_greater_than_task(handle, pair, seed=seed + i)
        t["oracle_circuit"] = oracle["heads"] if oracle else None
        tasks.append(t)
    return tasks


# Flat list of callables (handle) -> task for the comparison harness
GREATER_THAN_TASKS = [
    (lambda h, _p=pair: _make_greater_than_task(h, _p))
    for pair in _YEAR_PAIRS
]

