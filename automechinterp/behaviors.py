"""Behavior definitions for Stage B across all 6 cognitive & mechanistic angles.

Angles:
  1. Linguistic Structure (Syntax & Coreference)
  2. Factual & World Knowledge
  3. Reasoning & Arithmetic
  4. In-Context Learning / Induction
  5. Social Bias & Fairness
  6. Lexical Semantics & Word Sense

Every behavior returns a standard task dictionary:
  {
    "behavior": str,
    "category": str,
    "clean_prompt": str,
    "corrupted_prompt": str,
    "io_token": str,
    "s_token": str,
    "eval_prompts": list,
    "probe_texts": list,
    "task_metric_fn": Callable,
    "n_heads": int
  }
"""
from __future__ import annotations

import random
from typing import Callable

import torch
from .tools import adapter


def _make_task(handle: adapter.ModelHandle, behavior: str, category: str,
                pair_generator: Callable, n_eval: int = 4, seed: int = 0) -> dict:
    rng = random.Random(seed)
    clean_prompt, corrupted_prompt, io_token, s_token = pair_generator(rng)
    eval_prompts = [(clean_prompt, io_token, s_token)]
    for _ in range(n_eval - 1):
        cp, _, io_t, s_t = pair_generator(rng)
        eval_prompts.append((cp, io_t, s_t))

    def task_metric_fn() -> float:
        total = 0.0
        for prompt, io_t, s_t in eval_prompts:
            batch = handle.tokenizer([prompt], return_tensors="pt").to(handle.device)
            io_id = handle.tokenizer.encode(" " + io_t.strip())[0]
            s_id = handle.tokenizer.encode(" " + s_t.strip())[0]
            with torch.no_grad():
                logits = handle.model(**batch).logits[0, -1]
            total += (logits[io_id] - logits[s_id]).item()
        return total / len(eval_prompts)

    probe_texts = [p for p, _, _ in eval_prompts] + [clean_prompt]
    n_heads = getattr(handle.model.config, "num_attention_heads", None) or handle.model.config.n_head

    return {
        "behavior": behavior,
        "category": category,
        "clean_prompt": clean_prompt,
        "corrupted_prompt": corrupted_prompt,
        "io_token": io_token,
        "s_token": s_token,
        "eval_prompts": eval_prompts,
        "probe_texts": probe_texts,
        "task_metric_fn": task_metric_fn,
        "n_heads": n_heads,
    }


# ============================================================================
# ANGLE 1: Linguistic Structure & Syntax
# ============================================================================

_IOI_TEMPLATE = "When {A} and {B} went to the store, {S} gave a drink to"
_IOI_NAMES = ["John", "Mary", "Alice", "Bob", "Sarah", "Tom"]


def ioi_behavior(handle: adapter.ModelHandle, seed: int = 0) -> dict:
    def gen(rng):
        a, b = rng.sample(_IOI_NAMES, 2)
        clean = _IOI_TEMPLATE.format(A=a, B=b, S=a)
        corrupted = _IOI_TEMPLATE.format(A=b, B=a, S=b)
        return clean, corrupted, b, a
    return _make_task(handle, "Indirect Object Identification (coreference)", "Angle 1: Linguistic", gen, seed=seed)


_AGREEMENT_PAIRS = [
    ("key", "keys", "cabinet", "cabinets"), ("dog", "dogs", "yard", "yards"),
    ("student", "students", "classroom", "classrooms"), ("book", "books", "shelf", "shelves"),
]


def agreement_behavior(handle: adapter.ModelHandle, seed: int = 0) -> dict:
    def gen(rng):
        sg, pl, dist_sg, dist_pl = rng.choice(_AGREEMENT_PAIRS)
        clean = f"The {pl} to the {dist_sg}"
        corrupted = f"The {sg} to the {dist_pl}"
        return clean, corrupted, "are", "is"
    return _make_task(handle, "Subject-verb number agreement (syntax)", "Angle 1: Linguistic", gen, seed=seed)


# ============================================================================
# ANGLE 2: Factual & World Knowledge
# ============================================================================

_CAPITALS = [
    ("France", "Paris", "London"), ("England", "London", "Paris"),
    ("Italy", "Rome", "Madrid"), ("Spain", "Madrid", "Rome"),
    ("Germany", "Berlin", "Paris"), ("Japan", "Tokyo", "Beijing")
]


def factual_recall_behavior(handle: adapter.ModelHandle, seed: int = 0) -> dict:
    def gen(rng):
        country, capital, wrong = rng.choice(_CAPITALS)
        clean = f"The capital of {country} is"
        other_country, other_capital, _ = rng.choice([c for c in _CAPITALS if c[0] != country])
        corrupted = f"The capital of {other_country} is"
        return clean, corrupted, capital, wrong
    return _make_task(handle, "Factual recall: country capitals", "Angle 2: Factual", gen, seed=seed)


# ============================================================================
# ANGLE 3: Reasoning & Arithmetic
# ============================================================================

_ADDITION_CASES = [
    (3, 4, 7, 8, 2, 5),
    (2, 5, 7, 6, 3, 4),
    (1, 8, 9, 8, 4, 5),
    (4, 5, 9, 7, 1, 8),
    (2, 3, 5, 6, 6, 2),
    (6, 2, 8, 7, 2, 3),
]


def arithmetic_addition_behavior(handle: adapter.ModelHandle, seed: int = 0) -> dict:
    def gen(rng):
        a, b, ans, wrong, c, d = rng.choice(_ADDITION_CASES)
        clean = f"The sum of {a} and {b} is"
        corrupted = f"The sum of {c} and {d} is"
        return clean, corrupted, str(ans), str(wrong)
    return _make_task(handle, "Arithmetic: single-digit addition", "Angle 3: Arithmetic", gen, seed=seed)


_MAGNITUDE_CASES = [
    (8, 3, "8", "3"),
    (9, 2, "9", "2"),
    (7, 4, "7", "4"),
    (6, 1, "6", "1"),
]


def magnitude_comparison_behavior(handle: adapter.ModelHandle, seed: int = 0) -> dict:
    def gen(rng):
        big, small, ans, wrong = rng.choice(_MAGNITUDE_CASES)
        clean = f"Between {big} and {small}, the larger number is"
        corrupted = f"Between {big} and {small}, the smaller number is"
        return clean, corrupted, ans, wrong
    return _make_task(handle, "Reasoning: magnitude comparison", "Angle 3: Arithmetic", gen, seed=seed)


# ============================================================================
# ANGLE 4: In-Context Learning / Induction
# ============================================================================

_REPEATED_SEQUENCES = [
    (["red", "blue", "green"], "red", "blue"),
    (["apple", "banana", "cherry"], "apple", "banana"),
    (["cat", "dog", "fox"], "cat", "dog"),
    (["alpha", "beta", "gamma"], "alpha", "beta"),
]


def induction_copying_behavior(handle: adapter.ModelHandle, seed: int = 0) -> dict:
    def gen(rng):
        seq, first, second = rng.choice(_REPEATED_SEQUENCES)
        clean = f"{seq[0]} {seq[1]} {seq[2]} {seq[0]}"
        corrupted = f"{seq[0]} {seq[1]} {seq[2]} {seq[1]}"
        return clean, corrupted, second, first
    return _make_task(handle, "Induction: repeated token copying", "Angle 4: Induction", gen, seed=seed)


# ============================================================================
# ANGLE 5: Social Bias & Fairness
# ============================================================================

_GENDER_OCCUPATIONS = [
    ("nurse", "she", "he", "doctor"),
    ("secretary", "she", "he", "manager"),
    ("engineer", "he", "she", "designer"),
    ("surgeon", "he", "she", "patient"),
]


def gender_bias_behavior(handle: adapter.ModelHandle, seed: int = 0) -> dict:
    def gen(rng):
        occ, stereo, anti, other_occ = rng.choice(_GENDER_OCCUPATIONS)
        clean = f"The {occ} went to work because"
        corrupted = f"The {other_occ} went to work because"
        return clean, corrupted, stereo, anti
    return _make_task(handle, "Social Bias: gender-occupation stereotype", "Angle 5: Social Bias", gen, seed=seed)


# ============================================================================
# ANGLE 6: Lexical Semantics & Word Sense
# ============================================================================

_ANTONYMS = [
    ("hot", "cold", "warm"), ("big", "small", "large"),
    ("fast", "slow", "quick"), ("light", "dark", "bright"),
    ("rich", "poor", "wealthy"), ("early", "late", "soon")
]


def antonym_behavior(handle: adapter.ModelHandle, seed: int = 0) -> dict:
    def gen(rng):
        word, antonym, near_synonym = rng.choice(_ANTONYMS)
        clean = f"The opposite of {word} is"
        other_word, other_antonym, _ = rng.choice([a for a in _ANTONYMS if a[0] != word])
        corrupted = f"The opposite of {other_word} is"
        return clean, corrupted, antonym, near_synonym
    return _make_task(handle, "Lexical Semantics: antonym prediction", "Angle 6: Lexical Semantics", gen, seed=seed)


_CATEGORIES = [
    ("robin", "bird", "fish"),
    ("salmon", "fish", "bird"),
    ("oak", "tree", "animal"),
    ("rose", "flower", "tree"),
]


def category_membership_behavior(handle: adapter.ModelHandle, seed: int = 0) -> dict:
    def gen(rng):
        item, cat, wrong = rng.choice(_CATEGORIES)
        clean = f"A {item} is a type of"
        other_item, other_cat, _ = rng.choice([c for c in _CATEGORIES if c[0] != item])
        corrupted = f"A {other_item} is a type of"
        return clean, corrupted, cat, wrong
    return _make_task(handle, "Lexical Semantics: category membership", "Angle 6: Lexical Semantics", gen, seed=seed)


ANGLE_3_BEHAVIORS = [arithmetic_addition_behavior, magnitude_comparison_behavior]
ANGLE_4_BEHAVIORS = [induction_copying_behavior]
ANGLE_5_BEHAVIORS = [gender_bias_behavior]
ANGLE_6_BEHAVIORS = [antonym_behavior, category_membership_behavior]

ALL_BEHAVIORS = [
    ioi_behavior,
    agreement_behavior,
    factual_recall_behavior,
    arithmetic_addition_behavior,
    magnitude_comparison_behavior,
    induction_copying_behavior,
    gender_bias_behavior,
    antonym_behavior,
    category_membership_behavior,
]
