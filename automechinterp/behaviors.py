"""Behavior definitions for Stage B (Sec. 4.4: "Behavior suite... across
factual recall, arithmetic, syntactic agreement, and hallucination triggers").

Every behavior returns a task dict shaped exactly like stage_a.build_ioi_task's
output -- {clean_prompt, corrupted_prompt, io_token, s_token, eval_prompts,
probe_texts, task_metric_fn, n_heads, behavior} -- because the whole hierarchy
(hierarchy.py, every Tier N/L/C/V tool) is already generic to "predict the
correct token over a specific wrong token given a clean/corrupted prompt
pair"; IOI was never a special case, just the first instance. That genericity
is what makes a multi-behavior Layer Atlas possible without new plumbing.

Scaled down from the proposal's 24 tasks to 4 -- one per category named in
Sec. 4.4 -- to keep a full Stage B run fast enough to actually execute here;
the categories are real and distinct, not padding.
"""
from __future__ import annotations

import random

import torch

from .tools import adapter


def _make_task(handle: adapter.ModelHandle, behavior: str, category: str,
                pair_generator, n_eval: int = 4, seed: int = 0) -> dict:
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
            io_id = handle.tokenizer.encode(" " + io_t)[0]
            s_id = handle.tokenizer.encode(" " + s_t)[0]
            with torch.no_grad():
                logits = handle.model(**batch).logits[0, -1]
            total += (logits[io_id] - logits[s_id]).item()
        return total / len(eval_prompts)

    probe_texts = [p for p, _, _ in eval_prompts] + [clean_prompt]
    n_heads = getattr(handle.model.config, "num_attention_heads", None) or handle.model.config.n_head

    return {
        "behavior": behavior, "category": category,
        "clean_prompt": clean_prompt, "corrupted_prompt": corrupted_prompt,
        "io_token": io_token, "s_token": s_token,
        "eval_prompts": eval_prompts, "probe_texts": probe_texts,
        "task_metric_fn": task_metric_fn, "n_heads": n_heads,
    }


# -- IOI (coreference / entity tracking) -------------------------------------
_IOI_TEMPLATE = "When {A} and {B} went to the store, {S} gave a drink to"
_IOI_NAMES = ["John", "Mary", "Alice", "Bob", "Sarah", "Tom"]


def ioi_behavior(handle: adapter.ModelHandle, seed: int = 0) -> dict:
    def gen(rng):
        a, b = rng.sample(_IOI_NAMES, 2)
        clean = _IOI_TEMPLATE.format(A=a, B=b, S=a)
        corrupted = _IOI_TEMPLATE.format(A=b, B=a, S=b)
        return clean, corrupted, b, a
    return _make_task(handle, "Indirect Object Identification (coreference)", "coreference", gen, seed=seed)


# -- Subject-verb number agreement (syntactic) -------------------------------
_AGREEMENT_PAIRS = [
    ("key", "keys", "cabinet", "cabinets"), ("dog", "dogs", "yard", "yards"),
    ("student", "students", "classroom", "classrooms"), ("book", "books", "shelf", "shelves"),
]


def agreement_behavior(handle: adapter.ModelHandle, seed: int = 0) -> dict:
    def gen(rng):
        sg, pl, dist_sg, dist_pl = rng.choice(_AGREEMENT_PAIRS)
        # clean: plural subject, singular distractor noun in the relative clause -> "are" is correct
        clean = f"The {pl} to the {dist_sg}"
        # corrupted: singular subject, plural distractor -> "is" is correct instead
        corrupted = f"The {sg} to the {dist_pl}"
        return clean, corrupted, "are", "is"
    return _make_task(handle, "Subject-verb number agreement (syntax)", "syntactic_agreement", gen, seed=seed)


# -- Factual recall (world knowledge) ----------------------------------------
_CAPITALS = [("France", "Paris", "London"), ("England", "London", "Paris"),
             ("Italy", "Rome", "Madrid"), ("Spain", "Madrid", "Rome")]


def factual_recall_behavior(handle: adapter.ModelHandle, seed: int = 0) -> dict:
    def gen(rng):
        country, capital, wrong = rng.choice(_CAPITALS)
        clean = f"The capital of {country} is"
        other_country, other_capital, _ = rng.choice([c for c in _CAPITALS if c[0] != country])
        corrupted = f"The capital of {other_country} is"
        return clean, corrupted, capital, wrong
    return _make_task(handle, "Factual recall: country capitals (world knowledge)", "factual_recall", gen, seed=seed)


# -- Antonym prediction (lexical semantics) ----------------------------------
_ANTONYMS = [("hot", "cold", "warm"), ("big", "small", "large"),
             ("fast", "slow", "quick"), ("light", "dark", "bright")]


def antonym_behavior(handle: adapter.ModelHandle, seed: int = 0) -> dict:
    def gen(rng):
        word, antonym, near_synonym = rng.choice(_ANTONYMS)
        clean = f"The opposite of {word} is"
        other_word, other_antonym, _ = rng.choice([a for a in _ANTONYMS if a[0] != word])
        corrupted = f"The opposite of {other_word} is"
        return clean, corrupted, antonym, near_synonym
    return _make_task(handle, "Antonym prediction (lexical semantics)", "lexical_semantics", gen, seed=seed)


ALL_BEHAVIORS = [ioi_behavior, agreement_behavior, factual_recall_behavior, antonym_behavior]
