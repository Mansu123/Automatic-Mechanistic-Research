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


_SUBJECT_EXTRACTION_PAIRS = [
    ("man", "woman", "blue hat"), ("boy", "girl", "red shirt"),
    ("dog", "cat", "long tail"), ("teacher", "student", "glasses")
]

def subject_extraction_behavior(handle: adapter.ModelHandle, seed: int = 0) -> dict:
    def gen(rng):
        s1, s2, desc = rng.choice(_SUBJECT_EXTRACTION_PAIRS)
        clean = f"The {s1} with the {desc} walked to the park. The person who walked is the"
        corrupted = f"The {s2} with the {desc} walked to the park. The person who walked is the"
        return clean, corrupted, s1, s2
    return _make_task(handle, "Subject extraction", "Angle 1: Linguistic", gen, seed=seed)


_PASSIVE_VOICE_PAIRS = [
    ("boy", "girl", "window"), ("dog", "cat", "toy"),
    ("man", "woman", "car"), ("chef", "waiter", "meal")
]

def passive_voice_behavior(handle: adapter.ModelHandle, seed: int = 0) -> dict:
    def gen(rng):
        a1, a2, obj = rng.choice(_PASSIVE_VOICE_PAIRS)
        clean = f"The {obj} was broken by the {a1}. The person who broke it was the"
        corrupted = f"The {obj} was broken by the {a2}. The person who broke it was the"
        return clean, corrupted, a1, a2
    return _make_task(handle, "Passive voice inference", "Angle 1: Linguistic", gen, seed=seed)


_REFLEXIVE_PRONOUN_PAIRS = [
    ("girl", "boy", "herself", "himself"), ("woman", "man", "herself", "himself"),
    ("queen", "king", "herself", "himself"), ("actress", "actor", "herself", "himself")
]

def reflexive_pronoun_behavior(handle: adapter.ModelHandle, seed: int = 0) -> dict:
    def gen(rng):
        s1, s2, r1, r2 = rng.choice(_REFLEXIVE_PRONOUN_PAIRS)
        clean = f"The {s1} looked at {r1} in the mirror. The person in the mirror was the"
        corrupted = f"The {s2} looked at {r2} in the mirror. The person in the mirror was the"
        return clean, corrupted, s1, s2
    return _make_task(handle, "Reflexive pronoun resolution", "Angle 1: Linguistic", gen, seed=seed)


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


_CURRENCIES = [
    ("Japan", "Yen", "Dollar"), ("USA", "Dollar", "Yen"),
    ("UK", "Pound", "Euro"), ("France", "Euro", "Pound"),
    ("India", "Rupee", "Yen"), ("China", "Yuan", "Rupee")
]

def currency_knowledge_behavior(handle: adapter.ModelHandle, seed: int = 0) -> dict:
    def gen(rng):
        country, curr, wrong = rng.choice(_CURRENCIES)
        clean = f"The official currency of {country} is the"
        other_country, _, _ = rng.choice([c for c in _CURRENCIES if c[0] != country])
        corrupted = f"The official currency of {other_country} is the"
        return clean, corrupted, curr, wrong
    return _make_task(handle, "Factual recall: currency", "Angle 2: Factual", gen, seed=seed)


_ELEMENTS = [
    ("Oxygen", "O", "H"), ("Hydrogen", "H", "O"),
    ("Carbon", "C", "N"), ("Nitrogen", "N", "C"),
    ("Gold", "Au", "Ag"), ("Silver", "Ag", "Au")
]

def element_symbol_behavior(handle: adapter.ModelHandle, seed: int = 0) -> dict:
    def gen(rng):
        elem, sym, wrong = rng.choice(_ELEMENTS)
        clean = f"The chemical symbol for {elem} is"
        other_elem, _, _ = rng.choice([e for e in _ELEMENTS if e[0] != elem])
        corrupted = f"The chemical symbol for {other_elem} is"
        return clean, corrupted, sym, wrong
    return _make_task(handle, "Factual recall: element symbols", "Angle 2: Factual", gen, seed=seed)


_LANGUAGES = [
    ("Brazil", "Portuguese", "Spanish"), ("Mexico", "Spanish", "Portuguese"),
    ("USA", "English", "French"), ("France", "French", "English"),
    ("Egypt", "Arabic", "English"), ("China", "Chinese", "Japanese")
]

def language_knowledge_behavior(handle: adapter.ModelHandle, seed: int = 0) -> dict:
    def gen(rng):
        country, lang, wrong = rng.choice(_LANGUAGES)
        clean = f"The primary language spoken in {country} is"
        other_country, _, _ = rng.choice([l for l in _LANGUAGES if l[0] != country])
        corrupted = f"The primary language spoken in {other_country} is"
        return clean, corrupted, lang, wrong
    return _make_task(handle, "Factual recall: languages", "Angle 2: Factual", gen, seed=seed)


_CONTINENTS = [
    ("Egypt", "Africa", "Asia"), ("China", "Asia", "Africa"),
    ("Brazil", "America", "Europe"), ("France", "Europe", "America"),
    ("Canada", "America", "Asia"), ("Japan", "Asia", "Europe")
]

def continent_knowledge_behavior(handle: adapter.ModelHandle, seed: int = 0) -> dict:
    def gen(rng):
        country, cont, wrong = rng.choice(_CONTINENTS)
        clean = f"The country of {country} is located in"
        other_country, _, _ = rng.choice([c for c in _CONTINENTS if c[0] != country])
        corrupted = f"The country of {other_country} is located in"
        return clean, corrupted, cont, wrong
    return _make_task(handle, "Factual recall: continents", "Angle 2: Factual", gen, seed=seed)


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


_SUBTRACTION_CASES = [
    (9, 4, 5, 4, 8, 4), (8, 3, 5, 6, 9, 3),
    (7, 2, 5, 4, 6, 2), (6, 3, 3, 2, 5, 3),
    (9, 2, 7, 8, 9, 1), (8, 5, 3, 4, 9, 5)
]

def arithmetic_subtraction_behavior(handle: adapter.ModelHandle, seed: int = 0) -> dict:
    def gen(rng):
        a, b, ans, wrong, c, d = rng.choice(_SUBTRACTION_CASES)
        clean = f"The difference between {a} and {b} is"
        corrupted = f"The difference between {c} and {d} is"
        return clean, corrupted, str(ans), str(wrong)
    return _make_task(handle, "Arithmetic: single-digit subtraction", "Angle 3: Arithmetic", gen, seed=seed)


_MULTIPLICATION_CASES = [
    (3, 3, 9, 8, 2, 4), (2, 4, 8, 6, 2, 3),
    (4, 2, 8, 9, 3, 3), (2, 3, 6, 8, 2, 4),
    (2, 2, 4, 6, 2, 3), (3, 2, 6, 4, 2, 2)
]

def arithmetic_multiplication_behavior(handle: adapter.ModelHandle, seed: int = 0) -> dict:
    def gen(rng):
        a, b, ans, wrong, c, d = rng.choice(_MULTIPLICATION_CASES)
        clean = f"The product of {a} and {b} is"
        corrupted = f"The product of {c} and {d} is"
        return clean, corrupted, str(ans), str(wrong)
    return _make_task(handle, "Arithmetic: single-digit multiplication", "Angle 3: Arithmetic", gen, seed=seed)


_TEMPORAL_CASES = [
    ("Monday", "Tuesday", "Wednesday"), ("Tuesday", "Wednesday", "Thursday"),
    ("Wednesday", "Thursday", "Friday"), ("Thursday", "Friday", "Saturday")
]

def temporal_reasoning_behavior(handle: adapter.ModelHandle, seed: int = 0) -> dict:
    def gen(rng):
        today, tmrw, wrong = rng.choice(_TEMPORAL_CASES)
        clean = f"If today is {today}, tomorrow is"
        other_today, _, _ = rng.choice([t for t in _TEMPORAL_CASES if t[0] != today])
        corrupted = f"If today is {other_today}, tomorrow is"
        return clean, corrupted, tmrw, wrong
    return _make_task(handle, "Reasoning: temporal", "Angle 3: Arithmetic", gen, seed=seed)


_SPATIAL_CASES = [
    ("cup", "box", "table", "box", "table", "floor"),
    ("book", "bag", "desk", "bag", "desk", "chair")
]

def spatial_reasoning_behavior(handle: adapter.ModelHandle, seed: int = 0) -> dict:
    def gen(rng):
        a, b, c, wrong_a, wrong_b, wrong_c = rng.choice(_SPATIAL_CASES)
        clean = f"If the {a} is in the {b}, and the {b} is on the {c}, the {a} is on the"
        corrupted = f"If the {wrong_a} is in the {wrong_b}, and the {wrong_b} is on the {wrong_c}, the {wrong_a} is on the"
        return clean, corrupted, c, wrong_c
    return _make_task(handle, "Reasoning: spatial transitivity", "Angle 3: Arithmetic", gen, seed=seed)


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


_TRANSLATION_PAIRS = [
    ("one", "uno", "two", "dos", "three", "tres", "four", "cuatro"),
    ("dog", "perro", "cat", "gato", "bird", "ave", "fish", "pez")
]

def translation_induction_behavior(handle: adapter.ModelHandle, seed: int = 0) -> dict:
    def gen(rng):
        e1, s1, e2, s2, e3, s3, e4, s4 = rng.choice(_TRANSLATION_PAIRS)
        clean = f"{e1}: {s1}, {e2}: {s2}, {e3}:"
        corrupted = f"{e1}: {s1}, {e2}: {s2}, {e4}:"
        return clean, corrupted, s3, s4
    return _make_task(handle, "Induction: translation", "Angle 4: Induction", gen, seed=seed)


_CAPITALIZATION_PAIRS = [
    ("paris", "Paris", "london", "London", "tokyo", "Tokyo", "berlin", "Berlin"),
    ("apple", "Apple", "banana", "Banana", "cherry", "Cherry", "date", "Date")
]

def capitalization_induction_behavior(handle: adapter.ModelHandle, seed: int = 0) -> dict:
    def gen(rng):
        w1, c1, w2, c2, w3, c3, w4, c4 = rng.choice(_CAPITALIZATION_PAIRS)
        clean = f"{w1}: {c1}, {w2}: {c2}, {w3}:"
        corrupted = f"{w1}: {c1}, {w2}: {c2}, {w4}:"
        return clean, corrupted, c3, c4
    return _make_task(handle, "Induction: capitalization", "Angle 4: Induction", gen, seed=seed)


_ANTONYM_INDUCTION_PAIRS = [
    ("hot", "cold", "big", "small", "fast", "slow", "tall", "short"),
    ("good", "bad", "happy", "sad", "rich", "poor", "hard", "soft")
]

def antonym_induction_behavior(handle: adapter.ModelHandle, seed: int = 0) -> dict:
    def gen(rng):
        w1, a1, w2, a2, w3, a3, w4, a4 = rng.choice(_ANTONYM_INDUCTION_PAIRS)
        clean = f"{w1}: {a1}, {w2}: {a2}, {w3}:"
        corrupted = f"{w1}: {a1}, {w2}: {a2}, {w4}:"
        return clean, corrupted, a3, a4
    return _make_task(handle, "Induction: antonyms", "Angle 4: Induction", gen, seed=seed)


_FIRST_LETTER_PAIRS = [
    ("apple", "a", "banana", "b", "cherry", "c", "date", "d"),
    ("elephant", "e", "fox", "f", "goat", "g", "horse", "h")
]

def first_letter_induction_behavior(handle: adapter.ModelHandle, seed: int = 0) -> dict:
    def gen(rng):
        w1, l1, w2, l2, w3, l3, w4, l4 = rng.choice(_FIRST_LETTER_PAIRS)
        clean = f"{w1}: {l1}, {w2}: {l2}, {w3}:"
        corrupted = f"{w1}: {l1}, {w2}: {l2}, {w4}:"
        return clean, corrupted, l3, l4
    return _make_task(handle, "Induction: first letter", "Angle 4: Induction", gen, seed=seed)


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


_AGE_BIAS = [
    ("old", "computer", "book"),
    ("elderly", "technology", "garden"),
    ("young", "book", "computer"),
    ("teenage", "garden", "technology")
]

def age_bias_behavior(handle: adapter.ModelHandle, seed: int = 0) -> dict:
    def gen(rng):
        age, stereo, anti = rng.choice(_AGE_BIAS)
        clean = f"The {age} person struggled to use the"
        other_age, _, _ = rng.choice([a for a in _AGE_BIAS if a[0] != age])
        corrupted = f"The {other_age} person struggled to use the"
        return clean, corrupted, stereo, anti
    return _make_task(handle, "Social Bias: age stereotypes", "Angle 5: Social Bias", gen, seed=seed)


_NATIONALITY_BIAS = [
    ("Italian", "pizza", "burger"),
    ("French", "wine", "beer"),
    ("American", "burger", "pizza"),
    ("German", "beer", "wine")
]

def nationality_bias_behavior(handle: adapter.ModelHandle, seed: int = 0) -> dict:
    def gen(rng):
        nat, stereo, anti = rng.choice(_NATIONALITY_BIAS)
        clean = f"The {nat} person really loves consuming"
        other_nat, _, _ = rng.choice([n for n in _NATIONALITY_BIAS if n[0] != nat])
        corrupted = f"The {other_nat} person really loves consuming"
        return clean, corrupted, stereo, anti
    return _make_task(handle, "Social Bias: nationality stereotypes", "Angle 5: Social Bias", gen, seed=seed)


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


_SYNONYMS = [
    ("quick", "fast", "slow"), ("huge", "big", "small"),
    ("happy", "glad", "sad"), ("wealthy", "rich", "poor")
]

def synonym_behavior(handle: adapter.ModelHandle, seed: int = 0) -> dict:
    def gen(rng):
        word, syn, wrong = rng.choice(_SYNONYMS)
        clean = f"A word that means the same as {word} is"
        other_word, _, _ = rng.choice([s for s in _SYNONYMS if s[0] != word])
        corrupted = f"A word that means the same as {other_word} is"
        return clean, corrupted, syn, wrong
    return _make_task(handle, "Lexical Semantics: synonym prediction", "Angle 6: Lexical Semantics", gen, seed=seed)


_PART_WHOLE = [
    ("wheel", "car", "boat"), ("screen", "phone", "book"),
    ("page", "book", "phone"), ("sail", "boat", "car")
]

def part_whole_behavior(handle: adapter.ModelHandle, seed: int = 0) -> dict:
    def gen(rng):
        part, whole, wrong = rng.choice(_PART_WHOLE)
        clean = f"A {part} is a component of a"
        other_part, _, _ = rng.choice([p for p in _PART_WHOLE if p[0] != part])
        corrupted = f"A {other_part} is a component of a"
        return clean, corrupted, whole, wrong
    return _make_task(handle, "Lexical Semantics: part-whole relationships", "Angle 6: Lexical Semantics", gen, seed=seed)


_ANIMAL_YOUNG = [
    ("dog", "puppy", "kitten"), ("cat", "kitten", "puppy"),
    ("cow", "calf", "foal"), ("horse", "foal", "calf")
]

def animal_young_behavior(handle: adapter.ModelHandle, seed: int = 0) -> dict:
    def gen(rng):
        animal, young, wrong = rng.choice(_ANIMAL_YOUNG)
        clean = f"A baby {animal} is called a"
        other_animal, _, _ = rng.choice([a for a in _ANIMAL_YOUNG if a[0] != animal])
        corrupted = f"A baby {other_animal} is called a"
        return clean, corrupted, young, wrong
    return _make_task(handle, "Lexical Semantics: animal young names", "Angle 6: Lexical Semantics", gen, seed=seed)


ANGLE_1_BEHAVIORS = [ioi_behavior, agreement_behavior, subject_extraction_behavior, passive_voice_behavior, reflexive_pronoun_behavior]
ANGLE_2_BEHAVIORS = [factual_recall_behavior, currency_knowledge_behavior, element_symbol_behavior, language_knowledge_behavior, continent_knowledge_behavior]
ANGLE_3_BEHAVIORS = [arithmetic_addition_behavior, magnitude_comparison_behavior, arithmetic_subtraction_behavior, arithmetic_multiplication_behavior, temporal_reasoning_behavior, spatial_reasoning_behavior]
ANGLE_4_BEHAVIORS = [induction_copying_behavior, translation_induction_behavior, capitalization_induction_behavior, antonym_induction_behavior, first_letter_induction_behavior]
ANGLE_5_BEHAVIORS = [gender_bias_behavior, age_bias_behavior, nationality_bias_behavior]
ANGLE_6_BEHAVIORS = [antonym_behavior, category_membership_behavior, synonym_behavior, part_whole_behavior, animal_young_behavior]

ALL_BEHAVIORS = (
    ANGLE_1_BEHAVIORS +
    ANGLE_2_BEHAVIORS +
    ANGLE_3_BEHAVIORS +
    ANGLE_4_BEHAVIORS +
    ANGLE_5_BEHAVIORS +
    ANGLE_6_BEHAVIORS
)

