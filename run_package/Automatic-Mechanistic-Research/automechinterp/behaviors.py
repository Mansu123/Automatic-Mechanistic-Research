"""Behavior definitions for Stage B across all cognitive & mechanistic angles.

Angles:
  1.  Linguistic Structure (Syntax & Coreference)
  2.  Factual & World Knowledge
  3.  Reasoning & Arithmetic
  4.  In-Context Learning / Induction
  5.  Social Bias & Fairness
  6.  Lexical Semantics & Word Sense
  7.  Sentiment & Emotion
  8.  Code & Formal Reasoning
  9.  Commonsense & Physical Reasoning
  10. Multilingual / Cross-lingual
  11. Entity Tracking & Discourse
  12. Negation & Logic
  13. Quantitative Comparison
  14. Temporal & Sequential Ordering
  15. Analogical Reasoning
  16. Morphology & Word Formation
  17. Pragmatics & Implicature
  18. Scientific & Technical Knowledge
  19. Geography & Spatial Knowledge
  20. Arts, Literature & Culture
  21. Economics, Politics & Law
  22. Idioms & Figurative Language
  23. Counting & Set Facts
  24. Units & Measurement
  25. History & Chronology

~200 behaviors total. Angles 1-12 are hand-written closures; angles 13-25 are
built declaratively from data rows via ``_pair_task`` (one row per task,
same (handle, seed) -> task-dict signature). ``ANGLE_BEHAVIORS`` maps each
angle number to its list; ``ALL_BEHAVIORS`` is the flattened, angle-ordered
sequence every stage iterates over.

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
                pair_generator: Callable, n_eval: int = 20, seed: int = 0) -> dict:
    """Finite-catalog engineering tasks: never invent independent samples by repetition.

    Train/discovery and evaluation prompts are disjoint. The small, related template
    pools are explicitly NOT a validated scientific stress benchmark.
    """
    from .eval.causal_measurements import score_contrast
    import hashlib
    import json
    rng = random.Random(seed)
    from .eval.task_repairs import repaired_pairs
    repaired = repaired_pairs(behavior)
    if repaired is not None:
        pair_generator = lambda rng: rng.choice(repaired)
    pool, seen = [], set()
    for _ in range(2000):
        row = tuple(pair_generator(rng))
        if row[0] not in seen and row[0] != row[1] and row[2] != row[3]:
            seen.add(row[0]); pool.append(row)
        if len(pool) >= 100:
            break
    if len(pool) < 2:
        raise ValueError(f"{behavior}: fewer than two distinct valid prompt pairs")
    train_count = min(6, max(1, len(pool)//4))
    train = pool[:train_count]
    train_texts = {text for row in train for text in row[:2]}
    heldout = [row for row in pool[train_count:] if not train_texts.intersection(row[:2])][:n_eval]
    if not heldout:
        # There is no defensible held-out estimate for this tiny catalog.
        heldout = []
    cp, xp, positive, negative = train[0]
    eval_prompts = [(c, a, b) for c, _, a, b in heldout]
    def task_metric_fn():
        # Profiling is discovery: it never consumes the held-out verification examples.
        return sum(score_contrast(handle, c, a, b)["margin"] for c, _, a, b in train) / len(train)
    serial = {"train": train, "evaluation": heldout, "seed": seed}
    return {
        "behavior": behavior, "category": category, "clean_prompt": cp,
        "corrupted_prompt": xp, "io_token": positive, "s_token": negative,
        "eval_prompts": eval_prompts, "eval_pairs": heldout,
        "discovery_pairs": train,
        "probe_texts": list(dict.fromkeys(text for row in train for text in row[:2])),
        "contrastive_pos": [r[0] for r in train], "contrastive_neg": [r[1] for r in train],
        "task_metric_fn": task_metric_fn, "n_heads": adapter.get_num_heads(handle),
        "stress_prompts": [], "stress_data_status": "unvalidated",
        "dataset_hash": hashlib.sha256(json.dumps(serial, sort_keys=True).encode()).hexdigest(),
        "data_audit": {"scope": "finite_catalog_engineering_evaluation",
            "requested_evaluation_examples": n_eval, "unique_pool_prompts": len(pool),
            "unique_evaluation_examples": len(eval_prompts), "discovery_examples": len(train),
            "prompt_overlap": 0, "independent_stress_validated": False,
            "limitation": "Template/entity correlations remain; counts are unique prompts, not independent mechanisms. No Confirmed claim without separately validated stress data."},
    }


def _pair_task(behavior: str, category: str,
               pairs: list[tuple[str, str, str, str]]) -> Callable:
    """Build a behavior factory from an explicit list of
    (clean_prompt, corrupted_prompt, io_token, s_token) tuples.

    Keeps the high-volume angles (13+) declarative: each task is one data
    row rather than a near-identical hand-written closure. The returned
    callable has the same (handle, seed) -> task-dict signature every other
    behavior in this module exposes, so ALL_BEHAVIORS stays uniform.
    """
    slug = "".join(c if c.isalnum() else "_" for c in behavior.split(":")[-1].strip().lower())

    def build(handle: adapter.ModelHandle, seed: int = 0) -> dict:
        def gen(rng):
            return rng.choice(pairs)
        return _make_task(handle, behavior, category, gen, seed=seed)

    build.__name__ = f"{slug}_behavior"
    build.__qualname__ = build.__name__
    return build


# ============================================================================
# ANGLE 1: Linguistic Structure & Syntax
# ============================================================================

_IOI_TEMPLATE = "When {A} and {B} went to the store, {S} gave a drink to"
_IOI_NAMES = [
    "John", "Mary", "Alice", "Bob", "Sarah", "Tom",
    "David", "Emma", "James", "Olivia", "Michael", "Sophia",
    "Daniel", "Emily",
]


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
    ("car", "cars", "garage", "garages"), ("apple", "apples", "basket", "baskets"),
    ("bird", "birds", "tree", "trees"), ("letter", "letters", "mailbox", "mailboxes"),
    ("flower", "flowers", "garden", "gardens"), ("doctor", "doctors", "hospital", "hospitals"),
    ("worker", "workers", "factory", "factories"), ("child", "children", "playground", "playgrounds"),
    ("teacher", "teachers", "school", "schools"), ("painting", "paintings", "museum", "museums"),
    ("boat", "boats", "harbor", "harbors"), ("computer", "computers", "office", "offices"),
    ("phone", "phones", "desk", "desks"), ("cup", "cups", "table", "tables"),
    ("shirt", "shirts", "closet", "closets"), ("plate", "plates", "kitchen", "kitchens"),
    ("lamp", "lamps", "bedroom", "bedrooms"), ("player", "players", "field", "fields"),
    ("passenger", "passengers", "train", "trains"), ("customer", "customers", "store", "stores"),
    ("soldier", "soldiers", "camp", "camps"),
]


def agreement_behavior(handle: adapter.ModelHandle, seed: int = 0) -> dict:
    def gen(rng):
        sg, pl, dist_sg, dist_pl = rng.choice(_AGREEMENT_PAIRS)
        clean = f"The {pl} near the {dist_sg}"
        corrupted = f"The {sg} near the {dist_pl}"
        return clean, corrupted, "are", "is"
    return _make_task(handle, "Subject-verb number agreement (syntax)", "Angle 1: Linguistic", gen, seed=seed)


_SUBJECT_EXTRACTION_PAIRS = [
    ("man", "woman", "blue hat"), ("boy", "girl", "red shirt"),
    ("dog", "cat", "long tail"), ("teacher", "student", "glasses"),
    ("doctor", "nurse", "white coat"), ("driver", "passenger", "leather jacket"),
    ("chef", "waiter", "striped apron"), ("singer", "dancer", "silver microphone"),
    ("pilot", "mechanic", "aviator sunglasses"), ("guard", "visitor", "black badge"),
    ("clerk", "customer", "green scarf"), ("baker", "patron", "floury hat"),
    ("artist", "model", "wooden palette"), ("officer", "driver", "gold watch"),
    ("coach", "player", "silver whistle"), ("runner", "walker", "yellow shoes"),
    ("builder", "architect", "hard hat"), ("farmer", "merchant", "straw hat"),
    ("sailor", "captain", "navy cap"), ("judge", "lawyer", "dark robe"),
    ("king", "knight", "golden crown"), ("priest", "monk", "silver cross"),
    ("actor", "director", "script folder"), ("writer", "editor", "fountain pen"),
    ("hunter", "guide", "camo jacket"),
]


def subject_extraction_behavior(handle: adapter.ModelHandle, seed: int = 0) -> dict:
    def gen(rng):
        s1, s2, desc = rng.choice(_SUBJECT_EXTRACTION_PAIRS)
        clean = f"The {s1} with the {desc} walked to the park. The one who walked is the"
        corrupted = f"The {s2} with the {desc} walked to the park. The one who walked is the"
        return clean, corrupted, s1, s2
    return _make_task(handle, "Subject extraction", "Angle 1: Linguistic", gen, seed=seed)


_PASSIVE_VOICE_PAIRS = [
    ("boy", "girl", "window"), ("dog", "cat", "toy"),
    ("man", "woman", "car"), ("chef", "waiter", "meal"),
    ("artist", "critic", "statue"), ("doctor", "nurse", "needle"),
    ("teacher", "student", "chalkboard"), ("mechanic", "driver", "engine"),
    ("baker", "helper", "cake"), ("guard", "thief", "safe"),
    ("author", "editor", "manuscript"), ("painter", "assistant", "canvas"),
    ("driver", "cyclist", "mirror"), ("builder", "apprentice", "wall"),
    ("farmer", "hand", "tractor"), ("tailor", "weaver", "fabric"),
    ("pilot", "copilot", "throttle"), ("captain", "sailor", "anchor"),
    ("clerk", "cashier", "receipt"), ("scientist", "intern", "flask"),
    ("photographer", "tourist", "camera"), ("gardener", "landscaper", "hedge"),
    ("shoemaker", "cobbler", "boot"), ("musician", "producer", "track"),
    ("hunter", "trapper", "trap"),
]


def passive_voice_behavior(handle: adapter.ModelHandle, seed: int = 0) -> dict:
    def gen(rng):
        a1, a2, obj = rng.choice(_PASSIVE_VOICE_PAIRS)
        clean = f"The {obj} was seen by the {a1}. The one who saw it was the"
        corrupted = f"The {obj} was seen by the {a2}. The one who saw it was the"
        return clean, corrupted, a1, a2
    return _make_task(handle, "Passive voice inference", "Angle 1: Linguistic", gen, seed=seed)


_REFLEXIVE_PRONOUN_PAIRS = [
    ("girl", "boy", "herself", "himself"), ("woman", "man", "herself", "himself"),
    ("queen", "king", "herself", "himself"), ("actress", "actor", "herself", "himself"),
    ("sister", "brother", "herself", "himself"), ("mother", "father", "herself", "himself"),
    ("daughter", "son", "herself", "himself"), ("niece", "nephew", "herself", "himself"),
    ("waitress", "waiter", "herself", "himself"), ("aunt", "uncle", "herself", "himself"),
    ("princess", "prince", "herself", "himself"), ("grandmother", "grandfather", "herself", "himself"),
    ("empress", "emperor", "herself", "himself"), ("hostess", "host", "herself", "himself"),
    ("lady", "gentleman", "herself", "himself"), ("heroine", "hero", "herself", "himself"),
    ("goddess", "god", "herself", "himself"), ("baroness", "baron", "herself", "himself"),
    ("duchess", "duke", "herself", "himself"), ("countess", "count", "herself", "himself"),
    ("niece", "uncle", "herself", "himself"), ("girl", "man", "herself", "himself"),
    ("woman", "boy", "herself", "himself"), ("sister", "father", "herself", "himself"),
    ("mother", "son", "herself", "himself"),
]


def reflexive_pronoun_behavior(handle: adapter.ModelHandle, seed: int = 0) -> dict:
    def gen(rng):
        s1, s2, r1, r2 = rng.choice(_REFLEXIVE_PRONOUN_PAIRS)
        clean = f"The {s1} looked at {r1} in the mirror. The person in the mirror was the"
        corrupted = f"The {s2} looked at {r2} in the mirror. The person in the mirror was the"
        return clean, corrupted, s1, s2
    return _make_task(handle, "Reflexive pronoun resolution", "Angle 1: Linguistic", gen, seed=seed)


_RELATIVE_CLAUSE_PAIRS = [
    ("author", "authors", "who lived nearby"), ("painter", "painters", "who won the prize"),
    ("scientist", "scientists", "who gave the talk"), ("driver", "drivers", "who parked outside"),
    ("student", "students", "who passed the test"), ("doctor", "doctors", "who visited yesterday"),
    ("teacher", "teachers", "who arrived early"), ("nurse", "nurses", "who helped today"),
    ("baker", "bakers", "who made the bread"), ("clerk", "clerks", "who filed the report"),
    ("actor", "actors", "who memorized the script"), ("farmer", "farmers", "who harvested the corn"),
    ("musician", "musicians", "who played the tune"), ("builder", "builders", "who repaired the roof"),
    ("officer", "officers", "who checked the door"), ("runner", "runners", "who finished the race"),
    ("waiter", "waiters", "who served the table"), ("gardener", "gardeners", "who watered the lawn"),
    ("singer", "singers", "who sang the song"), ("worker", "workers", "who cleaned the room"),
    ("pilot", "pilots", "who landed the plane"), ("mechanic", "mechanics", "who fixed the tire"),
    ("tailor", "tailors", "who sewed the coat"), ("cook", "cooks", "who prepared the soup"),
    ("guard", "guards", "who locked the gate"),
]


def relative_clause_behavior(handle: adapter.ModelHandle, seed: int = 0) -> dict:
    def gen(rng):
        sg, pl, clause = rng.choice(_RELATIVE_CLAUSE_PAIRS)
        clean = f"The {pl} {clause}"
        corrupted = f"The {sg} {clause}"
        return clean, corrupted, "are", "is"
    return _make_task(handle, "Agreement across a relative clause", "Angle 1: Linguistic", gen, seed=seed)


_COORDINATION_PAIRS = [
    ("apple", "orange"), ("cat", "dog"), ("red", "blue"), ("north", "south"),
    ("pen", "pencil"), ("chair", "table"), ("cup", "glass"), ("shirt", "jacket"),
    ("car", "truck"), ("bread", "butter"), ("knife", "fork"), ("salt", "pepper"),
    ("shoe", "sock"), ("brother", "sister"), ("sun", "moon"), ("day", "night"),
    ("king", "queen"), ("lock", "key"), ("needle", "thread"), ("bow", "arrow"),
    ("hammer", "nail"), ("paper", "envelope"), ("tea", "coffee"), ("milk", "juice"),
    ("summer", "winter"),
]


def coordination_behavior(handle: adapter.ModelHandle, seed: int = 0) -> dict:
    def gen(rng):
        a, b = rng.choice(_COORDINATION_PAIRS)
        clean = f"Complete the expression 'both {a} and {b}': both {a} and"
        corrupted = f"Complete the expression 'both {b} and {a}': both {b} and"
        return clean, corrupted, b, a
    return _make_task(handle, "Coordination completion", "Angle 1: Linguistic", gen, seed=seed)


_CAPITALS = [
    ("France", "Paris", "London"), ("England", "London", "Paris"),
    ("Italy", "Rome", "Madrid"), ("Spain", "Madrid", "Rome"),
    ("Germany", "Berlin", "Paris"), ("Japan", "Tokyo", "Beijing"),
    ("China", "Beijing", "Tokyo"), ("Russia", "Moscow", "Berlin"),
    ("Egypt", "Cairo", "Nairobi"), ("Canada", "Ottawa", "Washington"),
    ("Greece", "Athens", "Rome"), ("Portugal", "Lisbon", "Madrid"),
    ("Austria", "Vienna", "Berlin"), ("Poland", "Warsaw", "Prague"),
    ("Norway", "Oslo", "Stockholm"), ("Sweden", "Stockholm", "Oslo"),
    ("Finland", "Helsinki", "Oslo"), ("Denmark", "Copenhagen", "Stockholm"),
    ("Ireland", "Dublin", "London"), ("Belgium", "Brussels", "Amsterdam"),
    ("Netherlands", "Amsterdam", "Brussels"), ("Switzerland", "Bern", "Vienna"),
    ("Kenya", "Nairobi", "Cairo"), ("Mexico", "Mexico", "Madrid"),
    ("Brazil", "Brasilia", "Lisbon"), ("Argentina", "Buenos", "Santiago"),
    ("Chile", "Santiago", "Lima"), ("Peru", "Lima", "Bogota"),
    ("Colombia", "Bogota", "Lima"), ("Australia", "Canberra", "London"),
]


def factual_recall_behavior(handle: adapter.ModelHandle, seed: int = 0) -> dict:
    def gen(rng):
        country, capital, wrong = rng.choice(_CAPITALS)
        clean = f"The capital of {country} is"
        other_country, other_capital, _ = rng.choice([c for c in _CAPITALS if c[0] != country])
        corrupted = f"The capital of {other_country} is"
        return clean, corrupted, capital, other_capital
    return _make_task(handle, "Factual recall: country capitals", "Angle 2: Factual", gen, seed=seed)


_CURRENCIES = [
    ("Japan", "Yen", "Dollar"), ("USA", "Dollar", "Yen"),
    ("UK", "Pound", "Euro"), ("France", "Euro", "Pound"),
    ("India", "Rupee", "Yen"), ("China", "Yuan", "Rupee"),
    ("Russia", "Ruble", "Euro"), ("Germany", "Euro", "Dollar"),
    ("Italy", "Euro", "Pound"), ("Spain", "Euro", "Dollar"),
    ("Mexico", "Peso", "Dollar"), ("Brazil", "Real", "Peso"),
    ("Canada", "Dollar", "Pound"), ("Australia", "Dollar", "Yen"),
    ("Switzerland", "Franc", "Euro"), ("South Africa", "Rand", "Pound"),
    ("South Korea", "Won", "Yen"), ("Saudi Arabia", "Riyal", "Dinar"),
    ("Turkey", "Lira", "Euro"), ("Egypt", "Pound", "Riyal"),
    ("Sweden", "Krona", "Euro"), ("Norway", "Krone", "Euro"),
    ("Denmark", "Krone", "Euro"), ("Poland", "Zloty", "Euro"),
    ("Thailand", "Baht", "Rupee"),
]


def currency_knowledge_behavior(handle: adapter.ModelHandle, seed: int = 0) -> dict:
    def gen(rng):
        country, curr, wrong = rng.choice(_CURRENCIES)
        clean = f"The official currency of {country} is the"
        other_country, other_curr, _ = rng.choice([c for c in _CURRENCIES if c[1] != curr])
        corrupted = f"The official currency of {other_country} is the"
        return clean, corrupted, curr, other_curr
    return _make_task(handle, "Factual recall: currency", "Angle 2: Factual", gen, seed=seed)


_ELEMENTS = [
    ("Oxygen", "O", "H"), ("Hydrogen", "H", "O"),
    ("Carbon", "C", "N"), ("Nitrogen", "N", "C"),
    ("Gold", "Au", "Ag"), ("Silver", "Ag", "Au"),
    ("Iron", "Fe", "Cu"), ("Copper", "Cu", "Fe"),
    ("Helium", "He", "H"), ("Sodium", "Na", "K"),
    ("Potassium", "K", "Na"), ("Lead", "Pb", "Sn"),
    ("Tin", "Sn", "Pb"), ("Mercury", "Hg", "Ag"),
    ("Uranium", "U", "Pu"), ("Zinc", "Zn", "Cu"),
    ("Aluminum", "Al", "Fe"), ("Silicon", "Si", "C"),
    ("Phosphorus", "P", "S"), ("Sulfur", "S", "P"),
    ("Chlorine", "Cl", "Br"), ("Bromine", "Br", "Cl"),
    ("Iodine", "I", "Br"), ("Calcium", "Ca", "K"),
    ("Magnesium", "Mg", "Ca"),
]


def element_symbol_behavior(handle: adapter.ModelHandle, seed: int = 0) -> dict:
    def gen(rng):
        elem, sym, wrong = rng.choice(_ELEMENTS)
        clean = f"The chemical symbol for {elem} is"
        other_elem, other_sym, _ = rng.choice([e for e in _ELEMENTS if e[0] != elem])
        corrupted = f"The chemical symbol for {other_elem} is"
        return clean, corrupted, sym, other_sym
    return _make_task(handle, "Factual recall: element symbols", "Angle 2: Factual", gen, seed=seed)


_LANGUAGES = [
    ("Brazil", "Portuguese", "Spanish"), ("Mexico", "Spanish", "Portuguese"),
    ("USA", "English", "French"), ("France", "French", "English"),
    ("Egypt", "Arabic", "English"), ("China", "Chinese", "Japanese"),
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
    ("Canada", "America", "Asia"), ("Japan", "Asia", "Europe"),
]


def continent_knowledge_behavior(handle: adapter.ModelHandle, seed: int = 0) -> dict:
    def gen(rng):
        country, cont, wrong = rng.choice(_CONTINENTS)
        clean = f"The country of {country} is located in"
        other_country, _, _ = rng.choice([c for c in _CONTINENTS if c[0] != country])
        corrupted = f"The country of {other_country} is located in"
        return clean, corrupted, cont, wrong
    return _make_task(handle, "Factual recall: continents", "Angle 2: Factual", gen, seed=seed)


_PLANETS = [
    ("closest to the Sun", "Mercury", "Neptune"),
    ("the largest in the Solar System", "Jupiter", "Mars"),
    ("known as the Red Planet", "Mars", "Venus"),
    ("famous for its rings", "Saturn", "Earth"),
]


def planet_knowledge_behavior(handle: adapter.ModelHandle, seed: int = 0) -> dict:
    def gen(rng):
        desc, planet, wrong = rng.choice(_PLANETS)
        clean = f"The planet {desc} is"
        other_desc, _, _ = rng.choice([p for p in _PLANETS if p[0] != desc])
        corrupted = f"The planet {other_desc} is"
        return clean, corrupted, planet, wrong
    return _make_task(handle, "Factual recall: planets", "Angle 2: Factual", gen, seed=seed)


_LANDMARKS = [
    ("Eiffel Tower", "France", "Italy"), ("Colosseum", "Italy", "France"),
    ("Big Ben", "England", "Germany"), ("Great Wall", "China", "Japan"),
]


def landmark_country_behavior(handle: adapter.ModelHandle, seed: int = 0) -> dict:
    def gen(rng):
        landmark, country, wrong = rng.choice(_LANDMARKS)
        clean = f"The {landmark} is located in the country of"
        other_landmark, _, _ = rng.choice([l for l in _LANDMARKS if l[0] != landmark])
        corrupted = f"The {other_landmark} is located in the country of"
        return clean, corrupted, country, wrong
    return _make_task(handle, "Factual recall: landmark to country", "Angle 2: Factual", gen, seed=seed)


# ============================================================================
# ANGLE 3: Reasoning & Arithmetic
# ============================================================================

_ADDITION_CASES = [
    (3, 4, 7, 8, 2, 5), (2, 5, 7, 6, 3, 4),
    (1, 8, 9, 8, 4, 5), (4, 5, 9, 7, 1, 8),
    (2, 3, 5, 6, 6, 2), (6, 2, 8, 7, 2, 3),
    (1, 4, 5, 6, 2, 3), (5, 3, 8, 9, 4, 2),
    (3, 3, 6, 7, 1, 5), (4, 2, 6, 5, 3, 1),
    (2, 6, 8, 9, 5, 2), (7, 1, 8, 6, 3, 2),
    (1, 6, 7, 8, 2, 4), (5, 2, 7, 8, 3, 3),
    (4, 3, 7, 6, 1, 7), (2, 7, 9, 8, 4, 4),
    (3, 5, 8, 7, 6, 1), (1, 7, 8, 9, 3, 4),
    (6, 3, 9, 8, 5, 1), (5, 4, 9, 8, 2, 6),
    (1, 5, 6, 7, 4, 1), (2, 4, 6, 5, 3, 2),
    (4, 4, 8, 7, 5, 2), (3, 2, 5, 4, 1, 3),
    (7, 2, 9, 8, 4, 3),
]


def arithmetic_addition_behavior(handle: adapter.ModelHandle, seed: int = 0) -> dict:
    def gen(rng):
        a, b, ans, wrong, c, d = rng.choice(_ADDITION_CASES)
        clean = f"The sum of {a} and {b} is"
        corrupted = f"The sum of {c} and {d} is"
        return clean, corrupted, str(ans), str(wrong)
    return _make_task(handle, "Arithmetic: single-digit addition", "Angle 3: Arithmetic", gen, seed=seed)


_MAGNITUDE_CASES = [
    (8, 3, "8", "3"), (9, 2, "9", "2"), (7, 4, "7", "4"), (6, 1, "6", "1"),
    (9, 4, "9", "4"), (8, 2, "8", "2"), (7, 3, "7", "3"), (9, 5, "9", "5"),
    (6, 2, "6", "2"), (8, 5, "8", "5"), (7, 2, "7", "2"), (9, 6, "9", "6"),
    (8, 4, "8", "4"), (5, 1, "5", "1"), (6, 3, "6", "3"), (7, 1, "7", "1"),
    (9, 1, "9", "1"), (8, 1, "8", "1"), (5, 2, "5", "2"), (4, 1, "4", "1"),
    (9, 7, "9", "7"), (8, 6, "8", "6"), (7, 5, "7", "5"), (6, 4, "6", "4"),
    (5, 3, "5", "3"),
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
    (9, 2, 7, 8, 9, 1), (8, 5, 3, 4, 9, 5),
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
    (2, 2, 4, 6, 2, 3), (3, 2, 6, 4, 2, 2),
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
    ("Wednesday", "Thursday", "Friday"), ("Thursday", "Friday", "Saturday"),
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
    ("book", "bag", "desk", "bag", "desk", "chair"),
]


def spatial_reasoning_behavior(handle: adapter.ModelHandle, seed: int = 0) -> dict:
    def gen(rng):
        a, b, c, wrong_a, wrong_b, wrong_c = rng.choice(_SPATIAL_CASES)
        clean = f"If the {a} is in the {b}, and the {b} is on the {c}, the {a} is on the"
        corrupted = f"If the {wrong_a} is in the {wrong_b}, and the {wrong_b} is on the {wrong_c}, the {wrong_a} is on the"
        return clean, corrupted, c, wrong_c
    return _make_task(handle, "Reasoning: spatial transitivity", "Angle 3: Arithmetic", gen, seed=seed)


_PARITY_CASES = [
    (4, "even", "odd"), (7, "odd", "even"),
    (10, "even", "odd"), (3, "odd", "even"),
]


def parity_reasoning_behavior(handle: adapter.ModelHandle, seed: int = 0) -> dict:
    def gen(rng):
        n, ans, wrong = rng.choice(_PARITY_CASES)
        clean = f"The number {n} is"
        other_n, _, _ = rng.choice([p for p in _PARITY_CASES if p[0] != n])
        corrupted = f"The number {other_n} is"
        return clean, corrupted, ans, wrong
    return _make_task(handle, "Reasoning: parity (even/odd)", "Angle 3: Arithmetic", gen, seed=seed)


_ORDINAL_CASES = [
    ("first", "A", "B", "C", "A", "C"),
    ("last", "A", "B", "C", "C", "A"),
]


def ordinal_reasoning_behavior(handle: adapter.ModelHandle, seed: int = 0) -> dict:
    def gen(rng):
        pos, x, y, z, ans, wrong = rng.choice(_ORDINAL_CASES)
        clean = f"In the list {x}, {y}, {z}, the {pos} item is"
        other = [o for o in _ORDINAL_CASES if o[0] != pos][0]
        corrupted = f"In the list {other[1]}, {other[2]}, {other[3]}, the {other[0]} item is"
        return clean, corrupted, ans, wrong
    return _make_task(handle, "Reasoning: ordinal position", "Angle 3: Arithmetic", gen, seed=seed)


_DIVISION_CASES = [
    (6, 2, 3, 2), (8, 4, 2, 3), (9, 3, 3, 4), (10, 5, 2, 5),
]


def arithmetic_division_behavior(handle: adapter.ModelHandle, seed: int = 0) -> dict:
    def gen(rng):
        a, b, ans, wrong = rng.choice(_DIVISION_CASES)
        clean = f"{a} divided by {b} is"
        c, d, _, _ = rng.choice([x for x in _DIVISION_CASES if x[0] != a])
        corrupted = f"{c} divided by {d} is"
        return clean, corrupted, str(ans), str(wrong)
    return _make_task(handle, "Arithmetic: single-digit division", "Angle 3: Arithmetic", gen, seed=seed)


# ============================================================================
# ANGLE 4: In-Context Learning / Induction
# ============================================================================

_REPEATED_SEQUENCES = [
    (["red", "blue", "green"], "red", "blue"),
    (["apple", "banana", "cherry"], "apple", "banana"),
    (["cat", "dog", "fox"], "cat", "dog"),
    (["alpha", "beta", "gamma"], "alpha", "beta"),
    (["gold", "silver", "bronze"], "gold", "silver"),
    (["spring", "summer", "autumn"], "spring", "summer"),
    (["north", "south", "east"], "north", "south"),
    (["sun", "moon", "star"], "sun", "moon"),
    (["cup", "fork", "knife"], "cup", "fork"),
    (["desk", "chair", "table"], "desk", "chair"),
    (["rock", "paper", "scissor"], "rock", "paper"),
    (["pen", "book", "pad"], "pen", "book"),
    (["iron", "copper", "lead"], "iron", "copper"),
    (["lake", "river", "ocean"], "lake", "river"),
    (["oak", "pine", "maple"], "oak", "pine"),
    (["lion", "tiger", "bear"], "lion", "tiger"),
    (["rose", "tulip", "daisy"], "rose", "tulip"),
    (["mars", "venus", "jupiter"], "mars", "venus"),
    (["shirt", "pants", "coat"], "shirt", "pants"),
    (["hat", "boot", "glove"], "hat", "boot"),
    (["bread", "milk", "cheese"], "bread", "milk"),
    (["salt", "pepper", "sugar"], "salt", "pepper"),
    (["car", "bike", "bus"], "car", "bike"),
    (["train", "ship", "plane"], "train", "ship"),
    (["one", "two", "three"], "one", "two"),
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
    ("dog", "perro", "cat", "gato", "bird", "ave", "fish", "pez"),
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
    ("apple", "Apple", "banana", "Banana", "cherry", "Cherry", "date", "Date"),
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
    ("good", "bad", "happy", "sad", "rich", "poor", "hard", "soft"),
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
    ("elephant", "e", "fox", "f", "goat", "g", "horse", "h"),
]


def first_letter_induction_behavior(handle: adapter.ModelHandle, seed: int = 0) -> dict:
    def gen(rng):
        w1, l1, w2, l2, w3, l3, w4, l4 = rng.choice(_FIRST_LETTER_PAIRS)
        clean = f"{w1}: {l1}, {w2}: {l2}, {w3}:"
        corrupted = f"{w1}: {l1}, {w2}: {l2}, {w4}:"
        return clean, corrupted, l3, l4
    return _make_task(handle, "Induction: first letter", "Angle 4: Induction", gen, seed=seed)


_NUMBER_SEQUENCE_PAIRS = [
    ("2", "4", "6", "8", "10"),
    ("5", "10", "15", "20", "25"),
    ("1", "2", "3", "4", "5"),
    ("3", "6", "9", "12", "15"),
]


def number_sequence_induction_behavior(handle: adapter.ModelHandle, seed: int = 0) -> dict:
    def gen(rng):
        a, b, c, d, e = rng.choice(_NUMBER_SEQUENCE_PAIRS)
        clean = f"{a}, {b}, {c}, {d},"
        corrupted = f"{a}, {b}, {c},"
        return clean, corrupted, e, d
    return _make_task(handle, "Induction: arithmetic number sequence", "Angle 4: Induction", gen, seed=seed)


_UPPERCASE_SEQ_PAIRS = [
    ("A", "B", "C", "D"), ("X", "Y", "Z", "A"), ("M", "N", "O", "P"),
]


def alphabet_sequence_induction_behavior(handle: adapter.ModelHandle, seed: int = 0) -> dict:
    def gen(rng):
        a, b, c, d = rng.choice(_UPPERCASE_SEQ_PAIRS)
        clean = f"{a}, {b}, {c},"
        corrupted = f"{a}, {b},"
        return clean, corrupted, d, c
    return _make_task(handle, "Induction: alphabet sequence", "Angle 4: Induction", gen, seed=seed)


# ============================================================================
# ANGLE 5: Social Bias & Fairness
# ============================================================================

_GENDER_OCCUPATIONS = [
    ("nurse", "she", "he", "doctor"),
    ("secretary", "she", "he", "manager"),
    ("engineer", "he", "she", "designer"),
    ("surgeon", "he", "she", "patient"),
    ("teacher", "she", "he", "principal"),
    ("programmer", "he", "she", "assistant"),
    ("librarian", "she", "he", "curator"),
    ("mechanic", "he", "she", "driver"),
    ("receptionist", "she", "he", "executive"),
    ("carpenter", "he", "she", "architect"),
    ("housekeeper", "she", "he", "resident"),
    ("plumber", "he", "she", "homeowner"),
    ("cashier", "she", "he", "accountant"),
    ("electrician", "he", "she", "inspector"),
    ("assistant", "she", "he", "director"),
    ("machinist", "he", "she", "operator"),
    ("paralegal", "she", "he", "attorney"),
    ("welder", "he", "she", "supervisor"),
    ("stylist", "she", "he", "customer"),
    ("firefighter", "he", "she", "paramedic"),
    ("babysitter", "she", "he", "parent"),
    ("pilot", "he", "she", "attendant"),
    ("dental hygienist", "she", "he", "dentist"),
    ("miner", "he", "she", "geologist"),
    ("seamstress", "she", "he", "tailor"),
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
    ("teenage", "garden", "technology"),
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
    ("German", "beer", "wine"),
]


def nationality_bias_behavior(handle: adapter.ModelHandle, seed: int = 0) -> dict:
    def gen(rng):
        nat, stereo, anti = rng.choice(_NATIONALITY_BIAS)
        clean = f"The {nat} person really loves consuming"
        other_nat, _, _ = rng.choice([n for n in _NATIONALITY_BIAS if n[0] != nat])
        corrupted = f"The {other_nat} person really loves consuming"
        return clean, corrupted, stereo, anti
    return _make_task(handle, "Social Bias: nationality stereotypes", "Angle 5: Social Bias", gen, seed=seed)


_APPEARANCE_BIAS = [
    ("tall", "basketball", "chess"),
    ("short", "chess", "basketball"),
    ("strong", "boxing", "reading"),
    ("slim", "running", "wrestling"),
]


def appearance_bias_behavior(handle: adapter.ModelHandle, seed: int = 0) -> dict:
    def gen(rng):
        trait, stereo, anti = rng.choice(_APPEARANCE_BIAS)
        clean = f"The {trait} student was assumed to be good at"
        other_trait, _, _ = rng.choice([a for a in _APPEARANCE_BIAS if a[0] != trait])
        corrupted = f"The {other_trait} student was assumed to be good at"
        return clean, corrupted, stereo, anti
    return _make_task(handle, "Social Bias: appearance stereotypes", "Angle 5: Social Bias", gen, seed=seed)


_PROFESSION_GENDER_NEUTRAL = [
    ("teacher", "she", "he"), ("firefighter", "he", "she"),
    ("librarian", "she", "he"), ("pilot", "he", "she"),
]


def occupation_pronoun_bias_behavior(handle: adapter.ModelHandle, seed: int = 0) -> dict:
    def gen(rng):
        occ, stereo, anti = rng.choice(_PROFESSION_GENDER_NEUTRAL)
        clean = f"After the shift ended, the {occ} said that"
        other_occ, _, _ = rng.choice([p for p in _PROFESSION_GENDER_NEUTRAL if p[0] != occ])
        corrupted = f"After the shift ended, the {other_occ} said that"
        return clean, corrupted, stereo, anti
    return _make_task(handle, "Social Bias: occupation pronoun default", "Angle 5: Social Bias", gen, seed=seed)


# ============================================================================
# ANGLE 6: Lexical Semantics & Word Sense
# ============================================================================

_ANTONYMS = [
    ("hot", "cold", "warm"), ("big", "small", "large"),
    ("fast", "slow", "quick"), ("light", "dark", "bright"),
    ("rich", "poor", "wealthy"), ("early", "late", "soon"),
    ("happy", "sad", "glad"), ("strong", "weak", "tough"),
    ("young", "old", "new"), ("hard", "soft", "solid"),
    ("easy", "hard", "simple"), ("tall", "short", "high"),
    ("clean", "dirty", "fresh"), ("thick", "thin", "heavy"),
    ("wide", "narrow", "broad"), ("loud", "quiet", "noisy"),
    ("sweet", "sour", "sugar"), ("brave", "cowardly", "bold"),
    ("dry", "wet", "arid"), ("sharp", "dull", "pointed"),
    ("heavy", "light", "dense"), ("smooth", "rough", "flat"),
    ("deep", "shallow", "profound"), ("cheap", "expensive", "inexpensive"),
    ("safe", "dangerous", "secure"),
]


def antonym_behavior(handle: adapter.ModelHandle, seed: int = 0) -> dict:
    def gen(rng):
        word, antonym, near_synonym = rng.choice(_ANTONYMS)
        clean = f"The opposite of {word} is"
        other_word, other_antonym, _ = rng.choice([a for a in _ANTONYMS if a[0] != word])
        corrupted = f"The opposite of {other_word} is"
        return clean, corrupted, antonym, other_antonym
    return _make_task(handle, "Lexical Semantics: antonym prediction", "Angle 6: Lexical Semantics", gen, seed=seed)


_CATEGORIES = [
    ("robin", "bird", "fish"), ("salmon", "fish", "bird"),
    ("oak", "tree", "animal"), ("rose", "flower", "tree"),
    ("trout", "fish", "mammal"), ("sparrow", "bird", "reptile"),
    ("eagle", "bird", "fish"), ("pine", "tree", "flower"),
    ("tulip", "flower", "tree"), ("lion", "mammal", "bird"),
    ("tiger", "mammal", "reptile"), ("shark", "fish", "mammal"),
    ("beetle", "insect", "bird"), ("ant", "insect", "fish"),
    ("frog", "amphibian", "reptile"), ("lizard", "reptile", "mammal"),
    ("snake", "reptile", "amphibian"), ("carrot", "vegetable", "fruit"),
    ("apple", "fruit", "vegetable"), ("banana", "fruit", "grain"),
    ("wheat", "grain", "vegetable"), ("chair", "furniture", "vehicle"),
    ("table", "furniture", "appliance"), ("car", "vehicle", "furniture"),
    ("truck", "vehicle", "building"),
]


def category_membership_behavior(handle: adapter.ModelHandle, seed: int = 0) -> dict:
    def gen(rng):
        item, cat, wrong = rng.choice(_CATEGORIES)
        clean = f"A {item} is a type of"
        other_item, other_cat, _ = rng.choice([c for c in _CATEGORIES if c[1] != cat])
        corrupted = f"A {other_item} is a type of"
        return clean, corrupted, cat, other_cat
    return _make_task(handle, "Lexical Semantics: category membership", "Angle 6: Lexical Semantics", gen, seed=seed)


_SYNONYMS = [
    ("quick", "fast", "slow"), ("huge", "big", "small"),
    ("happy", "glad", "sad"), ("wealthy", "rich", "poor"),
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
    ("page", "book", "phone"), ("sail", "boat", "car"),
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
    ("cow", "calf", "foal"), ("horse", "foal", "calf"),
]


def animal_young_behavior(handle: adapter.ModelHandle, seed: int = 0) -> dict:
    def gen(rng):
        animal, young, wrong = rng.choice(_ANIMAL_YOUNG)
        clean = f"A baby {animal} is called a"
        other_animal, _, _ = rng.choice([a for a in _ANIMAL_YOUNG if a[0] != animal])
        corrupted = f"A baby {other_animal} is called a"
        return clean, corrupted, young, wrong
    return _make_task(handle, "Lexical Semantics: animal young names", "Angle 6: Lexical Semantics", gen, seed=seed)


_HYPERNYMS = [
    ("color", "red", "seven"), ("animal", "tiger", "table"),
    ("fruit", "mango", "hammer"), ("metal", "copper", "cotton"),
]


def hypernym_instance_behavior(handle: adapter.ModelHandle, seed: int = 0) -> dict:
    def gen(rng):
        cat, member, wrong = rng.choice(_HYPERNYMS)
        clean = f"An example of a {cat} is"
        other_cat, _, _ = rng.choice([h for h in _HYPERNYMS if h[0] != cat])
        corrupted = f"An example of a {other_cat} is"
        return clean, corrupted, member, wrong
    return _make_task(handle, "Lexical Semantics: hypernym to instance", "Angle 6: Lexical Semantics", gen, seed=seed)


_WORD_SENSE = [
    ("The bank of the river was muddy, so he sat on the", "grass", "money"),
    ("She deposited the check at the bank and withdrew some", "money", "grass"),
    ("The bat flew out of the cave at dusk, flapping its", "wings", "bases"),
    ("He swung the bat and ran toward first", "base", "wings"),
]


def word_sense_disambiguation_behavior(handle: adapter.ModelHandle, seed: int = 0) -> dict:
    def gen(rng):
        clean, ans, wrong = rng.choice(_WORD_SENSE)
        other = rng.choice([w for w in _WORD_SENSE if w[0] != clean])
        corrupted = other[0]
        return clean, corrupted, ans, wrong
    return _make_task(handle, "Lexical Semantics: word sense disambiguation", "Angle 6: Lexical Semantics", gen, seed=seed)


# ============================================================================
# ANGLE 7: Sentiment & Emotion
# ============================================================================

_SENTIMENT_CASES = [
    ("The movie was absolutely wonderful and moving.", "positive", "negative"),
    ("The food was cold, bland, and overpriced.", "negative", "positive"),
    ("I loved every single minute of the concert.", "positive", "negative"),
    ("The service was rude and painfully slow.", "negative", "positive"),
]


def sentiment_polarity_behavior(handle: adapter.ModelHandle, seed: int = 0) -> dict:
    def gen(rng):
        text, ans, wrong = rng.choice(_SENTIMENT_CASES)
        clean = f"{text} The overall sentiment of this review is"
        other, _, _ = rng.choice([s for s in _SENTIMENT_CASES if s[0] != text])
        corrupted = f"{other} The overall sentiment of this review is"
        return clean, corrupted, ans, wrong
    return _make_task(handle, "Sentiment: polarity classification", "Angle 7: Sentiment", gen, seed=seed)


_EMOTION_CASES = [
    ("She jumped up and cheered when she heard the news.", "happy", "sad"),
    ("He cried quietly after reading the letter.", "sad", "happy"),
    ("She screamed and slammed the door behind her.", "angry", "calm"),
    ("He took a deep breath and smiled peacefully.", "calm", "angry"),
]


def emotion_classification_behavior(handle: adapter.ModelHandle, seed: int = 0) -> dict:
    def gen(rng):
        text, ans, wrong = rng.choice(_EMOTION_CASES)
        clean = f"{text} The person is most likely feeling"
        other, _, _ = rng.choice([e for e in _EMOTION_CASES if e[0] != text])
        corrupted = f"{other} The person is most likely feeling"
        return clean, corrupted, ans, wrong
    return _make_task(handle, "Sentiment: emotion classification", "Angle 7: Sentiment", gen, seed=seed)


_REVIEW_RATING_CASES = [
    ("This product broke after a single day of use.", "low", "high"),
    ("Best purchase I have made all year, works perfectly.", "high", "low"),
    ("Complete waste of money, do not buy this.", "low", "high"),
    ("Exceeded my expectations in every possible way.", "high", "low"),
]


def review_rating_behavior(handle: adapter.ModelHandle, seed: int = 0) -> dict:
    def gen(rng):
        text, ans, wrong = rng.choice(_REVIEW_RATING_CASES)
        clean = f"{text} This customer would give a rating that is"
        other, _, _ = rng.choice([r for r in _REVIEW_RATING_CASES if r[0] != text])
        corrupted = f"{other} This customer would give a rating that is"
        return clean, corrupted, ans, wrong
    return _make_task(handle, "Sentiment: implied review rating", "Angle 7: Sentiment", gen, seed=seed)


_TONE_CASES = [
    ("Get out of my way right now!", "angry", "polite"),
    ("Would you mind passing the salt, please?", "polite", "angry"),
    ("I am so thrilled you could make it!", "excited", "bored"),
    ("Whatever, I guess we can do that.", "bored", "excited"),
]


def tone_detection_behavior(handle: adapter.ModelHandle, seed: int = 0) -> dict:
    def gen(rng):
        text, ans, wrong = rng.choice(_TONE_CASES)
        clean = f'The sentence "{text}" sounds'
        other, _, _ = rng.choice([t for t in _TONE_CASES if t[0] != text])
        corrupted = f'The sentence "{other}" sounds'
        return clean, corrupted, ans, wrong
    return _make_task(handle, "Sentiment: tone detection", "Angle 7: Sentiment", gen, seed=seed)


_PRAISE_CRITICISM_CASES = [
    ("Your presentation was clear, sharp, and inspiring.", "praise", "criticism"),
    ("The report was sloppy and full of careless errors.", "criticism", "praise"),
    ("You handled that difficult client with real skill.", "praise", "criticism"),
    ("This code is unreadable and poorly organized.", "criticism", "praise"),
]


def praise_criticism_behavior(handle: adapter.ModelHandle, seed: int = 0) -> dict:
    def gen(rng):
        text, ans, wrong = rng.choice(_PRAISE_CRITICISM_CASES)
        clean = f"{text} This feedback is best described as"
        other, _, _ = rng.choice([p for p in _PRAISE_CRITICISM_CASES if p[0] != text])
        corrupted = f"{other} This feedback is best described as"
        return clean, corrupted, ans, wrong
    return _make_task(handle, "Sentiment: praise vs criticism", "Angle 7: Sentiment", gen, seed=seed)


# ============================================================================
# ANGLE 8: Code & Formal Reasoning
# ============================================================================

_PY_OUTPUT_CASES = [
    ("print(2 + 3)", "5", "6"), ("print(10 - 4)", "6", "7"),
    ("print(3 * 3)", "9", "8"), ("print(8 // 2)", "4", "3"),
]


def python_output_behavior(handle: adapter.ModelHandle, seed: int = 0) -> dict:
    def gen(rng):
        code, ans, wrong = rng.choice(_PY_OUTPUT_CASES)
        clean = f"The Python statement {code} prints"
        other_code, _, _ = rng.choice([c for c in _PY_OUTPUT_CASES if c[0] != code])
        corrupted = f"The Python statement {other_code} prints"
        return clean, corrupted, ans, wrong
    return _make_task(handle, "Code: Python expression output", "Angle 8: Code", gen, seed=seed)


_BOOLEAN_CASES = [
    ("True and False", "False", "True"), ("True or False", "True", "False"),
    ("not True", "False", "True"), ("not False", "True", "False"),
]


def boolean_logic_behavior(handle: adapter.ModelHandle, seed: int = 0) -> dict:
    def gen(rng):
        expr, ans, wrong = rng.choice(_BOOLEAN_CASES)
        clean = f"The expression {expr} evaluates to"
        other_expr, _, _ = rng.choice([b for b in _BOOLEAN_CASES if b[0] != expr])
        corrupted = f"The expression {other_expr} evaluates to"
        return clean, corrupted, ans, wrong
    return _make_task(handle, "Code: boolean logic evaluation", "Angle 8: Code", gen, seed=seed)


_TYPE_CASES = [
    ('x = "hello"', "str", "int"), ("x = 42", "int", "str"),
    ("x = 3.5", "float", "str"), ("x = [1, 2]", "list", "int"),
]


def variable_type_behavior(handle: adapter.ModelHandle, seed: int = 0) -> dict:
    def gen(rng):
        assign, ans, wrong = rng.choice(_TYPE_CASES)
        clean = f"After {assign}, the type of x is"
        other, _, _ = rng.choice([t for t in _TYPE_CASES if t[0] != assign])
        corrupted = f"After {other}, the type of x is"
        return clean, corrupted, ans, wrong
    return _make_task(handle, "Code: variable type inference", "Angle 8: Code", gen, seed=seed)


_LIST_INDEX_CASES = [
    ("[10, 20, 30]", "0", "10", "30"), ("[10, 20, 30]", "-1", "30", "10"),
    ("[5, 6, 7, 8]", "1", "6", "8"), ("[5, 6, 7, 8]", "2", "7", "5"),
]


def list_index_behavior(handle: adapter.ModelHandle, seed: int = 0) -> dict:
    def gen(rng):
        lst, idx, ans, wrong = rng.choice(_LIST_INDEX_CASES)
        clean = f"Given lst = {lst}, the value of lst[{idx}] is"
        other = rng.choice([x for x in _LIST_INDEX_CASES if (x[0], x[1]) != (lst, idx)])
        corrupted = f"Given lst = {other[0]}, the value of lst[{other[1]}] is"
        return clean, corrupted, ans, wrong
    return _make_task(handle, "Code: list indexing", "Angle 8: Code", gen, seed=seed)


_COMPARISON_CASES = [
    ("5 > 3", "True", "False"), ("2 > 7", "False", "True"),
    ("4 == 4", "True", "False"), ("9 < 1", "False", "True"),
]


def comparison_operator_behavior(handle: adapter.ModelHandle, seed: int = 0) -> dict:
    def gen(rng):
        expr, ans, wrong = rng.choice(_COMPARISON_CASES)
        clean = f"The comparison {expr} is"
        other_expr, _, _ = rng.choice([c for c in _COMPARISON_CASES if c[0] != expr])
        corrupted = f"The comparison {other_expr} is"
        return clean, corrupted, ans, wrong
    return _make_task(handle, "Code: comparison operators", "Angle 8: Code", gen, seed=seed)


_STRING_METHOD_CASES = [
    ('"hello".upper()', "HELLO", "hello"), ('"WORLD".lower()', "world", "WORLD"),
    ('"abc".upper()', "ABC", "abc"), ('"XYZ".lower()', "xyz", "XYZ"),
]


def string_method_behavior(handle: adapter.ModelHandle, seed: int = 0) -> dict:
    def gen(rng):
        expr, ans, wrong = rng.choice(_STRING_METHOD_CASES)
        clean = f"In Python, {expr} returns"
        other_expr, _, _ = rng.choice([s for s in _STRING_METHOD_CASES if s[0] != expr])
        corrupted = f"In Python, {other_expr} returns"
        return clean, corrupted, ans, wrong
    return _make_task(handle, "Code: string method results", "Angle 8: Code", gen, seed=seed)


# ============================================================================
# ANGLE 9: Commonsense & Physical Reasoning
# ============================================================================

_AFFORDANCE_CASES = [
    ("cut a piece of paper", "scissors", "spoon"),
    ("eat a bowl of soup", "spoon", "scissors"),
    ("drive a nail into wood", "hammer", "pillow"),
    ("write on a whiteboard", "marker", "hammer"),
]


def object_affordance_behavior(handle: adapter.ModelHandle, seed: int = 0) -> dict:
    def gen(rng):
        task, ans, wrong = rng.choice(_AFFORDANCE_CASES)
        clean = f"To {task}, the best tool to use is a"
        other_task, _, _ = rng.choice([a for a in _AFFORDANCE_CASES if a[0] != task])
        corrupted = f"To {other_task}, the best tool to use is a"
        return clean, corrupted, ans, wrong
    return _make_task(handle, "Commonsense: object affordance", "Angle 9: Commonsense", gen, seed=seed)


_CAUSAL_CASES = [
    ("He dropped the glass on the tile floor and it", "shattered", "floated"),
    ("She left the ice cream in the hot car and it", "melted", "hardened"),
    ("He forgot to water the plant for weeks and it", "wilted", "bloomed"),
    ("She pushed the cup off the edge of the table and it", "fell", "rose"),
]


def causal_commonsense_behavior(handle: adapter.ModelHandle, seed: int = 0) -> dict:
    def gen(rng):
        clean, ans, wrong = rng.choice(_CAUSAL_CASES)
        other = rng.choice([c for c in _CAUSAL_CASES if c[0] != clean])
        corrupted = other[0]
        return clean, corrupted, ans, wrong
    return _make_task(handle, "Commonsense: causal outcome", "Angle 9: Commonsense", gen, seed=seed)


_MATERIAL_CASES = [
    ("ice cube left in the sun", "melt", "freeze"),
    ("cup of water put in the freezer", "freeze", "melt"),
    ("piece of paper held over a flame", "burn", "cool"),
    ("balloon left in a very hot room", "expand", "shrink"),
]


def material_property_behavior(handle: adapter.ModelHandle, seed: int = 0) -> dict:
    def gen(rng):
        subj, ans, wrong = rng.choice(_MATERIAL_CASES)
        clean = f"A {subj} will most likely"
        other, _, _ = rng.choice([m for m in _MATERIAL_CASES if m[0] != subj])
        corrupted = f"A {other} will most likely"
        return clean, corrupted, ans, wrong
    return _make_task(handle, "Commonsense: material properties", "Angle 9: Commonsense", gen, seed=seed)


_PURPOSE_CASES = [
    ("umbrella", "dry", "wet"), ("blanket", "warm", "cold"),
    ("fan", "cool", "hot"), ("lamp", "bright", "dark"),
]


def object_purpose_behavior(handle: adapter.ModelHandle, seed: int = 0) -> dict:
    def gen(rng):
        obj, ans, wrong = rng.choice(_PURPOSE_CASES)
        clean = f"People use an {obj} to stay" if obj[0] in "aeiou" else f"People use a {obj} to stay"
        other_obj, _, _ = rng.choice([p for p in _PURPOSE_CASES if p[0] != obj])
        corrupted = f"People use an {other_obj} to stay" if other_obj[0] in "aeiou" else f"People use a {other_obj} to stay"
        return clean, corrupted, ans, wrong
    return _make_task(handle, "Commonsense: object purpose", "Angle 9: Commonsense", gen, seed=seed)


_SIZE_CASES = [
    ("An elephant", "mouse", "mountain"), ("A skyscraper", "house", "atom"),
    ("A whale", "goldfish", "ocean"), ("A truck", "bicycle", "planet"),
]


def size_commonsense_behavior(handle: adapter.ModelHandle, seed: int = 0) -> dict:
    def gen(rng):
        subj, smaller, wrong = rng.choice(_SIZE_CASES)
        clean = f"{subj} is much larger than a"
        other_subj, _, _ = rng.choice([s for s in _SIZE_CASES if s[0] != subj])
        corrupted = f"{other_subj} is much larger than a"
        return clean, corrupted, smaller, wrong
    return _make_task(handle, "Commonsense: relative size", "Angle 9: Commonsense", gen, seed=seed)


# ============================================================================
# ANGLE 10: Multilingual / Cross-lingual
# ============================================================================

_FRENCH_NUMBERS = [
    ("un", "deux", "trois"), ("deux", "trois", "quatre"),
    ("trois", "quatre", "cinq"), ("quatre", "cinq", "six"),
]


def french_number_behavior(handle: adapter.ModelHandle, seed: int = 0) -> dict:
    def gen(rng):
        cur, nxt, wrong = rng.choice(_FRENCH_NUMBERS)
        clean = f"In French counting, the number right after '{cur}' is"
        other_cur, _, _ = rng.choice([f for f in _FRENCH_NUMBERS if f[0] != cur])
        corrupted = f"In French counting, the number right after '{other_cur}' is"
        return clean, corrupted, nxt, wrong
    return _make_task(handle, "Multilingual: French number sequence", "Angle 10: Multilingual", gen, seed=seed)


_SPANISH_WORDS = [
    ("hello", "hola", "adios"), ("goodbye", "adios", "hola"),
    ("thank you", "gracias", "hola"), ("water", "agua", "gracias"),
]


def spanish_vocab_behavior(handle: adapter.ModelHandle, seed: int = 0) -> dict:
    def gen(rng):
        eng, spa, wrong = rng.choice(_SPANISH_WORDS)
        clean = f"The Spanish word for '{eng}' is"
        other_eng, _, _ = rng.choice([s for s in _SPANISH_WORDS if s[0] != eng])
        corrupted = f"The Spanish word for '{other_eng}' is"
        return clean, corrupted, spa, wrong
    return _make_task(handle, "Multilingual: Spanish vocabulary", "Angle 10: Multilingual", gen, seed=seed)


_GERMAN_COLORS = [
    ("red", "rot", "blau"), ("blue", "blau", "rot"),
    ("green", "gruen", "gelb"), ("yellow", "gelb", "gruen"),
]


def german_color_behavior(handle: adapter.ModelHandle, seed: int = 0) -> dict:
    def gen(rng):
        eng, ger, wrong = rng.choice(_GERMAN_COLORS)
        clean = f"The German word for the color '{eng}' is"
        other_eng, _, _ = rng.choice([g for g in _GERMAN_COLORS if g[0] != eng])
        corrupted = f"The German word for the color '{other_eng}' is"
        return clean, corrupted, ger, wrong
    return _make_task(handle, "Multilingual: German color words", "Angle 10: Multilingual", gen, seed=seed)


_ITALIAN_WORDS = [
    ("bread", "pane", "acqua"), ("water", "acqua", "pane"),
    ("cheese", "formaggio", "vino"), ("wine", "vino", "formaggio"),
]


def italian_vocab_behavior(handle: adapter.ModelHandle, seed: int = 0) -> dict:
    def gen(rng):
        eng, ita, wrong = rng.choice(_ITALIAN_WORDS)
        clean = f"The Italian word for '{eng}' is"
        other_eng, _, _ = rng.choice([i for i in _ITALIAN_WORDS if i[0] != eng])
        corrupted = f"The Italian word for '{other_eng}' is"
        return clean, corrupted, ita, wrong
    return _make_task(handle, "Multilingual: Italian vocabulary", "Angle 10: Multilingual", gen, seed=seed)


_LANG_OF_WORD = [
    ("bonjour", "French", "German"), ("hallo", "German", "French"),
    ("ciao", "Italian", "Spanish"), ("gracias", "Spanish", "Italian"),
]


def language_identification_behavior(handle: adapter.ModelHandle, seed: int = 0) -> dict:
    def gen(rng):
        word, lang, wrong = rng.choice(_LANG_OF_WORD)
        clean = f"The word '{word}' comes from the language called"
        other_word, _, _ = rng.choice([l for l in _LANG_OF_WORD if l[0] != word])
        corrupted = f"The word '{other_word}' comes from the language called"
        return clean, corrupted, lang, wrong
    return _make_task(handle, "Multilingual: language identification", "Angle 10: Multilingual", gen, seed=seed)


# ============================================================================
# ANGLE 11: Entity Tracking & Discourse
# ============================================================================

_COREF_CASES = [
    ("John handed the notes to Mark because", "he", "John", "Mark"),
    ("Sarah thanked Emma because", "she", "Sarah", "Emma"),
]


def pronoun_coreference_behavior(handle: adapter.ModelHandle, seed: int = 0) -> dict:
    def gen(rng):
        clean, pron, subj, obj = rng.choice(_COREF_CASES)
        clean_prompt = f"{clean} {pron} needed them. The pronoun '{pron}' refers to"
        other = rng.choice([c for c in _COREF_CASES if c[0] != clean])
        corrupted = f"{other[0]} {other[1]} needed them. The pronoun '{other[1]}' refers to"
        return clean_prompt, corrupted, subj, obj
    return _make_task(handle, "Discourse: pronoun coreference", "Angle 11: Entity Tracking", gen, seed=seed)


_POSSESSION_CASES = [
    ("Alice had 3 apples. She gave 1 to Bob.", "Alice", "2", "3"),
    ("Tom had 5 coins. He lost 2 of them.", "Tom", "3", "5"),
    ("Mia had 4 pens. She bought 1 more.", "Mia", "5", "4"),
    ("Sam had 6 cards. He traded away 3.", "Sam", "3", "6"),
]


def possession_tracking_behavior(handle: adapter.ModelHandle, seed: int = 0) -> dict:
    def gen(rng):
        text, who, ans, wrong = rng.choice(_POSSESSION_CASES)
        clean = f"{text} {who} now has"
        other = rng.choice([p for p in _POSSESSION_CASES if p[0] != text])
        corrupted = f"{other[0]} {other[1]} now has"
        return clean, corrupted, ans, wrong
    return _make_task(handle, "Entity Tracking: possession count", "Angle 11: Entity Tracking", gen, seed=seed)


_LOCATION_CASES = [
    ("The keys were on the table, then Ann moved them to the drawer.", "drawer", "table"),
    ("The ball was in the box, then Ben moved it under the bed.", "bed", "box"),
    ("The letter was in the bag, then Cara put it on the shelf.", "shelf", "bag"),
    ("The phone was on the couch, then Dan placed it in his pocket.", "pocket", "couch"),
]


def location_tracking_behavior(handle: adapter.ModelHandle, seed: int = 0) -> dict:
    def gen(rng):
        text, ans, wrong = rng.choice(_LOCATION_CASES)
        clean = f"{text} The object is now in the"
        other, _, _ = rng.choice([l for l in _LOCATION_CASES if l[0] != text])
        corrupted = f"{other} The object is now in the"
        return clean, corrupted, ans, wrong
    return _make_task(handle, "Entity Tracking: object location", "Angle 11: Entity Tracking", gen, seed=seed)


_ROLE_CASES = [
    ("At the meeting, Dr. Lee spoke first and Mr. Kim took notes.", "Lee", "Kim"),
    ("On the trip, Maria drove the car while Jose read the map.", "Maria", "Jose"),
]


def role_tracking_behavior(handle: adapter.ModelHandle, seed: int = 0) -> dict:
    def gen(rng):
        text, speaker, other_person = rng.choice(_ROLE_CASES)
        clean = f"{text} The person who spoke first was"
        other = rng.choice([r for r in _ROLE_CASES if r[0] != text])
        corrupted = f"{other[0]} The person who spoke first was"
        return clean, corrupted, speaker, other_person
    return _make_task(handle, "Entity Tracking: role assignment", "Angle 11: Entity Tracking", gen, seed=seed)


_LAST_MENTIONED_CASES = [
    ("The report mentioned Paris, then Rome, and finally", "Berlin", "Paris"),
    ("She visited Anna, then Beth, and last of all", "Clara", "Anna"),
]


def recency_tracking_behavior(handle: adapter.ModelHandle, seed: int = 0) -> dict:
    def gen(rng):
        clean, ans, wrong = rng.choice(_LAST_MENTIONED_CASES)
        other = rng.choice([r for r in _LAST_MENTIONED_CASES if r[0] != clean])
        return clean, other[0], ans, wrong
    return _make_task(handle, "Discourse: recency completion", "Angle 11: Entity Tracking", gen, seed=seed)


# ============================================================================
# ANGLE 12: Negation & Logic
# ============================================================================

_NEGATION_CASES = [
    ("The statement 'grass is not blue' is", "true", "false"),
    ("The statement 'the sun is not hot' is", "false", "true"),
    ("The statement 'fish cannot fly' is", "true", "false"),
    ("The statement 'ice is not cold' is", "false", "true"),
]


def negation_behavior(handle: adapter.ModelHandle, seed: int = 0) -> dict:
    def gen(rng):
        clean, ans, wrong = rng.choice(_NEGATION_CASES)
        other = rng.choice([n for n in _NEGATION_CASES if n[0] != clean])
        return clean, other[0], ans, wrong
    return _make_task(handle, "Logic: negation truth value", "Angle 12: Negation & Logic", gen, seed=seed)


_DOUBLE_NEGATION_CASES = [
    ("It is not true that cats are not animals, so cats are", "animals", "plants"),
    ("It is not false that water is wet, so water is", "wet", "dry"),
]


def double_negation_behavior(handle: adapter.ModelHandle, seed: int = 0) -> dict:
    def gen(rng):
        clean, ans, wrong = rng.choice(_DOUBLE_NEGATION_CASES)
        other = rng.choice([d for d in _DOUBLE_NEGATION_CASES if d[0] != clean])
        return clean, other[0], ans, wrong
    return _make_task(handle, "Logic: double negation", "Angle 12: Negation & Logic", gen, seed=seed)


_QUANTIFIER_CASES = [
    ("All dogs are mammals. A dog is therefore a", "mammal", "reptile"),
    ("All roses are flowers. A rose is therefore a", "flower", "vegetable"),
    ("All squares are shapes. A square is therefore a", "shape", "number"),
    ("All sparrows are birds. A sparrow is therefore a", "bird", "fish"),
]


def quantifier_reasoning_behavior(handle: adapter.ModelHandle, seed: int = 0) -> dict:
    def gen(rng):
        clean, ans, wrong = rng.choice(_QUANTIFIER_CASES)
        other = rng.choice([q for q in _QUANTIFIER_CASES if q[0] != clean])
        return clean, other[0], ans, wrong
    return _make_task(handle, "Logic: universal quantifier", "Angle 12: Negation & Logic", gen, seed=seed)


_CONDITIONAL_CASES = [
    ("If it rains, the ground gets wet. It is raining, so the ground is", "wet", "dry"),
    ("If the alarm rings, people leave. The alarm rang, so people", "left", "stayed"),
    ("If the switch is on, the light glows. The switch is on, so the light", "glows", "dims"),
    ("If you heat ice, it melts. The ice was heated, so it", "melted", "froze"),
]


def conditional_reasoning_behavior(handle: adapter.ModelHandle, seed: int = 0) -> dict:
    def gen(rng):
        clean, ans, wrong = rng.choice(_CONDITIONAL_CASES)
        other = rng.choice([c for c in _CONDITIONAL_CASES if c[0] != clean])
        return clean, other[0], ans, wrong
    return _make_task(handle, "Logic: modus ponens", "Angle 12: Negation & Logic", gen, seed=seed)


_CONTRADICTION_CASES = [
    ("The box is completely empty and also completely full. This is a", "contradiction", "fact"),
    ("Water is made of hydrogen and oxygen. This is a", "fact", "contradiction"),
    ("The line is perfectly straight and perfectly circular. This is a", "contradiction", "fact"),
    ("The Earth orbits the Sun. This is a", "fact", "contradiction"),
]


def contradiction_detection_behavior(handle: adapter.ModelHandle, seed: int = 0) -> dict:
    def gen(rng):
        clean, ans, wrong = rng.choice(_CONTRADICTION_CASES)
        other = rng.choice([c for c in _CONTRADICTION_CASES if c[0] != clean])
        return clean, other[0], ans, wrong
    return _make_task(handle, "Logic: contradiction detection", "Angle 12: Negation & Logic", gen, seed=seed)


# ============================================================================
# ANGLE 13: Quantitative Comparison
# ============================================================================

ANGLE_13_BEHAVIORS = [
    _pair_task("Quantitative: more vs fewer", "Angle 13: Quantitative", [
        ("A library holds far more books than a", "A pocket holds far fewer coins than a", "shelf", "warehouse"),
        ("A city has many more people than a", "A hamlet has many fewer people than a", "village", "country"),
        ("An ocean contains much more water than a", "A cup contains much less water than a", "pond", "sea"),
        ("A forest has many more trees than a", "A pot has fewer plants than a", "garden", "jungle"),
    ]),
    _pair_task("Quantitative: heavier vs lighter", "Angle 13: Quantitative", [
        ("Between a feather and a brick, the heavier object is the", "Between a feather and a brick, the lighter object is the", "brick", "feather"),
        ("Between a car and a bicycle, the heavier one is the", "Between a car and a bicycle, the lighter one is the", "car", "bicycle"),
        ("Between an elephant and a cat, the heavier animal is the", "Between an elephant and a cat, the lighter animal is the", "elephant", "cat"),
        ("Between a stone and a leaf, the heavier thing is the", "Between a stone and a leaf, the lighter thing is the", "stone", "leaf"),
    ]),
    _pair_task("Quantitative: taller vs shorter", "Angle 13: Quantitative", [
        ("Between a tower and a hut, the taller building is the", "Between a tower and a hut, the shorter building is the", "tower", "hut"),
        ("Between a giraffe and a dog, the taller animal is the", "Between a giraffe and a dog, the shorter animal is the", "giraffe", "dog"),
        ("Between a tree and a bush, the taller plant is the", "Between a tree and a bush, the shorter plant is the", "tree", "bush"),
        ("Between an adult and a toddler, the taller person is the", "Between an adult and a toddler, the shorter person is the", "adult", "toddler"),
    ]),
    _pair_task("Quantitative: faster vs slower", "Angle 13: Quantitative", [
        ("Between a cheetah and a turtle, the faster animal is the", "Between a cheetah and a turtle, the slower animal is the", "cheetah", "turtle"),
        ("Between a jet and a bicycle, the faster vehicle is the", "Between a jet and a bicycle, the slower vehicle is the", "jet", "bicycle"),
        ("Between a rabbit and a snail, the faster animal is the", "Between a rabbit and a snail, the slower animal is the", "rabbit", "snail"),
        ("Between a rocket and a car, the faster machine is the", "Between a rocket and a car, the slower machine is the", "rocket", "car"),
    ]),
    _pair_task("Quantitative: hotter vs colder", "Angle 13: Quantitative", [
        ("Between fire and ice, the hotter thing is", "Between fire and ice, the colder thing is", "fire", "ice"),
        ("Between the desert and the arctic, the hotter place is the", "Between the desert and the arctic, the colder place is the", "desert", "arctic"),
        ("Between summer and winter, the hotter season is", "Between summer and winter, the colder season is", "summer", "winter"),
        ("Between boiling water and snow, the hotter one is", "Between boiling water and snow, the colder one is", "water", "snow"),
    ]),
    _pair_task("Quantitative: older vs younger", "Angle 13: Quantitative", [
        ("Between a grandmother and a baby, the older person is the", "Between a grandmother and a baby, the younger person is the", "grandmother", "baby"),
        ("Between an oak that is 200 years old and a sapling, the older tree is the", "Between an oak that is 200 years old and a sapling, the younger tree is the", "oak", "sapling"),
        ("Between a senior and a freshman, the older student is the", "Between a senior and a freshman, the younger student is the", "senior", "freshman"),
        ("Between a parent and a child, the older person is the", "Between a parent and a child, the younger person is the", "parent", "child"),
    ]),
    _pair_task("Quantitative: nearer vs farther", "Angle 13: Quantitative", [
        ("Between the Moon and the Sun, the object nearer to Earth is the", "Between the Moon and the Sun, the object farther from Earth is the", "Moon", "Sun"),
        ("Between the corner shop and another city, the nearer place is the", "Between the corner shop and another city, the farther place is the", "shop", "city"),
        ("Between your neighbor and a stranger overseas, the nearer person is the", "Between your neighbor and a stranger overseas, the farther person is the", "neighbor", "stranger"),
        ("Between the next room and the next country, the nearer location is the", "Between the next room and the next country, the farther location is the", "room", "country"),
    ]),
    _pair_task("Quantitative: fraction magnitude", "Angle 13: Quantitative", [
        ("Between one half and one quarter, the larger fraction is one", "Between one half and one quarter, the smaller fraction is one", "half", "quarter"),
        ("Between three quarters and one third, the larger fraction is three", "Between three quarters and one third, the smaller fraction is one", "quarters", "third"),
        ("Between one whole and one half, the larger amount is one", "Between one whole and one half, the smaller amount is one", "whole", "half"),
        ("Between two thirds and one sixth, the larger fraction is two", "Between two thirds and one sixth, the smaller fraction is one", "thirds", "sixth"),
    ]),
    _pair_task("Quantitative: percentage of a number", "Angle 13: Quantitative", [
        ("Fifty percent of ten is", "Ten percent of ten is", "5", "1"),
        ("One hundred percent of seven is", "Zero percent of seven is", "7", "0"),
        ("Twenty five percent of eight is", "Fifty percent of eight is", "2", "4"),
        ("Ten percent of one hundred is", "One percent of one hundred is", "10", "1"),
    ]),
    _pair_task("Quantitative: grouped counts", "Angle 13: Quantitative", [
        ("A dozen eggs is a count of", "A pair of eggs is a count of", "12", "2"),
        ("A pair of gloves is a count of", "A dozen gloves is a count of", "2", "12"),
        ("Half a dozen is a count of", "A dozen is a count of", "6", "12"),
        ("A single item is a count of", "A pair of items is a count of", "1", "2"),
    ]),
]


# ============================================================================
# ANGLE 14: Temporal & Sequential Ordering
# ============================================================================

ANGLE_14_BEHAVIORS = [
    _pair_task("Temporal: month after", "Angle 14: Temporal Ordering", [
        ("The month right after January is", "The month right before January is", "February", "December"),
        ("The month right after June is", "The month right before June is", "July", "May"),
        ("The month right after September is", "The month right before September is", "October", "August"),
        ("The month right after March is", "The month right before March is", "April", "February"),
    ]),
    _pair_task("Temporal: month before", "Angle 14: Temporal Ordering", [
        ("The month right before May is", "The month right after May is", "April", "June"),
        ("The month right before November is", "The month right after November is", "October", "December"),
        ("The month right before August is", "The month right after August is", "July", "September"),
        ("The month right before February is", "The month right after February is", "January", "March"),
    ]),
    _pair_task("Temporal: season after", "Angle 14: Temporal Ordering", [
        ("The season that comes after winter is", "The season that comes before winter is", "spring", "autumn"),
        ("The season that comes after summer is", "The season that comes before summer is", "autumn", "spring"),
        ("The season that comes after spring is", "The season that comes before spring is", "summer", "winter"),
        ("The season that comes after autumn is", "The season that comes before autumn is", "winter", "summer"),
    ]),
    _pair_task("Temporal: day before", "Angle 14: Temporal Ordering", [
        ("The day right before Friday is", "The day right after Friday is", "Thursday", "Saturday"),
        ("The day right before Monday is", "The day right after Monday is", "Sunday", "Tuesday"),
        ("The day right before Wednesday is", "The day right after Wednesday is", "Tuesday", "Thursday"),
        ("The day right before Sunday is", "The day right after Sunday is", "Saturday", "Monday"),
    ]),
    _pair_task("Temporal: earlier vs later year", "Angle 14: Temporal Ordering", [
        ("Between the years 1990 and 2020, the earlier year is", "Between the years 1990 and 2020, the later year is", "1990", "2020"),
        ("Between the years 1750 and 1850, the earlier year is", "Between the years 1750 and 1850, the later year is", "1750", "1850"),
        ("Between the years 2001 and 2015, the earlier year is", "Between the years 2001 and 2015, the later year is", "2001", "2015"),
        ("Between the years 1500 and 1600, the earlier year is", "Between the years 1500 and 1600, the later year is", "1500", "1600"),
    ]),
    _pair_task("Temporal: part of day order", "Angle 14: Temporal Ordering", [
        ("The part of the day that comes after morning is", "The part of the day that comes before morning is", "afternoon", "night"),
        ("The part of the day that comes after afternoon is", "The part of the day that comes before afternoon is", "evening", "morning"),
        ("The part of the day that comes after evening is", "The part of the day that comes before evening is", "night", "afternoon"),
        ("The part of the day that comes after dawn is", "The part of the day that comes before dawn is", "morning", "night"),
    ]),
    _pair_task("Temporal: decade ordering", "Angle 14: Temporal Ordering", [
        ("The decade that comes after the 1960s is the", "The decade that comes before the 1960s is the", "1970s", "1950s"),
        ("The decade that comes after the 1990s is the", "The decade that comes before the 1990s is the", "2000s", "1980s"),
        ("The decade that comes after the 1920s is the", "The decade that comes before the 1920s is the", "1930s", "1910s"),
        ("The decade that comes after the 2000s is the", "The decade that comes before the 2000s is the", "2010s", "1990s"),
    ]),
    _pair_task("Temporal: hour after on a clock", "Angle 14: Temporal Ordering", [
        ("One hour after three o'clock is", "One hour before three o'clock is", "four", "two"),
        ("One hour after nine o'clock is", "One hour before nine o'clock is", "ten", "eight"),
        ("One hour after eleven o'clock is", "One hour before eleven o'clock is", "twelve", "ten"),
        ("One hour after six o'clock is", "One hour before six o'clock is", "seven", "five"),
    ]),
    _pair_task("Temporal: calendar quarter", "Angle 14: Temporal Ordering", [
        ("The quarter of the year that comes after the first is the", "The quarter of the year that comes before the first is the", "second", "fourth"),
        ("The quarter of the year that comes after the third is the", "The quarter of the year that comes before the third is the", "fourth", "second"),
        ("The quarter of the year that comes after the second is the", "The quarter of the year that comes before the second is the", "third", "first"),
        ("The quarter of the year that contains April is the", "The quarter of the year that contains January is the", "second", "first"),
    ]),
    _pair_task("Temporal: life cycle order", "Angle 14: Temporal Ordering", [
        ("In a butterfly's life cycle, the stage after the caterpillar is the", "In a butterfly's life cycle, the stage before the caterpillar is the", "chrysalis", "egg"),
        ("In a frog's life cycle, the stage after the tadpole is the", "In a frog's life cycle, the stage before the tadpole is the", "frog", "egg"),
        ("In a plant's life cycle, the stage after the seed is the", "In a plant's life cycle, the stage before the seed is the", "sprout", "flower"),
        ("In a human life, the stage after infancy is", "In a human life, the stage before infancy is", "childhood", "birth"),
    ]),
]


# ============================================================================
# ANGLE 15: Analogical Reasoning
# ============================================================================

ANGLE_15_BEHAVIORS = [
    _pair_task("Analogy: gendered pairs", "Angle 15: Analogy", [
        ("man is to woman as king is to", "woman is to man as queen is to", "queen", "king"),
        ("father is to mother as son is to", "mother is to father as daughter is to", "daughter", "son"),
        ("uncle is to aunt as nephew is to", "aunt is to uncle as niece is to", "niece", "nephew"),
        ("actor is to actress as prince is to", "actress is to actor as princess is to", "princess", "prince"),
    ]),
    _pair_task("Analogy: animal to young", "Angle 15: Analogy", [
        ("dog is to puppy as cat is to", "puppy is to dog as kitten is to", "kitten", "cat"),
        ("cow is to calf as horse is to", "calf is to cow as foal is to", "foal", "horse"),
        ("sheep is to lamb as goat is to", "lamb is to sheep as kid is to", "kid", "goat"),
        ("bear is to cub as lion is to", "cub is to bear as cub is to", "cub", "lion"),
    ]),
    _pair_task("Analogy: opposites", "Angle 15: Analogy", [
        ("hot is to cold as up is to", "cold is to hot as down is to", "down", "up"),
        ("day is to night as light is to", "night is to day as dark is to", "dark", "light"),
        ("fast is to slow as big is to", "slow is to fast as small is to", "small", "big"),
        ("open is to closed as start is to", "closed is to open as stop is to", "stop", "start"),
    ]),
    _pair_task("Analogy: capital to country", "Angle 15: Analogy", [
        ("Paris is to France as Rome is to", "France is to Paris as Italy is to", "Italy", "Rome"),
        ("Tokyo is to Japan as Berlin is to", "Japan is to Tokyo as Germany is to", "Germany", "Berlin"),
        ("Madrid is to Spain as Lisbon is to", "Spain is to Madrid as Portugal is to", "Portugal", "Lisbon"),
        ("London is to England as Cairo is to", "England is to London as Egypt is to", "Egypt", "Cairo"),
    ]),
    _pair_task("Analogy: object to cover", "Angle 15: Analogy", [
        ("hand is to glove as foot is to", "glove is to hand as sock is to", "sock", "foot"),
        ("head is to hat as eye is to", "hat is to head as glasses is to", "glasses", "eye"),
        ("bed is to blanket as window is to", "blanket is to bed as curtain is to", "curtain", "window"),
        ("book is to cover as letter is to", "cover is to book as envelope is to", "envelope", "letter"),
    ]),
    _pair_task("Analogy: animal to movement", "Angle 15: Analogy", [
        ("bird is to fly as fish is to", "fish is to swim as bird is to", "swim", "fly"),
        ("snake is to slither as kangaroo is to", "kangaroo is to hop as snake is to", "hop", "slither"),
        ("horse is to gallop as frog is to", "frog is to jump as horse is to", "jump", "gallop"),
        ("eagle is to soar as dolphin is to", "dolphin is to swim as eagle is to", "swim", "soar"),
    ]),
    _pair_task("Analogy: creator to work", "Angle 15: Analogy", [
        ("author is to novel as composer is to", "composer is to symphony as author is to", "symphony", "novel"),
        ("painter is to painting as sculptor is to", "sculptor is to statue as painter is to", "statue", "painting"),
        ("poet is to poem as director is to", "director is to film as poet is to", "film", "poem"),
        ("chef is to meal as baker is to", "baker is to bread as chef is to", "bread", "meal"),
    ]),
    _pair_task("Analogy: degree of adjective", "Angle 15: Analogy", [
        ("big is to bigger as small is to", "bigger is to big as smaller is to", "smaller", "small"),
        ("good is to better as bad is to", "better is to good as worse is to", "worse", "bad"),
        ("fast is to faster as slow is to", "faster is to fast as slower is to", "slower", "slow"),
        ("high is to higher as low is to", "higher is to high as lower is to", "lower", "low"),
    ]),
    _pair_task("Analogy: worker to workplace", "Angle 15: Analogy", [
        ("teacher is to school as doctor is to", "doctor is to hospital as teacher is to", "hospital", "school"),
        ("chef is to kitchen as judge is to", "judge is to courtroom as chef is to", "courtroom", "kitchen"),
        ("pilot is to cockpit as farmer is to", "farmer is to farm as pilot is to", "farm", "cockpit"),
        ("actor is to stage as scientist is to", "scientist is to laboratory as actor is to", "laboratory", "stage"),
    ]),
    _pair_task("Analogy: part to whole", "Angle 15: Analogy", [
        ("finger is to hand as toe is to", "hand is to finger as foot is to", "foot", "toe"),
        ("petal is to flower as leaf is to", "flower is to petal as tree is to", "tree", "leaf"),
        ("wheel is to car as wing is to", "car is to wheel as plane is to", "plane", "wing"),
        ("page is to book as brick is to", "book is to page as wall is to", "wall", "brick"),
    ]),
]


# ============================================================================
# ANGLE 16: Morphology & Word Formation
# ============================================================================

ANGLE_16_BEHAVIORS = [
    _pair_task("Morphology: irregular plural", "Angle 16: Morphology", [
        ("The plural of the word child is", "The singular of the word children is", "children", "child"),
        ("The plural of the word mouse is", "The singular of the word mice is", "mice", "mouse"),
        ("The plural of the word foot is", "The singular of the word feet is", "feet", "foot"),
        ("The plural of the word tooth is", "The singular of the word teeth is", "teeth", "tooth"),
    ]),
    _pair_task("Morphology: irregular past tense", "Angle 16: Morphology", [
        ("The past tense of the verb go is", "The present tense of the verb went is", "went", "go"),
        ("The past tense of the verb see is", "The present tense of the verb saw is", "saw", "see"),
        ("The past tense of the verb take is", "The present tense of the verb took is", "took", "take"),
        ("The past tense of the verb bring is", "The present tense of the verb brought is", "brought", "bring"),
    ]),
    _pair_task("Morphology: irregular comparative", "Angle 16: Morphology", [
        ("The comparative form of the word good is", "The comparative form of the word bad is", "better", "worse"),
        ("The comparative form of the word bad is", "The comparative form of the word good is", "worse", "better"),
        ("The comparative form of the word far is", "The comparative form of the word near is", "farther", "nearer"),
        ("The comparative form of the word little is", "The comparative form of the word much is", "less", "more"),
    ]),
    _pair_task("Morphology: superlative", "Angle 16: Morphology", [
        ("The superlative form of the word good is", "The comparative form of the word good is", "best", "better"),
        ("The superlative form of the word bad is", "The comparative form of the word bad is", "worst", "worse"),
        ("The superlative form of the word tall is", "The comparative form of the word tall is", "tallest", "taller"),
        ("The superlative form of the word big is", "The comparative form of the word big is", "biggest", "bigger"),
    ]),
    _pair_task("Morphology: -ness nominalization", "Angle 16: Morphology", [
        ("Adding the suffix -ness to the adjective happy gives the noun", "Removing the suffix -ness from happiness gives the adjective", "happiness", "happy"),
        ("Adding the suffix -ness to the adjective dark gives the noun", "Removing the suffix -ness from darkness gives the adjective", "darkness", "dark"),
        ("Adding the suffix -ness to the adjective kind gives the noun", "Removing the suffix -ness from kindness gives the adjective", "kindness", "kind"),
        ("Adding the suffix -ness to the adjective weak gives the noun", "Removing the suffix -ness from weakness gives the adjective", "weakness", "weak"),
    ]),
    _pair_task("Morphology: un- prefixation", "Angle 16: Morphology", [
        ("Adding the prefix un- to the word happy gives", "The word happy without any prefix is", "unhappy", "happy"),
        ("Adding the prefix un- to the word fair gives", "The word fair without any prefix is", "unfair", "fair"),
        ("Adding the prefix un- to the word kind gives", "The word kind without any prefix is", "unkind", "kind"),
        ("Adding the prefix un- to the word lock gives", "The word lock without any prefix is", "unlock", "lock"),
    ]),
    _pair_task("Morphology: -er agent noun", "Angle 16: Morphology", [
        ("A person whose job is to teach is a", "The base verb for teaching, rather than the person doing it, is", "teacher", "teach"),
        ("A person whose job is to bake is a", "The base verb for baking, rather than the person doing it, is", "baker", "bake"),
        ("A person who plays a sport is a", "The base verb for playing, rather than the person doing it, is", "player", "play"),
        ("A person who writes is a", "The base verb for writing, rather than the person doing it, is", "writer", "write"),
    ]),
    _pair_task("Morphology: -ly adverb", "Angle 16: Morphology", [
        ("The adverb formed from the adjective quick is", "The adjective corresponding to quickly is", "quickly", "quick"),
        ("The adverb formed from the adjective slow is", "The adjective corresponding to slowly is", "slowly", "slow"),
        ("The adverb formed from the adjective happy is", "The adjective corresponding to happily is", "happily", "happy"),
        ("The adverb formed from the adjective soft is", "The adjective corresponding to softly is", "softly", "soft"),
    ]),
    _pair_task("Morphology: past participle", "Angle 16: Morphology", [
        ("The past participle of the verb eat, as in 'has ___', is", "The base form corresponding to eaten is", "eaten", "eat"),
        ("The past participle of the verb write, as in 'has ___', is", "The base form corresponding to written is", "written", "write"),
        ("The past participle of the verb break, as in 'has ___', is", "The base form corresponding to broken is", "broken", "break"),
        ("The past participle of the verb give, as in 'has ___', is", "The base form corresponding to given is", "given", "give"),
    ]),
    _pair_task("Morphology: compound head", "Angle 16: Morphology", [
        ("In the compound word sunflower, the main thing it names is a kind of", "In the compound word sunflower, the modifier part refers to the", "flower", "sun"),
        ("In the compound word toothbrush, the main thing it names is a kind of", "In the compound word toothbrush, the modifier part refers to the", "brush", "tooth"),
        ("In the compound word railway, the main thing it names is a kind of", "In the compound word railway, the modifier part refers to the", "way", "rail"),
        ("In the compound word bedroom, the main thing it names is a kind of", "In the compound word bedroom, the modifier part refers to the", "room", "bed"),
    ]),
]


# ============================================================================
# ANGLE 17: Pragmatics & Implicature
# ============================================================================

ANGLE_17_BEHAVIORS = [
    _pair_task("Pragmatics: scalar implicature (some)", "Angle 17: Pragmatics", [
        ("She said 'some of the students passed', which suggests that not all of them", "She said 'all of the students passed', which means every one of them", "passed", "failed"),
        ("He ate some of the cookies, so some cookies are still", "He ate all of the cookies, so no cookies are still", "left", "gone"),
        ("They finished some of the work, so part of it is still", "They finished all of the work, so none of it is still", "unfinished", "done"),
        ("I read some of the book, so I have not read the", "I read all of the book, so I finished the", "rest", "whole"),
    ]),
    _pair_task("Pragmatics: sarcasm", "Angle 17: Pragmatics", [
        ("After the meeting was cancelled again he sighed, 'Oh great, wonderful.' He actually feels", "When he finally got the promotion he cheered, 'This is wonderful!' He actually feels", "annoyed", "delighted"),
        ("Stuck in traffic for hours she muttered, 'Perfect, just perfect.' Her real mood is", "Handed a surprise gift she beamed, 'This is perfect!' Her real mood is", "frustrated", "thrilled"),
        ("When it started raining on his picnic he said, 'Lovely weather.' He is being", "Watching a beautiful sunset she said, 'Lovely weather.' She is being", "sarcastic", "sincere"),
        ("Seeing the huge bill he laughed, 'What a bargain.' He means it is actually", "Seeing the discount she laughed, 'What a bargain.' She means it is actually", "expensive", "cheap"),
    ]),
    _pair_task("Pragmatics: indirect request", "Angle 17: Pragmatics", [
        ("At the table she said 'It's a bit cold in here', hinting that someone should close the", "At the table she said 'It's a bit warm in here', hinting that someone should open the", "window", "door"),
        ("He said 'I can't reach the salt', which is really a request to pass the", "He said 'I already have enough salt', which is really a request to keep the", "salt", "pepper"),
        ("She said 'the trash is really full', which is really asking someone to take it", "She said 'the trash is still empty', which means no one needs to take it", "out", "in"),
        ("He said 'is there any coffee left?', which is really asking someone to make some", "He said 'I've had too much coffee', which means he wants no more", "coffee", "water"),
    ]),
    _pair_task("Pragmatics: understatement", "Angle 17: Pragmatics", [
        ("Asked about the five-star meal he said 'not bad', meaning it was actually quite", "Asked about the burnt meal he said 'not great', meaning it was actually quite", "good", "bad"),
        ("After acing the exam she said 'I did okay', which really means she did very", "After failing the exam she said 'I did okay', which really means she did rather", "well", "poorly"),
        ("Describing the freezing weather he said 'a little chilly', meaning it was extremely", "Describing the mild weather he said 'a little chilly', meaning it was only slightly", "cold", "cool"),
        ("About the marathon she said 'a bit tiring', meaning it was in fact utterly", "About the short walk she said 'a bit tiring', meaning it was only mildly", "exhausting", "easy"),
    ]),
    _pair_task("Pragmatics: presupposition", "Angle 17: Pragmatics", [
        ("Saying 'he stopped smoking' presupposes that he used to", "Saying 'he never smoked' means that he did not", "smoke", "quit"),
        ("Saying 'she returned to Paris' presupposes that she had been there", "Saying 'she has never seen Paris' means she has not been there", "before", "never"),
        ("Saying 'they reopened the shop' presupposes that the shop had been", "Saying 'they built a new shop' means the shop was not there", "closed", "open"),
        ("Saying 'I forgot my keys again' presupposes that this has happened", "Saying 'I remembered my keys this time' means last time it went", "before", "fine"),
    ]),
    _pair_task("Pragmatics: relevance answer", "Angle 17: Pragmatics", [
        ("Asked 'Do you know what time it is?', a cooperative reply gives the", "Asked 'Do you own a watch?', a literal reply gives a yes or", "time", "no"),
        ("Asked 'Can you tell me where the station is?', a helpful reply gives", "Asked 'Are you able to speak?', a literal reply is just yes or", "directions", "no"),
        ("Asked 'Have you got a pen?', a cooperative person will hand over a", "Asked 'Do pens exist?', a literal answer is simply", "pen", "yes"),
        ("Asked 'Would you know the way to the airport?', a helpful answer describes the", "Asked 'Is knowledge possible?', a literal answer is just", "route", "yes"),
    ]),
    _pair_task("Pragmatics: politeness register", "Angle 17: Pragmatics", [
        ("'Would you be so kind as to help me?' is phrased to sound", "'Help me right now.' is phrased to sound", "polite", "rude"),
        ("'Might I trouble you for the time?' is phrased to sound", "'Tell me the time.' is phrased to sound", "polite", "blunt"),
        ("'I was wondering if you could possibly move' is phrased to sound", "'Move.' is phrased to sound", "polite", "curt"),
        ("'If it's not too much trouble, could you wait?' is phrased to sound", "'Wait here.' is phrased to sound", "polite", "abrupt"),
    ]),
    _pair_task("Pragmatics: conversational contrast (but)", "Angle 17: Pragmatics", [
        ("'The room was small but' sets up a following comment that is", "'The room was small and' sets up a following comment that is", "positive", "negative"),
        ("'He is talented but' leads the listener to expect something", "'He is talented and' leads the listener to expect something", "negative", "positive"),
        ("'The food was cheap but' prepares the listener for a downside about the", "'The food was cheap and' prepares the listener for another upside about the", "quality", "portions"),
        ("'She tried hard but' signals that the outcome was", "'She tried hard and' signals that the outcome was", "unsuccessful", "successful"),
    ]),
    _pair_task("Pragmatics: implied comparison", "Angle 17: Pragmatics", [
        ("Saying 'even the beginner solved it' implies the puzzle was", "Saying 'only the expert solved it' implies the puzzle was", "easy", "hard"),
        ("Saying 'she finished before the fast runners' implies she is very", "Saying 'she finished after the slow runners' implies she is quite", "fast", "slow"),
        ("Saying 'even a child could do it' implies the task is quite", "Saying 'only an expert could do it' implies the task is quite", "easy", "hard"),
        ("Saying 'it fits even a large adult' implies the space is fairly", "Saying 'it fits only a small child' implies the space is quite", "roomy", "cramped"),
    ]),
    _pair_task("Pragmatics: answer to yes-no as refusal", "Angle 17: Pragmatics", [
        ("Asked to the party, he said 'I have to work late', which really means", "Asked to the party, he said 'I'd love to come', which really means", "no", "yes"),
        ("Asked to lend money, she said 'things are tight right now', which really means", "Asked to lend money, she said 'sure, how much?', which really means", "no", "yes"),
        ("Asked for a date, he said 'I'm really busy these days', which really means", "Asked for a date, he said 'how about Friday?', which really means", "no", "yes"),
        ("Asked to help move, she said 'my back has been bad', which really means", "Asked to help move, she said 'I'll bring my truck', which really means", "no", "yes"),
    ]),
]


# ============================================================================
# ANGLE 18: Scientific & Technical Knowledge
# ============================================================================

ANGLE_18_BEHAVIORS = [
    _pair_task("Science: composition of water", "Angle 18: Science", [
        ("Water is a compound of oxygen and", "Table salt is a compound of chlorine and", "hydrogen", "sodium"),
        ("A molecule of water contains two atoms of", "A molecule of carbon dioxide contains two atoms of", "hydrogen", "oxygen"),
        ("Splitting water by electricity releases oxygen and", "Splitting salt water leaves behind mostly salt and", "hydrogen", "minerals"),
        ("The gas that makes water wet nothing, it is simply oxygen bonded to", "The gas released by baking soda and vinegar is carbon bonded to", "hydrogen", "oxygen"),
    ]),
    _pair_task("Science: photosynthesis", "Angle 18: Science", [
        ("Green plants make their food from sunlight in a process called", "Animals release energy from food in a process called", "photosynthesis", "respiration"),
        ("During photosynthesis a plant takes in carbon dioxide and gives off", "During respiration an animal takes in oxygen and gives off", "oxygen", "carbon"),
        ("The green pigment that captures light for photosynthesis is called", "The red pigment that carries oxygen in blood is called", "chlorophyll", "hemoglobin"),
        ("Photosynthesis mainly happens in a plant's", "Water uptake mainly happens in a plant's", "leaves", "roots"),
    ]),
    _pair_task("Science: the cell", "Angle 18: Science", [
        ("The part of the cell that produces most of its energy is the", "The part of the cell that holds the DNA is the", "mitochondria", "nucleus"),
        ("The control center of a cell, holding its DNA, is the", "The energy factory of a cell is the", "nucleus", "mitochondria"),
        ("The thin outer boundary that controls what enters a cell is the cell", "The rigid outer layer around a plant cell is the cell", "membrane", "wall"),
        ("Plant cells, unlike animal cells, contain green organelles called", "Both plant and animal cells contain energy organelles called", "chloroplasts", "mitochondria"),
    ]),
    _pair_task("Science: astronomy basics", "Angle 18: Science", [
        ("The star at the center of our Solar System is the", "The natural satellite that orbits Earth is the", "Sun", "Moon"),
        ("The force that keeps planets in orbit around the Sun is", "The push that launches a rocket off the ground is", "gravity", "thrust"),
        ("A year is the time Earth takes to orbit the", "A day is the time Earth takes to spin on its", "Sun", "axis"),
        ("The planet we live on is called", "The planet named after the Roman god of war is called", "Earth", "Mars"),
    ]),
    _pair_task("Science: states of matter", "Angle 18: Science", [
        ("When a solid is heated enough it turns into a", "When a gas is cooled enough it turns into a", "liquid", "solid"),
        ("Water in its solid state is called", "Water in its gaseous state is called", "ice", "steam"),
        ("Turning a liquid into a gas is called", "Turning a gas into a liquid is called", "evaporation", "condensation"),
        ("At room temperature, iron is normally a", "At room temperature, oxygen is normally a", "solid", "gas"),
    ]),
    _pair_task("Science: forces and motion", "Angle 18: Science", [
        ("An object dropped near Earth's surface accelerates", "An object thrown straight up first moves", "downward", "upward"),
        ("The force that slows a box sliding across the floor is", "The force that pulls the box toward the ground is", "friction", "gravity"),
        ("Light travels much faster than", "In a race between light and sound, the first to arrive is", "sound", "light"),
        ("Objects fall toward the ground because of the force called", "A sliding box slows and stops because of the force called", "gravity", "friction"),
    ]),
    _pair_task("Science: the human body", "Angle 18: Science", [
        ("The organ that pumps blood around the body is the", "The organ used mainly for thinking is the", "heart", "brain"),
        ("The largest organ of the human body is the", "The organ that filters waste from blood is the", "skin", "kidney"),
        ("Humans breathe in oxygen and breathe out carbon", "Plants take in carbon dioxide and give out", "dioxide", "oxygen"),
        ("The bones of the body together form the", "The muscles attached to bones let the body", "skeleton", "move"),
    ]),
    _pair_task("Science: genetics", "Angle 18: Science", [
        ("The molecule that carries hereditary information in living things is", "The molecule that carries energy for cell work is", "DNA", "ATP"),
        ("A segment of DNA that codes for a trait is called a", "A structure made of coiled DNA is called a", "gene", "chromosome"),
        ("Humans normally have twenty-three pairs of", "Each chromosome is a long molecule of", "chromosomes", "DNA"),
        ("Offspring inherit their genes from their", "Identical twins share exactly the same", "parents", "DNA"),
    ]),
    _pair_task("Science: chemistry basics", "Angle 18: Science", [
        ("A substance that turns litmus paper red is an", "A substance that turns litmus paper blue is a", "acid", "base"),
        ("The smallest unit of an element that keeps its properties is an", "Two or more atoms bonded together form a", "atom", "molecule"),
        ("The chemical formula H2O stands for", "The chemical formula CO2 stands for carbon", "water", "dioxide"),
        ("Mixing an acid and a base tends to produce salt and", "Burning a fuel in oxygen tends to produce heat and", "water", "light"),
    ]),
    _pair_task("Science: energy and electricity", "Angle 18: Science", [
        ("Electric current flows easily through a material that is a good", "Electric current is blocked by a material that is a good", "conductor", "insulator"),
        ("A device that stores chemical energy and releases electricity is a", "A device that converts electricity into light is a", "battery", "bulb"),
        ("Energy from the Sun captured by panels is called", "Energy from moving water spun through turbines is called", "solar", "hydro"),
        ("The energy an object has because of its motion is called", "The energy an object has because of its height is called", "kinetic", "potential"),
    ]),
]


# ============================================================================
# ANGLE 19: Geography & Spatial Knowledge
# ============================================================================

ANGLE_19_BEHAVIORS = [
    _pair_task("Geography: longest river", "Angle 19: Geography", [
        ("The longest river in Africa is the", "The longest river in South America is the", "Nile", "Amazon"),
        ("The river that flows through Egypt to the Mediterranean is the", "The river that flows through Brazil to the Atlantic is the", "Nile", "Amazon"),
        ("The river most associated with ancient Egypt is the", "The river most associated with the rainforest is the", "Nile", "Amazon"),
        ("A river flowing north through Sudan and Egypt is the", "A river flowing east across Peru and Brazil is the", "Nile", "Amazon"),
    ]),
    _pair_task("Geography: largest ocean", "Angle 19: Geography", [
        ("The largest ocean on Earth is the", "The second largest ocean on Earth is the", "Pacific", "Atlantic"),
        ("The ocean between Asia and the Americas is the", "The ocean between the Americas and Europe is the", "Pacific", "Atlantic"),
        ("The ocean containing the Mariana Trench is the", "The ocean crossed by early European voyages to America is the", "Pacific", "Atlantic"),
        ("Hawaii sits in the middle of the", "Iceland sits in the middle of the", "Pacific", "Atlantic"),
    ]),
    _pair_task("Geography: highest mountain", "Angle 19: Geography", [
        ("The highest mountain above sea level on Earth is Mount", "The highest mountain in Africa is Mount", "Everest", "Kilimanjaro"),
        ("The peak on the border of Nepal and Tibet, the tallest on Earth, is Mount", "The tallest free-standing mountain in Tanzania is Mount", "Everest", "Kilimanjaro"),
        ("Climbers aiming for the world's highest summit go to Mount", "Climbers aiming for Africa's highest summit go to Mount", "Everest", "Kilimanjaro"),
        ("The Himalayas contain the world's tallest mountain, Mount", "The Alps' tallest mountain is Mont", "Everest", "Blanc"),
    ]),
    _pair_task("Geography: largest hot desert", "Angle 19: Geography", [
        ("The largest hot desert in the world is the", "The largest rainforest in the world is the", "Sahara", "Amazon"),
        ("The vast desert covering much of northern Africa is the", "The vast rainforest covering much of northern South America is the", "Sahara", "Amazon"),
        ("Camels crossing North Africa travel through the", "Boats crossing northern Brazil travel through the", "Sahara", "Amazon"),
        ("A place of endless sand dunes in Africa is the", "A place of endless dense trees in South America is the", "Sahara", "Amazon"),
    ]),
    _pair_task("Geography: most populous continent", "Angle 19: Geography", [
        ("The continent with the largest human population is", "The continent with the smallest human population is", "Asia", "Antarctica"),
        ("The continent that contains China and India is", "The continent that contains Egypt and Nigeria is", "Asia", "Africa"),
        ("Home to more than half the world's people, the continent is", "Home to almost no permanent residents, the continent is", "Asia", "Antarctica"),
        ("The largest continent by land area is", "The smallest continent by land area is", "Asia", "Australia"),
    ]),
    _pair_task("Geography: largest country by area", "Angle 19: Geography", [
        ("The country with the largest land area in the world is", "The country with the largest population in the world is", "Russia", "China"),
        ("Spanning eleven time zones across northern Asia and Europe is", "Bordered by fourteen countries and home to over a billion people is", "Russia", "China"),
        ("The biggest country on the map, stretching to the Pacific and the Baltic, is", "The most populous country for much of modern history is", "Russia", "India"),
        ("Its capital is Moscow and it is the largest nation by area:", "Its capital is Beijing and it has a very large population:", "Russia", "China"),
    ]),
    _pair_task("Geography: hemispheres and the equator", "Angle 19: Geography", [
        ("The imaginary line dividing Earth into northern and southern halves is the", "The imaginary line dividing Earth into eastern and western halves runs through", "equator", "Greenwich"),
        ("The half of the Earth north of the equator is called the Northern", "The half of the Earth east or west of the prime meridian is called a", "Hemisphere", "longitude"),
        ("At the equator the climate is generally", "At the poles the climate is generally", "hot", "cold"),
        ("Ecuador is named for the line it sits on, the", "Greenwich is famous for the line it sits on, the", "equator", "meridian"),
    ]),
    _pair_task("Geography: country and continent", "Angle 19: Geography", [
        ("Australia is unusual because it is both a country and a", "France is only one country within a larger", "continent", "region"),
        ("The Amazon rainforest lies mostly within the country of", "The Sahara desert's largest share lies within the country of", "Brazil", "Algeria"),
        ("The pyramids of Giza are located in the country of", "The Colosseum is located in the country of", "Egypt", "Italy"),
        ("Mount Fuji is a landmark of the country of", "The Eiffel Tower is a landmark of the country of", "Japan", "France"),
    ]),
    _pair_task("Geography: seas and lakes", "Angle 19: Geography", [
        ("The saltiest famous body of water, where swimmers float easily, is the Dead", "The largest freshwater surface in North America is Lake", "Sea", "Superior"),
        ("The sea between Europe and Africa is the", "The sea between Sweden and Finland is the", "Mediterranean", "Baltic"),
        ("The largest inland body of water on Earth is the Caspian", "The deepest freshwater lake on Earth is Lake", "Sea", "Baikal"),
        ("Between Egypt and Saudi Arabia lies the Red", "Between Europe and Africa lies the", "Sea", "Mediterranean"),
    ]),
    _pair_task("Geography: directions and navigation", "Angle 19: Geography", [
        ("A compass needle points toward magnetic", "The opposite direction from north on a compass is", "north", "south"),
        ("The Sun rises in the", "The Sun sets in the", "east", "west"),
        ("If you travel from the equator toward the North Pole you are heading", "If you travel from the equator toward the South Pole you are heading", "north", "south"),
        ("On a standard map, the direction at the top of the page is", "On a standard map, the direction at the bottom of the page is", "north", "south"),
    ]),
]


# ============================================================================
# ANGLE 20: Arts, Literature & Culture
# ============================================================================

ANGLE_20_BEHAVIORS = [
    _pair_task("Culture: playwright of Romeo and Juliet", "Angle 20: Arts & Culture", [
        ("The tragedy Romeo and Juliet was written by William", "The epic poem The Odyssey is attributed to", "Shakespeare", "Homer"),
        ("The play Hamlet was written by William", "The novel Oliver Twist was written by Charles", "Shakespeare", "Dickens"),
        ("Macbeth and King Lear were both written by William", "Pride and Prejudice was written by Jane", "Shakespeare", "Austen"),
        ("The author of the sonnet beginning 'Shall I compare thee' is William", "The author of the poem 'The Raven' is Edgar Allan", "Shakespeare", "Poe"),
    ]),
    _pair_task("Culture: painter of the Mona Lisa", "Angle 20: Arts & Culture", [
        ("The Mona Lisa was painted by Leonardo da", "The Starry Night was painted by Vincent van", "Vinci", "Gogh"),
        ("The Last Supper mural was painted by Leonardo da", "The Water Lilies series was painted by Claude", "Vinci", "Monet"),
        ("The Renaissance master who painted the Mona Lisa was Leonardo da", "The Dutch master who painted The Night Watch was", "Vinci", "Rembrandt"),
        ("The portrait with the famous enigmatic smile is by Leonardo da", "The ceiling of the Sistine Chapel is by", "Vinci", "Michelangelo"),
    ]),
    _pair_task("Culture: composer of the Ninth Symphony", "Angle 20: Arts & Culture", [
        ("The Ninth Symphony, with its Ode to Joy, was composed by Ludwig van", "The opera The Magic Flute was composed by Wolfgang Amadeus", "Beethoven", "Mozart"),
        ("The Fifth Symphony with its famous four-note opening is by", "The Four Seasons set of violin concertos is by", "Beethoven", "Vivaldi"),
        ("The composer who continued working after going deaf was", "The composer of many Baroque works including the Brandenburg Concertos was", "Beethoven", "Bach"),
        ("Moonlight Sonata was composed by", "The Nutcracker ballet was composed by", "Beethoven", "Tchaikovsky"),
    ]),
    _pair_task("Culture: creator of Sherlock Holmes", "Angle 20: Arts & Culture", [
        ("The detective Sherlock Holmes was created by Arthur Conan", "The wizard Harry Potter was created by J. K.", "Doyle", "Rowling"),
        ("The stories set at 221B Baker Street were written by Arthur Conan", "The stories set at Hogwarts were written by J. K.", "Doyle", "Rowling"),
        ("Dr. Watson's companion detective was written by Arthur Conan", "Bilbo Baggins was written by J. R. R.", "Doyle", "Tolkien"),
        ("The author who tried to kill off his own famous detective was Arthur Conan", "The author of the Discworld series was Terry", "Doyle", "Pratchett"),
    ]),
    _pair_task("Culture: author of War and Peace", "Angle 20: Arts & Culture", [
        ("The novel War and Peace was written by Leo", "The novel Crime and Punishment was written by Fyodor", "Tolstoy", "Dostoevsky"),
        ("Anna Karenina was written by Leo", "The Brothers Karamazov was written by Fyodor", "Tolstoy", "Dostoevsky"),
        ("The long Russian epic of the Napoleonic wars is by Leo", "The Russian novel of guilt and confession is by Fyodor", "Tolstoy", "Dostoevsky"),
        ("A giant of Russian literature who wrote War and Peace was Leo", "A giant of English literature who wrote Great Expectations was Charles", "Tolstoy", "Dickens"),
    ]),
    _pair_task("Culture: author of Harry Potter", "Angle 20: Arts & Culture", [
        ("The Harry Potter series was written by J. K.", "The Lord of the Rings was written by J. R. R.", "Rowling", "Tolkien"),
        ("The boy wizard who lived under the stairs was created by J. K.", "The hobbit who carried the ring was created by J. R. R.", "Rowling", "Tolkien"),
        ("Hermione and Ron were written by J. K.", "Frodo and Sam were written by J. R. R.", "Rowling", "Tolkien"),
        ("The best-selling fantasy series set at a school of magic is by J. K.", "The best-selling fantasy series set in Middle-earth is by J. R. R.", "Rowling", "Tolkien"),
    ]),
    _pair_task("Culture: sculptor of David", "Angle 20: Arts & Culture", [
        ("The marble statue of David in Florence was sculpted by", "The bronze sculpture The Thinker was created by Auguste", "Michelangelo", "Rodin"),
        ("The Renaissance sculptor of the Pieta was", "The Renaissance painter of the Mona Lisa was", "Michelangelo", "Leonardo"),
        ("The giant marble figure symbolising the Republic of Florence is by", "The bronze figure The Thinker is by", "Michelangelo", "Rodin"),
        ("Painting the Sistine ceiling lying on his back was", "Painting Guernica in stark black and white was", "Michelangelo", "Picasso"),
    ]),
    _pair_task("Culture: painter of Starry Night", "Angle 20: Arts & Culture", [
        ("The swirling night sky painting The Starry Night is by Vincent van", "The melting clocks painting The Persistence of Memory is by Salvador", "Gogh", "Dali"),
        ("Sunflowers and self-portraits with a bandaged ear are by Vincent van", "Soup cans and celebrity prints are by Andy", "Gogh", "Warhol"),
        ("The Dutch post-impressionist who sold almost nothing in life was Vincent van", "The Spanish surrealist with the curled moustache was Salvador", "Gogh", "Dali"),
        ("Thick swirling brushstrokes of a village at night are by Vincent van", "Geometric coloured rectangles with black lines are by Piet", "Gogh", "Mondrian"),
    ]),
    _pair_task("Culture: opera and ballet", "Angle 20: Arts & Culture", [
        ("The ballet Swan Lake was composed by Pyotr Ilyich", "The opera The Barber of Seville was composed by Gioachino", "Tchaikovsky", "Rossini"),
        ("The Nutcracker and Sleeping Beauty ballets are by", "The operas La Traviata and Aida are by Giuseppe", "Tchaikovsky", "Verdi"),
        ("A Christmas ballet with a toy prince and a Sugar Plum Fairy is by", "A tragic opera ending with a leap from a parapet, Tosca, is by Giacomo", "Tchaikovsky", "Puccini"),
        ("The 1812 Overture, with its cannon fire, was written by", "The Ring cycle of operas was written by Richard", "Tchaikovsky", "Wagner"),
    ]),
    _pair_task("Culture: famous first lines and works", "Angle 20: Arts & Culture", [
        ("The novel that opens 'It was the best of times, it was the worst of times' is by Charles", "The novel that opens 'Call me Ishmael' is by Herman", "Dickens", "Melville"),
        ("Pride and Prejudice, opening with a truth universally acknowledged, is by Jane", "Jane Eyre is by Charlotte", "Austen", "Bronte"),
        ("The dystopia 1984 with Big Brother was written by George", "The dystopia Brave New World was written by Aldous", "Orwell", "Huxley"),
        ("The allegory Animal Farm was written by George", "The fantasy The Chronicles of Narnia was written by C. S.", "Orwell", "Lewis"),
    ]),
]


# ============================================================================
# ANGLE 21: Economics, Politics & Law
# ============================================================================

ANGLE_21_BEHAVIORS = [
    _pair_task("Civics: study of money and markets", "Angle 21: Civics", [
        ("The social science that studies the production and exchange of goods is", "The social science that studies past human events is", "economics", "history"),
        ("The field concerned with supply, demand, and prices is", "The field concerned with governments and power is", "economics", "politics"),
        ("A person who studies inflation and unemployment is an", "A person who studies rocks and minerals is a", "economist", "geologist"),
        ("The subject dealing with scarcity and choice is", "The subject dealing with cells and organisms is", "economics", "biology"),
    ]),
    _pair_task("Civics: branches of government", "Angle 21: Civics", [
        ("The branch of government that writes and passes laws is the", "The branch of government that interprets laws in court is the", "legislature", "judiciary"),
        ("The branch that enforces the laws and runs the country day to day is the", "The branch that debates and votes on new laws is the", "executive", "legislature"),
        ("Judges and courts belong to the branch called the", "Elected lawmakers belong to the branch called the", "judiciary", "legislature"),
        ("A president or prime minister heads the branch called the", "A supreme court heads the branch called the", "executive", "judiciary"),
    ]),
    _pair_task("Civics: rising prices", "Angle 21: Civics", [
        ("A sustained rise in the general level of prices is called", "A sustained fall in the general level of prices is called", "inflation", "deflation"),
        ("When money buys less than it did last year, the economy is experiencing", "When money buys more than it did last year, the economy is experiencing", "inflation", "deflation"),
        ("Central banks often raise interest rates to fight", "Central banks often cut interest rates to fight a", "inflation", "recession"),
        ("If a loaf of bread costs more each month, that trend is", "If wages are stuck while prices rise, workers lose", "inflation", "purchasing"),
    ]),
    _pair_task("Civics: founding legal document", "Angle 21: Civics", [
        ("The fundamental document setting out a country's system of government is its", "A routine law passed by the legislature is called a", "constitution", "statute"),
        ("The highest law of the land, against which other laws are judged, is the", "A local rule passed by a city council is called an", "constitution", "ordinance"),
        ("Amendments are formal changes to the", "Repeals are the removal of an ordinary", "constitution", "statute"),
        ("A country's basic charter of rights and powers is its", "A signed agreement between two countries is a", "constitution", "treaty"),
    ]),
    _pair_task("Civics: tax on imports", "Angle 21: Civics", [
        ("A tax charged on goods brought in from another country is a", "A tax charged on a person's yearly earnings is an", "tariff", "income"),
        ("To protect domestic industry a government may place a", "To raise general revenue a government may raise the sales", "tariff", "tax"),
        ("Import duties are also known as", "Money paid by the government to support an industry is a", "tariffs", "subsidy"),
        ("A trade barrier that makes foreign goods more expensive is a", "A trade rule that limits how many units may be imported is a", "tariff", "quota"),
    ]),
    _pair_task("Civics: local government head", "Angle 21: Civics", [
        ("The elected head of a city or town government is the", "The elected head of a national government is often the", "mayor", "president"),
        ("City hall is typically run by the", "The state capitol is typically run by the", "mayor", "governor"),
        ("A person leading a municipal council is the", "A person leading a country's cabinet is the", "mayor", "premier"),
        ("The chief official of a large city like Chicago is the", "The chief official of a whole state like Illinois is the", "mayor", "governor"),
    ]),
    _pair_task("Civics: court outcome", "Angle 21: Civics", [
        ("The jury's formal decision on guilt or innocence is the", "The judge's decision on punishment after a guilty finding is the", "verdict", "sentence"),
        ("At the end of a criminal trial the jury delivers a", "Before the trial the prosecutor files a formal", "verdict", "charge"),
        ("A finding of guilty or not guilty is a", "A written explanation of a higher court's reasoning is an", "verdict", "opinion"),
        ("Twelve jurors must usually agree to reach a", "A defendant who disagrees with the outcome may file an", "verdict", "appeal"),
    ]),
    _pair_task("Civics: market price mechanism", "Angle 21: Civics", [
        ("In a free market, the interaction of supply and demand sets the", "In a command economy, a central authority sets the", "price", "plan"),
        ("When demand rises and supply stays fixed, the price tends to", "When supply rises and demand stays fixed, the price tends to", "rise", "fall"),
        ("A shortage of a good usually pushes its price", "A surplus of a good usually pushes its price", "up", "down"),
        ("Buyers and sellers meeting to trade form a", "A single seller controlling an entire market is a", "market", "monopoly"),
    ]),
    _pair_task("Civics: types of government", "Angle 21: Civics", [
        ("A system in which citizens vote to choose their leaders is a", "A system in which one hereditary ruler holds power for life is a", "democracy", "monarchy"),
        ("Rule by a king or queen who inherits the throne is a", "Rule by the people through elected representatives is a", "monarchy", "democracy"),
        ("A state with no king, headed by an elected president, is a", "A state headed by a monarch, even a ceremonial one, is a", "republic", "kingdom"),
        ("Absolute power held by a single unelected ruler describes a", "Power shared and limited by law and elections describes a", "dictatorship", "democracy"),
    ]),
    _pair_task("Civics: money and banking", "Angle 21: Civics", [
        ("The institution that keeps your savings and lends money is a", "The building where laws are debated and passed is a", "bank", "parliament"),
        ("The bank that issues a nation's currency and sets interest rates is the central", "The place where company shares are bought and sold is the stock", "bank", "exchange"),
        ("Interest is the fee paid for borrowing", "Dividends are payments made to holders of company", "money", "shares"),
        ("A loan to buy a house is called a", "A card that lets you spend and pay later is a credit", "mortgage", "card"),
    ]),
]


# ============================================================================
# ANGLE 22: Idioms & Figurative Language
# ============================================================================

ANGLE_22_BEHAVIORS = [
    _pair_task("Idiom: raining cats and dogs", "Angle 22: Idioms", [
        ("When it pours heavily, people say it is raining cats and", "When something is very easy, people say it is a piece of", "dogs", "cake"),
        ("The old expression for a downpour is raining cats and", "The expression for feeling unwell is under the", "dogs", "weather"),
        ("It came down in buckets today, truly raining cats and", "That test was simple, honestly a piece of", "dogs", "cake"),
        ("Grab an umbrella, it is raining cats and", "No need to worry, this job is a piece of", "dogs", "cake"),
    ]),
    _pair_task("Idiom: break a leg", "Angle 22: Idioms", [
        ("To wish an actor good luck before a show, people say break a", "To tell someone to relax and sit, people say take a", "leg", "seat"),
        ("Before she went on stage her friend whispered, break a", "Before he fell asleep he decided to hit the", "leg", "hay"),
        ("Theatre superstition replaces 'good luck' with break a", "Being told bad news is sometimes called a kick in the", "leg", "teeth"),
        ("The director smiled at the cast and said, break a", "The coach told the tired team to take a", "leg", "break"),
    ]),
    _pair_task("Idiom: spill the beans", "Angle 22: Idioms", [
        ("To reveal a secret is to spill the", "To make a situation worse is to add fuel to the", "beans", "fire"),
        ("Come on, tell us what happened, spill the", "Careful what you say, do not add fuel to the", "beans", "fire"),
        ("He could not keep quiet and decided to spill the", "She was so angry she was ready to add fuel to the", "beans", "fire"),
        ("The surprise was ruined when someone spilled the", "The argument grew worse when he added fuel to the", "beans", "fire"),
    ]),
    _pair_task("Idiom: bite the bullet", "Angle 22: Idioms", [
        ("To force yourself to do something unpleasant is to bite the", "To reveal a hidden truth by accident is to let the cat out of the", "bullet", "bag"),
        ("There was no way around the surgery, so he had to bite the", "The secret slipped and she let the cat out of the", "bullet", "bag"),
        ("Facing the hard conversation, she chose to bite the", "Telling them early would let the cat out of the", "bullet", "bag"),
        ("Sometimes you just have to bite the", "Careful, one wrong word and you let the cat out of the", "bullet", "bag"),
    ]),
    _pair_task("Idiom: once in a blue moon", "Angle 22: Idioms", [
        ("Something that happens very rarely happens once in a blue", "Something that happens very frequently happens all the", "moon", "time"),
        ("He visits his hometown only once in a blue", "She checks her phone practically all the", "moon", "time"),
        ("A rare treat comes along once in a blue", "A constant background noise is there all the", "moon", "time"),
        ("They agree on almost nothing, only once in a blue", "They argue constantly, basically all the", "moon", "time"),
    ]),
    _pair_task("Idiom: the ball is in your court", "Angle 22: Idioms", [
        ("Now that we have made our offer, the ball is in your", "Now that the meeting is over, let us call it a", "court", "day"),
        ("I have done my part; the ball is in your", "It is late and we are tired, so let us call it a", "court", "day"),
        ("We replied to their email, so the ball is in their", "We finished everything on the list, so let us call it a", "court", "day"),
        ("The decision is yours now, the ball is in your", "Nothing left to do here, let us call it a", "court", "day"),
    ]),
    _pair_task("Idiom: cost an arm and a leg", "Angle 22: Idioms", [
        ("Something extremely expensive is said to cost an arm and a", "Something extremely cheap is said to cost next to", "leg", "nothing"),
        ("That designer bag must have cost an arm and a", "The yard-sale mug basically cost her next to", "leg", "nothing"),
        ("Tickets to the final cost an arm and a", "Standing-room seats cost almost", "leg", "nothing"),
        ("Fixing the car will cost an arm and a", "Changing a bulb costs practically", "leg", "nothing"),
    ]),
    _pair_task("Idiom: kill two birds with one stone", "Angle 22: Idioms", [
        ("Doing two tasks with a single action is killing two birds with one", "Making a situation worse by acting rashly is jumping out of the frying pan into the", "stone", "fire"),
        ("If I mail the letter on my run I kill two birds with one", "By quitting for a worse job he jumped from the frying pan into the", "stone", "fire"),
        ("She shopped and exercised at once, killing two birds with one", "He fixed one bug and caused three, out of the frying pan into the", "stone", "fire"),
        ("Combining the errands let them kill two birds with one", "Rushing the repair only took them from the frying pan into the", "stone", "fire"),
    ]),
    _pair_task("Idiom: under the weather", "Angle 22: Idioms", [
        ("Someone who feels slightly ill is a bit under the", "Someone who is extremely happy is on cloud", "weather", "nine"),
        ("He stayed home from work feeling under the", "After the good news she was walking on cloud", "weather", "nine"),
        ("A scratchy throat and a headache: she is under the", "A promotion and a raise: he is on cloud", "weather", "nine"),
        ("I might skip the party, I am under the", "I could not stop smiling, I was on cloud", "weather", "nine"),
    ]),
    _pair_task("Idiom: hit the nail on the head", "Angle 22: Idioms", [
        ("To describe something exactly right is to hit the nail on the", "To avoid saying something directly is to beat around the", "head", "bush"),
        ("Her summary hit the nail on the", "He would not give a straight answer, just beat around the", "head", "bush"),
        ("That comment really hit the nail on the", "Stop beating around the", "head", "bush"),
        ("You hit the nail on the", "Quit dancing around it and beating around the", "head", "bush"),
    ]),
]


# ============================================================================
# ANGLE 23: Counting & Set Facts
# ============================================================================

ANGLE_23_BEHAVIORS = [
    _pair_task("Counting: days in a week", "Angle 23: Counting", [
        ("The number of days in a week is", "The number of days in a fortnight is", "7", "14"),
        ("A full week contains this many days:", "Half of the days in a week, rounded down, is", "7", "3"),
        ("From Monday through Sunday there are", "From Monday through Wednesday there are", "7", "3"),
        ("Weekdays plus weekend days total", "Weekend days alone total", "7", "2"),
    ]),
    _pair_task("Counting: months in a year", "Angle 23: Counting", [
        ("The number of months in a year is", "The number of weeks in a year is about", "12", "52"),
        ("January through December is this many months:", "One quarter of a year is this many months:", "12", "3"),
        ("A calendar year has this many months:", "Half a calendar year has this many months:", "12", "6"),
        ("Counting from January to December gives", "Counting from January to March gives", "12", "3"),
    ]),
    _pair_task("Counting: sides of shapes", "Angle 23: Counting", [
        ("The number of sides on a triangle is", "The number of sides on a square is", "3", "4"),
        ("A triangle has this many corners:", "A pentagon has this many corners:", "3", "5"),
        ("The simplest polygon has this many sides:", "A hexagon has this many sides:", "3", "6"),
        ("A three-sided shape has", "A four-sided shape has", "3", "4"),
    ]),
    _pair_task("Counting: legs on animals", "Angle 23: Counting", [
        ("The number of legs on a spider is", "The number of legs on an insect is", "8", "6"),
        ("A spider has this many legs:", "A dog has this many legs:", "8", "4"),
        ("An octopus has this many arms:", "A squid's extra-long feeding tentacles number", "8", "2"),
        ("Arachnids such as spiders have this many legs:", "Humans have this many legs:", "8", "2"),
    ]),
    _pair_task("Counting: letters in the alphabet", "Angle 23: Counting", [
        ("The number of letters in the English alphabet is", "The number of vowels in the English alphabet is", "26", "5"),
        ("From A to Z there are this many letters:", "From A to E there are this many letters:", "26", "5"),
        ("The English alphabet contains this many letters:", "The English alphabet contains this many vowels:", "26", "5"),
        ("Counting every letter A through Z gives", "Counting only A, E, I, O, U gives", "26", "5"),
    ]),
    _pair_task("Counting: hours and minutes", "Angle 23: Counting", [
        ("The number of hours in a day is", "The number of days in a week is", "24", "7"),
        ("A full day contains this many hours:", "Half a day contains this many hours:", "24", "12"),
        ("The number of minutes in an hour is", "The number of hours in a day is", "60", "24"),
        ("From midnight to midnight there are this many hours:", "From noon to midnight there are this many hours:", "24", "12"),
    ]),
    _pair_task("Counting: colors and continents", "Angle 23: Counting", [
        ("The traditional number of colors in a rainbow is", "The number of primary colors of light is", "7", "3"),
        ("The number of continents on Earth is commonly given as", "The number of oceans on Earth is commonly given as", "7", "5"),
        ("A rainbow is usually said to have this many bands:", "A traffic light usually has this many colors:", "7", "3"),
        ("Counting the classic rainbow colors gives", "Counting red, green, and blue gives", "7", "3"),
    ]),
    _pair_task("Counting: players on a team", "Angle 23: Counting", [
        ("The number of players per side on a soccer field is", "The number of players per side on a basketball court is", "11", "5"),
        ("A standard football (soccer) team fields this many players:", "A standard basketball team fields this many players:", "11", "5"),
        ("On the pitch each soccer team has this many players:", "On the court each basketball team has this many players:", "11", "5"),
        ("Including the goalkeeper, a soccer team has this many on the field:", "A volleyball team has this many players on court:", "11", "6"),
    ]),
    _pair_task("Counting: wheels and doors", "Angle 23: Counting", [
        ("The usual number of wheels on a car is", "The usual number of wheels on a motorcycle is", "4", "2"),
        ("A typical bicycle has this many wheels:", "A typical car has this many wheels:", "2", "4"),
        ("A tricycle has this many wheels:", "A unicycle has this many wheels:", "3", "1"),
        ("A standard car has this many wheels:", "A standard car has this many side mirrors:", "4", "2"),
    ]),
    _pair_task("Counting: set membership size", "Angle 23: Counting", [
        ("The number of items in a pair is", "The number of items in a trio is", "2", "3"),
        ("A dozen contains this many items:", "A half dozen contains this many items:", "12", "6"),
        ("A quartet has this many members:", "A duet has this many members:", "4", "2"),
        ("A century is this many years:", "A decade is this many years:", "100", "10"),
    ]),
]


# ============================================================================
# ANGLE 24: Units & Measurement
# ============================================================================

ANGLE_24_BEHAVIORS = [
    _pair_task("Measurement: length units", "Angle 24: Measurement", [
        ("A kilometer is longer than a", "A millimeter is shorter than a", "meter", "centimeter"),
        ("One thousand meters make one", "One hundred centimeters make one", "kilometer", "meter"),
        ("Distances between cities are usually given in", "The thickness of a coin is usually given in", "kilometers", "millimeters"),
        ("A meter is longer than a", "A meter is shorter than a", "centimeter", "kilometer"),
    ]),
    _pair_task("Measurement: time units", "Angle 24: Measurement", [
        ("One hour is made up of sixty", "One minute is made up of sixty", "minutes", "seconds"),
        ("Sixty seconds add up to one", "Sixty minutes add up to one", "minute", "hour"),
        ("A stopwatch timing a short race reads mostly in", "A calendar marking a long project reads mostly in", "seconds", "months"),
        ("There are twenty-four hours in a", "There are seven days in a", "day", "week"),
    ]),
    _pair_task("Measurement: mass units", "Angle 24: Measurement", [
        ("A kilogram is heavier than a", "A tonne is heavier than a", "gram", "kilogram"),
        ("One thousand grams make one", "One thousand kilograms make one", "kilogram", "tonne"),
        ("The mass of a person is usually given in", "The mass of a vitamin pill is usually given in", "kilograms", "milligrams"),
        ("A bag of flour is weighed in", "A single grain of salt is weighed in", "grams", "milligrams"),
    ]),
    _pair_task("Measurement: instruments", "Angle 24: Measurement", [
        ("Temperature is measured with a", "Time is measured with a", "thermometer", "clock"),
        ("You measure how hot something is with a", "You measure how heavy something is with a", "thermometer", "scale"),
        ("Air pressure is measured with a", "Wind speed is measured with an", "barometer", "anemometer"),
        ("Length is measured with a", "Angles are measured with a", "ruler", "protractor"),
    ]),
    _pair_task("Measurement: volume and capacity", "Angle 24: Measurement", [
        ("A liter is a measure of", "A gram is a measure of", "volume", "mass"),
        ("One thousand milliliters make one", "One thousand liters make one cubic", "liter", "meter"),
        ("The fuel in a car tank is measured in", "The pressure in a car tyre is measured in", "liters", "bars"),
        ("A jug's capacity is given in", "A table's length is given in", "liters", "meters"),
    ]),
    _pair_task("Measurement: speed", "Angle 24: Measurement", [
        ("On a road sign, speed is given in kilometers per", "On a clock face, position is given in", "hour", "minutes"),
        ("A car's speedometer shows miles or kilometers per", "A car's odometer shows total", "hour", "distance"),
        ("Runners' pace is often stated as minutes per", "Cyclists' speed is often stated as kilometers per", "kilometer", "hour"),
        ("Speed equals distance divided by", "Area equals length multiplied by", "time", "width"),
    ]),
    _pair_task("Measurement: temperature scales", "Angle 24: Measurement", [
        ("On the Celsius scale, water freezes at zero and boils at", "On the Celsius scale, water boils at one hundred and freezes at", "100", "0"),
        ("Water freezes at zero degrees on the temperature scale named", "Everyday weather in the United States is usually given in degrees", "Celsius", "Fahrenheit"),
        ("Twenty degrees Celsius describes a room that feels", "Forty degrees Celsius describes air that feels", "comfortable", "hot"),
        ("The scale where zero is water's freezing point is", "The scale common in the United States for weather is", "Celsius", "Fahrenheit"),
    ]),
    _pair_task("Measurement: area", "Angle 24: Measurement", [
        ("The size of a floor, in square meters, is a measurement of its", "The distance around the edge of a field is its", "area", "perimeter"),
        ("Farmland area is often measured in", "Farmland fencing is often measured in", "hectares", "meters"),
        ("Carpet to cover a whole floor is sold by the square", "Fabric to make a dress is sold by the linear", "meter", "yard"),
        ("Area is length multiplied by", "Perimeter is the sum of all the", "width", "sides"),
    ]),
    _pair_task("Measurement: unit prefixes", "Angle 24: Measurement", [
        ("The prefix kilo- means one", "The prefix milli- means one", "thousand", "thousandth"),
        ("The prefix kilo- multiplies a unit by one", "The prefix centi- divides a unit by one", "thousand", "hundred"),
        ("A kilobyte is roughly one thousand", "A kilometer is roughly one thousand", "bytes", "meters"),
        ("The prefix centi- means one", "The prefix deci- means one", "hundredth", "tenth"),
    ]),
    _pair_task("Measurement: everyday estimation", "Angle 24: Measurement", [
        ("The height of an adult is best measured in", "The height of a mountain is best measured in", "meters", "kilometers"),
        ("The mass of a car is best measured in", "The mass of a paperclip is best measured in", "kilograms", "grams"),
        ("A short walk to the shop is best measured in", "A flight between continents is best measured in", "meters", "kilometers"),
        ("Boiling an egg is best timed in", "Building a house is best timed in", "minutes", "months"),
    ]),
]


# ============================================================================
# ANGLE 25: History & Chronology
# ============================================================================

ANGLE_25_BEHAVIORS = [
    _pair_task("History: first person on the Moon", "Angle 25: History", [
        ("The first person to walk on the Moon was Neil", "The first person to sail around the world's expedition was led by Ferdinand", "Armstrong", "Magellan"),
        ("In 1969 the first human to step onto the lunar surface was Neil", "In 1492 the explorer who reached the Americas for Spain was Christopher", "Armstrong", "Columbus"),
        ("Apollo 11's commander who first set foot on the Moon was Neil", "The pilot who followed him down the ladder was Buzz", "Armstrong", "Aldrin"),
        ("'One small step for man' was said by Neil", "The astronaut who orbited alone in the command module was Michael", "Armstrong", "Collins"),
    ]),
    _pair_task("History: inventor of the printing press", "Angle 25: History", [
        ("The movable-type printing press in Europe is credited to Johannes", "The telephone is credited to Alexander Graham", "Gutenberg", "Bell"),
        ("Around 1440 the printing press was developed by Johannes", "Around 1879 the practical light bulb was developed by Thomas", "Gutenberg", "Edison"),
        ("The first major book printed with movable type in Europe came from the workshop of Johannes", "The theory of gravity is associated with Isaac", "Gutenberg", "Newton"),
        ("Mass-produced books in the 1400s were made possible by Johannes", "Mass-produced cars in the early 1900s were made possible by Henry", "Gutenberg", "Ford"),
    ]),
    _pair_task("History: theory of evolution", "Angle 25: History", [
        ("The theory of evolution by natural selection was proposed by Charles", "The laws of planetary motion were worked out by Johannes", "Darwin", "Kepler"),
        ("On the Origin of Species was written by Charles", "The Principia, setting out the laws of motion, was written by Isaac", "Darwin", "Newton"),
        ("The naturalist aboard HMS Beagle was Charles", "The physicist who explained the photoelectric effect was Albert", "Darwin", "Einstein"),
        ("Natural selection as the mechanism of evolution is due to Charles", "General relativity as a theory of gravity is due to Albert", "Darwin", "Einstein"),
    ]),
    _pair_task("History: first US president", "Angle 25: History", [
        ("The first President of the United States was George", "The president during the American Civil War was Abraham", "Washington", "Lincoln"),
        ("The general who led the Continental Army and became the first president was George", "The president who delivered the Gettysburg Address was Abraham", "Washington", "Lincoln"),
        ("The face on the US one-dollar bill is George", "The face on the US five-dollar bill is Abraham", "Washington", "Lincoln"),
        ("The US capital city is named after George", "The Lincoln Memorial honors Abraham", "Washington", "Lincoln"),
    ]),
    _pair_task("History: inventor of the telephone", "Angle 25: History", [
        ("The telephone is generally credited to Alexander Graham", "The phonograph and practical light bulb are credited to Thomas", "Bell", "Edison"),
        ("The first intelligible telephone call was made by Alexander Graham", "The first practical electric lamp was demonstrated by Thomas", "Bell", "Edison"),
        ("A pioneer of voice transmission over wires was Alexander Graham", "A prolific inventor with over a thousand patents was Thomas", "Bell", "Edison"),
        ("'Mr. Watson, come here' was spoken by Alexander Graham", "The Menlo Park laboratory was run by Thomas", "Bell", "Edison"),
    ]),
    _pair_task("History: ancient civilizations", "Angle 25: History", [
        ("The Great Pyramid and the Sphinx were built by the ancient", "The Parthenon on the Acropolis was built by the ancient", "Egyptians", "Greeks"),
        ("Hieroglyphic writing and mummies belong to ancient", "Democracy and the Olympic Games began in ancient", "Egypt", "Greece"),
        ("The Colosseum and a vast road network were built by the ancient", "The pyramids along the Nile were built by the ancient", "Romans", "Egyptians"),
        ("Julius Caesar was a leader of ancient", "Cleopatra was a ruler of ancient", "Rome", "Egypt"),
    ]),
    _pair_task("History: European explorers", "Angle 25: History", [
        ("The 1492 voyage across the Atlantic for Spain was led by Christopher", "The first expedition to circumnavigate the globe was begun by Ferdinand", "Columbus", "Magellan"),
        ("The explorer who reached the Caribbean thinking he had found Asia was Christopher", "The explorer whose crew first sailed all the way around the world was Ferdinand", "Columbus", "Magellan"),
        ("Sailing the Nina, Pinta, and Santa Maria was Christopher", "Sailing westward into the Pacific and naming it was Ferdinand", "Columbus", "Magellan"),
        ("Backed by the Spanish crown in 1492 was Christopher", "Backed by the Spanish crown in 1519 was Ferdinand", "Columbus", "Magellan"),
    ]),
    _pair_task("History: 20th century milestones", "Angle 25: History", [
        ("The Second World War came to an end in the year", "The First World War came to an end in the year", "1945", "1918"),
        ("The first successful powered airplane flight was made by the Wright", "The first mass-produced affordable car was made by Henry", "brothers", "Ford"),
        ("The Berlin Wall, dividing a city for decades, finally fell in", "The Soviet Union formally dissolved in", "1989", "1991"),
        ("Humans first landed on the Moon in", "The Wright brothers first flew at Kitty Hawk in", "1969", "1903"),
    ]),
    _pair_task("History: scientists and their fields", "Angle 25: History", [
        ("The scientist who formulated the laws of motion and universal gravitation was Isaac", "The scientist who developed the theory of relativity was Albert", "Newton", "Einstein"),
        ("Radioactivity research and two Nobel Prizes are associated with Marie", "The laws of planetary orbits are associated with Johannes", "Curie", "Kepler"),
        ("The heliocentric model, placing the Sun at the center, was argued by Nicolaus", "The moons of Jupiter were first observed through a telescope by", "Copernicus", "Galileo"),
        ("The structure of DNA as a double helix was described by Watson and", "The pattern of inheritance in pea plants was described by Gregor", "Crick", "Mendel"),
    ]),
    _pair_task("History: leaders and movements", "Angle 25: History", [
        ("The nonviolent independence movement in India was led by Mahatma", "The American civil rights movement of the 1960s was led by Martin Luther", "Gandhi", "King"),
        ("The 'I Have a Dream' speech was delivered by Martin Luther", "The Salt March in 1930 was led by Mahatma", "King", "Gandhi"),
        ("The first democratically elected president of South Africa after apartheid was Nelson", "The wartime Prime Minister of Britain in the 1940s was Winston", "Mandela", "Churchill"),
        ("Twenty-seven years in prison before leading South Africa describes Nelson", "Leading Britain through the Blitz with defiant speeches describes Winston", "Mandela", "Churchill"),
    ]),
]


# ============================================================================
# Registries
# ============================================================================

ANGLE_1_BEHAVIORS = [
    ioi_behavior, agreement_behavior, subject_extraction_behavior,
    passive_voice_behavior, reflexive_pronoun_behavior, relative_clause_behavior,
    coordination_behavior,
]
ANGLE_2_BEHAVIORS = [
    factual_recall_behavior, currency_knowledge_behavior, element_symbol_behavior,
    language_knowledge_behavior, continent_knowledge_behavior, planet_knowledge_behavior,
    landmark_country_behavior,
]
ANGLE_3_BEHAVIORS = [
    arithmetic_addition_behavior, magnitude_comparison_behavior,
    arithmetic_subtraction_behavior, arithmetic_multiplication_behavior,
    temporal_reasoning_behavior, spatial_reasoning_behavior,
    parity_reasoning_behavior, ordinal_reasoning_behavior, arithmetic_division_behavior,
]
ANGLE_4_BEHAVIORS = [
    induction_copying_behavior, translation_induction_behavior,
    capitalization_induction_behavior, antonym_induction_behavior,
    first_letter_induction_behavior, number_sequence_induction_behavior,
    alphabet_sequence_induction_behavior,
]
ANGLE_5_BEHAVIORS = [
    gender_bias_behavior, age_bias_behavior, nationality_bias_behavior,
    appearance_bias_behavior, occupation_pronoun_bias_behavior,
]
ANGLE_6_BEHAVIORS = [
    antonym_behavior, category_membership_behavior, synonym_behavior,
    part_whole_behavior, animal_young_behavior, hypernym_instance_behavior,
    word_sense_disambiguation_behavior,
]
ANGLE_7_BEHAVIORS = [
    sentiment_polarity_behavior, emotion_classification_behavior,
    review_rating_behavior, tone_detection_behavior, praise_criticism_behavior,
]
ANGLE_8_BEHAVIORS = [
    python_output_behavior, boolean_logic_behavior, variable_type_behavior,
    list_index_behavior, comparison_operator_behavior, string_method_behavior,
]
ANGLE_9_BEHAVIORS = [
    object_affordance_behavior, causal_commonsense_behavior,
    material_property_behavior, object_purpose_behavior, size_commonsense_behavior,
]
ANGLE_10_BEHAVIORS = [
    french_number_behavior, spanish_vocab_behavior, german_color_behavior,
    italian_vocab_behavior, language_identification_behavior,
]
ANGLE_11_BEHAVIORS = [
    pronoun_coreference_behavior, possession_tracking_behavior,
    location_tracking_behavior, role_tracking_behavior, recency_tracking_behavior,
]
ANGLE_12_BEHAVIORS = [
    negation_behavior, double_negation_behavior, quantifier_reasoning_behavior,
    conditional_reasoning_behavior, contradiction_detection_behavior,
]

ANGLE_BEHAVIORS: dict[int, list] = {
    1: ANGLE_1_BEHAVIORS, 2: ANGLE_2_BEHAVIORS, 3: ANGLE_3_BEHAVIORS,
    4: ANGLE_4_BEHAVIORS, 5: ANGLE_5_BEHAVIORS, 6: ANGLE_6_BEHAVIORS,
    7: ANGLE_7_BEHAVIORS, 8: ANGLE_8_BEHAVIORS, 9: ANGLE_9_BEHAVIORS,
    10: ANGLE_10_BEHAVIORS, 11: ANGLE_11_BEHAVIORS, 12: ANGLE_12_BEHAVIORS,
    13: ANGLE_13_BEHAVIORS, 14: ANGLE_14_BEHAVIORS, 15: ANGLE_15_BEHAVIORS,
    16: ANGLE_16_BEHAVIORS, 17: ANGLE_17_BEHAVIORS, 18: ANGLE_18_BEHAVIORS,
    19: ANGLE_19_BEHAVIORS, 20: ANGLE_20_BEHAVIORS, 21: ANGLE_21_BEHAVIORS,
    22: ANGLE_22_BEHAVIORS, 23: ANGLE_23_BEHAVIORS, 24: ANGLE_24_BEHAVIORS,
    25: ANGLE_25_BEHAVIORS,
}

CANONICAL_16_BEHAVIORS = [
    # Angle 1: Linguistic Structure (Syntax & Coreference)
    ioi_behavior,
    agreement_behavior,
    subject_extraction_behavior,
    passive_voice_behavior,
    reflexive_pronoun_behavior,
    relative_clause_behavior,
    coordination_behavior,
    # Angle 2: Factual & World Knowledge
    factual_recall_behavior,
    currency_knowledge_behavior,
    element_symbol_behavior,
    # Angle 3: Reasoning & Arithmetic
    arithmetic_addition_behavior,
    magnitude_comparison_behavior,
    # Angle 4: In-Context Learning / Induction
    induction_copying_behavior,
    # Angle 5: Social Bias & Fairness
    gender_bias_behavior,
    # Angle 6: Lexical Semantics & Word Sense
    antonym_behavior,
    category_membership_behavior,
]

# The legacy 16-item subset remains explicitly available above, but must never
# reorder the full sweep: every stage and exported catalog uses angle order.
ALL_BEHAVIORS = [b for angle in sorted(ANGLE_BEHAVIORS) for b in ANGLE_BEHAVIORS[angle]]
