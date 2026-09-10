"""Authoritative 25-angle/203-behavior catalog shared by execution and review."""
from ..behaviors import ALL_BEHAVIORS, ANGLE_BEHAVIORS

ANGLE_NAMES = [
    "Linguistic Structure (Syntax & Coreference)", "Factual & World Knowledge",
    "Reasoning & Arithmetic", "In-Context Learning / Induction", "Social Bias & Fairness",
    "Lexical Semantics & Word Sense", "Sentiment & Emotion", "Code & Formal Reasoning",
    "Commonsense & Physical Reasoning", "Multilingual / Cross-lingual",
    "Entity Tracking & Discourse", "Negation & Logic", "Quantitative Comparison",
    "Temporal & Sequential Ordering", "Analogical Reasoning", "Morphology & Word Formation",
    "Pragmatics & Implicature", "Scientific & Technical Knowledge", "Geography & Spatial Knowledge",
    "Arts, Literature & Culture", "Economics, Politics & Law", "Idioms & Figurative Language",
    "Counting & Set Facts", "Units & Measurement", "History & Chronology",
]
BUILDERS = {b.__name__: b for b in ALL_BEHAVIORS}
ANGLE_BY_BEHAVIOR = {b.__name__: a for a, builders in ANGLE_BEHAVIORS.items() for b in builders}
COUNTS = {a: len(bs) for a, bs in ANGLE_BEHAVIORS.items()}
assert len(BUILDERS) == len(ALL_BEHAVIORS) == 203 and len(COUNTS) == 25


def metadata(behavior_id):
    angle = ANGLE_BY_BEHAVIOR[behavior_id]
    return {"behavior_id": behavior_id, "angle": angle, "angle_name": ANGLE_NAMES[angle-1]}
