# Subject-verb number agreement (syntax)

**Angle / category:** Angle 1: Linguistic  
**Target model:** Qwen/Qwen2.5-0.5B-Instruct  
**Tool calls spent:** 73

## Task definition

- Clean prompt: `The books to the shelf`
- Corrupted prompt: `The book to the shelves`
- Expected token: `are`  vs. contrast token: `is`
- Held-out eval prompts: 4

## Network Analyst: flagged layers

Flagged 24 layer(s): [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]

## Layer Agent findings

| Layer | Fraction recovered | Spawned Component Agent | High superposition |
|---|---|---|---|
| L0 | -0.880 | False | False |
| L1 | -0.830 | False | False |
| L2 | -0.840 | False | False |
| L3 | -0.840 | False | False |
| L4 | -0.860 | False | False |
| L5 | -0.890 | False | False |
| L6 | -0.860 | False | False |
| L7 | -0.810 | False | False |
| L8 | -0.890 | False | False |
| L9 | -0.800 | False | False |
| L10 | -0.810 | False | False |
| L11 | -0.830 | False | False |
| L12 | -0.770 | False | False |
| L13 | -0.800 | False | False |
| L14 | -0.670 | False | False |
| L15 | -0.640 | False | False |
| L16 | -0.620 | False | False |
| L17 | -0.590 | False | False |
| L18 | -0.620 | False | False |
| L19 | 1.000 | True | False |
| L20 | 1.020 | True | False |
| L21 | 1.000 | False | False |
| L22 | 1.030 | False | False |
| L23 | 1.000 | False | False |

## Claimed circuit

`[(19, 4)]`

## Verification (Skeptic -> Judge)

**Verdict: Probable**

Reasoning: passes core tests but verification suite was not fully run

Evidence transcript:
```
claimed circuit: [(19, 4)]
ablate_component: ablate_component L19H4 (mean): avg_delta=-2.188 over 4 prompts -> SPECIFIC EFFECT
exclusion_ablation: exclusion_ablation: max single-head drop outside claim = 0.875 -> INCOMPLETE (unexplained heads found)
minimality_check: minimality_check: [L19H4:2.188] -> MINIMAL
counterexample_search: counterexample_search('Subject-verb number agreement (syntax)', budget=20): 6/20 COUNTEREXAMPLES FOUND
```
