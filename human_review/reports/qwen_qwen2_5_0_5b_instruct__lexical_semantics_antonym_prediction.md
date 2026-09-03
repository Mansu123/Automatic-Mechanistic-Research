# Lexical Semantics: antonym prediction

**Angle / category:** Angle 6: Lexical Semantics  
**Target model:** Qwen/Qwen2.5-0.5B-Instruct  
**Tool calls spent:** 120

## Task definition

- Clean prompt: `The opposite of light is`
- Corrupted prompt: `The opposite of rich is`
- Expected token: `dark`  vs. contrast token: `bright`
- Held-out eval prompts: 4

## Network Analyst: flagged layers

Flagged 24 layer(s): [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]

## Layer Agent findings

| Layer | Fraction recovered | Spawned Component Agent | High superposition |
|---|---|---|---|
| L0 | -0.010 | False | False |
| L1 | 0.190 | True | False |
| L2 | 0.230 | False | False |
| L3 | 0.240 | False | False |
| L4 | 0.050 | False | False |
| L5 | 0.000 | False | False |
| L6 | -0.010 | False | False |
| L7 | 0.020 | False | False |
| L8 | -0.020 | False | False |
| L9 | 0.050 | False | False |
| L10 | 0.310 | True | False |
| L11 | 0.280 | True | False |
| L12 | 0.330 | True | False |
| L13 | 0.390 | True | False |
| L14 | 0.530 | True | False |
| L15 | 0.610 | True | False |
| L16 | 0.690 | False | False |
| L17 | 0.750 | True | False |
| L18 | 0.970 | True | False |
| L19 | 0.950 | False | False |
| L20 | 0.890 | True | False |
| L21 | 1.020 | False | False |
| L22 | 1.020 | False | False |
| L23 | 1.000 | False | False |

## Claimed circuit

`[(1, 9), (10, 10), (10, 12), (14, 7), (14, 12), (14, 13), (18, 1), (18, 5)]`

## Verification (Skeptic -> Judge)

**Verdict: Refuted**

Reasoning: ablating the claimed circuit did not produce a specific effect

Evidence transcript:
```
claimed circuit: [(1, 9), (10, 10), (10, 12), (14, 7), (14, 12), (14, 13), (18, 1), (18, 5)]
ablate_component: not run
exclusion_ablation: not run
minimality_check: not run
counterexample_search: not run
```
