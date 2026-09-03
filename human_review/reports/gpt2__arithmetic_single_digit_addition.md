# Arithmetic: single-digit addition

**Angle / category:** Angle 3: Arithmetic  
**Target model:** gpt2  
**Tool calls spent:** 44

## Task definition

- Clean prompt: `The sum of 4 and 5 is`
- Corrupted prompt: `The sum of 1 and 8 is`
- Expected token: `9`  vs. contrast token: `7`
- Held-out eval prompts: 4

## Network Analyst: flagged layers

Flagged 8 layer(s): [2, 3, 5, 6, 8, 9, 10, 11]

## Layer Agent findings

| Layer | Fraction recovered | Spawned Component Agent | High superposition |
|---|---|---|---|
| L2 | 0.040 | False | False |
| L3 | 0.060 | False | False |
| L5 | 0.080 | False | False |
| L6 | 0.080 | False | False |
| L8 | 0.320 | True | False |
| L9 | 0.560 | True | False |
| L10 | 1.000 | False | False |
| L11 | 1.000 | True | False |

## Claimed circuit

`[(9, 1)]`

## Verification (Skeptic -> Judge)

**Verdict: Refuted**

Reasoning: ablating the claimed circuit did not produce a specific effect

Evidence transcript:
```
claimed circuit: [(9, 1)]
ablate_component: ablate_component L9H1 (mean): avg_delta=+0.006 over 4 prompts -> weak/no effect
exclusion_ablation: not run
minimality_check: not run
counterexample_search: not run
```
