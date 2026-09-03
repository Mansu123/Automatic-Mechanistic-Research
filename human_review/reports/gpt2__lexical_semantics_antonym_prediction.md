# Lexical Semantics: antonym prediction

**Angle / category:** Angle 6: Lexical Semantics  
**Target model:** gpt2  
**Tool calls spent:** 47

## Task definition

- Clean prompt: `The opposite of light is`
- Corrupted prompt: `The opposite of rich is`
- Expected token: `dark`  vs. contrast token: `bright`
- Held-out eval prompts: 4

## Network Analyst: flagged layers

Flagged 12 layer(s): [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

## Layer Agent findings

| Layer | Fraction recovered | Spawned Component Agent | High superposition |
|---|---|---|---|
| L0 | 0.050 | False | False |
| L1 | 0.070 | False | False |
| L2 | 0.090 | False | False |
| L3 | -0.010 | False | False |
| L4 | -0.070 | False | False |
| L5 | -0.010 | False | False |
| L6 | 0.060 | False | False |
| L7 | -0.200 | False | False |
| L8 | -0.120 | False | False |
| L9 | 0.170 | True | False |
| L10 | 0.920 | False | False |
| L11 | 1.000 | True | False |

## Claimed circuit

`[(9, 2), (9, 7)]`

## Verification (Skeptic -> Judge)

**Verdict: Refuted**

Reasoning: ablating the claimed circuit did not produce a specific effect

Evidence transcript:
```
claimed circuit: [(9, 2), (9, 7)]
ablate_component: ablate_component L9H2 (mean): avg_delta=+0.029 over 4 prompts -> weak/no effect | ablate_component L9H7 (mean): avg_delta=-0.733 over 4 prompts -> SPECIFIC EFFECT
exclusion_ablation: not run
minimality_check: not run
counterexample_search: not run
```
