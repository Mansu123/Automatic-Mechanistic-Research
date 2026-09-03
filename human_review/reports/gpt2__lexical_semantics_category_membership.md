# Lexical Semantics: category membership

**Angle / category:** Angle 6: Lexical Semantics  
**Target model:** gpt2  
**Tool calls spent:** 59

## Task definition

- Clean prompt: `A rose is a type of`
- Corrupted prompt: `A salmon is a type of`
- Expected token: `flower`  vs. contrast token: `tree`
- Held-out eval prompts: 4

## Network Analyst: flagged layers

Flagged 11 layer(s): [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11]

## Layer Agent findings

| Layer | Fraction recovered | Spawned Component Agent | High superposition |
|---|---|---|---|
| L0 | -0.010 | False | False |
| L1 | -0.000 | False | False |
| L2 | -0.010 | False | False |
| L3 | -0.010 | False | False |
| L4 | -0.010 | False | False |
| L5 | 0.020 | False | False |
| L6 | 0.100 | False | False |
| L7 | 0.170 | True | False |
| L8 | 0.400 | True | False |
| L9 | 0.670 | True | False |
| L11 | 1.000 | True | False |

## Claimed circuit

`[(7, 8), (8, 8), (8, 11), (9, 6), (11, 3)]`

## Verification (Skeptic -> Judge)

**Verdict: Refuted**

Reasoning: ablating the claimed circuit did not produce a specific effect

Evidence transcript:
```
claimed circuit: [(7, 8), (8, 8), (8, 11), (9, 6), (11, 3)]
ablate_component: ablate_component L7H8 (mean): avg_delta=-0.004 over 4 prompts -> weak/no effect | ablate_component L8H8 (mean): avg_delta=-0.067 over 4 prompts -> weak/no effect | ablate_component L8H11 (mean): avg_delta=-0.075 over 4 prompts -> weak/no effect | ablate_component L9H6 (mean): avg_delta=-0.004 over 4 prompts -> weak/no effect | ablate_component L11H3 (mean): avg_delta=-0.076 over 4 prompts -> weak/no effect
exclusion_ablation: not run
minimality_check: not run
counterexample_search: not run
```
