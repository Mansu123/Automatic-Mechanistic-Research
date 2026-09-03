# Induction: repeated token copying

**Angle / category:** Angle 4: Induction  
**Target model:** gpt2  
**Tool calls spent:** 81

## Task definition

- Clean prompt: `alpha beta gamma alpha`
- Corrupted prompt: `alpha beta gamma beta`
- Expected token: `beta`  vs. contrast token: `alpha`
- Held-out eval prompts: 4

## Network Analyst: flagged layers

Flagged 10 layer(s): [0, 1, 2, 3, 4, 5, 7, 8, 9, 11]

## Layer Agent findings

| Layer | Fraction recovered | Spawned Component Agent | High superposition |
|---|---|---|---|
| L0 | 1.000 | True | False |
| L1 | 1.000 | False | False |
| L2 | 1.000 | False | False |
| L3 | 1.000 | False | False |
| L4 | 1.000 | False | False |
| L5 | 1.000 | False | False |
| L7 | 1.000 | True | False |
| L8 | 1.000 | True | False |
| L9 | 1.000 | True | False |
| L11 | 1.000 | True | False |

## Claimed circuit

`[(0, 1), (0, 4), (0, 5), (0, 9), (0, 11), (7, 9), (8, 0), (8, 6), (8, 7), (8, 8), (8, 10), (9, 1), (9, 7), (11, 0), (11, 7), (11, 8), (11, 10)]`

## Verification (Skeptic -> Judge)

**Verdict: Refuted**

Reasoning: ablating the claimed circuit did not produce a specific effect

Evidence transcript:
```
claimed circuit: [(0, 1), (0, 4), (0, 5), (0, 9), (0, 11), (7, 9), (8, 0), (8, 6), (8, 7), (8, 8), (8, 10), (9, 1), (9, 7), (11, 0), (11, 7), (11, 8), (11, 10)]
ablate_component: ablate_component L0H1 (mean): avg_delta=+0.037 over 4 prompts -> weak/no effect | ablate_component L0H4 (mean): avg_delta=-0.007 over 4 prompts -> weak/no effect | ablate_component L0H5 (mean): avg_delta=+0.026 over 4 prompts -> weak/no effect | ablate_component L0H9 (mean): avg_delta=+0.003 over 4 prompts -> weak/no effect | ablate_component L0H11 (mean): avg_delta=-0.027 over 4 prompts -> weak/no effect | ablate_component L7H9 (mean): avg_delta=-0.008 over 4 prompts -> weak/no effect | ablate_component L8H0 (mean): avg_delta=-0.008 over 4 prompts -> weak/no effect | ablate_component L8H6 (mean): avg_delta=-0.048 over 4 prompts -> weak/no effect | ablate_component L8H7 (mean): avg_delta=+0.009 over 4 prompts -> weak/no effect | ablate_component L8H8 (mean): avg_delta=+0.009 over 4 prompts -> weak/no effect | ablate_component L8H10 (mean): avg_delta=-0.010 over 4 prompts -> weak/no effect | ablate_component L9H1 (mean): avg_delta=-0.004 over 4 prompts -> weak/no effect | ablate_component L9H7 (mean): avg_delta=-0.013 over 4 prompts -> weak/no effect | ablate_component L11H0 (mean): avg_delta=-0.035 over 4 prompts -> weak/no effect | ablate_component L11H7 (mean): avg_delta=-0.016 over 4 prompts -> weak/no effect | ablate_component L11H8 (mean): avg_delta=+0.243 over 4 prompts -> SPECIFIC EFFECT | ablate_component L11H10 (mean): avg_delta=-0.068 over 4 prompts -> weak/no effect
exclusion_ablation: not run
minimality_check: not run
counterexample_search: not run
```
