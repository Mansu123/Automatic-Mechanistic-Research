# Subject-verb number agreement (syntax)

**Angle / category:** Angle 1: Linguistic  
**Target model:** gpt2  
**Tool calls spent:** 64

## Task definition

- Clean prompt: `The books to the shelf`
- Corrupted prompt: `The book to the shelves`
- Expected token: `are`  vs. contrast token: `is`
- Held-out eval prompts: 4

## Network Analyst: flagged layers

Flagged 11 layer(s): [0, 1, 3, 4, 5, 6, 7, 8, 9, 10, 11]

## Layer Agent findings

| Layer | Fraction recovered | Spawned Component Agent | High superposition |
|---|---|---|---|
| L0 | -0.220 | False | False |
| L1 | -0.230 | False | False |
| L3 | -0.220 | False | False |
| L4 | -0.190 | False | False |
| L5 | -0.040 | False | False |
| L6 | 0.150 | False | False |
| L7 | 0.590 | True | False |
| L8 | 0.670 | True | False |
| L9 | 0.680 | True | False |
| L10 | 0.960 | False | False |
| L11 | 1.000 | True | False |

## Claimed circuit

`[(7, 4), (8, 5), (11, 10)]`

## Verification (Skeptic -> Judge)

**Verdict: Refuted**

Reasoning: ablating the claimed circuit did not produce a specific effect

Evidence transcript:
```
claimed circuit: [(7, 4), (8, 5), (11, 10)]
ablate_component: ablate_component L7H4 (mean): avg_delta=-0.334 over 4 prompts -> SPECIFIC EFFECT | ablate_component L8H5 (mean): avg_delta=-0.077 over 4 prompts -> weak/no effect | ablate_component L11H10 (mean): avg_delta=-0.157 over 4 prompts -> weak/no effect
exclusion_ablation: not run
minimality_check: not run
counterexample_search: not run
```
