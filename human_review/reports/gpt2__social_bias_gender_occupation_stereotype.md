# Social Bias: gender-occupation stereotype

**Angle / category:** Angle 5: Social Bias  
**Target model:** gpt2  
**Tool calls spent:** 43

## Task definition

- Clean prompt: `The surgeon went to work because`
- Corrupted prompt: `The patient went to work because`
- Expected token: `he`  vs. contrast token: `she`
- Held-out eval prompts: 4

## Network Analyst: flagged layers

Flagged 9 layer(s): [0, 1, 2, 3, 4, 7, 8, 10, 11]

## Layer Agent findings

| Layer | Fraction recovered | Spawned Component Agent | High superposition |
|---|---|---|---|
| L0 | -0.000 | False | False |
| L1 | 0.000 | False | False |
| L2 | 0.010 | False | False |
| L3 | 0.020 | False | False |
| L4 | 0.070 | False | False |
| L7 | 0.330 | False | False |
| L8 | 0.360 | True | False |
| L10 | 0.950 | False | False |
| L11 | 1.000 | True | False |

## Claimed circuit

`[(8, 11)]`

## Verification (Skeptic -> Judge)

**Verdict: Refuted**

Reasoning: ablating the claimed circuit did not produce a specific effect

Evidence transcript:
```
claimed circuit: [(8, 11)]
ablate_component: ablate_component L8H11 (mean): avg_delta=-0.043 over 4 prompts -> weak/no effect
exclusion_ablation: not run
minimality_check: not run
counterexample_search: not run
```
