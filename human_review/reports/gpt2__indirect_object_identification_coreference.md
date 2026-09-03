# Indirect Object Identification (coreference)

**Angle / category:** Angle 1: Linguistic  
**Target model:** gpt2  
**Tool calls spent:** 52

## Task definition

- Clean prompt: `When Bob and Tom went to the store, Bob gave a drink to`
- Corrupted prompt: `When Tom and Bob went to the store, Tom gave a drink to`
- Expected token: `Tom`  vs. contrast token: `Bob`
- Held-out eval prompts: 4

## Network Analyst: flagged layers

Flagged 12 layer(s): [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

## Layer Agent findings

| Layer | Fraction recovered | Spawned Component Agent | High superposition |
|---|---|---|---|
| L0 | -0.000 | False | False |
| L1 | -0.000 | False | False |
| L2 | -0.000 | False | False |
| L3 | -0.000 | False | False |
| L4 | -0.000 | False | False |
| L5 | 0.000 | False | False |
| L6 | -0.000 | False | False |
| L7 | 0.060 | False | False |
| L8 | 0.140 | False | False |
| L9 | 0.450 | True | False |
| L10 | 0.940 | True | False |
| L11 | 1.000 | True | False |

## Claimed circuit

`[(9, 9), (10, 10)]`

## Verification (Skeptic -> Judge)

**Verdict: Refuted**

Reasoning: ablating the claimed circuit did not produce a specific effect

Evidence transcript:
```
claimed circuit: [(9, 9), (10, 10)]
ablate_component: ablate_component L9H9 (mean): avg_delta=-0.031 over 4 prompts -> weak/no effect | ablate_component L10H10 (mean): avg_delta=-0.254 over 4 prompts -> SPECIFIC EFFECT
exclusion_ablation: not run
minimality_check: not run
counterexample_search: not run
```
