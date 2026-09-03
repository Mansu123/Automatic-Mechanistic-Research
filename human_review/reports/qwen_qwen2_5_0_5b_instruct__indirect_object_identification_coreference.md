# Indirect Object Identification (coreference)

**Angle / category:** Angle 1: Linguistic  
**Target model:** Qwen/Qwen2.5-0.5B-Instruct  
**Tool calls spent:** 70

## Task definition

- Clean prompt: `When Bob and Tom went to the store, Bob gave a drink to`
- Corrupted prompt: `When Tom and Bob went to the store, Tom gave a drink to`
- Expected token: `Tom`  vs. contrast token: `Bob`
- Held-out eval prompts: 4

## Network Analyst: flagged layers

Flagged 24 layer(s): [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]

## Layer Agent findings

| Layer | Fraction recovered | Spawned Component Agent | High superposition |
|---|---|---|---|
| L0 | -0.060 | False | False |
| L1 | -0.050 | False | False |
| L2 | -0.050 | False | False |
| L3 | -0.050 | False | False |
| L4 | -0.060 | False | False |
| L5 | -0.080 | False | False |
| L6 | -0.060 | False | False |
| L7 | -0.110 | False | False |
| L8 | -0.090 | False | False |
| L9 | -0.110 | False | False |
| L10 | -0.140 | False | False |
| L11 | -0.120 | False | False |
| L12 | -0.140 | False | False |
| L13 | -0.110 | False | False |
| L14 | -0.090 | False | False |
| L15 | -0.150 | False | False |
| L16 | -0.090 | False | False |
| L17 | -0.120 | False | False |
| L18 | -0.090 | False | False |
| L19 | -0.300 | False | False |
| L20 | 0.390 | True | False |
| L21 | 0.890 | False | False |
| L22 | 1.090 | False | False |
| L23 | 1.000 | True | False |

## Claimed circuit

`[(20, 9), (23, 2)]`

## Verification (Skeptic -> Judge)

**Verdict: Refuted**

Reasoning: ablating the claimed circuit did not produce a specific effect

Evidence transcript:
```
claimed circuit: [(20, 9), (23, 2)]
ablate_component: ablate_component L20H9 (mean): avg_delta=-0.922 over 4 prompts -> SPECIFIC EFFECT | ablate_component L23H2 (mean): avg_delta=+0.016 over 4 prompts -> weak/no effect
exclusion_ablation: not run
minimality_check: not run
counterexample_search: not run
```
