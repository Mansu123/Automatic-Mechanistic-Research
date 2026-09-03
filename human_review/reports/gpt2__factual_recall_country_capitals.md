# Factual recall: country capitals

**Angle / category:** Angle 2: Factual  
**Target model:** gpt2  
**Tool calls spent:** 49

## Task definition

- Clean prompt: `The capital of Spain is`
- Corrupted prompt: `The capital of Germany is`
- Expected token: `Madrid`  vs. contrast token: `Rome`
- Held-out eval prompts: 4

## Network Analyst: flagged layers

Flagged 12 layer(s): [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

## Layer Agent findings

| Layer | Fraction recovered | Spawned Component Agent | High superposition |
|---|---|---|---|
| L0 | -0.000 | False | False |
| L1 | -0.000 | False | False |
| L2 | -0.010 | False | False |
| L3 | -0.010 | False | False |
| L4 | -0.010 | False | False |
| L5 | -0.010 | False | False |
| L6 | 0.010 | False | False |
| L7 | 0.000 | False | False |
| L8 | 0.070 | False | False |
| L9 | 0.730 | True | False |
| L10 | 0.990 | False | False |
| L11 | 1.000 | True | False |

## Claimed circuit

`[(9, 8)]`

## Verification (Skeptic -> Judge)

**Verdict: Probable**

Reasoning: passes core tests but verification suite was not fully run

Evidence transcript:
```
claimed circuit: [(9, 8)]
ablate_component: ablate_component L9H8 (mean): avg_delta=-0.406 over 4 prompts -> SPECIFIC EFFECT
exclusion_ablation: exclusion_ablation: max single-head drop outside claim = 0.755 -> INCOMPLETE (unexplained heads found)
minimality_check: minimality_check: [L9H8:0.406] -> MINIMAL
counterexample_search: counterexample_search('Factual recall: country capitals', budget=20): 9/20 COUNTEREXAMPLES FOUND
```
