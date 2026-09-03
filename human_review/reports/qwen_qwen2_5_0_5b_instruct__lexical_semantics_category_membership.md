# Lexical Semantics: category membership

**Angle / category:** Angle 6: Lexical Semantics  
**Target model:** Qwen/Qwen2.5-0.5B-Instruct  
**Tool calls spent:** 97

## Task definition

- Clean prompt: `A rose is a type of`
- Corrupted prompt: `A salmon is a type of`
- Expected token: `flower`  vs. contrast token: `tree`
- Held-out eval prompts: 4

## Network Analyst: flagged layers

Flagged 23 layer(s): [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]

## Layer Agent findings

| Layer | Fraction recovered | Spawned Component Agent | High superposition |
|---|---|---|---|
| L0 | -0.000 | False | False |
| L1 | -0.000 | False | False |
| L2 | 0.000 | False | False |
| L3 | -0.000 | False | False |
| L4 | 0.000 | False | False |
| L5 | 0.010 | False | False |
| L6 | 0.020 | False | False |
| L7 | 0.020 | False | False |
| L8 | 0.000 | False | False |
| L9 | 0.030 | False | False |
| L10 | 0.040 | False | False |
| L12 | 0.030 | False | False |
| L13 | 0.030 | False | False |
| L14 | 0.220 | True | False |
| L15 | 0.240 | True | False |
| L16 | 0.230 | True | False |
| L17 | 0.250 | True | False |
| L18 | 0.300 | True | False |
| L19 | 0.350 | True | False |
| L20 | 0.890 | True | False |
| L21 | 0.990 | False | False |
| L22 | 1.020 | False | False |
| L23 | 1.000 | False | False |

## Claimed circuit

`[(14, 13), (20, 6)]`

## Verification (Skeptic -> Judge)

**Verdict: Probable**

Reasoning: passes core tests but verification suite was not fully run

Evidence transcript:
```
claimed circuit: [(14, 13), (20, 6)]
ablate_component: ablate_component L14H13 (mean): avg_delta=-0.203 over 4 prompts -> SPECIFIC EFFECT | ablate_component L20H6 (mean): avg_delta=-0.297 over 4 prompts -> SPECIFIC EFFECT
exclusion_ablation: exclusion_ablation: max single-head drop outside claim = 2.312 -> INCOMPLETE (unexplained heads found)
minimality_check: minimality_check: [L14H13:0.203; L20H6:0.297] -> MINIMAL
counterexample_search: counterexample_search('Lexical Semantics: category membership', budget=20): 1/20 COUNTEREXAMPLES FOUND
```
