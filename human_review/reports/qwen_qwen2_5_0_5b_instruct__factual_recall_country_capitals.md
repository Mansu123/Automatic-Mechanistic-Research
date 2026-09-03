# Factual recall: country capitals

**Angle / category:** Angle 2: Factual  
**Target model:** Qwen/Qwen2.5-0.5B-Instruct  
**Tool calls spent:** 59

## Task definition

- Clean prompt: `The capital of Spain is`
- Corrupted prompt: `The capital of Germany is`
- Expected token: `Madrid`  vs. contrast token: `Rome`
- Held-out eval prompts: 4

## Network Analyst: flagged layers

Flagged 22 layer(s): [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 16, 17, 18, 19, 20, 21, 22, 23]

## Layer Agent findings

| Layer | Fraction recovered | Spawned Component Agent | High superposition |
|---|---|---|---|
| L0 | -0.010 | False | False |
| L1 | 0.000 | False | False |
| L2 | 0.000 | False | False |
| L3 | -0.010 | False | False |
| L4 | -0.020 | False | False |
| L5 | -0.010 | False | False |
| L6 | 0.000 | False | False |
| L7 | 0.000 | False | False |
| L8 | -0.010 | False | False |
| L9 | -0.010 | False | False |
| L11 | -0.010 | False | False |
| L12 | -0.040 | False | False |
| L13 | -0.040 | False | False |
| L14 | -0.050 | False | False |
| L16 | -0.050 | False | False |
| L17 | 0.040 | False | False |
| L18 | 0.040 | False | False |
| L19 | 0.050 | False | False |
| L20 | 0.150 | True | False |
| L21 | 0.990 | False | False |
| L22 | 1.000 | False | False |
| L23 | 1.000 | False | False |

## Claimed circuit

No component-level claim survived Layer Agent triage (null result).

## Verification (Skeptic -> Judge)

Not run -- no claim reached the Skeptic.
