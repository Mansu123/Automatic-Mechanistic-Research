# Indirect Object Identification (coreference)

**Angle / category:** Angle 1: Linguistic  
**Target model:** Qwen/Qwen2.5-3B-Instruct  
**Tool calls spent:** 62

## Task definition

- Clean prompt: `When Bob and Tom went to the store, Bob gave a drink to`
- Corrupted prompt: `When Tom and Bob went to the store, Tom gave a drink to`
- Expected token: `Tom`  vs. contrast token: `Bob`
- Held-out eval prompts: 4

## Network Analyst: flagged layers

Flagged 36 layer(s): [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35]

## Layer Agent findings

| Layer | Fraction recovered | Spawned Component Agent | High superposition |
|---|---|---|---|
| L0 | -0.000 | False | False |
| L1 | 0.000 | False | False |
| L2 | 0.000 | False | False |
| L3 | 0.010 | False | False |
| L4 | 0.010 | False | False |
| L5 | 0.020 | False | False |
| L6 | 0.000 | False | False |
| L7 | 0.000 | False | False |
| L8 | 0.000 | False | False |
| L9 | -0.010 | False | False |
| L10 | 0.000 | False | False |
| L11 | 0.000 | False | False |
| L12 | 0.010 | False | False |
| L13 | 0.010 | False | False |
| L14 | 0.020 | False | False |
| L15 | 0.030 | False | False |
| L16 | 0.030 | False | False |
| L17 | 0.030 | False | False |
| L18 | 0.020 | False | False |
| L19 | 0.020 | False | False |
| L20 | 0.010 | False | False |
| L21 | 0.010 | False | False |
| L22 | 0.010 | False | False |
| L23 | 0.010 | False | False |
| L24 | 0.010 | False | False |
| L25 | 0.030 | False | False |
| L26 | 0.010 | False | False |
| L27 | 0.000 | False | False |
| L28 | 0.030 | False | False |

## Claimed circuit

No component-level claim survived Layer Agent triage (null result).

## Verification (Skeptic -> Judge)

Not run -- no claim reached the Skeptic.
