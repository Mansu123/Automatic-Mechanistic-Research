# Subject-verb number agreement (syntax)

**Angle / category:** Angle 1: Linguistic  
**Target model:** Qwen/Qwen2.5-3B-Instruct  
**Tool calls spent:** 62

## Task definition

- Clean prompt: `The books to the shelf`
- Corrupted prompt: `The book to the shelves`
- Expected token: `are`  vs. contrast token: `is`
- Held-out eval prompts: 4

## Network Analyst: flagged layers

Flagged 36 layer(s): [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35]

## Layer Agent findings

| Layer | Fraction recovered | Spawned Component Agent | High superposition |
|---|---|---|---|
| L0 | -0.760 | False | False |
| L1 | -0.790 | False | False |
| L2 | -0.780 | False | False |
| L3 | -0.830 | False | False |
| L4 | -0.830 | False | False |
| L5 | -0.820 | False | False |
| L6 | -0.850 | False | False |
| L7 | -0.860 | False | False |
| L8 | -0.850 | False | False |
| L9 | -0.850 | False | False |
| L10 | -0.820 | False | False |
| L11 | -0.860 | False | False |
| L12 | -0.930 | False | False |
| L13 | -0.930 | False | False |
| L14 | -1.010 | False | False |
| L15 | -0.960 | False | False |
| L16 | -0.930 | False | False |
| L17 | -0.920 | False | False |
| L18 | -0.850 | False | False |
| L19 | -0.810 | False | False |
| L20 | -0.780 | False | False |
| L21 | -0.750 | False | False |
| L22 | -0.720 | False | False |
| L23 | -0.640 | False | False |
| L24 | -0.500 | False | False |
| L25 | -0.440 | False | False |
| L26 | 0.030 | False | False |
| L27 | 0.060 | False | False |
| L28 | 0.060 | False | False |

## Claimed circuit

No component-level claim survived Layer Agent triage (null result).

## Verification (Skeptic -> Judge)

Not run -- no claim reached the Skeptic.
