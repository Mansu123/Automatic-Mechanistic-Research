# S-EAP (Synergy-Aware EAP) — empirical benchmark

Run: 2026-09-01, CPU, `python3 experiments/seap/seap_benchmark.py`

## Method

`Syn(i,j) = ( ∇_{A_i}L_clean − ∇_{A_i}L_ablate(j) ) · (A_i^clean − A_i^corrupt)`

Finite-difference screen for second-order head interactions. One backward pass
per candidate head `j` yields the interaction row for all `i`. Validated against
exact coalition patching: `Syn_exact(i,j) = m({i,j}) − m({i}) − m({j}) + m({})`
where `m(S)` = IOI logit-diff with head-set `S` mean-ablated.

Implementation: `tools/tier_c.py::run_synergy_eap`, `tools/adapter.py::ablate_head_set`.

## Results

### GPT-2 small, IOI (published ground truth)

- ACDC first-order recall: **1/9** GT heads, 1/8 backup name-movers — blind spot reproduced
- `Spearman(|S-EAP Syn|, |exact Syn|)` = **+0.653, p = 8.9e-11** (78 pairs) — valid magnitude screen
- `Spearman(signed, signed)` = **−0.118, p = 0.30** — **sign is NOT recovered**
- Largest exact interactions are all among φ₁≈0 heads:
  `(10,7)+(11,10)` negNM+negNM `Syn=+0.62`; `(10,10)+(11,10)` backup+negNM `Syn=−0.35`
- Faithfulness (fraction of clean−corrupt gap the circuit alone keeps):
  ACDC 0.314 → ACDC + top-4 synergy pairs **0.355**

### Qwen2.5-0.5B-Instruct, IOI (no ground truth)

- `Spearman(|S-EAP Syn|, |exact Syn|)` = **+0.20, p = 0.30** (28 pairs) — not significant
- Faithfulness: ACDC 0.530 → +synergy **0.576**
- Underpowered: 8 EAP-only candidates, no known backup structure to find

## Verdict

- S-EAP is a valid **magnitude** screen on GPT-2; use as shortlist → confirm sign
  with exact `ablate_head_set` coalition patching.
- Does not replicate on 0.5B model with a naive candidate set.
- Faithfulness gains small (+0.04) — not a dramatic unlock.
- Workshop-shaped mixed result, not ICML main-track.

## Pipeline integration (2026-09)

`component_agent.py` now runs a second-order pass after `run_acdc`:
`direct_logit_attribution` (write-direction seed) -> `run_synergy_eap` over
the layer + neighbours -> merge any head with |Syn| >= 0.05 to a circuit
head into `state["circuit"]`.

**Stage A, GPT-2 IOI, heuristic backend:**

| | per-layer ACDC only | + S-EAP second-order pass |
|---|---|---|
| claimed circuit | `[(9,9),(10,10)]` | `[(9,6),(9,9),(10,7),(10,10),(11,10)]` |
| circuit recall vs GT_HEADS | 11% (1/9) | **44% (4/9)** |
| recovers negative name movers (10,7),(11,10) | no | **yes** (the heads the README flagged as missed) |
| Judge verdict | — | **Refuted** |

The recall jump is real, but it exposes a **matching blind spot in the
Skeptic**: `do_ablate` tests each claimed head with individual
`ablate_component` and requires ALL to show a specific effect. Backup /
negative-mover heads (9,9), (10,10) show ~0 individual effect *by
definition*, so the first-order Skeptic refutes the whole (more complete)
claim. Fix: the Skeptic's ablation + minimality tests need to use
`ablate_head_set` on synergy-recovered heads as coalitions with their
partner, not singletons -- the same first-order->second-order upgrade,
applied to verification. Not yet done.

## Caveats / TODO before writeup

- Single prompt pair only — needs paired permutation test across `eval_prompts`
- Mean ablation throughout — compare against resampling ablation (OOD concern)
- Add a 2nd circuit with known ground truth (Greater-Than on GPT-2)
- Larger candidate sets on Qwen; FDR / Benjamini-Hochberg across tested pairs
