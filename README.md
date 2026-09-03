# AutoMechInterp

A runnable implementation of the AutoMechInterp research proposal: a hierarchical
multi-agent system that autonomously performs mechanistic interpretability on a
neural network — profiling it layer by layer, discovering circuits, decomposing
them into sparse features, diffing fine-tuned models against their base, and
adversarially verifying every claim before trusting it.

## Abstract

Automated interpretability today is either one flat agent asked to do too
much (MAIA 2024's documented confirmation bias) or a fixed statistical
pipeline that can't decide what to investigate next (Dalvi et al. 2019).
AutoMechInterp addresses the systems-level version of that problem with a
hierarchical multi-agent architecture whose structure mirrors the target
network's own structure — a Network Analyst, per-layer Layer Agents,
per-head Component Agents — plus a structurally separate Skeptic/Judge pair
whose entire job is to falsify every claim before it gets trusted, rather
than an agent grading its own work.

That architecture alone is an engineering contribution, not a theoretical
one. On top of it, this project adds a genuine mathematical layer:
**Synergy-Aware Circuit Discovery (S-ACDC)** — a cooperative-game-theoretic
extension to ACDC/EAP-style circuit discovery that formally targets a
documented blind spot neither algorithm solves: components whose individual
ablation effect is near zero because another component compensates for them
(e.g. GPT-2's IOI task has published "backup name-mover" heads that exist
for exactly this reason), which greedy single-edge pruning cannot detect by
construction. See [§ The theoretical layer](#the-theoretical-layer-synergy-aware-circuit-discovery-s-acdc)
below for the formal definition. Honest current status: the first piece
(the **S-EAP** gradient screen) is implemented and benchmarked with a
mixed result — a valid interaction-*magnitude* screen on GPT-2 (Spearman
0.65 vs. exact coalition patching) that loses sign information and did not
replicate on a 0.5B model; the full statistically-controlled S-ACDC loop
is still a specification. Details in [§ Status update](#status-update-2026-09-s-eap-implemented)
and `experiments/seap/`.

## Research question this solves

> Can a hierarchical multi-agent LLM system — whose agent structure mirrors the
> target network's architecture — autonomously analyze a model layer by layer,
> discover and causally verify the circuits within it, and produce a verified
> mechanistic account, without a human researcher driving each step?

Prior automated-interpretability systems fall short in specific, documented ways:

| Prior work | What it couldn't do |
|---|---|
| Dalvi et al. 2019 | Fixed statistical pipeline — can't decide what to test next |
| SASC (2023) | One-pass, single-shot — never follows up on a weak explanation |
| FIND / AIA (2023) | Only tested on synthetic functions, no real network internals |
| MAIA (2024) | Single agent, documented confirmation bias, correlational only |

Run `python3 main.py --gap-matrix` to see exactly which class/function in this
codebase closes each of these gaps.

A second, narrower question sits underneath the systems-level one above:

> Even with a working multi-agent pipeline, does the circuit-discovery
> algorithm it calls (ACDC, EAP) have blind spots — and can they be fixed
> with something more principled than a bigger threshold?

`stage_a.py`'s own ground-truth comparison already shows evidence of this:
against GPT-2's published IOI circuit, this pipeline's standard `run_acdc`
recovers only 1 of 9 ground-truth heads (11% recall — see [§ What a real run
actually found](#what-a-real-run-actually-found-gpt-2-walkthrough)), missing
both of the IOI paper's documented "backup" heads entirely. That's not a bug
in this repo specifically — it's a known structural limitation of testing
each component's ablation effect one at a time (see [§ The theoretical
layer](#the-theoretical-layer-synergy-aware-circuit-discovery-s-acdc)).

## Quickstart — how it runs

**The mental model.** `main.py` is the entrypoint; `run_eval_matrix.py` and the
`human_review/` scripts are thin wrappers around it for multi-model sweeps and
scoring. `main.py` builds a *task* (a clean prompt, a corrupted prompt, and a metric —
e.g. IOI: does the model predict the un-repeated name?), then runs the agent
hierarchy on a *target model*: the Network Analyst flags interesting layers →
a Layer Agent per layer runs in parallel → Component Agents find the
responsible heads → the Skeptic tries to falsify the claimed circuit → the
Judge rules Confirmed / Probable / Speculative / Refuted. The "brain" driving
the agents is a deterministic rule table by default (**no API key, no GPU,
no downloads beyond the target model**); swap in a local or API LLM with one
flag. Everything writes a PNG + JSON (and often a Markdown report) to
`output/`.

### Setup

```bash
cd "/Users/mansuba/Desktop/MI MAIN RESEARCH/  2 Automatic-Mechanistic-Research-main"
python3 -m venv .venv && source .venv/bin/activate      # optional
pip install -r requirements.txt
```

### Every command, and what it produces

| Command | What it does | Output |
|---|---|---|
| `python3 main.py` | **Stage A** — full hierarchy on GPT-2 / IOI, scored against the published ground-truth circuit, with negative controls | console |
| `python3 main.py --graph` | Stage A + saved plots | `output/stage_a_*.png/.json` |
| `python3 main.py --ablations --graph` | Sec. 4.7 ablations: hierarchy vs flat, verification on/off, Network-Analyst flagging strategy | `output/ablations_*` |
| `python3 main.py --stage-b --graph` | **Stage B** — run the hierarchy across every behavior, assemble a cross-behavior Layer Atlas | `output/stage_b_atlas_*` |
| `python3 main.py --stage-c --graph` | **Stage C** — CKA layer alignment + causal transplant between GPT-2 and a biomedical fine-tune | `output/stage_c_diff_*` |
| `python3 main.py --stage-d` | **Stage D** — write one human-readable report per (model, behavior) for scoring | `human_review/reports/*.md`, `output/eval/stage_d_*.json` |
| `python3 main.py --stage-d --ai-judge anthropic` | Stage D + auto-score every report with the RubricJudge (needs `ANTHROPIC_API_KEY`; `openai` also works) | same + `ai_scores` in the JSON |
| `python3 main.py --techniques` | run the **~70-technique toolkit** (logit lens, patching, probing, SAEs, steering, model diffing, …) on one model + task | `experiments/techniques/techniques_*.md/.json` |
| `python3 main.py --gap-matrix` | print the related-work gap table (which class/function closes each prior system's limitation) and exit | console |
| `python3 main.py --smoke-test-qwen` | prove the local-LLM agent backend works, using a tiny Qwen model | console |
| `python3 run_eval_matrix.py --list` | show the ≤7B **evaluation model matrix** (families, sizes, tiers, ladders) | console |
| `python3 run_eval_matrix.py --ladder pythia --purge-downloads` | run Stage D across a whole size ladder, one model at a time, deleting each after | reports + `output/eval/eval_matrix_*.json` |
| `python3 human_review/agent_layer_matrix.py` | build the agent × layer matrix for every report in `human_review/reports/` | `output/eval/agent_layer_matrix.md` |
| `python3 human_review/aggregate_scores.py` | combine the 3 human reviewers' CSVs, report inter-rater agreement | `output/eval/human_review_aggregate.csv` |

**Common modifiers** (any `main.py` run):

```bash
python3 main.py --stage-d --target-model Qwen/Qwen2.5-0.5B-Instruct   # interpret a different model
python3 main.py --backend hf_local --heavy-model Qwen/Qwen2.5-3B-Instruct   # real LLM drives the agents
python3 main.py --graph 2>&1 | tee "output/run_$(date +%Y%m%d_%H%M%S).log"   # keep the full transcript
```

Deeper detail on each: [§ Running it](#running-it) · [§ Evaluation model matrix](#evaluation-model-matrix-cross-model--scaling-sweep)
· [§ The technique toolkit](#the-technique-toolkit-automechinterptechniques) ·
[`COMMANDS_eval_matrix.md`](COMMANDS_eval_matrix.md) · [`human_review/README.md`](human_review/README.md)

## How it works

The system mirrors the target network's own structure instead of using one flat
agent with every tool in its context at once:

```
Orchestrator
  -> Network Analyst        profiles the whole model, flags "interesting" layers
       -> Layer Agent (x N)     one per flagged layer, runs in parallel
            -> Component Agent   spawned only if a layer's effect is attention-driven;
                                  finds the specific head(s) responsible, then
                                  decomposes them into sparse (SAE) features
       -> Skeptic                tries to FALSIFY the claimed circuit
            -> Judge               reads Prover's claim + Skeptic's evidence,
                                    assigns Confirmed / Probable / Speculative / Refuted
```

Nothing is trusted just because an agent found it. The **Skeptic** is a
structurally separate agent that never sees the Component Agent's reasoning —
only the claimed circuit — and tries to break it four ways:

1. **`ablate_component`** — does removing it actually hurt the behavior?
2. **`exclusion_ablation`** — does ablating everything *outside* the claim also
   hurt it? (if so, the claim is incomplete)
3. **`minimality_check`** — does every claimed head matter on its own? (if not,
   the claim is padded)
4. **`counterexample_search`** — is there an input where the claimed circuit is
   active but doesn't explain the behavior?

Only a circuit that survives all four gets labeled **Confirmed** by the Judge.

Every agent (Network Analyst, Layer Agent, Component Agent, Skeptic, Judge) is
driven by the same interface — `LLMBackend.decide(system, evidence, tools) ->
next_action` — so the "brain" behind the decisions is swappable with zero
changes to agent logic: a hand-written rule table by default, or a real
open-source LLM (Qwen2.5-7B-Instruct, or a smaller Qwen model — this machine
comfortably handles up to ~3-4B) when you have the hardware for it.

## The theoretical layer: Synergy-Aware Circuit Discovery (S-ACDC)

> **Status: partially implemented and benchmarked (2026-09).** The
> gradient-based screen (`run_synergy_eap`) and the coalition primitive
> (`ablate_head_set`) now run; the full statistically-controlled loop
> (FDR correction, split-half replication, iterate-to-convergence) is
> still a specification. See [§ Status update](#status-update-2026-09-s-eap-implemented)
> below for what the first benchmark actually found — a mixed result,
> reported here rather than silently dropped because the whole point of
> this project's Skeptic/Judge machinery is not trusting a claim until
> it's been checked, and that standard applies to this repo's own design
> process too.

**The gap.** `run_acdc` and `run_eap` (`tools/tier_c.py`) both measure a
component's *marginal* effect: ablate it alone, see what breaks. That's a
first-order test. It is structurally blind to components whose effect only
appears jointly — a component with a "backup" that compensates when the
primary one is removed will show near-zero effect when tested alone, so
greedy single-edge pruning removes it, even though it's a real part of the
circuit. This isn't hypothetical: Wang et al.'s published GPT-2 IOI circuit
(`stage_a.py`'s `GT_NEGATIVE_NAME_MOVERS = [(10,7), (11,10)]`) names two
heads that exist for exactly this reason, and this repo's own Stage A runs
currently miss both of them (11% circuit recall — see the walkthrough
below).

**The idea.** Cooperative game theory already has the right object for
this: a component's Shapley value decomposes into first-order (marginal)
and higher-order (interaction) terms. Full Shapley computation is
exponential in the number of components, which is why nobody runs it as-is
here — the contribution is making the *search* for interaction terms
adaptive and statistically rigorous instead of exhaustive, using machinery
this repo already has (the adversarial Skeptic, `eval_prompts`' held-out
split, `ToolCallBudget`).

**Formal definition**, for a candidate set of components `V` (EAP's
top-`M` ranked, kept small for tractability):

- Coalition value `m(S)` = task metric with only components in `S ⊆ V`
  active (the rest ablated) — the same quantity `_ioi_metric_fn` already
  computes, generalized from single heads to sets.
- Marginal effect (what `run_acdc`/`run_eap` already compute):
  `φ₁(i) = m({i}) − m(∅)`.
- **Synergy score** (the new quantity — a 2nd-order Harsanyi dividend):
  `Syn(i,j) = m({i,j}) − m(∅) − φ₁(i) − φ₁(j)`.
  A true backup pair looks like `φ₁(i) ≈ 0, φ₁(j) ≈ 0, Syn(i,j) ≫ 0` —
  exactly the shape single-edge ACDC cannot see.

**Algorithm:**

1. Run `run_eap` / `run_acdc` as they exist today for the first-order pass
   — nothing about this step changes.
2. **Adversarially-guided coalition search**: rather than testing all
   `C(M,2)` pairs, the Skeptic proposes which pairs to test — prioritizing
   components EAP scored non-trivially but below the single-edge
   threshold, and components sharing a layer or overlapping attention
   patterns (`get_attention_pattern` already computes the latter). Bounds
   the search to on the order of 100-200 pairs instead of an unbounded
   sweep.
3. **Statistical test, not a hand threshold**: compute `Syn(i,j)` across
   the sampled prompt pairs, test `H₀: E[Syn] = 0` via a paired
   permutation/bootstrap test, and apply Benjamini-Hochberg FDR correction
   across every pair tested — the multiple-comparisons control `run_acdc`'s
   fixed threshold (`ABLATION_EFFECT_THRESHOLD`) doesn't have today.
4. **Held-out replication**: split each task's `eval_prompts` (already
   generated by `behaviors.py`'s `_make_task`, currently only used for
   post-hoc scoring) into a discovery half and a held-out half. A synergy
   pair only enters the circuit if it's significant on *both*
   independently — closing the gap where pruning decisions are currently
   validated on the same data they were found on.
5. **Iterate**: add confirmed pairs to the circuit, re-run marginal
   effects on the updated circuit (a newly-included backup component can
   change what the remaining components' effects look like), repeat until
   no new significant pair survives FDR correction or the budget runs out.

**What this would concretely change**: a direct, falsifiable comparison
against this repo's own existing benchmark — does S-ACDC recover `(10,7)`
and `(11,10)` where plain `run_acdc` currently doesn't, on the exact same
GPT-2 IOI task `stage_a.py` already runs.

**Reuses as-is**: `run_eap`'s ranking, `ablate_component`'s hook mechanism
(extended to ablate sets, not singletons), `eval_prompts`' existing
train/held-out structure, `ToolCallBudget`. **Would be new**: the
synergy-score computation, the FDR-corrected significance test, the
split-half replication gate, and the iterate-to-convergence loop — likely
living alongside `run_acdc`/`run_eap` in `tools/tier_c.py`, called from
`component_agent.py` as an optional follow-up pass after the existing
EAP-then-ACDC sequence.

**Honest caveats**: cooperative-game-theoretic (Shapley/Harsanyi) feature
attribution is well-established in ML broadly (SHAP and its variants); this
specific combination — adversarially-bounded pairwise synergy testing,
FDR-corrected, with split-half replication, applied to transformer circuit
ablation — is, as far as this project is aware, not something already
published, but "not aware of" isn't "doesn't exist," and that should be
checked against current literature before this is written up as a novel
contribution rather than merely a useful addition. Computational cost is
`O(M²)` forward passes plus resampling for the hypothesis test, so `M`
needs to stay small (~10-15) to remain tractable on CPU.

### Status update (2026-09): S-EAP implemented

The first concrete piece is built and benchmarked. `tools/tier_c.py` now
has **`run_synergy_eap`** — a finite-difference approximation to the
pairwise Hessian term:

```
Syn(i,j) ≈ ( ∇_{A_i}L_clean − ∇_{A_i}L_ablate(j) ) · (A_i^clean − A_i^corrupt)
```

One backward pass per candidate head `j` yields the interaction row for
every `i` at once (so it is `O(M)` backward passes, not `O(M²)`).
`tools/adapter.py::ablate_head_set` provides the exact coalition value
`m(S)` used both as ground truth and, eventually, by the full S-ACDC loop.

**What the benchmark found** (`experiments/seap/`, GPT-2 IOI + Qwen2.5-0.5B,
CPU):

| | GPT-2 IOI | Qwen2.5-0.5B IOI |
|---|---|---|
| ACDC first-order recall vs GT | 1/9 heads (blind spot reproduced) | no ground truth |
| `Spearman(\|S-EAP Syn\|, \|exact Syn\|)` | **+0.65, p=9e-11** (78 pairs) | +0.20, p=0.30 (28 pairs) |
| `Spearman(signed, signed)` | −0.12, p=0.30 — **sign not recovered** | +0.14, p=0.49 |
| circuit faithfulness, ACDC → + top synergy heads | 0.31 → 0.36 | 0.53 → 0.58 |

On GPT-2 the exact coalition analysis does surface what first-order ACDC
misses — the largest interactions are all among `φ₁ ≈ 0` heads
(`(10,7)+(11,10)` negative name movers, `Syn=+0.62`; `(10,10)+(11,10)`
backup + negative mover, `Syn=−0.35`).

**Honest read:** S-EAP is a usable *magnitude* screen on GPT-2 but loses
sign information (cooperation vs. cancellation), so it can only shortlist
pairs for exact `ablate_head_set` confirmation — it is not a standalone
discovery method. It did not replicate on the 0.5B model (underpowered:
8 EAP-only candidates, no known backup structure to find). Faithfulness
gains are small (+0.04). This is a workshop-scale mixed result, not a
main-track contribution. Before any writeup it needs: the paired
permutation test across `eval_prompts`, resampling-ablation comparison,
and a second circuit with published ground truth (Greater-Than on GPT-2).
Full details and reproduction in `experiments/seap/FINDINGS.md`.

## The technique toolkit (`automechinterp/techniques/`)

A model-agnostic library covering the learnmechinterp.com curriculum — one
module per category, every function driven by forward/backward hooks on the
same `adapter.ModelHandle`, so it runs on GPT-2, Qwen2, Llama, Pythia
unchanged.

~70 techniques across 13 categories. Runnable ones use forward/backward hooks
only; the rest are honest `NotImplementedError` stubs that name exactly what
they'd need (a training run, a labelled corpus, a weight edit).

| module | techniques |
|---|---|
| `basic` | logit lens, tuned lens (fast ridge fit), Jacobian lens (single-readout approx), DLA, attention-pattern classification |
| `causal` | activation patching, attribution patching, path patching, **self-repair** (ablate → re-measure others' DLA), interchange intervention, causal-mediator selection, integrated-gradients (refined) attribution |
| `probing` | linear probe, k-sparse probe, MDL probe, probe-direction causal test, **geometry-of-truth**, per-head attention probes, LAT reading vectors (RepE) |
| `steering` | activation addition (CAA), ablation steering, affine steering, multi-layer steering, function vectors, **unsupervised** (MELBO) steering vectors |
| `editing` | **LEACE** (closed-form concept erasure), inference-time concept ablation, hook-based localized fact edit; ROME / machine-unlearning = stubs (weight edits) |
| `superposition` | toy SAE trainer (any model), gated SAE + SAE evaluation vs PCA, feature dashboards, **feature steering**, temporal features, public SAE decompose (GPT-2); transcoder / crosscoder = stubs |
| `feature_geometry` | activation dimensionality (participation ratio), feature-direction angles / cliques, principal-component manifold, on- vs off-manifold steering |
| `model_diffing` | logit-diff amplification, per-layer activation drift, fine-tuning traces (GPT-2 auto-pairs with the bio stand-in); feature-level diffing = stub |
| `hidden_state` | logit-lens trajectory, **Patchscopes**, SelfIE read-back, concept-injection introspection; activation-oracle = stub |
| `circuits` | ACDC, EAP, S-EAP, QK/OV decomposition, faithfulness / completeness / minimality, **copy-suppression** detection, attribution graph, entity binding; universality = stub |
| `blackbox` | counterfactual resampling (per-token importance), minimal-pair contrast |
| `weight_space` | parameter SVD / effective rank, weight-norm profile, E/U tying, parameter-space head grouping; interpretable-training = stub |
| `safety` | refusal direction, Mahalanobis activation-anomaly monitor, **evaluation-awareness** probe; deception / sleeper-agent = stubs (need labelled data) |

Run every applicable technique on any model and get a Markdown + JSON report:

```bash
python3 main.py --techniques                       # target model from config
python3 main.py --techniques --target-model Qwen/Qwen2.5-0.5B-Instruct
# or:  from automechinterp.techniques.runner import run_all; run_all("gpt2")
```

Stubs are reported as `needs-more` with the exact requirement (training run,
dataset, weight-edit machinery), never faked.

### Which agent runs which technique

Every agent in `run_hierarchy` is an LLM policy over a **fixed tool menu** —
the "brain" (heuristic rule table or a real LLM) only chooses *which* of its
allowed tools to call next and when to stop. The core pipeline agents and
the six deep-technique agents each own a disjoint slice of the toolkit:

```
Orchestrator
  → Network Analyst          tier_n: profile_network, layerwise_cka_scan, redundancy_scan, depth_probe_sweep
      ‖ Weight Agent          weight_space: parameter_svd (attn_out + mlp_out), weight_norm_profile,
      ‖ Safety Agent                        embedding_unembedding_alignment
  → Layer Agent (×N)          tier_l: logit_lens, patch_layer, attn_mlp_attribution, sae_layer_profile
      ‖ Lens Agent            basic.logit_lens · basic.jacobian_lens · hidden_state.logit_lens_trajectory · hidden_state.patchscopes
      ‖ Probe Agent           probing.linear_probe · probing.sparse_probe · probing.mdl_probe · probing.probe_direction_causal_test
      ‖ Feature Agent         superposition.train_toy_sae · superposition.public_sae_decompose ·
                              feature_geometry.feature_direction_geometry · feature_geometry.activation_dimensionality
  → Component Agent (×M)      tier_c: run_eap → run_acdc → run_synergy_eap (S-EAP), get_attention_pattern, run_sae_decompose
      ‖ Steering Agent        steering.activation_addition · steering.ablation_steering · editing.leace
  → Skeptic                   tier_v: ablate_component, exclusion_ablation, minimality_check, counterexample_search
  → Judge                     no tools — reads claim + Skeptic evidence, assigns the verdict
```

| Agent | Scope | When | Techniques it may call | What its output answers |
|---|---|---|---|---|
| **Network Analyst** | whole model | stage 0 | `profile_network`, `layerwise_cka_scan`, `redundancy_scan`, `depth_probe_sweep` | which layers are worth a deeper look |
| **Weight Agent** | whole model | ‖ Network Analyst | `parameter_svd` (attn-out & MLP-out), `weight_norm_profile`, `embedding_unembedding_alignment` | where capacity / effective rank concentrates; E/U tying — from weights alone |
| **Safety Agent** | whole model | ‖ Network Analyst | `refusal_direction`, `activation_anomaly_monitor` (Mahalanobis), `copy_suppression` scan; `deception_detection` / `sleeper_agent_scan` = stubs | is there a refusal direction; any Negative-Name-Mover heads; OOD-input flag |
| **Layer Agent** | one layer | per flagged layer, parallel | `logit_lens`, `patch_layer`, `attn_mlp_attribution`, `sae_layer_profile` | is this layer causally load-bearing, attention- or MLP-driven |
| **Lens Agent** | one layer | per load-bearing layer | `logit_lens`, `jacobian_lens`, `logit_lens_trajectory`, `patchscopes` read-back | *what* the residual stream here represents (not just that it matters) |
| **Probe Agent** | one layer | per load-bearing layer | `linear_probe`, `sparse_probe` (k-sparse), `mdl_probe`, `probe_direction_causal_test` | is the task variable linearly encoded here, how distributed, does the direction *do* anything |
| **Feature Agent** | one layer | per load-bearing layer | `train_toy_sae`, `public_sae_decompose` (GPT-2), `feature_direction_geometry`, `activation_dimensionality` | decompose the layer into sparse features; how superposed |
| **Component Agent** | one layer's heads | if a layer is attention-driven | `run_eap` → `run_acdc` → `run_synergy_eap` (S-EAP 2nd-order), `get_attention_pattern`, `run_sae_decompose` | which specific head(s) carry the behaviour; what they represent |
| **Steering Agent** | claimed circuit | after a circuit claim, before the Judge | `activation_addition` (CAA), `ablation_steering`, `leace` | does *constructively* moving the direction move the behaviour the way the claim predicts |
| **Skeptic** | claimed circuit | after the claim | `ablate_component`, `exclusion_ablation`, `minimality_check`, `counterexample_search` | can the claim be falsified four ways |
| **Judge** | — | last | none | Confirmed / Probable / Speculative / Refuted |

The six deep-technique agents (Weight, Safety, Lens, Probe, Feature, Steering)
are toggled with `AMI_DEEP_TECHNIQUES=0`; Stage B/D keep them off (200
behaviours × the full suite is too slow). Everything they call also runs
standalone via `python3 main.py --techniques` (see [§ The technique
toolkit](#the-technique-toolkit-automechinterptechniques) — ~70 techniques,
of which the agents wire up the ~20 most load-bearing).

On GPT-2 IOI the Safety Agent's `copy_suppression` scan independently flags
`L10H7` / `L11H10` (the negative name movers), and the Steering Agent's
`ablation_steering` on the answer-name direction drops the logit-diff where
the Skeptic's component ablation alone was inconclusive.

## Project layout

```
main.py                              entrypoint / CLI
requirements.txt                     pip dependencies

automechinterp/
  config.py                          all tunables: which model is being interpreted,
                                      which LLM drives the agents, budgets, thresholds
  gap_work.py                        the gap table above, as data — each entry names
                                      the prior system, its limitation, and the class/
                                      function in this repo that closes it
  llm_backends.py                    the "brain" behind every agent:
                                        HeuristicBackend  - rule-based, no LLM needed
                                        HFLocalBackend    - runs a real open-source LLM
                                                            (Qwen2.5-7B-Instruct by default)
                                        OpenAIBackend / AnthropicBackend - optional, need
                                                            an API key, off by default
  hierarchy.py                       wires the whole pipeline together: Network Analyst
                                      -> Layer Agents -> Component Agents -> Skeptic ->
                                      Judge, with a shared tool-call budget and a
                                      hypothesis ledger that dedupes claims
  stage_a.py                         Stage A: builds the IOI task on GPT-2, runs the
                                      pipeline, scores it against the published
                                      ground-truth circuit, runs negative controls
  behaviors.py                       4 distinct task definitions (coreference, syntax,
                                      factual recall, lexical semantics) used by Stage B
                                      -- every one reuses the SAME hierarchy machinery,
                                      since it was never IOI-specific to begin with
  stage_b.py                         Stage B: runs the full pipeline once per behavior
                                      and assembles a cross-behavior Layer Atlas
  finetune_stand_in.py               builds the base/fine-tuned model pair Stage C
                                      needs: briefly fine-tunes GPT-2 on a small
                                      biomedical corpus (a fast, local stand-in for the
                                      proposal's Mistral-7B/BioMistral-7B pair)
  stage_c.py                         Stage C: CKA layer-alignment between base and
                                      fine-tuned model, then causal layer-transplant
                                      confirmation
  ablations.py                       Sec. 4.7 ablation studies: hierarchy (flat vs
                                      scoped context), verification (self-confirmation
                                      vs Prover-Skeptic-Judge), Network Analyst
                                      (guided vs uniform vs random layer flagging)
  visualize.py                       every plotting function (Stage A/B/C, ablations) --
                                      saves a PNG + JSON pair to output/
  ablations.py, stage_b.py, stage_c.py all reuse hierarchy.py / the Tier N-V tools
                                      directly; none of them needed new plumbing

  agents/
    base.py                          the shared ReAct loop every agent runs (ask the
                                      backend what to do next, call the tool, log the
                                      result, repeat until budget runs out)
    network_analyst.py                Stage-0 whole-model profiling; decides which
                                      layers are worth a deeper look
    layer_agent.py                   diagnoses one layer: is it causally load-bearing,
                                      attention- or MLP-driven, and (if SAEs are
                                      available for this model) how superposed?
    component_agent.py               finds which specific attention head(s) inside a
                                      layer are responsible (EAP then exact ACDC), then
                                      decomposes the finding into sparse SAE features
    skeptic.py                       the adversarial verifier described above
    judge.py                         reads the Prover's claim + Skeptic's evidence,
                                      assigns the final verdict
    orchestrator.py                  the hypothesis ledger (dedupes claims found by
                                      multiple agents) + budget-aware layer selection

  tools/
    adapter.py                       the architecture-agnostic core: works on any
                                      HuggingFace causal LM by discovering its layer
                                      stack via named_modules() — no assumption that
                                      it's a GPT-2-shaped model. register_model,
                                      capture_activations, layer_similarity,
                                      ablate_layer, swap_layer, transplant_layer
                                      (weight-level, for Stage C), freeze_and_probe,
                                      patch_layer, plus head-level patching primitives
    tier_n.py                        Network-Level tools (profile_network,
                                      layerwise_cka_scan, redundancy_scan,
                                      depth_probe_sweep)
    tier_l.py                        Layer-Level tools (logit_lens, patch_layer,
                                      attn_mlp_attribution, sae_layer_profile,
                                      layer_role_card)
    tier_c.py                        Component-Level tools, transformer-specific
                                      (run_acdc, run_eap, get_attention_pattern,
                                      get_max_activating_examples, activation_patch,
                                      run_sae_decompose)
    tier_v.py                        Verification tools, used only by the Skeptic
                                      (ablate_component, exclusion_ablation,
                                      minimality_check, counterexample_search,
                                      interchange_intervention, steer_with_vector)
    sae.py                           SAE tool internals: loads pretrained SAEs via
                                      SAELens and sources activations via
                                      TransformerLens (see "SAE tools" section below
                                      for why that specific combination is required)
    digest.py                        turns raw tensors into short text summaries
                                      before they ever reach an agent's context
                                      (e.g. "L12H5: 83% mass units->tens digit")
    cache.py                         content-addressed cache so identical tool calls
                                      (e.g. the same CKA scan) aren't recomputed

  stage_d_eval.py                    Stage D: run the hierarchy per behavior, write a
                                      human-readable report per (model, behavior) via
                                      eval/report_writer.py, optionally AI-score it
  eval/
    rubrics.py                       the 15 scoring dimensions (source of truth for
                                      both the AI RubricJudge and the human reviewers)
    report_writer.py                 deterministic hierarchy-result -> Markdown report
    model_matrix.py                  the <=7B evaluation model matrix as data: ~25 HF
                                      model ids + families / sizes / tiers / ladders
  techniques/                        ~70-technique MI toolkit, one module per
                                      learnmechinterp.com category (basic, causal,
                                      probing, steering, editing, superposition,
                                      feature_geometry, model_diffing, hidden_state,
                                      circuits, blackbox, weight_space, safety);
                                      runner.py runs them all and writes a report

run_eval_matrix.py                   repo-root runner: Stage D across a model size
                                      ladder, one model at a time, download -> run ->
                                      delete (see COMMANDS_eval_matrix.md)
human_review/                        the human half of the two-arm evaluation:
                                      reports/ to score, rubrics.md, scoring_template.csv,
                                      aggregate_scores.py (inter-rater agreement),
                                      agent_layer_matrix.py (agent x layer matrix)
```

## What research problem each piece is actually solving

- **`adapter.py`** solves *"most interpretability tooling assumes a transformer"* —
  it locates a model's decoder-block list generically (`transformer.h` for GPT-2,
  `model.layers` for Qwen2/Llama-family) so the same tool works on either.
- **`tier_c.py`'s `run_acdc`/`run_eap`** solve *"how do you find which specific
  component matters, not just which layer"* — exact per-head causal patching
  (ACDC) and a fast gradient-based approximation (EAP), the same two methods
  the proposal names.
- **`tools/sae.py`** solves *"a causally-important head has an effect size, but
  what does it actually represent?"* — sparse feature decomposition gives a
  finding semantic content, not just a number.
- **`skeptic.py` + `judge.py`** solve *"how do you stop an agent from convincing
  itself"* — the MAIA confirmation-bias problem — by making falsification a
  separate agent's entire job.
- **`llm_backends.py`** solves *"does this depend on a specific paid API"* — the
  agent logic is identical whether the backend is a free rule table or a local
  open-source model.
- **`ablations.py`** solves *"why should anyone believe the hierarchy/verification
  design actually helps"* — measures it directly instead of asserting it.

## Which models this runs, and how

| Role | Model | Why |
|---|---|---|
| **Target model** (the network being interpreted) | GPT-2 small (124M) | Has a published, well-studied ground-truth circuit (IOI) to score against |
| **Reasoning backend** (the agents' "brain") | Heuristic rule table by default, or a Qwen model (`hf_local`) | Open-source, no API key, swappable via one flag |

Both are configurable via `--target-model` / `--backend` (or the equivalent env
vars in `automechinterp/config.py`) — nothing is hardcoded to GPT-2 or Qwen
specifically; `adapter.py` would work the same way against, say,
`mistralai/Mistral-7B-v0.1` as the target model. On this development machine,
`hf_local` has been verified up through a 3B Qwen model comfortably; a 7B
model works but expect a long first-time download and a hot laptop on a
fanless machine — see the CLI section below for the practical guidance this
was learned from.

## SAE tools

`sae_layer_profile` (Tier L) and `run_sae_decompose` (Tier C) were the two
tools originally left unimplemented. They use SAELens' public pretrained
GPT-2 residual-stream SAE release (`gpt2-small-res-jb`, Joseph Bloom) — the
same public-artifact role GemmaScope plays for Gemma 2 in the original
proposal.

**One real gotcha worth documenting**: this SAE was trained on activations
from a TransformerLens `HookedTransformer`, which applies weight processing
(LayerNorm folding, weight centering) that plain `transformers.GPT2Model`
does not. Feeding it activations captured the normal way (via `adapter.py`'s
plain-HF hooks) produces badly wrong numbers — verified: FVU (fraction of
variance unexplained) ≈ 16 and L0 (active features/token) ≈ 2000, i.e. the
SAE barely reconstructing noise. Sourcing the same activations from a
TransformerLens model instead fixes it completely — FVU ≈ 0.0000, L0 ≈ 50-60,
matching the SAE's published dashboard numbers. `tools/sae.py` is the only
module in this codebase that loads a `TransformerLens` model for this reason;
everything else stays on the architecture-agnostic HF path. This is also why
SAE support is currently GPT-2-only: it needs both a TransformerLens
implementation of the target model AND a matching public SAE release.

In the hierarchy, a Layer Agent runs `sae_layer_profile` on any causally
load-bearing layer to check for superposition, and a Component Agent runs
`run_sae_decompose` on a confirmed head to report which sparse features it
correlates with — turning "L9H9 matters" into "L9H9's output correlates with
SAE features {24120, 1721, 20971, ...}".

## Ablation studies (Sec. 4.7)

`python3 main.py --ablations --graph` runs three real comparisons against the
same GPT-2 IOI task, not a table of predicted numbers:

- **Hierarchy ablation**: a flat single agent's evidence log grows linearly as
  it checks more layers (122 → 1665 characters over 12 layers); the
  hierarchical version's largest single Layer Agent never exceeds ~482
  characters, because each agent only ever sees its own scope. **3.45x context
  reduction**, and — more importantly — a qualitatively different growth
  curve (linear vs. flat), which is the actual claim Sec. 3.1 makes.
- **Verification ablation**: a self-confirmation baseline (no Skeptic) accepts
  100% of 4 deliberately fabricated claims by construction; the real
  Prover-Skeptic-Judge pipeline, run on the exact same claims, refutes all 4
  (**0% false-confirmation rate**).
- **Network Analyst ablation**: CKA+redundancy-guided layer flagging vs.
  uniform-all-layers vs. random-subset (averaged over 5 draws). Honest
  finding: on GPT-2's shallow 12-layer stack, all three converge (100% /
  100% / 96% recall) — guided flagging only excludes 1 of 12 layers here, so
  there's little room to show an advantage. The proposal's own framing
  predicts this trade-off should widen on deeper models; this codebase's
  `network_analyst.py` already switches strategy by depth (exhaustive scan
  under ~16 layers, CKA-guided above that) for exactly this reason.

## Stage B — Layer Atlas (Sec. 4.4)

`python3 main.py --stage-b --graph` runs the identical hierarchy across 4
distinct behaviors (scaled down from the proposal's 24, one per named
category — coreference, syntactic agreement, factual recall, lexical
semantics) and assembles a **Layer Atlas**: for every layer, which behaviors'
Component Agents found a *confirmed circuit* there (not just "was
investigated" — Network Analyst flagging is deliberately permissive, so the
atlas is built from the much more selective circuit-level evidence).

A real, coherent finding from an actual run: layers 0-6 have no confirmed
circuit for any tested behavior (early layers build general representations,
not task-specific circuits); layer 9 is a cross-behavior **hub** (circuits
found for 3 of 4 behaviors — coreference, factual recall, lexical semantics);
layers 7, 8, and 10 each have exactly one behavior-specific circuit. This
matches the broader interpretability literature's picture of GPT-2's
mid-to-late layers doing general-purpose work before output-specific layers
take over.

## Stage C — Cross-Model Layer Diff (Sec. 4.5)

`python3 main.py --stage-c --graph` runs the proposal's Mistral-7B vs
BioMistral-7B methodology at a scale this machine can actually run: GPT-2
base vs. `finetune_stand_in.py`'s output, a GPT-2 briefly fine-tuned locally
on a small biomedical corpus (same architecture before/after by construction,
same "domain adaptation without changing the base model" relationship
BioMistral has to Mistral).

1. **Alignment**: per-layer CKA between base and fine-tuned activations on a
   shared general+biomedical probe set.
2. **Causal confirmation**: `adapter.transplant_layer()` — a new primitive,
   distinct from the existing `swap_layer()` (which swaps *activations*
   within one model) — substitutes a layer's actual *weights* from the
   fine-tuned model into the base model and re-measures a biomedical-capability
   metric (logit preference for domain-appropriate completions).

A real result from an actual run: CKA drift concentrates sharply in the last
3 layers (L11 drops to 0.756 vs. >0.97 everywhere else — representational
change is NOT spread evenly). Transplanting those three layers'
weights transfers 10-12% of the base/fine-tuned capability gap each, vs. only
1% for a control transplant of the least-changed layer (L0) — a genuine
causal confirmation that the CKA-flagged layers are the ones that matter, not
just correlated with the diff.

## How the evaluation works (Stage A)

`stage_a.py` runs the **Indirect Object Identification (IOI)** task — "When John
and Mary went to the store, John gave a drink to ___" should complete with the
un-repeated name — because Wang et al. (2022) already published GPT-2's exact
circuit for it (which heads, which layers), giving us ground truth to check
the pipeline's output against instead of just trusting it.

For every run it reports:

- **Layer-localization recall** — did the Network Analyst flag the real
  ground-truth layers?
- **Circuit recall / precision** — of the heads the pipeline claimed, how many
  are real, and how many of the real ones did it find?
- **Judge verdict** — Confirmed / Probable / Speculative / Refuted on the final
  merged claim
- **Negative-control refutation rate** — 4 deliberately wrong claims are
  injected straight into the Skeptic (bypassing discovery); a correct Skeptic
  refutes all 4. This measures the false-confirmation rate directly.
- **Tool-call budget spent** and **cache hit rate** — the cost-efficiency
  metrics the proposal asks for.

## What a real run actually found (GPT-2 walkthrough)

This is a real result from this codebase, not a mockup — the saved
[`output/`](output/) folder in this repo has the full log and the plot from
the run described below.

Running `python3 main.py --graph` with the heuristic backend against GPT-2
small produces, step by step:

1. **Network Analyst** runs `profile_network`, `layerwise_cka_scan`, then an
   exhaustive `redundancy_scan` (GPT-2's 12 layers are cheap enough to check
   all of them rather than trust the coarse CKA signal alone) and flags 11 of
   12 layers as worth investigating — **100% layer-localization recall**
   against the published ground-truth layers `[7, 8, 9, 10, 11]`.
2. **11 Layer Agents run in parallel.** Most immediately report near-zero
   causal effect from `patch_layer` and stop (no wasted deeper analysis).
   Layers 9, 10, and 11 show real effect — the plot's left panel shows this
   ramping up cleanly through the ground-truth layers (0.45, 0.94, 1.00
   fraction of the clean/corrupted gap recovered).
3. **Component Agents spawn** on those three layers. `run_eap` (cheap
   gradient approximation) flags `L9H9` with a score of `+2.22` — an order of
   magnitude above every other head in that layer. `run_acdc` (exact
   per-head patching) confirms it: `L9H9` alone recovers `+0.29` of the
   metric gap. **`L9H9` is a real entry in Wang et al.'s published GPT-2 IOI
   circuit** (a name-mover head) — the pipeline found it without being told
   what to look for. A Component Agent then runs `run_sae_decompose` on the
   finding, reporting which SAE features it correlates with.
4. **The Skeptic tries to break the claim** on 6 fresh held-out prompts the
   discovery agents never saw. Verdicts vary run to run since IOI prompts are
   resampled each time — sometimes Refuted (a real signal that didn't
   generalize past a stricter check — e.g. `L9H9` at `+0.192`, just under the
   `0.2` cutoff), sometimes **Confirmed** (Stage B's antonym-prediction
   behavior, for instance, survived the full falsification suite on `L9H7`).
   Both outcomes are the verification step working as intended.
5. **4 fabricated claims are injected straight into the Skeptic**, bypassing
   discovery entirely. All 4 are correctly refuted — **0% false-confirmation
   rate**.

|  | Value |
|---|---|
| Layer-localization recall | 100% |
| Circuit recall / precision | 11% / 50% (found `L9H9`, one real head, out of 9 ground-truth heads; 1 of 2 claimed heads was real) |
| Negative controls refuted | 4/4 (0% false-confirmation rate) |
| Tool calls spent | 45-50 / 120 budget |

## Output files

`--graph` (or a plain redirect/`tee`) saves results into `output/`:

- **`stage_a_<model>_<timestamp>.png/.json`** — per-layer causal effect,
  headline metrics, run summary
- **`ablations_<model>_<timestamp>.png/.json`** — the three Sec. 4.7
  comparisons above
- **`stage_b_atlas_<model>_<timestamp>.png/.json`** — the cross-behavior Layer
  Atlas heatmap
- **`stage_c_diff_<model>_<timestamp>.png/.json`** — the CKA alignment +
  causal transplant confirmation
- **`eval/stage_d_<model>_<timestamp>.json`** — Stage D: per-report verdicts,
  applicable rubrics, optional AI scores; the reports themselves land in
  **`human_review/reports/<model>__<behavior>.md`**
- **`eval/eval_matrix_<timestamp>.json`** — `run_eval_matrix.py` sweep summary
  (per-model verdict tallies, GB freed by `--purge-downloads`)
- **`eval/agent_layer_matrix.md`** — `human_review/agent_layer_matrix.py`:
  agent × layer matrix per report
- **`eval/human_review_aggregate.csv`** — `human_review/aggregate_scores.py`:
  per-rubric means + inter-rater agreement
- **`experiments/techniques/techniques_<model>_<timestamp>.md/.json`** —
  `--techniques` toolkit run
- **`*.log`** — full terminal transcripts from example runs (every agent
  decision, tool call, and verdict), saved via `tee`

This repo includes real examples of all of these from actual runs — open them
directly to see what a run looks like without installing anything.

## Running it

No API key needed for the default setup — everything downloads and runs locally.

```bash
cd "/Users/mansuba/Desktop/MI MAIN RESEARCH/  2 Automatic-Mechanistic-Research-main"
pip install -r requirements.txt

python3 main.py                 # Stage A, heuristic backend, GPT-2 target
python3 main.py --graph         # same, plus saves a PNG + JSON to output/
python3 main.py --ablations --graph   # Sec. 4.7 ablation studies
python3 main.py --stage-b --graph     # multi-behavior Layer Atlas
python3 main.py --stage-c --graph     # cross-model layer diff (base vs fine-tuned)
python3 main.py --stage-d              # write per-(model,behavior) reports for human review
python3 main.py --stage-d --ai-judge anthropic   # + auto-score them (needs ANTHROPIC_API_KEY)
python3 main.py --techniques           # run the ~70-technique toolkit on one model + task
python3 main.py --gap-matrix    # just print the related-work gap table
python3 main.py --smoke-test-qwen   # proves the open-source-LLM backend works,
                                     # using a small Qwen model so it's fast
```

**CLI flags** (all optional — most have an equivalent env var in
`automechinterp/config.py`, the flags just make one-off overrides easier):

| Flag | What it changes | Default |
|---|---|---|
| `--target-model <id>` | which model is **interpreted** (any HF causal LM id) | `gpt2` |
| `--backend <kind>` | which brain drives the **agents**: `heuristic`, `hf_local`, `openai`, `anthropic` | `heuristic` |
| `--heavy-model <id>` | which open-source model to load when `--backend hf_local` | `Qwen/Qwen2.5-7B-Instruct` |
| `--graph` | save a PNG + JSON of the results | off |
| `--output-dir <dir>` | where `--graph` saves to | `output` |
| `--ablations` | run Sec. 4.7 ablations instead of Stage A | off |
| `--stage-b` | run the multi-behavior Layer Atlas instead of Stage A | off |
| `--stage-c` | run the cross-model layer diff instead of Stage A | off |
| `--stage-d` | write per-(model, behavior) reports to `human_review/reports/` instead of Stage A | off |
| `--ai-judge <kind>` | with `--stage-d`, also RubricJudge-score every report: `openai` \| `anthropic` (needs that API key) | off |
| `--techniques` | run the technique toolkit instead of Stage A | off |

**Switch the target model** (the network being interpreted) — note: the
published IOI ground truth only exists for GPT-2, so recall/precision are
reported as `N/A` for anything else, honestly, not guessed:

```bash
python3 main.py --target-model gpt2-medium --graph
```

**Use a real open-source LLM as the agents' brain** instead of the rule table
— still no API key. This machine has been verified comfortable up through a
3B-parameter Qwen model; a 7B model works but expect a genuinely long
first-time download (multi-GB) and a hot laptop if you're on a fanless
machine — that's normal, not a bug, and only affects the one-time download +
inference speed, not correctness:

```bash
pip install accelerate
python3 main.py --backend hf_local --heavy-model "Qwen/Qwen2.5-3B-Instruct" --graph
```

**Save the full terminal transcript alongside the graph:**

```bash
python3 main.py --graph 2>&1 | tee "output/stage_a_$(date +%Y%m%d_%H%M%S).log"
```

### Evaluation model matrix (cross-model / scaling sweep)

`python3 main.py --stage-d` interprets **one** target model. To run the same
behavior battery across a **size ladder** of models (≤ 7B) — the mentor's
scaling ask, so rubric 15 (`scaling_coherence`) has something to score —
use the matrix runner. It needs no API key (agents stay on the `heuristic`
backend); only the target models download. This is the piece to hand to a
teammate with spare compute.

**It runs strictly one model at a time** — download → Stage D → write that
model's `output/eval/stage_d_<model>_*.json` → free RAM → (with
`--purge-downloads`) delete the weights from the HF cache → next model. Only
one model's weights are ever on disk / in RAM at once, so the whole ladder
fits on a laptop.

**Copy-paste runbook for a teammate:** [`COMMANDS_eval_matrix.md`](COMMANDS_eval_matrix.md)
— setup, wiring check, the ladder commands, resume, and troubleshooting.

```bash
python3 run_eval_matrix.py --list        # the menu: models, sizes, tiers, ladders

# 0. cheap wiring check for a whole tier: download → introspect → delete, no Stage D
python3 run_eval_matrix.py --tier laptop --check --purge-downloads

# 1. quick correctness check (3 behaviors, smallest models)
python3 run_eval_matrix.py --tier laptop --max-params 200M --behaviors 3

# 2. a clean within-family scaling ladder — recommended — deleting each model after it finishes
python3 run_eval_matrix.py --ladder pythia  --purge-downloads   # 70M → 160M → 410M → 1B → 1.4B → 2.8B → 6.9B
python3 run_eval_matrix.py --ladder qwen2.5 --purge-downloads   # 0.5B → 1.5B → 3B → 7B
python3 run_eval_matrix.py --ladder gpt2    --purge-downloads   # 124M → 355M → 774M → 1.5B

# 3. everything a laptop CPU can handle; or the 2–7B tier on a CUDA box
python3 run_eval_matrix.py --tier laptop --purge-downloads
python3 run_eval_matrix.py --tier workstation --dtype float16 --skip-done --purge-downloads

# explicit picks
python3 run_eval_matrix.py --models gpt2-large,EleutherAI/pythia-1.4b
```

Every report lands in `human_review/reports/` as `<model>__<behavior>.md`
(sizes never collide), a per-model `output/eval/stage_d_<model>_*.json` is
written, and a combined `output/eval/eval_matrix_<ts>.json` summary is
updated after each model. `--skip-done` resumes a partial sweep.

`--purge-downloads` deletes each model from the shared HF cache
(`~/.cache/huggingface/hub`) once its result is saved, and reports total GB
freed. Models that were **already cached before the run** are left alone
unless you also pass `--purge-preexisting`.

The model list lives in [`automechinterp/eval/model_matrix.py`](automechinterp/eval/model_matrix.py) —
all ungated except the Llama-3 / Gemma-2 entries (`huggingface-cli login` +
`--include-gated`). `--check` validates that `adapter.py` discovers each
model's decoder stack (layer count vs. the matrix's expectation) before you
commit to a long sweep. After a sweep, `python3 human_review/agent_layer_matrix.py`
regenerates the agent×layer matrices over the new reports.

**Attention backend.** `adapter.register_model` loads every target with
`eager` attention (so Tier C can read per-head attention patterns). On
`transformers >= 5.x` the eager path is numerically unstable for GPT-NeoX
(Pythia) and returns NaN logits — the adapter detects this on a one-line
sanity forward and falls back to SDPA, printing a `[adapter] …` line. Under
SDPA everything still works (layer patching, probes, ablation, logit lens,
the task metric) except `get_attention_pattern`, which raises a clear
"unavailable for this model" that the Component Agent handles like any other
tool error. GPT-2 / GPT-Neo / OPT / Qwen2.5 keep full eager support.

Only needed if you want a paid API model driving the agents instead:

```bash
pip install openai   # or anthropic
export OPENAI_API_KEY=sk-...
python3 main.py --backend openai
```

## What's implemented vs. not (from the full research proposal)

**Implemented:** the full agent hierarchy, all Tier N/L/C/V tools including
SAE decomposition, Prover-Skeptic-Judge verification, cost tracking,
content-addressed caching, Stage A (GPT-2 ground-truth validation), Stage B
(multi-behavior Layer Atlas), Stage C (cross-model layer diff with causal
transplant confirmation), and the Sec. 4.7 ablation studies (hierarchy,
verification, Network Analyst flagging strategy).

**Not yet implemented:** Stage B/C at the proposal's original scale (Gemma 2
9B, Mistral-7B/BioMistral-7B — the methodology is proven identical on small
stand-ins instead, deliberately, after a 3B-model download/thermal test on
this machine made the full-scale versions impractical here; a teammate with
GPU compute can now run the Stage-D half of this across a ≤7B size ladder
via `run_eval_matrix.py`); vision-model
support (ResNet/AlexNet-style CNN classifiers, and eventually object
detectors like YOLO — a fundamentally different output space that the
Skeptic/Judge causal-patching metric isn't built for yet); the
cost-mechanism ablation (with/without caching — already visible via
`cache_hit_rate` in every run, not yet a standalone comparison); a formal
backbone-comparison ablation (heuristic vs. `hf_local` — demonstrated
informally, not yet run as a scored comparison); and the full
**Synergy-Aware Circuit Discovery (S-ACDC)** loop — FDR-corrected
significance testing, split-half replication, iterate-to-convergence.
Its first piece, the **S-EAP** gradient screen and the `ablate_head_set`
coalition primitive, *is* now implemented and benchmarked (GPT-2 + Qwen,
mixed result — see [§ Status update](#status-update-2026-09-s-eap-implemented)
and `experiments/seap/`).
